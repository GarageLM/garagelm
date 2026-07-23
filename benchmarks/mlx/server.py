"""GarageLM local chat server: OpenAI-compatible API over the MLX port,
with the real rotating KV cache -- served by the machine that trained it.

  uv run python benchmarks/mlx/server.py                      # Metal, full speed
  uv run python benchmarks/mlx/server.py --cpu                # polite mode while training runs
  uv run python benchmarks/mlx/server.py --model-dir benchmarks/mlx/converted/09-flagship-2

Endpoints: POST /v1/chat/completions (stream and non-stream), GET /v1/models.
Any OpenAI-client chat UI pointed at http://localhost:8080/v1 works.

The chat template matches SFT training exactly (System:/User:/Assistant:
lines; <|endoftext|> token -- inserted as a TOKEN, never as text --
terminates assistant turns). Requests are serialized with a lock (MLX
inference is not concurrency-safe); each request does one prefill into a
fresh cache, then single-token decodes, which is exactly the regime the
RotatingCache supports.
"""

import argparse
import json
import os
import queue
import sys
import threading
import time
import uuid

import mlx.core as mx
import tiktoken
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

MLX_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, MLX_DIR)
from model import GPT  # noqa: E402

app = FastAPI(title="GarageLM server")
STATE = {"cfg": None, "enc": None, "name": "", "load_error": None}
# MLX streams are bound to the thread that creates them: the model must be
# loaded AND run in one dedicated engine thread (anything else eventually
# dies with "There is no Stream(cpu, 0) in current thread" when a request
# lands on a different worker thread). Handlers talk to the engine through
# this queue; per-request token queues carry results back. The single
# engine thread also serializes requests, so no lock is needed.
REQUESTS = queue.Queue()


def engine_loop(model_dir, dtype_name, use_cpu, ready):
    try:
        if use_cpu:
            mx.set_default_device(mx.cpu)
        with open(os.path.join(model_dir, "config.json")) as f:
            cfg = json.load(f)
        model = GPT(cfg)
        model.load_weights(os.path.join(model_dir, "weights.safetensors"))
        model.set_dtype(getattr(mx, dtype_name))
        mx.eval(model.parameters())
        STATE["cfg"] = cfg
    except Exception as e:
        STATE["load_error"] = e
        return
    finally:
        ready.set()

    eot = STATE["enc"].eot_token
    while True:
        prompt, max_tokens, temperature, top_k, out = REQUESTS.get()
        try:
            caches = model.make_cache()
            logits = model(mx.array([prompt]), caches)[:, -1, :]
            for _ in range(max_tokens):
                tok = sample_token(logits, temperature, top_k)
                if tok == eot:
                    break
                out.put(tok)
                logits = model(mx.array([[tok]]), caches)[:, -1, :]
        except Exception as e:
            out.put(e)
        finally:
            out.put(None)


def submit(prompt, max_tokens, temperature, top_k):
    """Enqueue a generation job; returns the queue of token ids
    (terminated by None, or an Exception then None)."""
    out = queue.Queue()
    REQUESTS.put((prompt, max_tokens, temperature, top_k, out))
    return out


def encode_conversation(enc, messages):
    toks = []
    for m in messages:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if role == "system":
            toks += enc.encode_ordinary(f"System: {content}\n")
        elif role == "user":
            toks += enc.encode_ordinary(f"User: {content}\n")
        elif role == "assistant":
            toks += enc.encode_ordinary(f"Assistant: {content}") + [enc.eot_token]
            toks += enc.encode_ordinary("\n")
    toks += enc.encode_ordinary("Assistant:")
    return toks


def sample_token(logits, temperature, top_k):
    logits = logits.astype(mx.float32)
    if temperature <= 0:
        return int(mx.argmax(logits, axis=-1).item())
    logits = logits / temperature
    if top_k:
        kth = mx.sort(logits, axis=-1)[:, -top_k]
        logits = mx.where(logits < kth[:, None], mx.array(-float("inf")), logits)
    return int(mx.random.categorical(logits).item())


def decode_prefix(enc, toks):
    """Decode a growing token list safely (byte-level BPE can split UTF-8;
    an incomplete trailing char is simply withheld until its bytes arrive)."""
    return enc.decode_bytes(toks).decode("utf-8", errors="ignore")


@app.get("/")
def ui():
    with open(os.path.join(MLX_DIR, "ui.html")) as f:
        return HTMLResponse(f.read())


@app.get("/v1/models")
def models():
    return {"object": "list",
            "data": [{"id": STATE["name"], "object": "model", "owned_by": "garagelm"}]}


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    enc, cfg = STATE["enc"], STATE["cfg"]
    messages = body.get("messages", [])
    temperature = float(body.get("temperature", 0.7))
    top_k = int(body.get("top_k", 50))
    max_tokens = int(body.get("max_tokens") or 256)
    stream = bool(body.get("stream", False))

    prompt = encode_conversation(enc, messages)
    budget = cfg["block_size"] - 8
    max_tokens = max(1, min(max_tokens, budget - min(len(prompt), budget - 32)))
    prompt = prompt[-(budget - max_tokens):]

    rid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    base = {"id": rid, "created": created, "model": STATE["name"]}

    if not stream:
        t0 = time.perf_counter()
        q = submit(prompt, max_tokens, temperature, top_k)
        out = []
        while True:
            item = q.get()
            if item is None:
                break
            if isinstance(item, Exception):
                return JSONResponse({"error": str(item)}, status_code=500)
            out.append(item)
        dt = time.perf_counter() - t0
        text = decode_prefix(enc, out).strip()
        finish = "stop" if len(out) < max_tokens else "length"
        return JSONResponse({**base, "object": "chat.completion",
            "choices": [{"index": 0, "finish_reason": finish,
                         "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(out),
                      "total_tokens": len(prompt) + len(out),
                      "tokens_per_second": round(len(out) / max(dt, 1e-6), 1)}})

    def sse():
        # this generator only drains the engine's queue -- pure Python
        # (tiktoken decode + json), safe on any threadpool thread
        q = submit(prompt, max_tokens, temperature, top_k)
        chunk = {**base, "object": "chat.completion.chunk"}
        yield "data: " + json.dumps({**chunk, "choices": [
            {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}) + "\n\n"
        toks, emitted, n = [], "", 0
        while True:
            item = q.get()
            if item is None:
                break
            if isinstance(item, Exception):
                yield f": error {item}\n\n"
                break
            toks.append(item)
            n += 1
            full = decode_prefix(enc, toks)
            delta, emitted = full[len(emitted):], full
            if delta:
                yield "data: " + json.dumps({**chunk, "choices": [
                    {"index": 0, "delta": {"content": delta}, "finish_reason": None}]}) + "\n\n"
        finish = "stop" if n < max_tokens else "length"
        yield "data: " + json.dumps({**chunk, "choices": [
            {"index": 0, "delta": {}, "finish_reason": finish}]}) + "\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default=os.path.join(MLX_DIR, "converted", "10-sft"))
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--cpu", action="store_true",
                   help="run on CPU (slow but leaves Metal free for training)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args()

    STATE.update(enc=tiktoken.get_encoding("gpt2"),
                 name=os.path.basename(args.model_dir.rstrip("/")))
    ready = threading.Event()
    threading.Thread(target=engine_loop, daemon=True,
                     args=(args.model_dir, args.dtype, args.cpu, ready)).start()
    ready.wait()
    if STATE["load_error"] is not None:
        raise SystemExit(f"model load failed: {STATE['load_error']}")
    print(f"GarageLM server: {STATE['name']} [{args.dtype}"
          f"{', cpu' if args.cpu else ', metal'}] on http://{args.host}:{args.port}/v1")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
