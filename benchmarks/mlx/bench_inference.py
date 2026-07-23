"""MLX inference benchmark: the efficiency half of this project's three-way
tradeoff, measured for a converted checkpoint (see convert.py).

Per prompt length: TTFT (prefill + first-token), prefill tok/s, decode
tok/s, and KV-cache bytes after prefill and after decode -- measured from
the live cache arrays, not inferred. Plus peak Metal memory and, with
--val-bin, validation loss (so 4-bit quantization quality deltas are
quantified, not guessed).

Run this on an idle machine -- it competes with any MPS training run.

  uv run python benchmarks/mlx/bench_inference.py \
      --converted benchmarks/mlx/converted/04-scaling-checks-gqa \
      --val-bin experiments/04-scaling-checks/data/val.bin
  # add --bits 4 for the quantized variant
"""

import argparse
import json
import os
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from model import GPT

MLX_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(os.path.dirname(MLX_DIR), "results")


def peak_memory_gb():
    try:
        return mx.get_peak_memory() / 1e9
    except AttributeError:
        return mx.metal.get_peak_memory() / 1e9


def load_model(converted_dir, dtype, bits):
    with open(os.path.join(converted_dir, "config.json")) as f:
        cfg = json.load(f)
    model = GPT(cfg)
    model.load_weights(os.path.join(converted_dir, "weights.safetensors"))
    model.set_dtype(getattr(mx, dtype))
    if bits:
        nn.quantize(model, group_size=64, bits=bits)
    mx.eval(model.parameters())
    return model, cfg


def bench_lengths(model, cfg, prompt_lens, decode_tokens, seed=1234):
    rng = np.random.default_rng(seed)
    rows = []
    for P in prompt_lens:
        D = min(decode_tokens, cfg["block_size"] - P)
        if P >= cfg["block_size"] or D <= 0:
            print(f"skipping P={P}: exceeds block_size={cfg['block_size']}")
            continue
        prompt = mx.array(rng.integers(0, cfg["vocab_size"], size=(1, P)).astype(np.int32))

        # warmup (kernel compilation) with a fresh cache, then measure
        for measured in (False, True):
            caches = model.make_cache()
            t0 = time.perf_counter()
            logits = model(prompt, caches)
            tok = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(tok)
            ttft = time.perf_counter() - t0
            kv_after_prefill = model.kv_bytes(caches)

            t0 = time.perf_counter()
            for _ in range(D):
                logits = model(tok[:, None], caches)
                tok = mx.argmax(logits[:, -1, :], axis=-1)
                mx.eval(tok)
            decode_s = time.perf_counter() - t0
            kv_after_decode = model.kv_bytes(caches)

        rows.append({
            "prompt_len": P,
            "decode_tokens": D,
            "ttft_s": round(ttft, 4),
            "prefill_tok_per_s": round(P / ttft, 1),
            "decode_tok_per_s": round(D / decode_s, 1),
            "kv_mb_after_prefill": round(kv_after_prefill / 1e6, 2),
            "kv_mb_after_decode": round(kv_after_decode / 1e6, 2),
        })
        print(rows[-1], flush=True)
    return rows


def val_loss(model, cfg, val_bin, n_windows=32, seed=42):
    data = np.memmap(val_bin, dtype=np.uint16, mode="r")
    T = cfg["block_size"]
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(data) - T - 1, size=n_windows)
    total, count = 0.0, 0
    for s in starts:
        x = mx.array(data[s:s + T].astype(np.int32))[None, :]
        y = mx.array(data[s + 1:s + 1 + T].astype(np.int32))[None, :]
        caches = model.make_cache()
        logits = model(x, caches)
        losses = nn.losses.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        total += float(mx.mean(losses))
        count += 1
    return total / count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--converted", required=True, help="dir from convert.py")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--bits", type=int, default=None, choices=[4, 8],
                        help="quantize with mlx.nn.quantize(group_size=64)")
    parser.add_argument("--prompt-lens", default="64,256,512")
    parser.add_argument("--decode-tokens", type=int, default=256)
    parser.add_argument("--val-bin", default=None, help="val.bin for quality check")
    parser.add_argument("--val-windows", type=int, default=32)
    args = parser.parse_args()

    converted = os.path.abspath(args.converted)
    name = os.path.basename(converted.rstrip("/"))
    tag = f"q{args.bits}" if args.bits else args.dtype
    model, cfg = load_model(converted, args.dtype, args.bits)

    n_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"{name} [{tag}]: block_size={cfg['block_size']} "
          f"window={cfg.get('window')} global_every={cfg.get('global_every')}")

    prompt_lens = [int(p) for p in args.prompt_lens.split(",")]
    rows = bench_lengths(model, cfg, prompt_lens, args.decode_tokens)

    summary = {
        "model": name,
        "tag": tag,
        "dtype": args.dtype,
        "quant_bits": args.bits,
        "params": int(n_params),
        "block_size": cfg["block_size"],
        "window": cfg.get("window"),
        "global_every": cfg.get("global_every"),
        "peak_memory_gb": round(peak_memory_gb(), 2),
        "lengths": rows,
    }
    if args.val_bin:
        summary["val_loss"] = round(val_loss(model, cfg, args.val_bin, args.val_windows), 4)
        print(f"val_loss[{tag}] = {summary['val_loss']}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"mlx-{name}-{tag}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"written to {out_path}")


if __name__ == "__main__":
    main()
