"""G0 throughput / memory probe against a running mlx_lm.server.

  uv run python benchmarks/harness/probe_throughput.py --model mlx-community/Qwen3.5-9B-4bit \
      --quant q4 [--concurrencies 1,4,8,16] [--max-tokens 2048] [--url http://127.0.0.1:8421/v1]

For each concurrency C: C concurrent requests over 12 fixed hard prompts,
thinking on. Reports aggregate tok/s (sum completion tokens / wall), per-stream
tok/s, TTFT (one streamed request at C=1), server RSS peak (1Hz), and the
planned-peak memory formula from the model's config.json. Writes
benchmarks/results/harness-probe-<model>-<quant>.json and prints the G0 verdict.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from benchmarks.harness.client import ChatClient          # noqa: E402
from benchmarks.harness.config import Sampler             # noqa: E402
from benchmarks.harness.runner import RssMonitor, server_pid  # noqa: E402

PROMPTS = [
    "Find the number of ordered pairs of integers (a, b) with 1 <= a, b <= 100 such that a^2 + b^2 is divisible by 7.",
    "Let ABC be a triangle with AB=13, BC=14, CA=15. Find the distance between the incenter and circumcenter.",
    "How many positive integers n <= 1000 have the property that n and n+1 have the same number of positive divisors?",
    "Find the remainder when 2^2024 + 3^2024 is divided by 1000.",
    "A fair coin is flipped 10 times. What is the probability that no two consecutive flips are both heads? Give a reduced fraction.",
    "Find all real x with x^4 - 5x^2 + 4 = 0 and justify each step.",
    "Let f(x) = x^3 - 3x + 1. Find the sum of the squares of its real roots.",
    "How many ways can 8 rooks be placed on an 8x8 board so that no two attack each other and none is on the main diagonal?",
    "Find the smallest positive integer n such that n! ends in exactly 20 zeros, or prove none exists.",
    "In how many ways can the letters of MISSISSIPPI be arranged so that no two I's are adjacent?",
    "Compute the sum from k=1 to 100 of floor(sqrt(k)).",
    "Find the largest prime factor of 2^16 + 1 and explain the method.",
]
SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."


def kv_bytes_per_token(cfg: dict) -> dict:
    cfg = cfg.get("text_config", cfg)          # Qwen3.5 nests the LM config
    L = cfg.get("num_hidden_layers") or cfg.get("n_layer")
    kvh = cfg.get("num_key_value_heads") or cfg.get("n_kv_head")
    hd = cfg.get("head_dim") or (cfg["hidden_size"] // cfg["num_attention_heads"])
    lt = cfg.get("layer_types")
    interval = cfg.get("full_attention_interval")
    if lt:
        n_full = sum(1 for x in lt if x == "full_attention")
    elif interval:
        n_full = L // interval
    else:
        n_full = L
    return {"layers": L, "full_attention_layers": n_full, "kv_heads": kvh, "head_dim": hd,
            "kv_bytes_per_token_bf16": 2 * n_full * kvh * hd * 2}


async def probe(url, model, concurrencies, max_tokens, sampler):
    client = ChatClient(url, concurrency=max(concurrencies))
    out = {"concurrency": {}}
    health = await client.health()
    out["served"] = [m.get("id") for m in health.get("data", [])]
    # TTFT / single-stream via one streamed request
    c1 = await client.complete([{"role": "user", "content": PROMPTS[0] + SUFFIX}], sampler,
                               max_tokens=512, stream=True)
    out["ttft_s_stream_c1"] = c1.ttft_s
    for C in concurrencies:
        mon = RssMonitor(server_pid()); mon.start()
        msgs = [[{"role": "user", "content": PROMPTS[i % len(PROMPTS)] + SUFFIX}] for i in range(C)]
        t0 = time.time()
        res = await asyncio.gather(*(client.complete(m, sampler, max_tokens=max_tokens,
                                                     timeout_s=3600.0) for m in msgs))
        wall = time.time() - t0
        mon.stop()
        toks = [r.completion_tokens or 0 for r in res]
        errs = [r.error for r in res if r.error]
        per_stream = [t / r.wall_s for t, r in zip(toks, res) if r.wall_s]
        row = {"wall_s": wall, "completion_tokens": toks, "aggregate_tok_s": sum(toks) / wall,
               "per_stream_tok_s_mean": statistics.fmean(per_stream) if per_stream else 0,
               "prompt_tokens": [r.prompt_tokens for r in res],
               "finish": [r.finish_reason for r in res], "errors": errs,
               "peak_rss_gb": mon.peak_bytes / 1e9 if mon.peak_bytes else None}
        out["concurrency"][str(C)] = row
        print(f"  C={C:2d}: aggregate {row['aggregate_tok_s']:.1f} tok/s | per-stream {row['per_stream_tok_s_mean']:.1f} "
              f"| wall {wall:.0f}s | peak RSS {row['peak_rss_gb']} GB | errors {len(errs)}", flush=True)
    await client.aclose()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--quant", default="q4")
    p.add_argument("--url", default="http://127.0.0.1:8421/v1")
    p.add_argument("--concurrencies", default="1,4,8,16")
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--planned-max-tokens", type=int, default=32768)
    p.add_argument("--planned-prompt", type=int, default=512)
    p.add_argument("--prompt-cache-gb", type=float, default=2.0)
    args = p.parse_args()
    concs = [int(x) for x in args.concurrencies.split(",")]
    sampler = Sampler(temperature=0.6, top_p=0.95, top_k=20, max_tokens=args.max_tokens, enable_thinking=True)
    t0 = time.time()
    res = asyncio.run(probe(args.url, args.model, concs, args.max_tokens, sampler))
    # planned peak from config.json
    from huggingface_hub import snapshot_download
    cfg_dir = snapshot_download(args.model, allow_patterns=["config.json"])
    cfg = json.load(open(os.path.join(cfg_dir, "config.json")))
    kv = kv_bytes_per_token(cfg)
    snap = snapshot_download(args.model)
    weights_gb = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(snap)
                     for f in fs if f.endswith(".safetensors")) / 1e9
    import subprocess
    contention = subprocess.run("ps aux | awk 'NR>1 && $3>50 {print $3, $11}' | grep -v mlx_lm.server",
                                shell=True, capture_output=True, text=True).stdout.strip().splitlines()
    planned = {}
    for C in concs:
        planned[str(C)] = weights_gb + C * (args.planned_prompt + args.planned_max_tokens) * kv["kv_bytes_per_token_bf16"] / 1e9 + args.prompt_cache_gb + 4.0
    res.update({"model": args.model, "quant": args.quant, "date": time.strftime("%Y-%m-%d"),
                "probe_max_tokens": args.max_tokens, "sampler": sampler.__dict__, "kv": kv,
                "weights_gb": weights_gb, "planned_peak_gb_at_32k": planned,
                "contention_at_probe": contention,
                "elapsed_s": time.time() - t0})
    # G0 verdict
    c8 = res["concurrency"].get("8") or res["concurrency"].get(str(max(concs)))
    agg8 = c8["aggregate_tok_s"] if c8 else 0
    rss = max((r["peak_rss_gb"] or 0) for r in res["concurrency"].values())
    verdict = {"aggregate_tok_s_at_c8": agg8, "gate_tok_s": 150, "pass_tok_s": agg8 >= 150,
               "measured_peak_rss_gb": rss, "gate_rss_gb": 42, "pass_rss": rss <= 42,
               "planned_peak_gb_c8_32k": planned.get("8"), "gate_planned_gb": 40,
               "pass_planned": (planned.get("8") or 0) <= 40}
    verdict["valid_for_g0"] = not contention
    verdict["note"] = ("contended probe: other processes >50% CPU were running; throughput is a lower "
                       "bound and the G0 verdict is not pinned from it" if contention else "idle machine")
    res["g0"] = verdict
    short = args.model.split("/")[-1].lower()
    out = os.path.join(ROOT, "benchmarks", "results", f"harness-probe-{short}.json")
    json.dump(res, open(out, "w"), indent=2)
    print(json.dumps(verdict, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
