"""Pre-launch checks for a harness run (analog of the 15-step loss probe).

  uv run python benchmarks/harness/preflight.py --config experiments/15-harness/configs/<run>.yaml

1. contention check; 2. server health + served model matches config; 3. grader
self-test; 4. 5-item smoke at k=1 with a small max_tokens (writes a -smoke
results JSON, never a headline); 5. thinking flag echo (reasoning present).
Exit code != 0 on any failure.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from benchmarks.harness.client import ChatClient      # noqa: E402
from benchmarks.harness.config import load_config     # noqa: E402
from benchmarks.harness import runner                  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--smoke-items", type=int, default=5)
    p.add_argument("--smoke-max-tokens", type=int, default=8192)
    p.add_argument("--smoke-task", default="math500",
                   help="task for the smoke (default math500: easy, short traces; AIME items "
                        "think for 10-30k tokens and only test truncation)")
    p.add_argument("--min-correct", type=int, default=3)
    p.add_argument("--smoke-thinking", action="store_true",
                   help="run the smoke with thinking ON (default off: Qwen3.5-9B thinks past 8k tokens "
                        "even on Level 1-2 MATH-500 items, so a thinking smoke only measures truncation; "
                        "the thinking path itself is covered by the echo check above)")
    args = p.parse_args()
    cfg = load_config(args.config)
    fails = []

    busy = subprocess.run("ps aux | awk 'NR>1 && $3>50 {print $3, $11}' | grep -v mlx_lm.server",
                          shell=True, capture_output=True, text=True).stdout.strip()
    print(f"[preflight] contention: {'none' if not busy else busy}")
    if subprocess.run(["pgrep", "-f", "train.py"], capture_output=True).returncode == 0:
        fails.append("a train.py is running")

    async def health():
        c = ChatClient(cfg.server_url, 1)
        try:
            h = await c.health()
            served = [m.get("id") for m in h.get("data", [])]
            print(f"[preflight] server ok, models {served}")
            if not any(cfg.model in (s or "") or (s or "") in cfg.model for s in served):
                print(f"[preflight] WARNING served model does not match config.model {cfg.model}")
            r = await c.complete([{"role": "user", "content": "Say OK."}], cfg.sampler, max_tokens=64)
            print(f"[preflight] thinking echo: reasoning_chars={len(r.reasoning)} finish={r.finish_reason} "
                  f"usage={r.prompt_tokens}/{r.completion_tokens}")
            if cfg.sampler.enable_thinking and not r.reasoning:
                fails.append("thinking requested but no reasoning returned")
            if r.completion_tokens is None:
                fails.append("server returned no usage.completion_tokens")
        finally:
            await c.aclose()
    try:
        asyncio.run(health())
    except Exception as e:
        fails.append(f"server health failed: {e}")

    st = subprocess.run([sys.executable, os.path.join(HERE, "selftest.py")], capture_output=True, text=True)
    print(f"[preflight] selftest: {'ok' if st.returncode == 0 else 'FAIL'}")
    if st.returncode != 0:
        print(st.stdout[-2000:])
        fails.append("selftest failed")

    if not fails:
        out_dir = os.path.join(ROOT, "experiments", "15-harness", "runs", f"smoke-{cfg.run_id}")
        smoke_cfg = cfg.__class__(**{**cfg.to_dict(), "run_id": f"smoke-{cfg.run_id}", "k": 1,
                                     "task": args.smoke_task,
                                     "task_args": ({"limit": args.smoke_items, "levels": [1, 2]}
                                                   if args.smoke_task == "math500" else cfg.task_args),
                                     "sampler": (cfg.sampler if args.smoke_thinking else
                                                 cfg.sampler.__class__(**{**cfg.sampler.__dict__,
                                                                          "enable_thinking": False}))})
        agg = asyncio.run(runner.run(smoke_cfg, out_dir, args.smoke_items, True, args.smoke_max_tokens))
        path = runner.write_results(agg, smoke_cfg, None, force=True, smoke=True)
        n_ok = int(round(agg["accuracy"] * agg["n_items"]))
        print(f"[preflight] smoke: {n_ok}/{agg['n_items']} correct, trunc {agg['truncation_rate']:.2f}, "
              f"tokens/item {agg['completion_tokens_per_item_mean']:.0f}, agg {agg['aggregate_tok_s']:.0f} tok/s -> {path}")
        if n_ok < args.min_correct:
            fails.append(f"smoke below min-correct ({n_ok} < {args.min_correct})")
        if agg["n_errors"]:
            fails.append(f"{agg['n_errors']} request errors in smoke")

    print(f"[preflight] {'PASS' if not fails else 'FAIL: ' + '; '.join(fails)}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
