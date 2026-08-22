"""Q gate: is 4-bit close enough to 8-bit for the headline model?

  uv run python benchmarks/harness/quant_gate.py --q4 mlx-community/Qwen3.5-9B-4bit \
      --q8 mlx-community/Qwen3.5-9B-8bit [--nll-only]

Part 1 (in-process mlx, no server): teacher-forced NLL of both quantizations on
(a) 64 x 2048-token windows of the 05 val bin decoded to text and re-tokenized
with the model's tokenizer, and (b) 64 MATH-500 reference solutions. PASS if
NLL(q4) - NLL(q8) <= 0.02 nats on both. Part 2 (through the server; run it
twice, once per served quant, with runner.py on configs/math500-<quant>-k1.yaml
limited to the fixed 100-item slice) is adjudicated by `--compare a.json b.json`:
PASS if |acc delta| <= 3 points and exact McNemar p > 0.05. Fail -> ship q8.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

VAL_BIN = os.path.join(ROOT, "experiments", "05-data-frontier", "data", "val.bin")


def windows_text(n=64, block=2048, seed=1234):
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    toks = np.memmap(VAL_BIN, dtype=np.uint16, mode="r")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(toks) - block - 1, size=n)
    return [enc.decode([int(x) for x in toks[s:s + block]]) for s in starts]


def math500_solutions(n=64):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    return [f"Problem: {r['problem']}\n\nSolution: {r['solution']}" for r in list(ds)[:n]]


def nll_of(model_id, texts, max_tokens=2048):
    import mlx.core as mx
    from mlx_lm import load
    model, tok = load(model_id)
    total, count = 0.0, 0
    per_text = []
    for t in texts:
        ids = tok.encode(t)[:max_tokens]
        if len(ids) < 8:
            continue
        x = mx.array(ids)[None, :-1]
        y = mx.array(ids)[None, 1:]
        logits = model(x)
        lp = mx.log(mx.softmax(logits.astype(mx.float32), axis=-1))
        nll = -mx.take_along_axis(lp, y[..., None], axis=-1).squeeze(-1)
        s = float(nll.sum()); n = int(nll.size)
        per_text.append(s / n); total += s; count += n
        mx.eval(logits)
    del model
    mx.clear_cache()
    return total / max(count, 1), per_text


def compare(a_path, b_path):
    from benchmarks.harness.report import mcnemar_exact
    a, b = json.load(open(a_path)), json.load(open(b_path))
    ia, ib = a["per_item_cons_correct"], b["per_item_cons_correct"]
    common = sorted(set(ia) & set(ib))
    da = sum(1 for i in common if ia[i] and not ib[i])
    db = sum(1 for i in common if ib[i] and not ia[i])
    acc_a = sum(ia[i] for i in common) / len(common)
    acc_b = sum(ib[i] for i in common) / len(common)
    p = mcnemar_exact(da, db)
    out = {"n": len(common), "acc_a": acc_a, "acc_b": acc_b, "delta_pts": 100 * (acc_a - acc_b),
           "discordant": [da, db], "mcnemar_p": p,
           "pass": abs(acc_a - acc_b) <= 0.03 and p > 0.05}
    print(json.dumps(out, indent=1))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--q4", default="mlx-community/Qwen3.5-9B-4bit")
    p.add_argument("--q8", default="mlx-community/Qwen3.5-9B-8bit")
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--compare", nargs=2, metavar=("A_JSON", "B_JSON"))
    args = p.parse_args()
    if args.compare:
        compare(*args.compare)
        return
    texts = {"val_windows": windows_text(args.n), "math500_solutions": math500_solutions(args.n)}
    res = {"q4": args.q4, "q8": args.q8, "n": args.n, "date": time.strftime("%Y-%m-%d"), "nll": {}}
    for name, tx in texts.items():
        r = {}
        for tag, mid in (("q4", args.q4), ("q8", args.q8)):
            t0 = time.time()
            mean, per = nll_of(mid, tx)
            r[tag] = {"nll": mean, "per_text_mean": float(np.mean(per)), "seconds": time.time() - t0}
            print(f"  {name} {tag}: NLL {mean:.4f} ({time.time() - t0:.0f}s)", flush=True)
        r["delta_q4_minus_q8"] = r["q4"]["nll"] - r["q8"]["nll"]
        r["pass"] = r["delta_q4_minus_q8"] <= 0.02
        res["nll"][name] = r
    res["pass_nll"] = all(v["pass"] for v in res["nll"].values())
    short = args.q4.split("/")[-1].lower().replace("-4bit", "")
    out = os.path.join(ROOT, "benchmarks", "results", f"harness-quant-gate-{short}.json")
    json.dump(res, open(out, "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "nll"} | {"nll": {k: v["delta_q4_minus_q8"] for k, v in res["nll"].items()}}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
