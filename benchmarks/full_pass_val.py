"""Deterministic full-pass validation loss -- the lab's standard
adjudication evaluator (institutionalizing design-review F2: no verdict
from stochastic in-run estimates).

Protocol: non-overlapping 1024-token windows, stride = block, partial
tail dropped, fp32, every predicted token counted exactly once. One
number per checkpoint.

  uv run python benchmarks/full_pass_val.py \
      --experiment-dir experiments/12-budget-ablation/hybrid-50m \
      --val-bin experiments/05-data-frontier/data/val.bin
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BENCH_DIR, "results")


def result_name(exp_dir):
    parts = os.path.abspath(exp_dir).rstrip("/").split(os.sep)
    if "experiments" in parts:
        parts = parts[parts.index("experiments") + 1:]
    else:
        parts = parts[-1:]
    return "-".join(parts)


@torch.no_grad()
def full_pass_loss(model, tokens, device, block, micro=8):
    n_win = (len(tokens) - 1) // block
    total, count = 0.0, 0
    for w0 in range(0, n_win, micro):
        xs, ys = [], []
        for w in range(w0, min(w0 + micro, n_win)):
            k = w * block
            xs.append(torch.from_numpy(tokens[k:k + block].astype(np.int64)))
            ys.append(torch.from_numpy(tokens[k + 1:k + 1 + block].astype(np.int64)))
        x = torch.stack(xs).to(device)
        y = torch.stack(ys).to(device)
        logits, _ = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum")
        total += loss.item()
        count += y.numel()
        del x, y, logits, loss
        if device == "mps" and (w0 // micro) % 8 == 0:
            torch.mps.empty_cache()
    return total / count


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-dir", required=True)
    p.add_argument("--val-bin", required=True)
    # default 4: at block 1024, micro x block x 50257 fp32 logits must stay
    # under the documented ~1GB MPS envelope (8 x 1024 would be ~1.65GB)
    p.add_argument("--micro", type=int, default=4)
    args = p.parse_args()

    exp_dir = os.path.abspath(args.experiment_dir)
    sys.path.insert(0, exp_dir)
    for m in ("config", "model"):
        sys.modules.pop(m, None)
    from config import GPTConfig
    from model import GPT
    sys.path.pop(0)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ckpt = torch.load(os.path.join(exp_dir, "out", "ckpt.pt"), map_location=device)
    cfg = GPTConfig(**ckpt["model_cfg"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tokens = np.memmap(args.val_bin, dtype=np.uint16, mode="r")
    loss = full_pass_loss(model, tokens, device, cfg.block_size, args.micro)

    name = result_name(exp_dir)
    out = {"experiment": name, "full_pass_val_loss": round(loss, 4),
           "block_size": cfg.block_size, "val_bin": os.path.abspath(args.val_bin),
           "val_tokens": len(tokens),
           "protocol": "non-overlapping windows, stride=block, fp32, tail dropped"}
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{name}-fullpass.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"{name}: full-pass val loss {loss:.4f} (block {cfg.block_size}) -> {path}")


if __name__ == "__main__":
    main()
