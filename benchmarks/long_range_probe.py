"""Per-position validation loss: the long-range diagnostic for milestone 05.

Loss at position t means "predict token t+1 given tokens 0..t". A model
whose attention can exploit context beyond the local window should keep
improving as positions grow past the window size; a purely local model's
curve should flatten once positions exceed what the window (plus depth-
stacked receptive field) covers. If the gqa/hybrid curves separate from
sliding-window past position ~64-128, long-range attention is earning
something on this corpus -- exactly what TinyStories never showed.

Windows are sampled with a fixed seed, so every model sees the identical
set of val windows and curves are directly comparable.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BENCH_DIR, "results")


def result_name(exp_dir):
    """'experiments/05-data-frontier/gqa' -> '05-data-frontier-gqa'."""
    parts = os.path.abspath(exp_dir).rstrip("/").split(os.sep)
    if "experiments" in parts:
        parts = parts[parts.index("experiments") + 1:]
    else:
        parts = parts[-1:]
    return "-".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--data-bin", default=None,
                        help="val token bin; default <experiment-dir>/../data/val.bin")
    parser.add_argument("--windows", type=int, default=256,
                        help="number of val windows (block_size+1 tokens each)")
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--bucket", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    exp_dir = os.path.abspath(args.experiment_dir)
    sys.path.insert(0, exp_dir)
    from config import GPTConfig
    from model import GPT

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ckpt = torch.load(os.path.join(exp_dir, "out", "ckpt.pt"), map_location=device)
    model_cfg = GPTConfig(**ckpt["model_cfg"])
    model = GPT(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    data_bin = args.data_bin or os.path.join(exp_dir, "..", "data", "val.bin")
    data = np.memmap(data_bin, dtype=np.uint16, mode="r")
    T = model_cfg.block_size

    g = torch.Generator().manual_seed(args.seed)
    starts = torch.randint(0, len(data) - T - 1, (args.windows,), generator=g)

    loss_sum = torch.zeros(T, dtype=torch.float64)
    n_windows = 0
    with torch.no_grad():
        for i in range(0, args.windows, args.micro_batch):
            batch_starts = starts[i:i + args.micro_batch]
            x = torch.stack([torch.from_numpy(data[s:s + T].astype(np.int64)) for s in batch_starts]).to(device)
            y = torch.stack([torch.from_numpy(data[s + 1:s + 1 + T].astype(np.int64)) for s in batch_starts]).to(device)
            logits, _ = model(x, y)
            per_tok = F.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1), reduction="none"
            ).view(len(batch_starts), T)
            loss_sum += per_tok.sum(dim=0).cpu().double()  # MPS has no float64; widen on CPU
            n_windows += len(batch_starts)
            del x, y, logits, per_tok
            if device == "mps" and (i // args.micro_batch) % 16 == 0:
                torch.mps.empty_cache()

    per_position = (loss_sum / n_windows).numpy()
    buckets = {}
    for b0 in range(0, T, args.bucket):
        buckets[f"{b0}-{b0 + args.bucket - 1}"] = float(per_position[b0:b0 + args.bucket].mean())

    name = result_name(exp_dir)
    summary = {
        "experiment": name,
        "block_size": T,
        "windows": n_windows,
        "seed": args.seed,
        "bucket_size": args.bucket,
        "mean_loss": float(per_position.mean()),
        "bucket_mean_loss": buckets,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"{name}-per-position.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"{name}: mean val loss {summary['mean_loss']:.4f} over {n_windows} windows")
    for k, v in buckets.items():
        print(f"  pos {k:>11}: {v:.4f}")
    print(f"written to {out_path}")


if __name__ == "__main__":
    main()
