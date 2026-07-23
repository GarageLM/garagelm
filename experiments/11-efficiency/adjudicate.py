"""F1/F2 adjudication for the edu-score4 lane (design-review-20260723.md).

Implements the protocol pinned in this milestone's README BEFORE the lane
finished: re-derive the contaminated val-doc mask, build the clean val
subset, and compute a deterministic full-pass loss for both final
checkpoints on both subsets. Prints the gate verdict.

Run only after edu-score4/train.py completes (needs its ckpt.pt and an
idle GPU).
"""

import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(ROOT, "..", ".."))
VAL_BIN = os.path.join(REPO, "experiments", "05-data-frontier", "data", "val.bin")
POOL_BIN = os.path.join(ROOT, "edu-score4", "data", "train.bin")
CONTROL_CKPT = os.path.join(REPO, "experiments", "05-data-frontier", "hybrid", "out", "ckpt.pt")
LANE_CKPT = os.path.join(ROOT, "edu-score4", "out", "ckpt.pt")
RESULTS = os.path.join(REPO, "benchmarks", "results", "11-efficiency-adjudication.json")

EOT = 50256
FWE_VAL_TOKENS = 1_680_000  # val.bin layout: FineWeb tail first, cosmo tail after
# Review census located the contamination cluster at pool token offsets
# 209,891,173-210,121,069; search a wide margin around it.
CLUSTER = (205_000_000, 215_000_000)
WINDOW = 96  # mid-doc window tokens, byte-exact match, per the review's census
GATE_MARGIN = 0.02


def split_docs(arr):
    ends = np.where(arr == EOT)[0]
    docs, start = [], 0
    for e in ends:
        docs.append((start, int(e) + 1))
        start = int(e) + 1
    if start < len(arr):
        docs.append((start, len(arr)))
    return docs


def contaminated_docs(val):
    pool = np.memmap(POOL_BIN, dtype=np.uint16, mode="r")
    region = pool[CLUSTER[0]:CLUSTER[1]].tobytes()
    bad = []
    for s, e in split_docs(val):
        if s >= FWE_VAL_TOKENS:
            break  # cosmo region: census-verified clean
        if e - s < WINDOW + 2:
            continue
        mid = s + (e - s - WINDOW) // 2
        needle = val[mid:mid + WINDOW].tobytes()
        if region.find(needle) != -1:
            bad.append((s, e))
    return bad


def build_subset(val, exclude):
    if not exclude:
        return np.asarray(val)
    keep = np.ones(len(val), dtype=bool)
    for s, e in exclude:
        keep[s:e] = False
    return np.asarray(val)[keep]


def load_model(ckpt_path, device):
    sys.path.insert(0, os.path.join(ROOT, "edu-score4"))  # model.py identical to control's
    for m in ("config", "model"):
        sys.modules.pop(m, None)
    from config import GPTConfig
    from model import GPT
    sys.path.pop(0)
    ckpt = torch.load(ckpt_path, map_location=device)
    model = GPT(GPTConfig(**ckpt["model_cfg"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@torch.no_grad()
def full_pass_loss(model, tokens, device, block=1024, micro=8):
    """Non-overlapping windows, stride `block`, partial tail dropped, fp32.
    Every predicted token counted exactly once."""
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
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    val = np.memmap(VAL_BIN, dtype=np.uint16, mode="r")

    bad = contaminated_docs(val)
    bad_tokens = sum(e - s for s, e in bad)
    print(f"contaminated FineWeb val docs: {len(bad)} ({bad_tokens:,} tokens, "
          f"{bad_tokens / len(val):.1%} of val)")
    assert 150 <= len(bad) <= 260, "census far from review's 200 -- investigate before adjudicating"

    raw = np.asarray(val)
    clean = build_subset(val, bad)
    print(f"raw val: {len(raw):,} tokens | clean val: {len(clean):,} tokens")

    out = {"contaminated_docs": len(bad), "contaminated_tokens": int(bad_tokens),
           "gate_margin": GATE_MARGIN, "protocol": "non-overlapping 1024, stride 1024, fp32"}
    for name, ckpt in (("control", CONTROL_CKPT), ("edu-score4", LANE_CKPT)):
        model = load_model(ckpt, device)
        out[name] = {"raw_val": round(full_pass_loss(model, raw, device), 4),
                     "clean_val": round(full_pass_loss(model, clean, device), 4)}
        print(f"{name}: raw {out[name]['raw_val']:.4f}  clean {out[name]['clean_val']:.4f}")
        del model
        if device == "mps":
            torch.mps.empty_cache()

    threshold = out["control"]["clean_val"] - GATE_MARGIN
    win = out["edu-score4"]["clean_val"] <= threshold
    out["gate_threshold_clean"] = round(threshold, 4)
    out["verdict"] = "PASS" if win else "FAIL"
    contamination_bias = out["edu-score4"]["raw_val"] - out["edu-score4"]["clean_val"]
    out["lane_raw_minus_clean"] = round(contamination_bias, 4)
    print(f"\ngate: edu-score4 clean {out['edu-score4']['clean_val']:.4f} "
          f"vs threshold {threshold:.4f} -> {out['verdict']}")
    print(f"contamination bias on the lane (raw - clean): {contamination_bias:+.4f}")

    with open(RESULTS, "w") as f:
        json.dump(out, f, indent=2)
    print(f"written to {RESULTS}")


if __name__ == "__main__":
    main()
