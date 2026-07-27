"""DPO trainer: policy vs frozen reference, response-masked logprobs,
standard sigmoid DPO loss.

  uv run python experiments/12-post-training/dpo/train.py --sanity   # overfit 10 pairs (gate)
  uv run python experiments/12-post-training/dpo/train.py --resume   # full run

Init: policy AND reference both load the Stage-1 SFT checkpoint
(--init-ckpt). The reference never updates. Loss per pair:
  -log sigmoid( beta * ((pi_c - pi_r) - (ref_c - ref_r)) )
where each term is the summed response-token logprob of that side.
Metrics: DPO loss, implicit-reward margin, preference accuracy.
Resume machinery follows the lab standard (state every 200 steps).
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from model import GPT
from config import GPTConfig

ROOT = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "out")
DEFAULT_INIT = os.path.join(ROOT, "..", "sft-full", "out", "ckpt.pt")
STATE_EVERY = 200

BETA = 0.1
LR = 1e-6
WARMUP = 50
MICRO_PAIRS = 1   # sequences per forward = 2*MICRO_PAIRS (memory envelope)
ACCUM = 16        # 16 pairs per optimizer step (unchanged effective batch)
SEED = 1337


def get_device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


class PairData:
    def __init__(self):
        self.stream = np.memmap(os.path.join(DATA_DIR, "pairs.bin"),
                                dtype=np.uint16, mode="r")
        z = np.load(os.path.join(DATA_DIR, "index.npz"))
        self.idx = {k: z[k] for k in
                    ("c_start", "c_len", "c_resp", "r_start", "r_len", "r_resp")}
        n_val = int(z["n_val"])
        self.n_total = len(self.idx["c_start"])
        self.train_ids = np.arange(0, self.n_total - n_val)
        self.val_ids = np.arange(self.n_total - n_val, self.n_total)

    def fetch(self, pair_ids):
        """Returns padded (x, resp_mask, lengths) for 2*len(pair_ids)
        sequences ordered [c0, r0, c1, r1, ...]."""
        seqs, resps = [], []
        for i in pair_ids:
            for pre in ("c", "r"):
                s = int(self.idx[pre + "_start"][i])
                l = int(self.idx[pre + "_len"][i])
                seqs.append(self.stream[s:s + l].astype(np.int64))
                resps.append(int(self.idx[pre + "_resp"][i]))
        # bucket lengths to multiples of 128: bounded shape diversity keeps
        # the MPS allocator from fragmenting across thousands of unique
        # tensor shapes (the OOM cause on the first launch)
        L = max(len(s) for s in seqs)
        L = ((L + 127) // 128) * 128
        x = np.zeros((len(seqs), L), dtype=np.int64)
        mask = np.zeros((len(seqs), L), dtype=np.float32)
        for j, (s, r) in enumerate(zip(seqs, resps)):
            x[j, :len(s)] = s
            # loss on predicting tokens [r, len) -> positions r-1 .. len-2 of y
            mask[j, r:len(s)] = 1.0
        return torch.from_numpy(x), torch.from_numpy(mask)


def response_logprobs(model, x, resp_mask):
    """Summed logprob of response tokens for each sequence.

    gather + logsumexp instead of materializing a full-vocab log_softmax:
    halves the number of retained (B, L, 50257) tensors on the policy
    side, which is what blew the memory envelope at probe time."""
    logits, _ = model(x[:, :-1])
    logits = logits.float()
    tgt = x[:, 1:]
    picked = logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)   # (B, L-1)
    lse = torch.logsumexp(logits, dim=-1)                        # (B, L-1)
    m = resp_mask[:, 1:]
    return ((picked - lse) * m).sum(dim=1)


def dpo_step_stats(policy, ref, x, mask, device):
    x = x.to(device)
    mask = mask.to(device)
    with torch.no_grad():
        ref_lp = response_logprobs(ref, x, mask)
    pol_lp = response_logprobs(policy, x, mask)
    pc, pr = pol_lp[0::2], pol_lp[1::2]
    rc, rr = ref_lp[0::2], ref_lp[1::2]
    logits = BETA * ((pc - pr) - (rc - rr))
    loss = -F.logsigmoid(logits).mean()
    margin = (BETA * ((pc - rc) - (pr - rr))).mean()
    acc = (logits > 0).float().mean()
    return loss, margin.detach(), acc.detach()


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = GPT(GPTConfig(**ckpt["model_cfg"])).to(device)
    model.load_state_dict(ckpt["model"])
    return model


@torch.no_grad()
def pref_accuracy(policy, ref, data, ids, device, batch_pairs=4):
    policy.eval()
    hits, n = 0, 0
    for k in range(0, len(ids), batch_pairs):
        x, mask = data.fetch(ids[k:k + batch_pairs])
        x, mask = x.to(device), mask.to(device)
        pol = response_logprobs(policy, x, mask)
        rf = response_logprobs(ref, x, mask)
        logits = BETA * ((pol[0::2] - pol[1::2]) - (rf[0::2] - rf[1::2]))
        hits += (logits > 0).sum().item()
        n += len(logits)
        if device == "mps" and (k // batch_pairs) % 16 == 0:
            torch.mps.empty_cache()
    policy.train()
    return hits / max(n, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init-ckpt", default=DEFAULT_INIT)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--sanity", action="store_true",
                   help="overfit 10 pairs: loss < 0.1 and rising margin, then exit")
    p.add_argument("--epochs", type=float, default=1.0)
    args = p.parse_args()

    torch.manual_seed(SEED)
    device = get_device()
    data = PairData()
    policy = load_model(args.init_ckpt, device)
    ref = load_model(args.init_ckpt, device)
    ref.eval()
    for q in ref.parameters():
        q.requires_grad_(False)
    policy.train()
    opt = torch.optim.AdamW(policy.parameters(), lr=LR, weight_decay=0.0)

    if args.sanity:
        # Pre-registered gate (README): overfit 10 pairs, loss < 0.1,
        # per-cycle margin strictly increasing. Cycles all 10 pairs in
        # MICRO_PAIRS chunks (4 seqs/forward stays inside the MPS logits
        # envelope). SANITY_LR/CYCLES are calibration knobs, pinned here
        # before results per the design review.
        SANITY_LR, CYCLES = 5e-6, 12
        for g in opt.param_groups:
            g["lr"] = SANITY_LR
        ids = data.train_ids[:10]
        chunks = [ids[i:i + MICRO_PAIRS] for i in range(0, len(ids), MICRO_PAIRS)]
        cycle_margins, cycle_losses = [], []
        for cyc in range(CYCLES):
            tot_l = tot_m = 0.0
            for ch in chunks:
                opt.zero_grad(set_to_none=True)
                x, mask = data.fetch(ch)
                loss, margin, acc = dpo_step_stats(policy, ref, x, mask, device)
                loss.backward()
                opt.step()
                tot_l += loss.item() / len(chunks)
                tot_m += margin.item() / len(chunks)
            cycle_margins.append(tot_m)
            cycle_losses.append(tot_l)
            print(f"sanity cycle {cyc}: loss={tot_l:.4f} margin={tot_m:+.4f}",
                  flush=True)
            if device == "mps":
                torch.mps.empty_cache()
        increasing = all(b > a for a, b in zip(cycle_margins, cycle_margins[1:]))
        ok = cycle_losses[-1] < 0.1 and increasing
        print(f"SANITY {'OK' if ok else 'FAIL'}: final cycle loss "
              f"{cycle_losses[-1]:.4f} (<0.1 required), margins strictly "
              f"increasing: {increasing}")
        return

    rng = np.random.default_rng(SEED)
    order = rng.permutation(data.train_ids)
    n_steps = int(len(order) * args.epochs) // (MICRO_PAIRS * ACCUM)
    print(f"device={device} pairs={len(order):,} steps={n_steps:,} "
          f"({MICRO_PAIRS * ACCUM} pairs/step)", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    start_step, elapsed_prior = 0, 0.0
    state_path = os.path.join(OUT_DIR, "state.pt")
    if args.resume and os.path.exists(state_path):
        st = torch.load(state_path, map_location=device)
        policy.load_state_dict(st["model"])
        opt.load_state_dict(st["optimizer"])
        torch.set_rng_state(st["rng_cpu"].cpu())
        start_step, elapsed_prior = st["step"], st["elapsed"]
        print(f"resumed from step {start_step}", flush=True)

    t0 = time.time()
    cursor = start_step * MICRO_PAIRS * ACCUM
    for step in range(start_step, n_steps):
        lr = LR * min(1.0, (step + 1) / WARMUP) * \
             (0.5 * (1 + math.cos(math.pi * min(step / max(n_steps, 1), 1.0))) * 0.9 + 0.1)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        tot_loss = tot_margin = tot_acc = 0.0
        for a in range(ACCUM):
            if cursor + MICRO_PAIRS > len(order):
                rng2 = np.random.default_rng(SEED + 1 + cursor)
                order = rng2.permutation(data.train_ids)
                cursor = 0
            ids = order[cursor:cursor + MICRO_PAIRS]
            cursor += MICRO_PAIRS
            x, mask = data.fetch(ids)
            loss, margin, acc = dpo_step_stats(policy, ref, x, mask, device)
            (loss / ACCUM).backward()
            tot_loss += loss.item() / ACCUM
            tot_margin += margin.item() / ACCUM
            tot_acc += acc.item() / ACCUM
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

        if step % 50 == 0:
            print(f"step {step}: loss={tot_loss:.4f} margin={tot_margin:+.4f} "
                  f"acc={tot_acc:.2f} elapsed={elapsed_prior + time.time() - t0:.0f}s",
                  flush=True)
        if (step + 1) % STATE_EVERY == 0:
            torch.save({"model": policy.state_dict(), "optimizer": opt.state_dict(),
                        "step": step + 1, "elapsed": elapsed_prior + time.time() - t0,
                        "rng_cpu": torch.get_rng_state()},
                       state_path + ".tmp")
            os.replace(state_path + ".tmp", state_path)
        if device == "mps" and step % 2 == 0:
            torch.mps.empty_cache()

    acc = pref_accuracy(policy, ref, data, data.val_ids, device)
    print(f"held-out preference accuracy: {acc:.3f} (gate G3: > 0.60)")
    ckpt_cfg = torch.load(args.init_ckpt, map_location="cpu")["model_cfg"]
    torch.save({"model": policy.state_dict(), "model_cfg": ckpt_cfg},
               os.path.join(OUT_DIR, "ckpt.pt"))
    with open(os.path.join(OUT_DIR, "run_summary.json"), "w") as f:
        json.dump({"pref_accuracy": acc, "beta": BETA, "lr": LR,
                   "steps": n_steps, "pairs_per_step": MICRO_PAIRS * ACCUM,
                   "init_ckpt": os.path.abspath(args.init_ckpt)}, f, indent=2)
    print("checkpoint saved")


if __name__ == "__main__":
    main()
