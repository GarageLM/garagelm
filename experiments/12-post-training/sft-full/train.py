"""SFT for the flagship: masked cross-entropy over SmolTalk conversations.

Differences from the pretraining trainer (09):
- Initializes from a pretrained checkpoint (--init-ckpt, default the 09
  flagship) instead of random init.
- get_batch also returns the assistant-only loss mask; the loss is
  sum(CE * mask) / sum(mask) -- prompts and user turns are context only.
- Separate small val split (500 held-out conversations), same masked loss.

Resume machinery (state every 500 steps, --resume) inherited from 05/07.
"""

import argparse
import json
import math
import os
import pickle
import time

import numpy as np
import torch

from config import GPTConfig, TrainConfig
from model import GPT

ROOT = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "out")
DEFAULT_INIT = os.path.join(ROOT, "..", "..", "09-flagship-2", "out", "ckpt.pt")
STATE_EVERY = 500


def get_device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_data():
    with open(os.path.join(DATA_DIR, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    def mm(name, dtype):
        return np.memmap(os.path.join(DATA_DIR, name), dtype=dtype, mode="r")
    return (mm("tokens.bin", np.uint16), mm("mask.bin", np.uint8),
            mm("val_tokens.bin", np.uint16), mm("val_mask.bin", np.uint8),
            meta["vocab_size"])


def get_batch(tokens, mask, block_size, batch_size, device):
    ix = torch.randint(len(tokens) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(tokens[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(tokens[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    m = torch.stack([torch.from_numpy(mask[i + 1:i + 1 + block_size].astype(np.float32)) for i in ix])
    return x.to(device), y.to(device), m.to(device)


def masked_loss(logits, y, m):
    ce = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), y.view(-1), reduction="none"
    ).view_as(y.float())
    return (ce * m).sum() / m.sum().clamp(min=1.0)


@torch.no_grad()
def estimate_loss(model, splits, block_size, batch_size, eval_iters, device):
    model.eval()
    out = {}
    for name, (tokens, mask) in splits.items():
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y, m = get_batch(tokens, mask, block_size, batch_size, device)
            logits, _ = model(x)
            losses[k] = masked_loss(logits, y, m).item()
        out[name] = losses.mean().item()
    model.train()
    return out


def get_lr(it, cfg: TrainConfig):
    if it < cfg.warmup_iters:
        return cfg.learning_rate * (it + 1) / cfg.warmup_iters
    decay = min((it - cfg.warmup_iters) / max(1, cfg.max_iters - cfg.warmup_iters), 1.0)
    return cfg.min_lr + 0.5 * (1.0 + math.cos(math.pi * decay)) * (cfg.learning_rate - cfg.min_lr)


def build(init_ckpt):
    train_cfg = TrainConfig()
    torch.manual_seed(train_cfg.seed)
    device = get_device()
    tokens, mask, vtokens, vmask, vocab_size = load_data()
    model_cfg = GPTConfig(vocab_size=vocab_size)

    model = GPT(model_cfg).to(device)
    ckpt = torch.load(init_ckpt, map_location=device)
    assert ckpt["model_cfg"] == model_cfg.as_dict(), \
        f"init checkpoint config mismatch: {ckpt['model_cfg']} vs {model_cfg.as_dict()}"
    model.load_state_dict(ckpt["model"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay
    )
    splits = {"train": (tokens, mask), "val": (vtokens, vmask)}
    return model, optimizer, model_cfg, train_cfg, splits, device


def train_step(model, optimizer, model_cfg, train_cfg, splits, device):
    optimizer.zero_grad(set_to_none=True)
    tokens, mask = splits["train"]
    for _ in range(train_cfg.grad_accum_steps):
        x, y, m = get_batch(tokens, mask, model_cfg.block_size, train_cfg.micro_batch_size, device)
        logits, _ = model(x)
        loss = masked_loss(logits, y, m)
        (loss / train_cfg.grad_accum_steps).backward()
        del x, y, m, logits, loss
    torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
    optimizer.step()


def save_state(model, optimizer, next_iter, elapsed, model_cfg):
    tmp = os.path.join(OUT_DIR, "state.pt.tmp")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iter": next_iter,
        "elapsed": elapsed,
        "rng_cpu": torch.get_rng_state(),
        "rng_mps": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
        "model_cfg": model_cfg.as_dict(),
    }, tmp)
    os.replace(tmp, os.path.join(OUT_DIR, "state.pt"))


def main(resume=False, init_ckpt=DEFAULT_INIT):
    model, optimizer, model_cfg, train_cfg, splits, device = build(init_ckpt)
    print(f"device={device} params={model.num_params():,} init={os.path.abspath(init_ckpt)}")

    os.makedirs(OUT_DIR, exist_ok=True)
    start_iter, elapsed_prior = 0, 0.0
    state_path = os.path.join(OUT_DIR, "state.pt")
    if resume and os.path.exists(state_path):
        state = torch.load(state_path, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["rng_cpu"].cpu())
        if state.get("rng_mps") is not None and device == "mps":
            torch.mps.set_rng_state(state["rng_mps"].cpu())
        start_iter, elapsed_prior = state["iter"], state["elapsed"]
        print(f"resumed from iter {start_iter}", flush=True)

    t0 = time.time()
    for it in range(start_iter, train_cfg.max_iters):
        if it % train_cfg.eval_interval == 0:
            losses = estimate_loss(model, splits, model_cfg.block_size,
                                   train_cfg.micro_batch_size, train_cfg.eval_iters, device)
            print(f"iter {it}: train_loss={losses['train']:.4f} val_loss={losses['val']:.4f} "
                  f"elapsed={elapsed_prior + time.time() - t0:.1f}s", flush=True)
        lr = get_lr(it, train_cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        train_step(model, optimizer, model_cfg, train_cfg, splits, device)
        if (it + 1) % STATE_EVERY == 0:
            save_state(model, optimizer, it + 1, elapsed_prior + time.time() - t0, model_cfg)
        if device == "mps" and it % 50 == 0:
            torch.mps.empty_cache()

    losses = estimate_loss(model, splits, model_cfg.block_size,
                           train_cfg.micro_batch_size, train_cfg.eval_iters, device)
    wall = elapsed_prior + time.time() - t0
    print(f"final: train_loss={losses['train']:.4f} val_loss={losses['val']:.4f} wall_time={wall:.1f}s")

    torch.save({"model": model.state_dict(), "model_cfg": model_cfg.as_dict()},
               os.path.join(OUT_DIR, "ckpt.pt"))
    with open(os.path.join(OUT_DIR, "run_summary.json"), "w") as f:
        json.dump({"device": device, "params": model.num_params(),
                   "final_train_loss": losses["train"], "final_val_loss": losses["val"],
                   "wall_time_seconds": wall, "init_ckpt": os.path.abspath(init_ckpt),
                   "model_config": model_cfg.as_dict(), "train_config": train_cfg.as_dict()}, f, indent=2)
    print("checkpoint saved")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--resume", action="store_true")
    p.add_argument("--init-ckpt", default=DEFAULT_INIT)
    args = p.parse_args()
    main(resume=args.resume, init_ckpt=args.init_ckpt)
