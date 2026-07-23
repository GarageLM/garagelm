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
DATA_DIR = os.path.join(ROOT, "data")  # score-filtered pool built by data.py
OUT_DIR = os.path.join(ROOT, "out")

# Periodic resumable state (model+optimizer+RNG), separate from the final
# ckpt.pt. Added after the 05 gqa run hit the in-process MPS allocator
# pathology at ~iter 2500 (GPU-healthy system, 7x step slowdown, cured by
# process restart): --resume continues from the last saved state instead of
# losing hours. Loss curves across a resume are seamless because CPU+MPS RNG
# states are restored.
STATE_EVERY = 1000


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


def get_device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_data():
    with open(os.path.join(DATA_DIR, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    train_ids = np.memmap(os.path.join(DATA_DIR, "train.bin"), dtype=np.uint16, mode="r")
    val_ids = np.memmap(os.path.join(DATA_DIR, "val.bin"), dtype=np.uint16, mode="r")
    return train_ids, val_ids, meta["vocab_size"]


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, eval_iters, device):
    model.eval()
    out = {}
    for split, data in (("train", train_data), ("val", val_data)):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def get_lr(it, cfg: TrainConfig):
    if it < cfg.warmup_iters:
        return cfg.learning_rate * (it + 1) / cfg.warmup_iters
    decay_ratio = (it - cfg.warmup_iters) / max(1, cfg.max_iters - cfg.warmup_iters)
    decay_ratio = min(decay_ratio, 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


def build():
    """Shared setup used by both the throughput probe and the full run."""
    train_cfg = TrainConfig()
    torch.manual_seed(train_cfg.seed)

    device = get_device()
    train_data, val_data, vocab_size = load_data()
    model_cfg = GPTConfig(vocab_size=vocab_size)

    model = GPT(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay
    )
    return model, optimizer, model_cfg, train_cfg, train_data, val_data, device


def train_step(model, optimizer, model_cfg, train_cfg, train_data, device):
    """One optimizer step = grad_accum_steps micro-batches."""
    optimizer.zero_grad(set_to_none=True)
    for _ in range(train_cfg.grad_accum_steps):
        x, y = get_batch(train_data, model_cfg.block_size, train_cfg.micro_batch_size, device)
        _, loss = model(x, y)
        (loss / train_cfg.grad_accum_steps).backward()
        del x, y, loss
    torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
    optimizer.step()


def main(resume=False):
    model, optimizer, model_cfg, train_cfg, train_data, val_data, device = build()
    n_params = model.num_params()
    tokens_per_step = train_cfg.micro_batch_size * train_cfg.grad_accum_steps * model_cfg.block_size
    print(f"device={device} params={n_params:,} tokens/step={tokens_per_step:,}")

    os.makedirs(OUT_DIR, exist_ok=True)
    start_iter, elapsed_prior = 0, 0.0
    state_path = os.path.join(OUT_DIR, "state.pt")
    if resume and os.path.exists(state_path):
        state = torch.load(state_path, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["rng_cpu"].cpu())
        if state.get("rng_mps") is not None and device == "mps":
            torch.mps.set_rng_state(state["rng_mps"].cpu())  # loader maps it to device; API wants CPU bytes
        start_iter = state["iter"]
        elapsed_prior = state["elapsed"]
        print(f"resumed from iter {start_iter} (prior elapsed {elapsed_prior:.0f}s)", flush=True)

    t0 = time.time()
    losses = {"train": None, "val": None}

    for it in range(start_iter, train_cfg.max_iters):
        if it % train_cfg.eval_interval == 0:
            losses = estimate_loss(
                model, train_data, val_data, model_cfg.block_size,
                train_cfg.micro_batch_size, train_cfg.eval_iters, device,
            )
            print(f"iter {it}: train_loss={losses['train']:.4f} val_loss={losses['val']:.4f} "
                  f"elapsed={elapsed_prior + time.time() - t0:.1f}s", flush=True)

        lr = get_lr(it, train_cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        train_step(model, optimizer, model_cfg, train_cfg, train_data, device)

        if (it + 1) % STATE_EVERY == 0:
            save_state(model, optimizer, it + 1, elapsed_prior + time.time() - t0, model_cfg)

        if device == "mps" and it % 50 == 0:
            torch.mps.empty_cache()

    losses = estimate_loss(
        model, train_data, val_data, model_cfg.block_size,
        train_cfg.micro_batch_size, train_cfg.eval_iters, device,
    )
    wall_time = elapsed_prior + time.time() - t0
    print(f"final: train_loss={losses['train']:.4f} val_loss={losses['val']:.4f} wall_time={wall_time:.1f}s")

    ckpt_path = os.path.join(OUT_DIR, "ckpt.pt")
    torch.save({"model": model.state_dict(), "model_cfg": model_cfg.as_dict()}, ckpt_path)

    summary = {
        "device": device,
        "params": n_params,
        "final_train_loss": losses["train"],
        "final_val_loss": losses["val"],
        "wall_time_seconds": wall_time,
        "tokens_seen": tokens_per_step * train_cfg.max_iters,
        "model_config": model_cfg.as_dict(),
        "train_config": train_cfg.as_dict(),
    }
    with open(os.path.join(OUT_DIR, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="continue from out/state.pt if present")
    args = parser.parse_args()
    main(resume=args.resume)
