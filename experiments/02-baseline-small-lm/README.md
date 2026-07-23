# 02-baseline-small-lm

The control model every attention variant in `03-attention-variants/` gets
compared against. Modern decoder-only transformer — RoPE, RMSNorm, SwiGLU,
GQA — trained locally on a BPE-tokenized natural-language corpus, unlike
`01-baseline-smoke`'s toy char-level model. See `docs/literature/README.md`
entries 1-3 for the architecture background and `model.py` for the
implementation (`RMSNorm`, `SwiGLU`, `precompute_rope`/`apply_rope`,
`repeat_kv`/`GQAAttention`).

- `data.py` — downloads TinyStories' *valid* split (22.5MB, not the 2.2GB
  train split — too large to be worth it for this milestone), splits 95/5,
  encodes with `tiktoken`'s `gpt2` BPE (vocab_size=50257).
- `model.py` / `config.py` — the architecture and hyperparameters (below).
- `train.py` / `sample.py` — same structure as `01`'s.

## Dataset tradeoff (documented, not hidden)

Training loops over the ~5.26M-token TinyStories valid split multiple times
(3000 iters × 16 × 256 ≈ 12.3M tokens seen, ~2.3 epochs) rather than
requiring a fresh multi-GB download. The goal here is a fair, fixed-budget
architecture control for `03`'s comparisons, not a compute-optimal model —
this is nowhere near Chinchilla-optimal token counts, and that's an accepted
tradeoff for local, single-machine research.

## Architecture

| | |
|---|---|
| Params | **114,114,048** |
| Layers | 12 |
| Attention | GQA: 12 query heads, 4 kv heads (group size 3), RoPE (θ=10000) |
| n_embd | 768 |
| MLP | SwiGLU, hidden=2048 |
| Norm | RMSNorm |
| Vocab | 50257 (GPT-2 BPE via `tiktoken`) |
| Context | 256 |

(Planning-time back-of-envelope estimated ~101.5M; actual came out to 114.1M
— the estimate was arithmetic error, not a config change. Still comfortably
inside the 50–150M target.)

## What went wrong, and why the config differs from the first draft

The original plan called for `batch_size=32, block_size=512` (~4000 iters).
A throughput probe at that config hit a **pathological MPS allocator
regime**: 71.5s/iter and 63GB "allocated by driver" for a workload that
should need under 5GB, eventually OOM-ing. Root cause: `vocab_size=50257`
means the `lm_head` produces a `(batch × block_size × 50257)` logits tensor
every forward pass — at batch=32/block=512 that's ~3.3GB in float32 alone,
and MPS's caching allocator handled the resulting memory pressure badly.
Reducing to **`batch_size=16, block_size=256`** (a good fit for TinyStories
anyway — most stories are shorter than 256 BPE tokens, so 512 was diluting
each window with unrelated concatenated stories) measured a sane
848ms–1.08s/iter at ~12–15GB allocated, with memory plateauing (no leak).
`train.py` also periodically calls `torch.mps.empty_cache()` as a defensive
measure.

Separately, the actual training run took **~6 hours** against a ~54-minute
estimate from the clean throughput probe — not a bug, but resource
contention from another concurrent Claude Code session on the same machine
running an unrelated, CPU/GPU-heavy workload. Iteration speed varied by
roughly 8x across the run depending on what else was running. Worth knowing
before `03`'s variants: run this machine's other heavy workloads to
completion first, or expect multi-hour wall-clock times under contention.

## Result

| | |
|---|---|
| Final train loss | 1.6448 |
| Final val loss | 1.8665 |
| Wall-clock | 21,553.7s (~6.0h, under concurrent-session contention; ~54min estimated in isolation) |

Loss dropped cleanly from 10.96 (≈ln(50257), the correct random-init
baseline) to 1.64/1.87. Sample generation (`--prompt "Once upon a time"`)
produces genuinely coherent multi-sentence short-story text with consistent
characters — better than expected given the compute/data budget, likely
because TinyStories' simple vocabulary and structure is specifically
designed to be learnable by small models.

## Run it

```
uv run python experiments/02-baseline-small-lm/data.py
uv run python experiments/02-baseline-small-lm/train.py
uv run python experiments/02-baseline-small-lm/sample.py --prompt "Once upon a time"
```
