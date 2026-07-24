# Experiments

Milestone-based training/eval harness for prototyping new attention variants,
run entirely locally (see `CLAUDE.md` for hardware/framework decisions).
Milestones 00-04 are complete — each directory's README has its results.
The current phase (05-07) moves from TinyStories to a refined corpus and
adds the inference-efficiency benchmark; the honest yardstick is the
capability-per-FLOP frontier against published same-size models, stated in
`05-data-frontier/README.md`.

## Milestones

- **`00-setup/`** — environment/dependency notes: PyTorch+MPS as the primary
  framework, MLX for inference-latency measurement, Python pinned to 3.11/3.12
  via `uv` rather than system Python. No training code here, just the
  environment.

- **`01-baseline-smoke/`** — a tiny (~1–10M param) nanoGPT-style char-level
  model trained on something small (Shakespeare or TinyStories). The point is
  to prove the *full* local pipeline works end-to-end — data prep, tokenizer,
  training loop, sampling, eval loop — cheaply, before investing in anything
  bigger. This should train in minutes on the M4 Pro.

- **`02-baseline-small-lm/`** — the real control model every attention
  variant is compared against: a modern decoder-only transformer (RoPE,
  RMSNorm, SwiGLU, GQA as the baseline attention) at roughly 50–150M
  parameters, trained on a local-feasible pretraining slice (TinyStories in
  full, or a FineWeb-EDU sample sized to the available time budget).

- **`03-attention-variants/`** — one subdirectory per variant, each changing
  **only** the attention module vs. `02-baseline-small-lm/`'s architecture,
  per the fair-comparison rule in `CLAUDE.md` (same tokenizer, dataset,
  token/step budget, optimizer, seed):
  - `gqa-vs-mha/` — grouped-query vs. full multi-head attention
  - `mla/` — Multi-head Latent Attention
  - `sliding-window/` — local/windowed attention
  - `nsa-lite/` — a simplified, local approximation of Native Sparse
    Attention (window + mean-pooled compressed branch; the trainable
    top-k selective branch was scoped out)

  **Result: sliding-window won** (best perplexity at zero parameter cost;
  top-3 effectively tied; MLA clearly behind at this scale) — see
  `03-attention-variants/README.md` for the full comparison table.

- **`04-scaling-checks/`** — the `03` winner (sliding-window) vs. the GQA
  full-attention control at 231.9M params / 512 context / 49M tokens of the
  real TinyStories train split, with gradient accumulation.
  **Result: the ordering held** — sliding-window statistically tied full
  attention (PPL 4.10 vs 4.11) while leading at every intermediate
  checkpoint. See `04-scaling-checks/README.md` for the verdict and the
  corpus-limitation caveat.

- **`05-data-frontier/`** — the corpus upgrade and the deferred long-range
  attention question, run as one experiment: a ~1.78B-token refined pool
  (FineWeb-Edu sample-10BT + Cosmopedia-v2, see
  `docs/literature/data-quality-frontier.md`) finally gave full attention
  something a 64-token window can't see. Three 114M models at block 1024,
  100M tokens each: `gqa/`, `sliding-window/`, and `hybrid/` (new — local
  attention with full attention every 4th layer, Gemma-2/3 lineage).
  **Result: the story flipped, and hybrid won outright** — val loss 3.809
  vs gqa 3.868 vs sliding-window 3.928, at ~30% of full attention's KV
  cache; per-position curves show sliding-window flat past position ~64
  while hybrid keeps improving deep into the context. The `03`/`04`
  sliding-window "win" was a TinyStories artifact, as suspected. Full
  tables and verdict: `05-data-frontier/README.md`.

- **Milestone 06 — MLX inference benchmark** *(complete)* — lives in
  `benchmarks/mlx/`: an MLX port of this repo's GPT with real KV caches
  (bounded rotating cache for windowed layers), verified to ~1e-5 logit
  parity against PyTorch. Measured: MLX decode is ~10x PyTorch/MPS
  (~530 vs ~53 tok/s at 114M), hybrid's cache is exactly its analytic 30%
  of full, 4-bit costs ~0.01-0.016 nats for +35-75% speed. Results table in
  `benchmarks/README.md`, JSONs in `benchmarks/results/mlx-*.json`.

- **`11-efficiency/`** — the training-cycle program: A/B lanes vs the
  `05` hybrid control under pre-registered gates. **Result: four
  challenges, four negatives** — bf16 autocast (1.01x on MPS, first
  formal measurement of the null), Muon@default (−0.22 nats), 4-epoch
  small-pool reuse (−0.32 nats, 0.90-nat train/val gap), and elite
  score≥4 filtering (−0.045 nats on a decontaminated val, after the
  design-reviewer agent caught 11.5% val contamination in the lane's
  pool — see `design-review-20260723.md` and `adjudicate.py`). The
  incumbent recipe survived everything; flagship-3-via-adopted-levers is
  cancelled as specified. Next lever class: distillation/post-training.

- **`10-sft/`** — SmolTalk SFT of the `09` flagship (assistant-only loss
  masking, one epoch, 7.6h): masked val 2.54→1.70, **zero benchmark
  regression**, reliable chat formatting. Plus the `release/` HF tooling
  (self-contained `trust_remote_code` wrapper, export/smoke/upload).
  **Weights published**: `garagelm/hybrid-gpt-232m` and `-chat`.

- **`09-flagship-2/`** — the flagship recipe at **1.0B tokens** (3.55B-token
  extended pool, identical val set). **PPL 21.5 (beat 07's 22.75); now
  ahead-or-tied vs Pythia-160M everywhere; clearly ahead of gpt2 on ARC-E
  (PIQA ahead within noise)** — but doubling tokens bought only 0.055 nats: the
  data-quality lever is visibly saturating at this scale. Verdict in
  `09-flagship-2/README.md`.

- **`08-solidify/`** — seed replication + window/ratio mini-sweep for the
  `05` result. **Hybrid beats full attention in 3/3 seeds (mean gap
  −0.038 nats)**; w=128 is worse than w=64; a 2-global-layer variant ties
  the champion at ~15% cache (flagged future work, not adopted at depth
  16). Gate outcome: `09` keeps w=64 / globals-every-4th.

- **`07-flagship-slm/`** — the `05` winner (hybrid) at the 232M recipe,
  block 1024, 500M refined tokens (63h clean, resumable every 500 steps).
  **Result: matches Pythia-160M (300B tokens — 600x ours; within n=300
  eval resolution) on
  every task at matched evaluation**, ties-to-beats gpt2, beats
  Cerebras-GPT-111M across the board, and stays clearly behind
  SmolLM2-135M (2T tokens) as pre-registered. PPL 22.7 on the refined val
  set; ~310 tok/s fp16 decode (527+ at 4-bit) / 4.97MB KV cache in MLX. Full tables and the
  frontier-per-FLOP verdict: `07-flagship-slm/README.md`.

## Conventions

Every experiment directory gets its own config (not shared mutable globals)
and logs its results in the format described in `benchmarks/README.md`, so
runs stay directly comparable.
