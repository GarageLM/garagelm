# Experiments

Milestone-based training/eval harness for prototyping new attention variants,
run entirely locally (see `CLAUDE.md` for hardware/framework decisions).
Milestones 00-13 are complete — each directory's README has its results and
the pre-registered gates they were adjudicated against. `14-babylm/` is
pre-registered and on hold (no arm trained). `15-harness/` is active: the
inference-time lever (open small model + harness vs dated frontier scores),
substrate in `benchmarks/harness/`.

Two findings from `13-moe/` constrain how later milestones can be read.
**Validation loss and downstream capability came apart**: the memory-matched
dense arm beat its control by 0.069 nats on 2M deterministic tokens and
converted none of it into measurable task performance. And **the standing
quality suite cannot separate models at this scale** — at `--limit 300` the
standard error is ~2.9 points per proportion, ~2.05 on a difference of two
4-task averages, and 13's three arms (114M to 284M) span about 1σ. Any
milestone claiming a downstream result needs a larger limit first.

The honest yardstick throughout is the capability-per-FLOP frontier against
published same-size models, stated in `05-data-frontier/README.md`. Entries
below run newest-first after the corpus upgrade.

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

- **`13-moe/`** — MoE at the hardware floor: at a hard memory budget, is a
  sparse model worth its bytes? Frontier MoE comparisons match *active*
  parameters and treat memory as free; at this lab's floor memory **is** the
  budget, so both controls ran — `moe/` (8 experts x ffn 1024, top-2, ~284M
  total / ~114M active) against the active-matched `05` hybrid (114M) and a
  memory-matched `dense-284m/` (FFN-width-only scale-up).

  | Arm | Full-pass val | Decode tok/s | 4-task avg |
  |---|---|---|---|
  | control 114M | 3.8398 | **54.3** | **44.08** |
  | moe 284M/114M active | 3.8094 | 8.9‡ | 42.08 |
  | dense-284m | **3.7711** | 41.8 | 43.75 |

  **Verdict: 2.5x the parameters bought validation loss and nothing else
  measurable.** G-A PASS but by **0.0004 nats** — 1/50th of the 08
  seed-noise yardstick at n=1 seed, so indistinguishable from sitting on
  the threshold, and not citable as "MoE beats the control". G-M: not
  parity, dense better by 0.0383. G-R PASS (min expert load 0.1147 vs
  uniform 0.125, drop 0.0004). No downstream difference between arms is
  resolvable at n=300. ‡MoE decode is implementation-bound: 96 `nonzero()`
  host syncs per forward, per-forward not per-token, so T=4096 training
  amortized it and T=1 decode does not — design-review F4 called this in
  advance, so it is not an architectural verdict on MoE. Deferred lanes
  (LatentMoE, shared experts, router z-loss, MLX expert paging) stay
  deferred. Gates and full tables: `13-moe/README.md`. The moe arm is
  released as a reproducibility artifact:
  [`garagelm/hybrid-gpt-moe-284m-a114m`](https://huggingface.co/garagelm/hybrid-gpt-moe-284m-a114m).

- **`12-budget-ablation/`** — the external review's decisive experiment for
  the paper's central architecture claim: **does the hybrid-vs-full gap move
  with training budget?** Byte-identical code to the `05` runs (which supply
  the 100M points), seed 1337, one complete cosine schedule per budget.
  **Result: the gap is present at every budget and closing monotonically** —
  −0.072 (50M), −0.057 (100M), −0.041 (200M), decay ≈ 0.016 nats per
  doubling. Both pre-registered calls fired. **Verdict: restricted attention
  is a favorable inductive bias whose advantage decays with training** — the
  hybrid wins across the entire practical local-training regime, and the
  decay rate is consistent with the convergence view as the asymptote. This
  upgrades the paper's central claim from a tension datum to a measured
  decay curve.

- **`12-post-training/`** — the post-training deep-dive the `11` negatives
  pointed at, from the `09` base: full SmolTalk SFT (~310M tokens), then
  UltraFeedback DPO, then distillation feasibility probes.
  **G1 PASS** — masked full-pass val 1.4470, 0.238 nats better than
  `10-sft`'s 60M-token run under a pinned identical evaluator.
  **G2 FAIL, recorded as such** — ARC-Easy −4.0 on the n=300 slice,
  confirmed at −2.2 on the full 2,376-item set: **the alignment tax,
  measured**. 10x chat SFT bought 0.238 nats of chat quality at ~2.2
  ARC-Easy points, where the 60M-token SFT had paid roughly none. Fail
  action honored — chat-v2 did not ship from Stage 1.
  **G3 PASS** (held-out preference accuracy 0.652) and **G4 PASS**
  (capability preserved to within a point on every task): DPO bought
  alignment at zero further capability cost, and chat-v2 shipped from
  there. Distillation probes return a **GO** — GPT-2-XL leads our base by
  0.296 nats on an identical protocol, with no tokenizer friction. Also
  recorded for the paper's systems section: a new species of the MPS
  allocator pathology, where variable-length DPO batches fragment the
  allocator (57GB peak); length-bucketing to multiples of 128 cut it to
  43GB and ran 1.9x faster.

- **`11-efficiency/`** — the training-cycle program: A/B lanes vs the
  `05` hybrid control under pre-registered gates. **Result: four
  challenges, four negatives** — bf16 autocast (1.01x on MPS, first
  formal measurement of the null), Muon@default (−0.22 nats), 4-epoch
  small-pool reuse (−0.32 nats, 0.90-nat train/val gap), and elite
  score≥4 filtering (−0.045 nats on a decontaminated val, after the
  design-reviewer agent caught 11.5% val contamination in the lane's
  pool — see `design-review-20260723.md` and `adjudicate.py`). The
  incumbent recipe survived everything; flagship-3-via-adopted-levers is
  cancelled as specified. Next lever class: distillation/post-training,
  run as `12-post-training/`.

- **`10-sft/`** — SmolTalk SFT of the `09` flagship (assistant-only loss
  masking, one epoch, 7.6h): masked val 2.54→1.70, **zero benchmark
  regression**, reliable chat formatting. Plus the `release/` HF tooling
  (self-contained `trust_remote_code` wrapper, export/smoke/upload).
  **Weights published**: `garagelm/hybrid-gpt-232m` and `-chat`, plus
  fp16 and 4-bit MLX conversions of both (`-mlx`, `-mlx-4bit`, and the
  chat pair), built and logit-parity-gated by `release/mlx/`.

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
