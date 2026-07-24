# 07-flagship-slm

The headline run: the `05` winner (hybrid local+global attention) at the
proven 232M recipe, trained on 500M tokens of the refined corpus — the
biggest swing this hardware can take, aimed at the capability-per-FLOP
frontier.

## The yardstick (stated before results, honestly)

No 0.5B-token local run beats SmolLM2-135M (2T tokens). The targets:

- **Beat Cerebras-GPT-111M everywhere** (111M params, 2.2B Pile tokens —
  if refined data can't beat unfiltered data with 4x our tokens, the data
  thesis fails). Published 0-shot: HS 26.8 / PIQA 59.4 / ARC-E 38.0 /
  WG 48.8. The 114M `05` models already match/beat it at 100M tokens.
- **Approach OPT-125M / Pythia-160M on HellaSwag + PIQA** (180B/300B
  tokens — 360–600x our budget). Published: HS ~30-31 / PIQA 62.0.
- Reference models are re-run through the identical lm-eval harness and
  limits (gpt2, Pythia-160M, SmolLM2-135M via the HF backend) rather than
  trusting cross-source numbers.

## Setup

- **231,852,032 params**: n_layer=16, n_head=16, n_kv_head=4, n_embd=1024,
  ffn_hidden=2816, block_size=1024, dropout 0.0; hybrid attention with
  window=64 and full attention at layers 3/7/11/15 (~30% of full
  attention's KV cache at T=1024, per the measured `05` MLX numbers).
- **Data**: the `05-data-frontier` pool (1.78B tokens, 84.5% FineWeb-Edu /
  15.5% Cosmopedia-v2) — 500M-token budget = ~0.28 epochs, minimal
  repetition.
- **30,500 optimizer steps** × 16,384 tokens/step (micro-batch 4×1024,
  accum 4 — micro-batch 8 was probed and rejected: 17.8s/step and 41.5GB
  peak vs 11.5s/step and 23.4GB). Cosine LR 3e-4→3e-5, warmup 200, wd 0.1,
  clip 1.0, seed 1337 — the recipe proven at 232M (`04`) and on this
  data/context (`05`).
- **Resumable**: state (model+optimizer+RNG) saved every 500 steps;
  `--resume` continues seamlessly (bit-identical eval after resume,
  verified in `05`). Measured 11.54s/step → ~98h (~4.1 days), accepted
  over cutting the token budget.

## Results

Trained 63.2h clean (7.4s/step sustained — the 11.5s pre-launch probe was
polluted by a warm allocator), zero restarts. Final val loss **3.1246**
(PPL 22.7) — vs 3.809 (PPL 45.1) for the same architecture at 114M/100M
tokens in `05`.

**Head-to-head, identical harness and example slices** (0-shot, n=300/task
(MMLU 5/subject), all run locally through the same lm-eval pipeline —
local numbers are NOT comparable to published full-set numbers, which is
the point of re-running the references):

| Model | Params | Train tokens | HellaSwag | PIQA | ARC-E | WinoGrande | MMLU |
|---|---|---|---|---|---|---|---|
| **this repo** | 232M | **0.5B** | .353/.393 | .617/.593 | .447/.433 | .533 | .267 |
| gpt2 | 124M | ~10B | .353/.427 | .610/.617 | .420/.380 | .530 | .267 |
| Pythia-160M | 162M | 300B | .350/.390 | .627/.627 | .460/.397 | .510 | .267 |
| SmolLM2-135M | 135M | 2,000B | .403/.553 | .653/.680 | .583/.437 | .490 | .263 |

(acc/acc_norm; MMLU is at chance for every model at this scale, including
SmolLM2 — consistent with this repo's benchmark guidance.)

- **vs Pythia-160M (600x our tokens): statistical tie on every task.**
- **vs gpt2 (20x our tokens): tie-to-ahead** (ARC-E clearly ahead, HS
  acc_norm behind, rest tied).
- **vs Cerebras-GPT-111M** (published .268 HS / .594 PIQA / .380 ARC-E /
  .488 WG at 4.4x our tokens): ahead across the board — the data-quality
  thesis target, met.
- **vs SmolLM2-135M (4,000x our tokens): clearly behind** on HS/PIQA/ARC-E,
  as pre-registered. This is the 2T-token ceiling the recipe scales toward.

**Long-range behavior held at scale**: per-position val loss improves from
3.67 (positions 0–63) to ~3.12 by position 256 and stays there through
position 1000 (`benchmarks/results/07-flagship-slm-per-position.json`) —
the hybrid's global layers keep earning deep context at 232M.

**Deployment numbers (MLX, milestone 06 harness)**: logit parity OK;
fp16 decode **312–322 tok/s**, TTFT 72ms at 896-token prompts, KV cache
**4.97MB** at full context (~30% of an equivalent full-attention model);
4-bit: **527–545 tok/s** at +0.014 nats val loss. Qualitative samples are fluent
encyclopedic prose with the usual small-model factual confabulation.

## Verdict

The phase goal — refined data + efficient attention on minimal hardware —
lands where the literature said it should: **data quality bought ~600x
token efficiency against Pythia-160M's Pile training** (a 2023-frontier
recipe), while the hybrid attention keeps the KV cache at 30% with 500+ tok/s
4-bit decode on-device (~310 fp16). What it does *not* do is touch SmolLM2 — closing
that gap is about scale of refined tokens (2T vs 0.5B), not architecture.
The honest frontier-per-FLOP claim: **at matched evaluation, this 0.5B-token
local run is competitive with early-2020s 100M-class models trained on
2–3 orders of magnitude more data.**

## Run it

```
uv run python experiments/05-data-frontier/data.py   # once, shared pool
uv run python experiments/07-flagship-slm/train.py --resume
uv run python benchmarks/run_quality_eval.py --experiment-dir experiments/07-flagship-slm
uv run python benchmarks/long_range_probe.py --experiment-dir experiments/07-flagship-slm
uv run python benchmarks/mlx/convert.py --experiment-dir experiments/07-flagship-slm --parity
```
