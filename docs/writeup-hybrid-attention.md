# Hybrid Local+Global Attention at Small Scale: What a Laptop Can Prove

*llm-arch-explore, milestones 02–07 synthesis — July 2026*

## Abstract

We trained and compared attention variants of a modern decoder-only
transformer (RoPE, RMSNorm, SwiGLU, GQA) entirely on one Apple M4 Pro
(48GB), holding tokenizer, data, token budget, optimizer, and seed fixed so
that only the attention pattern varied. Three findings:

1. **Corpus structure, not architecture, decided the early results.** On
   TinyStories, sliding-window attention (w=64) tied full attention at two
   model scales (114M, 232M) — but per-position loss analysis shows even
   full attention extracts nothing beyond ~128 tokens of context there.
   The "win" was an artifact of a corpus with no long-range structure.
2. **On a refined corpus with real long-form text, a hybrid local+global
   pattern (full attention every 4th layer, sliding window elsewhere) beat
   both pure alternatives** — including full attention itself, replicated
   across 3 seeds (mean gap −0.038 nats, hybrid ahead in 3/3; seed-1337
   values 3.809 vs 3.868 at 114M/100M tokens) — at ~30% of full
   attention's KV cache. Pure sliding-window clearly lost (3.928),
   confirming the artifact.
3. **Scaled to 232M params / 500M refined tokens (63h local training), the
   hybrid statistically ties Pythia-160M — a model trained on 600x more
   tokens — on HellaSwag, PIQA, ARC-Easy, and WinoGrande under a matched
   evaluation harness**, while decoding at 527+ tok/s on-device (MLX fp16)
   with a 4.97MB KV cache at 1024 context.

The token-efficiency result quantifies the data-quality thesis
(FineWeb-Edu + Cosmopedia vs the Pile); the architecture result supports
the Gemma-2/3-lineage design at scales hobbyist hardware can verify.

## 1. Setup

- **Hardware**: single Apple M4 Pro, 48GB unified memory, no CUDA.
  Training on PyTorch/MPS; inference measured on MLX (both fp16 and 4-bit),
  with the MLX port gated on logit parity vs PyTorch (max |Δ| ~1e-5,
  100% top-1 agreement, prefill and cached decode).
- **Architecture (control)**: decoder-only GPT — RoPE (θ=10000), RMSNorm,
  SwiGLU, GQA (4 KV heads), weight tying, GPT-2-style residual-scaled
  init. 114M (12L/768d) and 232M (16L/1024d) configurations.
- **Fair-comparison rule**: within each experiment, every variant shares
  the tokenizer (tiktoken gpt2), data pool, token/step budget, optimizer
  (AdamW, cosine 3e-4→3e-5, wd 0.1, clip 1.0), and seed (1337). Only the
  attention module differs; all variants have byte-identical parameter
  counts (masks carry no parameters).
- **Evaluation**: validation loss/perplexity (primary); lm-evaluation-
  harness 0-shot at n=300/task via a custom `TemplateLM` adapter over the
  raw PyTorch models; a **per-position validation loss probe** (mean loss
  bucketed by token position over 256 seed-fixed windows) as the
  long-range diagnostic; MLX inference benchmark (TTFT, prefill/decode
  tok/s, live-measured KV bytes, 4-bit quality delta).

## 2. Part I — The TinyStories artifact (milestones 03–04)

At 114M/256-context/12M tokens (TinyStories), the variant ranking by val
perplexity was: sliding-window 6.43 ≤ NSA-lite 6.45 ≈ GQA 6.47 < MHA 6.63
< MLA 7.09. At 232M/512-context/49M tokens the tie held: sliding-window
4.10 vs full attention 4.11 (0.0014 nats — noise), with sliding-window
ahead at every intermediate checkpoint.

The per-position probe explains why: on TinyStories, **even full
attention's loss stops improving past position ~128** (1.72 at positions
0–63, 1.33 at 64–127, flat ±0.05 thereafter). Short, self-contained
stories give long-range attention nothing to exploit, so removing it costs
nothing. Any attention comparison on such a corpus is uninformative about
long-range capability — a caveat we registered before running Part II.

## 3. Part II — The flip on refined data (milestone 05)

**Corpus**: 1.78B-token pool, 84.5% FineWeb-Edu `sample-10BT` + 15.5%
Cosmopedia-v2 (synthetic textbooks), tokenized with tiktoken gpt2;
2M-token val held out stratified by source. **Models**: three 114M
configurations at block 1024, 100M tokens each — full-attention GQA,
pure sliding-window (w=64), and **hybrid** (w=64 everywhere except full
attention at layers 3/7/11).

| Model | Val loss | PPL | KV cache @1024 (measured, MLX) |
|---|---|---|---|
| hybrid | **3.8091** | **45.1** | 3.73 MB (29.7% of full) |
| GQA (full) | 3.8680 | 47.9 | 12.58 MB |
| sliding-window | 3.9278 | 50.8 | 0.77 MB (constant) |

Per-position loss separates the mechanisms: sliding-window flattens at
position ~64 and degrades toward far positions (3.88 → 3.99), while GQA
and hybrid keep improving deep into the context; the sliding-window↔hybrid
gap grows to ~0.15 nats by position 1000. The hybrid is at or below full
attention in essentially every bucket.

Two conclusions. First, the 03/04 ordering inverted exactly as the
artifact hypothesis predicted — local attention's earlier win did not
survive contact with long-range data. Second, **hybrid > full attention at
equal parameters** suggests mostly-local attention is a useful inductive
bias, with a few global layers sufficient to carry long-range information
— consistent with (and small-scale evidence for) the Gemma-2/3 and
character.ai interleaving designs.

The corpus swap also moved benchmarks that TinyStories never could:
ARC-Easy went from chance (.26) to .38 at just 100M tokens — matching
Cerebras-GPT-111M's published score with ~22x fewer tokens.

## 4. Part III — Scaling the winner (milestone 07)

The hybrid at 232M (16L/1024d, global layers 3/7/11/15), block 1024,
**500M tokens** from the same pool: 63.2h wall on the M4 Pro, zero
restarts, final val loss 3.1246 (PPL 22.7). Per-position behavior held
(3.67 → ~3.12 by position 256, sustained through 1000).

Head-to-head under the **identical harness, limits, and example slices**
(all models re-run locally; local numbers are deliberately not compared to
published full-set scores):

| Model | Params | Train tokens | HellaSwag | PIQA | ARC-E | WinoGrande |
|---|---|---|---|---|---|---|
| **this repo (hybrid)** | 232M | **0.5B** | .353/.393 | .617/.593 | .447/.433 | .533 |
| gpt2 | 124M | ~10B | .353/.427 | .610/.617 | .420/.380 | .530 |
| Pythia-160M | 162M | 300B | .350/.390 | .627/.627 | .460/.397 | .510 |
| SmolLM2-135M | 135M | 2,000B | .403/.553 | .653/.680 | .583/.437 | .490 |

(acc/acc_norm, 0-shot, n=300/task. MMLU is at chance for all four models,
including SmolLM2 — 0-shot MMLU does not discriminate below ~1B params.)

- **Tie with Pythia-160M on every task at 600x fewer tokens** — the
  refined-corpus effect, quantified against a Pile-trained 2023 recipe.
- Tie-to-ahead vs gpt2 (~20x our tokens); ahead of Cerebras-GPT-111M's
  published numbers across the board.
- **Clearly behind SmolLM2-135M** (4,000x our tokens) on HS/PIQA/ARC-E —
  pre-registered as the ceiling; the remaining gap is refined-token scale,
  not architecture.

**Deployment (MLX, measured)**: 527–535 tok/s fp16 decode / ~700 tok/s at
4-bit (+0.014 nats val loss); TTFT 72ms at 896-token prompts; 4.97MB KV
cache at full context vs ~16.8MB for an equivalent full-attention model.
The MLX runtime is ~10x the PyTorch/MPS harness for decode at these sizes
— on-device deployment numbers must be measured there.

## 4b. Postscript — scaling the token axis (milestone 09)

Doubling the flagship's budget to 1.0B tokens (identical val set) gave
val loss 3.070 vs 3.125 (PPL 21.5 vs 22.75), ARC-E .477 (+3.0), PIQA .633
(+1.6), HellaSwag flat — now ahead-or-tied vs Pythia-160M on every task
and ahead of gpt2 on ARC-E and PIQA. The honest reading: **the refined-
data lever produced its 600x efficiency win in the first 0.5B tokens and
is saturating**; closing the remaining gap to SmolLM2-class models locally
requires a different lever (scale, context, distillation, post-training),
not more of the same tokens.

## 5. Limitations and threats to validity

- ~~Single seed per configuration~~ **Resolved (milestone 08)**: the
  hybrid-vs-GQA comparison was replicated at seeds 1338 and 1339 —
  **hybrid wins in 3 of 3 seeds, mean gap −0.038 nats (range −0.028 to
  −0.059)**. The originally reported 0.059 was the most favorable seed;
  the mean is the honest number. A window/ratio mini-sweep also found
  w=128 clearly worse than w=64, and a sparser-globals variant (2 global
  layers, ~15% cache) tying the champion at 114M — promising but
  structurally untested at 16 layers, so not adopted for scaling. See
  `experiments/08-solidify/README.md`.
- **Benchmark slices**: n=300 per task using lm-eval's `--limit` (first-N,
  not random). Internally consistent across all compared models, but not
  comparable to published full-set numbers — which is why references were
  re-run locally.
- **Val-set domain**: val PPL is measured on the training distribution
  (held-out split); cross-corpus generalization was only probed via the
  external benchmarks.
- **Scale**: 114M–232M params, ≤0.5B tokens — far below compute-optimal.
  The hybrid>full result could weaken at larger scale or budget; the
  design's use in Gemma-2/3 argues it does not, but that is not this
  repo's evidence.
- **Un-swept hyperparameters**: window (64) and global-layer ratio (1:4)
  were inherited from the 03 winner and the literature, not tuned. MLA was
  only tested at 114M/TinyStories, where it lost; its cache advantage at
  longer contexts remains untested here.
- **Apple Silicon operational caveat**: long MPS training runs can hit an
  in-process allocator pathology (~7x step slowdown with a healthy GPU;
  observed twice). Mitigations that worked: periodic
  `torch.mps.empty_cache()`, keeping micro-batch logits tensors under
  ~1GB, resumable state every 500–1000 steps, and process restart on
  detection (a fresh-process matmul probe distinguishes it from system
  throttling).

## 6. Released artifacts

Weights on the Hugging Face Hub, loadable with `trust_remote_code=True`:
[garagelm/hybrid-gpt-232m](https://huggingface.co/garagelm/hybrid-gpt-232m)
(the 1B-token base model from §4b) and
[garagelm/hybrid-gpt-232m-chat](https://huggingface.co/garagelm/hybrid-gpt-232m-chat)
(SmolTalk SFT, no benchmark regression). Post-upload verification: fresh
clean-cache download, generation, and an lm-eval task run against the live
repo.

## 7. Reproduction

```
uv sync                                                  # Python 3.12 venv
uv run python experiments/05-data-frontier/data.py       # ~5.5GB download → 1.78B-token pool
uv run python experiments/05-data-frontier/<variant>/train.py --resume   # ~7h each
uv run python experiments/07-flagship-slm/train.py --resume              # ~63h
uv run python benchmarks/run_quality_eval.py --experiment-dir <dir>
uv run python benchmarks/long_range_probe.py --experiment-dir <dir>
uv run python benchmarks/mlx/convert.py --experiment-dir <dir> --parity
uv run python benchmarks/mlx/bench_inference.py --converted benchmarks/mlx/converted/<name> [--bits 4]
```

Per-experiment configs are frozen in each directory; results JSONs live in
`benchmarks/results/`. Milestone-level detail: `experiments/*/README.md`.
