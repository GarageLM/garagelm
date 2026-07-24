# llm-arch-explore (GarageLM)

**Mission: optimize frontier training and inference until they run on
minimal hardware.** Frontier labs scale techniques up; this lab compresses
them down — measuring which frontier methods (refined data, efficient
attention, distillation, modern optimizers, post-training) survive
miniaturization to a hard hardware floor, currently one consumer
Apple-Silicon machine. The tracked quantity is the **frontier lag**: what
capability, at what fraction of the original hardware, how many years
later. First data point: GPT-2-class capability (2019, TPU pods)
reproduced at ~1/5th the training FLOPs (6ND, one-epoch-WebText
accounting) on a $1,400–$2,200 desktop. Everything under a strict
fair-comparison discipline; quality, size, and latency always reported
together.

And the ceiling is deliberately open: efficiency techniques
disproportionately originate with the compute-constrained (Muon from the
speedrun community, MLA from GPU-starved DeepSeek, FlashAttention and
LoRA from academia) and transfer *up*. Best case, work done at the floor
pushes the frontier itself.

**Released weights**:
[`garagelm/hybrid-gpt-232m`](https://huggingface.co/garagelm/hybrid-gpt-232m)
(base, 1B tokens) and
[`garagelm/hybrid-gpt-232m-chat`](https://huggingface.co/garagelm/hybrid-gpt-232m-chat)
(SmolTalk SFT) — both load via `transformers` with `trust_remote_code=True`.

**Headline result so far** (full synthesis:
[`docs/writeup-hybrid-attention.md`](docs/writeup-hybrid-attention.md)):
a 232M-param **hybrid local+global attention** model trained on **0.5B
refined tokens in 63h on one M4 Pro** matches Pythia-160M (within our
n=300 eval resolution)
(300B tokens — 600x more) on HellaSwag/PIQA/ARC-Easy/WinoGrande under a
matched evaluation harness, at ~30% of full attention's KV cache, decoding on-device at ~310 tok/s
fp16 (530+ at 4-bit, MLX). Along the way: sliding-window attention's
apparent parity with full attention proved to be a corpus artifact
(TinyStories has no long-range structure — shown directly with per-position
loss curves), and the hybrid pattern beat *full attention itself* at equal
parameters on long-form refined data (3/3 seeds, in our under-trained
regime).

## Scope and hardware

All work runs locally on a single Apple M4 Pro (48GB unified memory, no CUDA).
No multi-GPU runs, no billion-parameter models: experiments target roughly
**1M–250M parameters** — big enough to show real architectural differences,
small enough to iterate in hours-to-days on this machine. Training is
PyTorch/MPS; inference latency is measured in MLX (parity-gated port in
`benchmarks/mlx/`).

## Completed milestones

1. **Literature review** (`docs/literature/`) — attention foundations
   (MHA→MQA→GQA), MLA, sparse/sliding-window lineage, the data-quality
   frontier (phi, FineWeb-Edu, SmolLM2, DCLM).
2. **00–02: pipeline + baseline** — char-level smoke test, then a 114M
   GQA/RoPE/RMSNorm/SwiGLU control model.
3. **03–04: attention shootout + scaling check** (TinyStories) — GQA vs MHA
   vs MLA vs sliding-window vs NSA-lite under a strict fair-comparison rule;
   sliding-window "won", and the win was later shown to be a corpus artifact.
4. **05: data frontier** — 1.78B-token FineWeb-Edu + Cosmopedia-v2 pool;
   three-way re-test at 1024 context introduced the **hybrid local+global**
   variant, which beat full attention at ~30% of its KV cache.
5. **06: MLX inference benchmark** — TTFT / prefill / decode / live KV-cache
   measurement / 4-bit quantization, logit-parity-gated against PyTorch.
6. **07: flagship** — the hybrid at 232M / 500M tokens; the head-to-head
   table vs gpt2, Pythia-160M, and SmolLM2-135M.
7. **08: solidify** — seed replication (hybrid > full attention in 3/3
   seeds) + window/ratio sweep; w=64 / 1-in-4 globals confirmed.
8. **09: flagship-2** — 1.0B tokens: PPL 21.5, ahead-or-tied vs
   Pythia-160M everywhere, clearly ahead of gpt2 on ARC-E (PIQA ahead
   within noise) — and the honest
   measurement that the token axis is saturating.
9. **10: SFT + release** — SmolTalk chat tuning (no benchmark regression)
   and the published weights above.
10. **11: efficiency** — four levers vs the 05 control, **four gated
    negatives** (bf16: no MPS speedup; Muon@default: −0.22; 4x small-pool
    reuse: −0.32; elite filter: −0.045 on decontaminated val, after the
    design-reviewer agent caught 11.5% val contamination). The recipe
    stands; next lever class is distillation + post-training.

See `experiments/README.md` for per-milestone results and
`benchmarks/README.md` for the evaluation methodology.

## Layout

- `docs/literature/` — notes and summaries on existing attention architectures and
  efficiency techniques (the research foundation before proposing new designs).
- `experiments/` — training/eval harness for prototyping new attention variants at
  small scale.
- `benchmarks/` — tooling to measure quality vs. size vs. latency tradeoffs across
  architecture variants.

See `CLAUDE.md` for the framework decisions and working conventions.
