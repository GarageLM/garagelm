# garagelm (Garage Language Models)

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
(chat v2: SmolTalk SFT + UltraFeedback DPO; v1 stays fetchable at HF
revision `v1`) — both load via `transformers` with `trust_remote_code=True`.

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
   frontier (phi, FineWeb-Edu, SmolLM2, DCLM), and a replication triage
   of the Kimi K3 tech report (`docs/literature/kimi-k3.md`) into a
   candidate milestone-13 program.
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
11. **12: post-training** — SFT at 10x data (+0.24 nats chat quality) +
    UltraFeedback DPO (65.2% held-out preference accuracy, zero
    DPO-stage capability cost), with the SFT **alignment tax measured**:
    −2.2 full-set ARC-Easy pts at 310M SFT tokens vs ~0 at 60M. IFEval
    vs SmolLM2-135M-Instruct under one matched harness (13.9% vs 21.4%
    prompt-strict). The DPO endpoint ships as **chat v2** (tax disclosed
    on the card; v1 remains at HF revision `v1`). Distillation probes
    return a **GO** for a milestone-13 program: GPT-2-XL leads our base
    by 0.296 nats on the shared val as a same-tokenizer logit teacher,
    and local synthetic generation prices at ~13.9M tokens/day.
12. **12b: budget ablation** — the review-requested decisive experiment
    for the paper's central architecture claim: hybrid vs full attention
    at 50M / 100M / 200M tokens, byte-identical code, pre-registered
    noise yardstick. The hybrid's edge is **present at every budget**
    (−0.072 / −0.057 / −0.041 nats) and **decays monotonically** (~0.016
    nats per token doubling; single-seed extrapolated crossover near
    1.3B tokens). Reading: restricted attention is a favorable inductive
    bias whose advantage shrinks as training grows; the claim is now a
    measured decay curve, not a tension datum.

See `experiments/README.md` for per-milestone results and
`benchmarks/README.md` for the evaluation methodology.

## Talk to the models

One command starts a local chat server + web UI over the published chat
model, running entirely on this machine (MLX, real rotating KV cache):

```
cd ~/Desktop/garagelm && ./serve.sh
```

It opens the browser at `localhost:8080`, converts the checkpoint on
first use (logit-parity-gated), and — if a training run owns the GPU —
automatically starts in polite CPU mode so research is never disturbed.
The default is the published chat model (v2: SFT + DPO). Variants:
`MODEL=09-flagship-2 ./serve.sh` serves the raw base model,
`MODEL=10-sft ./serve.sh` serves chat v1; `PORT=9000` moves the port.
Stop with Ctrl-C.

The server is OpenAI-compatible (`http://localhost:8080/v1`), so any
client library or chat UI can talk to it — which also makes it the lab's
standard tool for qualitative testing and scripted batch probes of any
checkpoint. Terminal alternatives: `experiments/10-sft/chat.py` (REPL)
and each experiment's `sample.py` (one-shot generation from any
`out/ckpt.pt`).

## Layout

- `docs/literature/` — notes and summaries on existing attention architectures and
  efficiency techniques (the research foundation before proposing new designs).
- `experiments/` — training/eval harness for prototyping new attention variants at
  small scale.
- `benchmarks/` — tooling to measure quality vs. size vs. latency tradeoffs across
  architecture variants.

See `CLAUDE.md` for the framework decisions and working conventions.
