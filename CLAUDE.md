# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

**Optimize frontier training and inference until they run on minimal
hardware** (working identity: GarageLM). The core thesis is
frontier-compression: take the techniques that define cutting-edge models
— refined data, efficient attention, distillation, modern optimizers and
post-training — and measure which survive miniaturization to a hard
hardware floor. Currently that floor is one consumer Apple-Silicon
machine; self-funded GPU/TPU tiers may widen the envelope later without
changing the thesis, so training/eval code stays portable (MPS playbook
and MLX stack are Mac-specific instantiations, not the method). The
tracked quantity is the frontier lag: what capability, at what fraction
of the original hardware, how many years later — with the ceiling open:
techniques discovered under constraint historically transfer up (Muon,
MLA, FlashAttention), so frontier-pushing contributions are in scope, not
ruled out. Attention was phase one (hybrid local+global, validated n=3
seeds); always the fair-comparison rule below, always quality, size, and
latency reported together.

## Current state

Milestones 00–13 are complete (see `experiments/README.md` for results).
Shipped: the hybrid local+global attention win (validated n=3 seeds, with
its decay curve measured in 12b), a ~1B-token 232M flagship, the MLX
inference stack, SFT+DPO post-training with the alignment tax measured,
and 13's MoE verdict. The decisions below are locked in; do not
re-litigate them without a reason grounded in something that changed.

**Next lever**: distillation is the priced, unspent one — 12's probes
returned a GO (GPT-2-XL leads our base by 0.296 nats as a same-tokenizer
logit teacher; local synthetic generation prices at ~13.9M tokens/day).

**Open blocker on every future comparison**: the standing quality suite
runs at `--limit 300`, where the standard error on a proportion is ~2.9
points and ~2.05 points on a difference of two 4-task averages. Milestone
13 could not distinguish a 114M model from a 284M one on it. Any milestone
whose claim rests on downstream tasks needs a larger `--limit` first, or it
can only report validation loss. 13 also showed loss and downstream
capability coming apart outright, so validation loss alone is not a
sufficient proxy.

**Hardware**: all work runs locally on a single Apple M4 Pro, 48GB unified
memory, no CUDA. No multi-GPU runs, no billion-parameter models. Target
model sizes are roughly 1M–250M parameters. Practical MPS training regime
established in 02/04: keep the per-forward logits tensor small (micro-batch
× block × 50257 fp32 well under 1GB), `del` batch tensors each step,
periodic `torch.mps.empty_cache()`; batch32×512 at 232M params triggers an
allocator pathology (63GB, 70x slowdown). Check for machine contention
(`ps aux`, >50% CPU processes) before launching multi-hour runs, and run
trainings sequentially, never concurrently.

**Framework**: PyTorch with the MPS backend is primary, for training and for
running `lm-evaluation-harness`. MLX measures **inference**
latency/throughput (`benchmarks/mlx/` — port, converter with logit-parity
gate, benchmark runner), since it outperforms PyTorch+MPS for on-device
inference and that's the deployment-relevant number this project's "least
performance impact" goal cares about.

**Python/env**: `uv`-managed venv pinned to Python 3.12 (system Python
3.14.3 lacks wheels for parts of this stack). Note: `transformers` is
pinned `>=5.0,<5.6` — mlx-lm 0.31.x breaks against transformers 5.13+'s
changed `AutoTokenizer.register` API.

**Common commands** (always `uv run` from the repo root; use absolute paths
when launching background jobs — a relative path re-resolved from a changed
cwd has bitten before):

```
uv run python experiments/<milestone>/data.py                # build data pool
uv run python experiments/<milestone>/<variant>/train.py     # train (hours)
uv run python experiments/<milestone>/<variant>/sample.py    # qualitative check
uv run python benchmarks/run_quality_eval.py --experiment-dir experiments/<m>/<v>
uv run python benchmarks/long_range_probe.py --experiment-dir experiments/<m>/<v>
uv run python benchmarks/mlx/convert.py --experiment-dir experiments/<m>/<v> --parity
uv run python benchmarks/mlx/bench_inference.py --converted benchmarks/mlx/converted/<name>
```

Before any multi-hour training run: sanity-check (forward shapes/NaN,
param count, short loss-decrease probe) and a ~10-step throughput probe
with an abort gate, both established patterns in this repo's history.
**Closing a training chain includes stopping its stall watchdog** — a
watchdog left armed after completion fires a false STALL ALARM when the
log goes quiet (has happened three times; the alarm text is
indistinguishable from a real stall until checked).

## Structure

- `docs/literature/` — research notes on existing attention architectures and
  efficiency techniques, read in the order laid out in
  `docs/literature/README.md`. This is the foundation new architecture
  proposals should be justified against: what problem an existing technique
  solves, what it costs, where it breaks down at scale or at small model
  sizes.
- `experiments/` — the milestone-based training/eval harness described in
  `experiments/README.md` (baseline smoke test → baseline small LM →
  attention variants → optional scaling checks).
- `benchmarks/` — the evaluation methodology described in
  `benchmarks/README.md`: quality, parameter count, and latency/throughput
  measured together for every variant produced in `experiments/`.

## Working conventions

- New architecture ideas should be grounded in `docs/literature/` — check
  whether a variant (or something close to it) has already been tried before
  proposing it as novel.
- **Fair-comparison rule**: every attention-variant experiment must hold the
  tokenizer, dataset, token/step budget, optimizer, and seed fixed against
  the baseline — only the attention module changes. This is the one rule
  that makes the experiment path scientifically valid; without it,
  comparisons between variants are meaningless.
- **Reproducibility**: every experiment run gets its own config file and its
  results get logged in a consistent format (see `benchmarks/README.md`) —
  no hand-edited globals shared mutably across runs.
- Every experiment result should be comparable: report quality, parameter
  count, and latency/throughput together, not quality in isolation. The
  project's stated goal is a three-way tradeoff (power vs. size vs.
  performance impact), so single-axis results are insufficient on their own.
  Concretely: validation perplexity, relevant `lm-evaluation-harness` tasks,
  tokens/sec (prefill and decode reported separately), time-to-first-token,
  peak memory, and (for cache-compressing variants like MLA/GQA) KV-cache
  size vs. sequence length.
- At the model sizes this repo actually trains (single-digit M to low
  hundreds-of-M params), MMLU/GPQA-class benchmarks sit near random chance
  and aren't discriminating. Use validation perplexity plus small-scale
  `lm-evaluation-harness` tasks (HellaSwag, PIQA, ARC-Easy, WinoGrande) for
  real signal; defer MMLU-Pro/GPQA/IFEval-class evals until a larger
  checkpoint exists.
