# Test-time compute and inference harnesses

Foundation for `experiments/15-harness/`. The question this note answers: how
much measured capability does inference-time compute buy a small open model,
what does it cost, and where does it stop working. Written 2026-08-21.

## The problem it solves

Pretraining scale is the lever this lab cannot pull past ~250M params on one
machine. Inference-time compute is the other lever: sample more, verify, search,
refine. The 2025-2026 record says it is the largest single lever available at
small scale, and that it is harness design, not weights, that moves the number.

## Results worth knowing (all sourced; re-pin dates before quoting)

- **HF search-and-learn (Dec 2024).** Llama-3.2 1B/3B Instruct with a process
  reward model and best-of-N / beam search / Diverse Verifier Tree Search on
  MATH-500: the 1B model approaches the 8B single-sample score, the 3B model
  passes 70B at large budgets. Weighted best-of-N and PRMs do the work;
  DVTS keeps diversity at large N. Cost: a trained verifier and N x decode.
  https://huggingface.co/learn/cookbook/en/search_and_learn
- **"Can 1B LLM surpass 405B LLM?" (arXiv 2502.06703).** Compute-optimal
  test-time scaling is policy-, PRM-, and difficulty-dependent; no single
  strategy wins. Read for the failure modes: PRMs overfit, voting plateaus.
- **Cost-Effective Agent Harnesses for ARC-AGI-1 (arXiv 2607.06764, Jul 2026).**
  DeepSeek V3.2, non-thinking, no ARC fine-tuning: one-shot 15.5% pass@2.
  Explorer-Definer pipeline (separate pattern discovery from executable
  transformation synthesis, verify on train pairs): 57.5% at $0.25/task.
  Reflective Orchestrator (explore new transformations when hypotheses fail
  on train pairs): 67.25% at $0.62/task. Same weights, +52 points from the
  harness alone. https://arxiv.org/abs/2607.06764
- **ARC Prize 2025 technical report (arXiv 2601.10904).** Compute-capped
  Kaggle track (no internet, ~$50/submission, open source required): NVARC
  24.03% on ARC-AGI-2 private at $0.20/task (test-time training + synthetic
  data), ARChitects 16.53%, MindsAI 12.64%. Frontier API leaderboard at the
  same time: ~85% (GPT-5.5), so the compute cap costs ~60 points. Paper
  prizes: TRM (7M params, recursive refinement, 45% ARC-AGI-1 / 8% ARC-AGI-2),
  SOAR (evolutionary program synthesis, the LLM fine-tuned on its own search
  traces), CompressARC (76K params, 20% / 4%). Theme: refinement loops.
  https://arxiv.org/abs/2601.10904
- **Majority voting / parallel vs sequential.** Across 2025-2026 surveys
  (arXiv 2512.02008 and the inference-time-scaling paper lists), parallel
  sampling with consensus beats sequential self-revision at equal tokens and
  plateaus later; majority vote is a usable verifier when exact answers exist.
  Tool-integrated verification (T1, arXiv 2504.04718) extends it to small
  models by letting code execution replace the reward model.

## What it costs

Tokens. Thinking traces on competition math run 10-30k tokens per sample;
k=8 on 90 AIME problems is ~10M generated tokens. On one M4 Pro the binding
quantity is aggregate batched decode throughput, not model quality. Every
harness claim therefore ships with tokens per item, wall-clock per item, and
peak memory beside accuracy (the three-way rule, applied to inference).

## Where it breaks down

- A harness cannot conjure capability the base has ~0 probability mass on:
  a model with no math or code in its corpus does not vote its way to AIME.
  This is why the lab's own 232M models are the floor row, not the contender.
- Voting needs an exact, cheap answer extractor; free-form tasks need a
  verifier, and a learned verifier is a training milestone in disguise.
- Contamination: public eval sets from 2019-2024 are inside 2026 pretraining
  corpora. Prefer the newest sets, report each model's data cutoff beside its
  score, and run semantics-preserving augmentations where the task allows
  (ARC colour permutation / transpose).
- Truncation policy changes the number: a trace cut at max_tokens must count
  as wrong and the truncation rate must be reported.
- Frontier comparisons cannot be re-run locally (closed APIs, no API budget
  here by decision), so they are quoted, dated, sourced, and marked as such.

## What this repo takes from it

Lane order by cost-to-headline: exact-match math with consensus first (no
verifier to build), then ARC-AGI program synthesis where train pairs are the
verifier for free and the documented harness lift is the largest, then GPQA
consensus and execution-verified code. One substrate (`benchmarks/harness/`),
one control (same model, k=1), and every k derived from one k=8 run by
subsampling. The traces with verifier verdicts are the synthetic corpus the
distillation lever (12's GO) was priced for; the two compose.
