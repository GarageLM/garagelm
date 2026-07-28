# 12-post-training

Milestone 12: the post-training deep-dive the milestone-11 negatives
pointed at. Init from the `09` base (1B-token flagship). Two training
stages plus cheap distillation feasibility probes; chat-v2 ships only if
the gates pass.

## Pre-registered gates (written before any run)

**Stage 1 — `sft-full/` (full SmolTalk `all` mix, ~310M formatted
tokens, ~1 epoch, LR 2e-5 cosine):**
- The new 500-conversation held-out val is a *different mix* than
  `10-sft`'s, so cross-val-set numbers don't compare. The gate therefore
  evaluates all three checkpoints on the SAME new val at gate time:
  - G1: `sft-full` final masked val < the `10-sft` checkpoint's masked
    val on this val, by ≥ 0.05 nats (10x data must beat 1x data).
    **Pinned evaluator (design-review F1)**:
    `benchmarks/full_pass_val.py --mask-bin` — deterministic full-pass
    masked CE over `sft-full/data/val_tokens.bin` + `val_mask.bin`,
    identical invocation for both checkpoints (and the `09` base for
    context). In-run estimates decide nothing.
  - G2: benchmark suite (HellaSwag/PIQA/ARC-E/WinoGrande, n=300,
    **metric = acc**) within ±2.8 pts (1×SE) of the `09` base on every
    task (no capability regression).
  - Fail action: chat-v2 does not ship from this stage; result recorded.
  - Disclosure (F7): val splits are positional (first-N conversations /
    last-N pairs), the lab's standing convention — stated, not sampled.

**Stage 2 — `dpo/` (ultrafeedback_binarized train_prefs, β=0.1, ~1
epoch, from the Stage-1 checkpoint):**
- Sanity before launch: overfit 10 pairs — final-cycle DPO loss below
  0.1 and per-cycle implicit-reward margin strictly increasing, cycling
  all 10 pairs in 2-pair chunks (calibration knobs pinned pre-result:
  sanity LR 5e-6, 12 cycles). No launch without it.
- Wall-clock (F3): unprobed at design time; a 5-step probe runs before
  launch. Expected ~13–21h at 1 epoch; **pre-stated fallback**: if the
  probe projects >24h, epochs drops to 0.5 (same gates).
- G3: held-out preference accuracy (implicit reward ranking on 1k
  held-out pairs) > 60%.
- G4: same benchmark non-regression as G2, vs the Stage-1 model.
- Honest scope: no LLM-judge win-rates at this scale; qualitative
  side-by-sides published in this README instead.

**Stage 3 — distillation probes (no training committed):**
- P1: GPT-2-XL (the only strong gpt2-tokenizer teacher) full-pass loss
  on our shared val; logit-KD is dead unless it clearly beats our 09
  base (< 3.0).
- P2: MLX generation throughput for 1–2 modern teachers → tokens/day of
  synthetic generation on this machine. Output: a go/no-go note for a
  milestone-13 distillation program.

## Results

**Stage 1 (sft-full)**: trained 38.2h (one external kill, resumed from
step 4000, ≤1h lost). Adjudicated by the pinned evaluator
(`full_pass_val.py --mask-bin`, identical invocation, new shared val):

| Checkpoint | Masked full-pass val |
|---|---|
| 09 base | 2.0732 |
| 10-sft (60M tokens) | 1.6847 |
| **sft-full (310M tokens)** | **1.4470** |

- **G1: PASS** — 0.238 nats better than 10-sft (gate ≥ 0.05).
- **G2: FAIL** — vs 09 base (acc): HellaSwag +0.6 ✓, PIQA −0.3 ✓,
  WinoGrande +0.3 ✓, **ARC-Easy −4.0 ✗** (.437 vs .477, tolerance
  ±2.8). The alignment tax, measured: 10x chat SFT bought 0.24 nats of
  chat quality at the price of science-question capability the 60M SFT
  had preserved (10-sft ARC-E was −1.7, within noise).
- **Fail action honored: chat-v2 does not ship from this stage.**
- Follow-up measurement (not gate revision; the verdict stands):
  full-set ARC-Easy (2,376 items, SE ±1.0): base .520 vs sft-full .498
  (acc, −2.2) and .471 vs .447 (acc_norm, −2.4) — **the regression is
  real but roughly half the slice estimate**. The measured alignment
  tax at 310M SFT tokens: ~2.2 ARC-Easy points for 0.238 nats of chat
  quality; the 60M-token SFT paid ~none. Remedy candidates: replay-mix
  Stage-1b (5–10% pretraining data in the SFT stream) and/or shorter
  SFT. `benchmarks/results/12-post-training-arc-fullset.json`.

**Stage 2 (dpo)**: 9.0h, one epoch over 53,952 pairs. First launch OOM'd
at ~step 50 — a NEW species of the MPS allocator pathology: variable-
length batches create thousands of unique tensor shapes and the
allocator fragments (57GB peak). Fix: bucket sequence lengths to
multiples of 128 + aggressive cache clearing → 43GB peak and 1.9x
FASTER (9.7s vs 18.4s/step). Recorded for the paper's systems section.

- **G3: PASS** — held-out preference accuracy **0.652** (gate > 0.60).
- **G4: PASS** — vs Stage-1 (acc): HellaSwag −0.3, PIQA −0.3,
  ARC-E −0.4, WinoGrande −0.7 — all within a point. Preference tuning
  preserved capability exactly.

**Stage 3 (distillation probes)**:

- **P1: PASS — logit-KD is viable.** GPT-2-XL full-pass on the shared
  val: **2.746** vs our 09 base **3.042** (identical protocol/val) —
  the only strong gpt2-tokenizer teacher leads our base by **0.296
  nats**. Distillable headroom exists with zero tokenizer friction.
  (`distill-probe-p1.json`)
- **P2**: Qwen3-1.7B-4bit generates at **161 tok/s ≈ 13.9M tokens/day**
  on this machine — a 100M-token synthetic corpus costs ~7 generation-
  days. (SmolLM2-1.7B-Instruct measurement returned empty output;
  one teacher datapoint suffices for pricing.) (`distill-probe-p2.json`)

## Milestone verdict

Post-training at this scale is a **usability lever, not a capability
lever** — now measured, not assumed: chat quality improved 0.24 nats
(G1) and preference alignment works cleanly (G3) at zero DPO-stage
capability cost (G4), but the SFT stage itself pays a real alignment
tax (G2: −2.2 ARC-Easy full-set points at 310M tokens vs ~0 at 60M) —
consistent with the FLAN scale-crossover, reproduced below any scale
that paper tested. **chat-v2 does not ship** (G2 fail action); the
Stage-1b replay-mix remedy was considered and **shelved by decision**
(2026-07-27): milestone 12 stands as pure science, v1 chat remains the
public model, and the tax curve stays at two measured points.
The distillation probes return a GO signal for milestone 13: a viable
same-tokenizer logit teacher with 0.3 nats of headroom, and a priced
synthetic-generation alternative.

## Chat benchmarks (added post-milestone; matched harness throughout)

**IFEval** (rule-verifiable instruction following; generation via the
MLX server for our no-cache checkpoints, HF backend for the reference;
`--apply_chat_template` everywhere):

| Model | prompt-strict | inst-strict |
|---|---|---|
| SmolLM2-135M-Instruct (2T pretrain + full post-train stack) | 21.4% | 35.6% |
| **sft-full (ours)** | **15.2%** | **26.7%** |
| dpo (ours) | 13.9% | 25.1% |

Readings: real instruction-following exists at 232M (~1/4 of atomic
constraints satisfied); our SFT model reaches ~71% of the reference's
prompt-strict score at ~2000x fewer pretraining tokens; and DPO
*slightly hurt* constraint-following (−1.3 prompt-strict) —
UltraFeedback optimizes preferred-sounding answers, not rule
compliance, a tension now measured at this scale. (SmolLM2-Instruct's
model card reports IFEval 29.9 under its own setup vs 21.4 on our
harness — the re-run-references-locally rule earning its keep again.)

**Extended loglikelihood suite** (n=300, acc): BoolQ .623/.637/.643 and
TruthfulQA-mc2 .393/.418/.416 (base/sft/dpo) — both real signal, both
slightly favoring the chat models. ARC-Challenge, OpenBookQA, and
CommonsenseQA sit at chance at this scale, consistent with the repo's
benchmark guidance. Full JSONs: `benchmarks/results/*-ext.json`,
`benchmarks/results/ifeval-*`.
