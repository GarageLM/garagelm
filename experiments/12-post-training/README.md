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

*(pending: design-review audit → data prep → runs)*
