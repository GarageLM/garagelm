# 13-moe

MoE at the hardware floor. The question frontier papers don't answer: at
a hard memory budget, is a sparse model worth its bytes? Frontier MoE
comparisons match **active** parameters (equal FLOPs, memory treated as
free). At this lab's floor, memory *is* the budget — so we run both
controls.

## Pre-registered gates (written before any run)

- **Arms** (05 pool, block 1024, 100M tokens, seed 1337, hybrid
  attention everywhere; only the FFN differs):
  - `control-active` — the existing 05 hybrid (114M dense; full-pass
    **3.8398**). No new training.
  - `dense-284m/` — FFN-width-only scale-up (ffn_hidden 8192), ~284M
    dense: the memory-matched control.
  - `moe/` — 8 experts × ffn_hidden 1024, top-2 (renormalized gates),
    Switch aux loss 0.01 (train-time only, never in val CE), fixed
    expert capacity 1.25 (fixed tensor shapes; overflow dropped and
    logged): ~284M total / ~114M active.
- **G-A (vs active-matched)**: MoE full-pass ≤ 3.8398 − 0.03. If 2.5x
  the parameters at equal FLOPs can't clear this, MoE is dead at the
  floor.
- **G-M (vs memory-matched)**: measurement, not adoption. Report the
  MoE−dense284 gap; |gap| < 0.02 (the 08 seed-noise yardstick) is
  declared "parity per byte."
- **G-R (router health)**: min expert EMA load ≥ 2% and mean drop rate
  ≤ 10% at every logged eval; violation = routing collapse → run
  stopped, reported as a negative.
- **Probe gates**: MoE step time ≤ 1.5x the 114M dense step (~4.1s →
  ≤ 6.2s), memory inside the envelope, param counts within 1% of
  analytic (~284M both new arms), router entropy ≈ ln 8 at init,
  15-step loss decrease, init drop-rate < 15%.
- **Adjudication**: `benchmarks/full_pass_val.py`, shared 05 val, all
  three checkpoints. In-run numbers decide nothing.
- **Deferred, explicitly**: LatentMoE / quantile balancing and other K3
  lanes (Attention Residuals, per-head Muon); shared-expert designs;
  router z-loss; MLX MoE inference port (expert paging) — until this
  milestone's verdict justifies them.

## Results

*(pending: sanity → design review → probes → runs)*

| Arm | Total params | Active | Full-pass val | Gate |
|---|---|---|---|---|
| control-active (05 hybrid) | 114M | 114M | 3.8398 | — |
| dense-284m | ~284M | ~284M | – | G-M reference |
| moe | ~284M | ~114M | – | G-A / G-M / G-R |
