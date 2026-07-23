# 11-efficiency

The training-cycle program: after `09` showed the token axis saturating,
the question became intelligence **per training hour**, not per token.
Four lanes, each an A/B against the `05` hybrid run (seed 1337, val
3.8091 / probe 3.8219), which serves as the already-trained control —
same 114M hybrid model, same 100M-token budget (6,100 steps), same val
set for every lane. The fair-comparison rule now holds everything fixed
*except the lever under test*.

## Pre-registered gates (written before results)

| Lane | Lever | Gate to run | Adoption rule for flagship-3 |
|---|---|---|---|
| `muon/` | Muon on hidden matrices, AdamW on embeddings/norms | sane probe | val ≤ control − 0.02 at equal steps |
| `bf16/` | autocast bf16 forward/backward | ≥1.2x throughput, no NaN | val within +0.01 of control |
| `epoch4/` | 4 epochs of a 25M-token stratified sub-pool | — | within +0.015 of control ⇒ multi-epoch declared safe |
| `edu-score4/` | FineWeb-Edu int_score≥4 only (same blend, same val) | — | val ≤ control − 0.02 |

**Flagship-3 target** (pre-registered): the adopted combination must match
`09`'s val loss (3.0700) in ≤ half its 127h wall-clock, or beat it at
equal wall-clock.

## Probe results (gates applied)

- **Muon**: NS5 orthogonality ‖OOᵀ−I‖∞ = 0.19 ✓, loss decreasing ✓;
  overhead **4.90s/step vs 4.1 control (+19%)** from fp32 Newton-Schulz on
  MPS — so Muon must beat the control by >19% token efficiency to win on
  wall-clock, not just on loss. **Lane runs.**
- **bf16**: trajectory parity fine (Δ0.009 @ 30 steps) but throughput
  **1.01x — no speedup on MPS at this size**. Gate says cut. **Lane does
  not run**; recorded as a negative result: MPS autocast bf16 buys nothing
  here (fp32 matmul throughput appears to be the same silicon path).
- **epoch4 / edu-score4**: data-only lanes, no gate probe needed. Sub-pool
  built (25M stratified); filtered pool building (survival rate reported
  by its data.py).

## Adjudication protocol (pinned 2026-07-23, BEFORE edu-score4 completion)

Per `design-review-20260723.md` (F1/F2), the edu-score4 gate is decided by
a **deterministic full-pass validation loss** computed identically for the
control (`05-data-frontier/hybrid/out/ckpt.pt`) and the lane's final
checkpoint by `adjudicate.py`:

- **Decontaminated val subset**: the 05 val minus the FineWeb documents
  whose mid-doc 96-token window byte-matches
  `edu-score4/data/train.bin` in the shard-001-tail cluster (the review's
  census: 200 docs / 215,377 tokens / 10.8% of val — re-derived by the
  script). The 3 symmetric near-duplicate docs (present in BOTH pools)
  stay in val. Raw-val numbers are reported alongside for the record.
- **Protocol**: non-overlapping 1024-token windows, stride 1024, partial
  tail dropped, fp32, every predicted token counted exactly once, no
  sampling. One number per checkpoint per subset.
- **Gate**: edu-score4 clean-val ≤ control clean-val − 0.02. **A miss is
  a miss** — no post-hoc margin softening; landing within noise of the
  threshold is a miss. Failure action: lever not adopted for flagship-3.
- The clean subset is score<4-heavy by construction (the removed docs are
  exactly val's score≥4 members); both arms are evaluated on the identical
  subset, so the comparison is fair — this composition shift is disclosed
  in any write-up of the result.

## Results

- **bf16**: CUT at probe (1.01x, gate ≥1.2x). Negative result stands:
  first formal measurement of the MPS autocast throughput null.
- **muon**: FAIL — final val **4.0247** vs gate ≤3.7891 (+0.216 vs
  control), despite an early lead through step 500; +19\% step time.
  **Scope caveat (review F3)**: the lane bundles the update rule with
  LR 0.02 (66.7x the AdamW LR) and no weight decay on the Muon group;
  the result falsifies "speedrun-standard Muon at default settings," not
  "Muon." A fair ceiling test needs an LR/wd sweep.
- **epoch4**: FAIL — final val **4.1314** vs gate ≤3.8241 (+0.322 vs
  control), train loss 3.2334 → **0.90-nat train/val gap**: clear
  4-epoch overfit of the 25M-token sub-pool. The gap, not just the miss,
  is the finding: small-pool reuse at 1:4 is far from free.
- **edu-score4**: FAIL — adjudicated by the pinned protocol
  (`adjudicate.py`, results in
  `benchmarks/results/11-efficiency-adjudication.json`): clean-val
  **3.8741** vs control clean-val 3.8290 (threshold 3.8090) — the elite
  filter (top 13.3\% of FineWeb-Edu by score) *underperforms* the broader
  score≥3 corpus by 0.045 nats at this budget. Diversity loss beats
  quality gain at 100M tokens. The contamination the design review caught
  was real and directional: removing the 203 leaked docs improved the
  control (3.8398→3.8290, they're harder-than-average) while worsening
  the lane (3.8646→3.8741, it had memorized them) — a differential of
  ~0.02 nats, exactly the gate margin. In-run eval noise across the last
  three evals (3.867→3.927→3.844) independently confirmed F2: the
  stochastic estimator swings 3x the margin.

## Milestone verdict

**Four pre-registered challenges, four negatives.** The 05/09 recipe
(AdamW fp32, fresh sub-epoch data, score≥3 corpus) survived every
efficiency lever tested against it: precision (bf16: no speed), optimizer
(Muon@default: −0.22 nats), data reuse (4x small pool: −0.32 nats), and
harder filtering (top-13\%: −0.045 nats even after decontamination). The
pre-registered flagship-3 target ("09 quality at half the wall-clock via
adopted levers") therefore has no adopted levers and is **cancelled as
specified** — the honest conclusion is that at this scale the incumbent
recipe is locally optimal among cheap training-cycle levers, and further
efficiency must come from a different class of lever (distillation,
post-training, architecture). Each negative is scoped in its bullet;
each cost ~7h and is worth more than a lucky positive.
