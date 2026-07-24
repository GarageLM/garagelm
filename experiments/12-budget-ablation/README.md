# 12-budget-ablation

The external review's "decisive experiment" for the paper's central
architecture claim: **does the hybrid-vs-full-attention gap change with
training budget?** The convergence view (arXiv:2606.15378) predicts the
gap closes as training grows; our seed-replicated result at 100M tokens
shows hybrid ahead. This measures the trend.

## Pre-registration (written before any run)

- **This is measurement, not adoption.** No pass/fail gate; the
  deliverable is gap(budget) at three points, whatever it shows. Gap
  persisting → strengthens the paper's inductive-bias reading; gap
  closing → confirms convergence at small scale. Both are reportable.
- **Arms**: hybrid vs GQA full attention, byte-identical code to the 05
  runs (which provide the 100M points), seed 1337, 05 pool, block 1024.
  Only `max_iters` differs per budget: 3050 (50M tokens) and 12200
  (200M tokens). Each budget runs its own complete cosine schedule
  (warmup 200, 3e-4 → 3e-5) — the standard convention for budget
  ablations; gaps are compared between arms at equal budget, never
  across budgets.
- **Adjudication protocol**: `benchmarks/full_pass_val.py` on the shared
  05 val set (deterministic, non-overlapping 1024 windows, fp32) for all
  six checkpoints, including re-evaluating the two 05 finals under the
  identical protocol. In-run estimates decide nothing.
- **Expected cost**: 2×~3.5h + 2×~14h ≈ 35h sequential.

## Results

*(pending: design-review audit → runs → full-pass evals)*

| Budget | GQA (full) | hybrid | gap (hybrid−GQA) |
|---|---|---|---|
| 50M | – | – | – |
| 100M (from 05) | – | – | – |
| 200M | – | – | – |
