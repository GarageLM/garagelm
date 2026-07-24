# Design review — 12-budget-ablation (pre-launch)

Reviewer: design-reviewer agent, 2026-07-24 (requested filename date).
Target: `experiments/12-budget-ablation/` — four runs (hybrid/gqa × 50M/200M
tokens) measuring gap(budget) around the existing 05 seed-1337 100M points,
adjudicated by the new `benchmarks/full_pass_val.py` on the shared 05 val.
No training was running during review (`ps aux` checked); all checks CPU-only.

## Verdict

**FIX-THEN-LAUNCH** — one blocker (B1), a README-only pre-registration fix.
The code, plumbing, budget math, and adjudication protocol all check out.

---

## Findings (ranked by cost-if-missed)

### B1 (blocker): no pre-registered seed-noise yardstick for the trend claim — the "decisive experiment" risks an undecidable readout

- Where: `experiments/12-budget-ablation/README.md:12-14` ("Gap persisting →
  … gap closing → …" with no threshold for either).
- Evidence: 08-solidify measured the seed noise of exactly this gap at the
  100M point (`experiments/08-solidify/README.md:14-24`): gap range across 3
  seeds **−0.028 to −0.059 nats** (spread 0.031), and the GQA arm alone has
  probe-mean seed spread **0.025 nats**. The 50M and 200M points are one
  seed each. A gap change between budgets smaller than ~0.03 nats is
  indistinguishable from seed noise, yet the README's two interpretations
  ("persisting" vs "closing") have no boundary — after 35h of compute, a
  result like −0.038 → −0.025 could be argued either way. That is the
  "we will see if it helps" pattern the gates audit exists to catch, on the
  experiment the README itself calls decisive for the paper's central claim.
- Cost if missed: the full 35h — not lost to a crash, but spent producing a
  readout whose interpretation is post-hoc.
- Fix (one sentence in README, before launch): pre-register the yardstick,
  e.g. "gap differences across budgets smaller than the 08 seed spread
  (0.031 nats; GQA-arm probe spread 0.025) are within single-seed noise and
  will be reported as 'no detectable trend'; 'closing' requires the 200M gap
  to shrink vs 100M by more than that spread (and symmetrically for
  'widening')." Numbers from 08 are already in-repo, so this stays
  pre-registered as long as it lands before results exist.

### N1 (note): `full_pass_val.py` default `--micro 8` breaches the documented MPS logits envelope

- Where: `benchmarks/full_pass_val.py:36,62` (`micro=8` default); README run
  command passes no `--micro`.
- Evidence (computed): logits per forward = 8 × 1024 × 50257 × 4 B =
  **1.65 GB** fp32, vs the CLAUDE.md envelope "micro-batch × block × 50257
  fp32 well under 1 GB". `--micro 4` gives 0.82 GB — the proven 05 training
  shape. Inference-only (no activation storage for backward), 1953 windows ≈
  245 forwards, so worst case is a slow/aborted eval re-run in minutes, not
  hours; but the batch32×512 allocator pathology (3.3 GB logits) shows MPS
  punishes large fp32 logits tensors nonlinearly.
- Recommendation: adjudicate with explicit `--micro 4` (results are
  batching-invariant — verified, check C3).

### N2 (note): 200M runs are ~2× the longest prior run in this lane — allocator-pathology exposure

- Where: `experiments/12-budget-ablation/*/train.py:24` (STATE_EVERY=1000),
  `:121-130` (resume + CPU/MPS RNG restore).
- Evidence: 05 gqa hit the in-process MPS allocator pathology at ~iter 2500
  of 6100 (documented at train.py:19-23); the 200M runs go to 12200. The
  mitigation machinery is present and verified intact (state.pt every 1000
  steps ≈ 68 min max loss, atomic `os.replace`, `--resume` restores model,
  optimizer, iter, elapsed, CPU+MPS RNG). Watch s/step at the 500-iter eval
  prints; process restart + `--resume` is the established cure.
- No action required pre-launch; this is an operational watch item.

### N3 (note): relative `ROOT` — launch discipline

- Where: `*/train.py:14` — `ROOT = os.path.dirname(__file__)` (not abspath).
  Launched with a relative script path, DATA_DIR/OUT_DIR resolve against the
  cwd at each `open()`. Same pattern as 05 (verbatim copy) and CLAUDE.md
  already mandates absolute paths for background launches (this exact class
  of bug cost idle hours once). Absolute resolution verified correct for all
  four dirs (check C4). Launch with absolute paths from the repo root.

### N4 (note): val.bin internals — one mixed window, 127-token tail

- Evidence (computed): val = 2,000,000 tokens → 1953 windows, 1,999,872
  predicted tokens, 127-token tail dropped. The fineweb/cosmo boundary at
  token 1,680,000 falls inside window 1640, so one window of 1953 mixes
  sources. Identical for all six checkpoints — no differential effect.
  Also: `data.py`'s tail split cuts at a token offset, not a document
  boundary, so the document straddling each split point contributes its head
  to train and its tail to val — a few hundred tokens of shared context at
  most, a pre-existing 05 property applied identically to every arm and
  budget; it cannot move the gap.

### N5 (note): `full_pass_val.py` not yet in `benchmarks/README.md`

- The README pins adjudication to this script; `benchmarks/README.md` (the
  stated home of the evaluation methodology) doesn't document the protocol
  yet. Documentation-only; fix any time.

---

## Audit 1 — Fairness

Ran `diff -r` of every 12 dir against its named 05 control (check C1).
**Exactly two changed lines per dir, no extra/missing files:**

| Dir | config.py | train.py |
|---|---|---|
| hybrid-50m | `max_iters: 6100 → 3050` (line 27) | `DATA_DIR` re-pointed to 05 pool (line 15) |
| hybrid-200m | `max_iters: 6100 → 12200` (line 27) | same (line 15) |
| gqa-50m | `max_iters: 6100 → 3050` (line 25) | same (line 15) |
| gqa-200m | `max_iters: 6100 → 12200` (line 25) | same (line 15) |

`model.py` and `sample.py` byte-identical to 05 in all four dirs. Both
changes are the declared levers (budget; shared pool). Tokenizer (gpt2 via
meta.pkl), dataset (same train.bin/val.bin bytes), optimizer, seed 1337,
micro 4 × accum 4 × block 1024 all shared — fair-comparison rule holds.

Param parity (check C6): instantiating both arms **from the 12 copies** at
vocab 50257 gives 114,114,048 params each, equal to both 05
`run_summary.json` values. Attention masks carry no parameters.

Data-stream identity: `get_batch` draws from the global CPU RNG seeded 1337;
model init consumes identical RNG in both arms (same shapes/order; RMSNorm
inits are constant); the eval schedule (`it % 500`) and eval_iters=20 consume
identical RNG at equal iteration count in every run. So at any step i, all
runs (both arms, both budgets, and the 05 100M runs) have seen byte-identical
batches — the 50M checkpoint is the 100M trajectory's data stream truncated,
differing only by the intended LR schedule. This is exactly what a budget
ablation wants.

## Audit 2 — Math

- **LR schedule / max_iters coupling** (train.py:75-81): `get_lr` reads
  `cfg.max_iters` directly — each budget gets its own complete cosine, as the
  README pins. Computed endpoints (check C5): lr(0)=1.50e-06 ramp,
  lr(199)=lr(200)=3.00e-04 peak, lr(max_iters−1)=3.0000e-05 ≈ min_lr for both
  3050 and 12200. Warmup 200 = **6.6%** of the 3050-step run (1.6% at 12200)
  — inside the conventional 1–10% band; sensible. **Nothing else in
  TrainConfig couples to max_iters**: eval_interval/eval_iters/STATE_EVERY
  are absolute counts (3050 is not a multiple of 1000 — last state.pt at
  3000, final ckpt.pt at 3050; fine).
- **full_pass_val window/stride arithmetic** (benchmarks/full_pass_val.py:36-55),
  proven by execution (check C3): window w predicts absolute tokens
  [w·block+1, (w+1)·block]; coverage test over N ∈ {203, 1024, 1025, 2048,
  2049, 2,000,000} confirms contiguous, non-overlapping, every predicted
  token exactly once, tail < block dropped, no out-of-bounds read (last y
  index = n_win·block ≤ N−1). Uniform-logits stub returns exactly ln(V)
  (4.615120 vs 4.615121), proving correct count-weighting; tiny real hybrid
  GPT gives identical loss at micro 1/3/12 vs manual per-window recompute
  (max delta 3e-7). Degenerate N ≤ block would divide by zero — irrelevant
  at the 2M-token val, noted for reuse.
- **No last-token-logits trap**: both arms' `forward(idx, targets=None)`
  returns full-sequence logits (hybrid model.py:181-195, gqa
  model.py:155-168), so `model(x)` in full_pass_val is shape-correct.
- **fp32 claim holds**: 05 hybrid ckpt params are torch.float32 (check C6);
  the eval model is constructed fp32 and `load_state_dict` casts to param
  dtype regardless. Dropout is 0.0 and `model.eval()` is set — deterministic.
- **Mask spot-check** (build_sliding_window_mask, T=16, w=4; check C3): row 0
  → {0}; row 3 → {0..3}; row 4 → {1..4}; row 15 → {12..15}. Causal ∧
  (i−j) < w exactly — no coverage gap. RoPE table computed for its own
  head_dim with cat(freqs,freqs) layout and matching rotate_half — the 03-era
  slicing bug is absent (unchanged 05 code).
- **Grad accumulation** (train.py:100-109): loss/accum per micro-batch, clip
  and optimizer step once per optimizer step, LR set per optimizer step —
  correct, and `max_iters` counts optimizer steps as the budget math assumes.

## Audit 3 — Plumbing

- **DATA_DIR resolution** (check C4, executed): all four dirs resolve
  `ROOT/../../05-data-frontier/data` →
  `/Users/anthonytrevino/Desktop/llm-arch-explore/experiments/05-data-frontier/data`;
  train.bin (3,553,800,186 B = 1,776,900,093 tokens), val.bin (4,000,000 B =
  2,000,000 tokens), meta.pkl all present. The historical
  one-directory-too-high bug is not reproduced. (See N3 on launch discipline.)
- **No stale state**: none of the four 12 dirs has an `out/` — nothing for
  `--resume` to pick up accidentally, no copied 05 checkpoints.
- **Results naming** (check C7, executed): the six adjudication outputs —
  `12-budget-ablation-{hybrid,gqa}-{50m,200m}-fullpass.json` and
  `05-data-frontier-{hybrid,gqa}-fullpass.json` — are mutually unique and
  collide with nothing in `benchmarks/results/` (the 05 re-evals get the
  `-fullpass` suffix, distinct from the existing `05-data-frontier-*.json`).
- **Module shadowing in full_pass_val** (lines 66-71): exp_dir is inserted at
  sys.path[0] and stale `config`/`model` modules are popped; `benchmarks/`
  contains no config.py/model.py to shadow. Each checkpoint is evaluated in
  its own process — the right arm's model class is guaranteed.
- **Resume machinery**: present and complete in all four train.py copies
  (STATE_EVERY=1000, atomic tmp+rename, optimizer + CPU/MPS RNG state,
  `--resume` flag; lines 24-38, 121-130, 183).
- **05 source checkpoints for the 100M row**: both
  `experiments/05-data-frontier/{hybrid,gqa}/out/ckpt.pt` exist (456 MB each)
  with `{"model", "model_cfg"}` keys — loadable by full_pass_val (verified by
  actually loading the hybrid one on CPU).

## Audit 4 — Budget

- **Token math** (check C5): tokens/step = 4×4×1024 = 16,384.
  3050 → 49,971,200 (≈50M ✓); 12200 → 199,884,800 (≈200M ✓); the existing
  6100-step 05 runs → 99,942,400, matching their run_summary `tokens_seen`
  exactly — the three budget points are exact 1:2:4.
- **Pool sizing**: 200M budget = 11.2% of the 1,776,900,093-token pool
  (0.112 epochs) — comfortably sub-epoch, consistent with dropout 0.0
  reasoning; the 50M runs see 2.8%.
- **Contamination**: `data.py:85-99` writes train.bin as each shard minus its
  val tail and val.bin as the tails — val tokens are excluded from train.bin
  by construction, and `get_batch` samples train.bin only. The claimed
  reasoning verifies (token-offset split caveat in N4 — non-differential).
- **Memory regime**: training shape identical to the proven 05 runs
  (micro 4 × 1024 × 50257 fp32 logits = 0.82 GB < 1 GB envelope). Eval
  default breaches it (N1).
- **Wall-clock**: measured 05 s/step = 25,449/6100 = **4.17** (hybrid) and
  24,909/6100 = **4.08** (gqa). At 4.0–4.2 s/step: 50M runs 3.4–3.6 h, 200M
  runs 13.6–14.2 h, four-run total **33.9–35.6 h** — the README's
  "2×~3.5h + 2×~14h ≈ 35h" is honest. Sequential-only per CLAUDE.md; check
  machine contention before each launch.

## Audit 5 — Gates

- Pre-registration exists in the README before any run, correctly frames the
  experiment as measurement (no adoption gate), pins arms, seed, pool, block,
  the single lever, the per-budget-cosine convention, equal-budget-only
  comparisons, and — critically — a deterministic adjudication protocol
  (full_pass_val, shared 05 val, fp32, non-overlapping 1024 windows) applied
  to **all six** checkpoints including re-evaluating the 05 finals, so the
  100M row is protocol-identical rather than reusing in-run numbers. The
  script's implemented protocol matches the pinned one exactly (Audit 2).
- The eval plan covers the claim: the claim is a val-loss gap trend, and
  full-pass val loss is the right instrument; no long-range claim is made, so
  the per-position probe is not required (it exists for these arms from
  05/08 if the paper wants it).
- **Gap: no noise yardstick for interpreting the trend — B1 above.** With
  the yardstick sentence added, the decision rule is unambiguous and both
  outcomes remain reportable.

---

## Checks run (all CPU-only; script preserved in session scratchpad)

- **C1** `diff -r` of all four 12 dirs vs their 05 controls (excluding
  out/__pycache__): output reproduced in Audit 1 — 2 lines per dir.
- **C2** Environment sweep: data file sizes; 05 out/ contents; no out/ in 12
  dirs; `benchmarks/results/` listing; `ps aux` — no training running.
- **C3** Executed `full_pass_loss` correctness suite: coverage proof at six
  N values; uniform-logits stub == ln(V) to 1e-6; micro 1/3/12 vs manual
  per-window recompute agree to 3e-7; mask rows 0/3/4/15 exact.
- **C4** Path resolution: DATA_DIR from each of the four train.py locations
  → the 05 data dir; train.bin/val.bin/meta.pkl exist.
- **C5** Arithmetic: token budgets (exact values above), pool fraction
  (11.2%), LR endpoints for both cosines, warmup fractions, wall-clock at
  4.0/4.1/4.2 s/step, eval logits memory at micro 8 vs 4.
- **C6** Instantiated both arms from the 12 copies at vocab 50257: params
  114,114,048 == 114,114,048 == 05 run summaries; loaded 05 hybrid ckpt.pt on
  CPU: float32 params, keys {model, model_cfg}, block_size 1024.
- **C7** result_name() on all six checkpoint dirs: six unique `-fullpass.json`
  names, zero collisions with existing results files.

## Verdict

**FIX-THEN-LAUNCH** — blockers:

1. **B1**: add the pre-registered seed-noise yardstick (from 08's measured
   spread: gap range 0.031 nats across seeds, GQA-arm probe spread 0.025) to
   `experiments/12-budget-ablation/README.md` with an explicit
   persisting/closing decision boundary, before any run starts.

Once B1 lands, launch is safe as designed. Recommended (non-blocking):
adjudicate with `--micro 4` (N1); launch with absolute paths, sequentially,
after a contention check (N3); watch s/step on the 200M runs and use
`--resume` on allocator slowdown (N2).
