# Design review — 11-efficiency (2026-07-23)

Reviewer: design-reviewer (pre-registration adversary).
Target: `experiments/11-efficiency/` — four lanes (`muon/`, `bf16/`, `epoch4/`,
`edu-score4/`) vs the frozen control `experiments/05-data-frontier/hybrid/`
(seed 1337, final val 3.8091, probe 3.8219).
Review conditions: `edu-score4/train.py --resume` was TRAINING ON MPS during this
review (PID 84410, launched 09:41, chained after epoch4). All executed checks were
strictly CPU-only (byte compares, mmap searches, path resolution, pure-math LR
checks). No model forwards were run; no MPS allocation.

Status at review time: bf16 cut at probe (per its gate), muon and epoch4 completed,
edu-score4 mid-run (~iter 5000+, val ~3.89). This review is therefore part
retrospective calibration, part live audit of the lane whose verdict lands next.

## Verdict

**FIX-THEN-LAUNCH** — the edu-score4 *run* is scientifically fine to let finish
(model, config, seed, and training data lever are all clean); the blockers are on
**adjudication**: the gate as currently constituted cannot honestly decide the
edu-score4 race, and no flagship-3 decision should be taken from it until both
blockers below are addressed. Neither requires retraining.

## Findings (ranked by cost-if-missed)

### F1 — BLOCKER: edu-score4's training pool contains 10.8% of the val set verbatim (bias in the lane's own favor)

`edu-score4/data.py:55-77` streams FineWeb-Edu shards `000`–`003`
(`FWE_SHARDS`, line 25) start-to-finish, keeping `int_score >= 4` rows until
250M filtered tokens (`FWE_TARGET`, line 29). But the 05 val set's FineWeb
portion is the **tail of shard 001** (`05-data-frontier/data.py:28-29`: 1,680,000
tokens held out of `fineweb-edu-001`'s tail; `concat_bins`, lines 85-99). At the
~12% score>=4 survival rate, shards 000+001 yield only ~210M filtered tokens
— so the builder consumed **all of shard 001, including the held-out val tail**,
and continued into shard 002. Every score>=4 document that 05 held out as val
was thereby re-included in this lane's training data.

Measured (CPU census, 96-token mid-doc windows, byte-exact, alignment-checked):

- **200 of 1,668** FineWeb val documents found verbatim in
  `edu-score4/data/train.bin` — **215,377 val tokens = 10.8% of the 2M val set**.
- They form a contiguous cluster at train.bin token offsets
  **209,891,173–210,121,069** — exactly where shard 001's filtered tail lands,
  confirming the mechanism (not coincidental duplication).
- 3 further val docs (14,709 tokens, 0.7%) match at other offsets; all 3 are
  also present in the **control's** train.bin (corpus-internal near-duplicates)
  — symmetric, not a fairness issue, no action.
- Cosmopedia val portion: **0/15 sampled docs** in the pool (the cosmo take,
  `data.py:84-90`, reads ~46M tokens from the shard head; the 320k val tail is
  never reached). Clean.
- `epoch4/data/subpool.bin`: 0/40 val docs found. Clean.

Why it's the top finding: contaminated tokens are expected to be sampled ~0.34x
each during the 100M-token run (100M / 296.5M pool), and the bias is
*directional* — it lowers this lane's val loss and only this lane's. The win
gate's margin is 0.02 nats; a memorization effect of even 0.05–0.2 nats on the
contaminated 10.8% shifts total val by ~0.005–0.02 nats — up to the entire
margin. Cost if missed: the lane "wins" its gate spuriously, the lever is
adopted into flagship-3 (pre-registered target references 09's 127h run; the
adopted-combo run is a multi-day commitment), and the headline data-quality
conclusion is wrong.

**Fix (post-hoc, no retrain):** adjudicate on the decontaminated val subset —
drop the 200 identified docs (re-derivable in ~60s: any FineWeb val doc whose
mid-doc 96-token window byte-matches `edu-score4/data/train.bin`), and evaluate
**both** final checkpoints (`05-data-frontier/hybrid/out/ckpt.pt` and
`edu-score4/out/ckpt.pt`) on that same clean subset with the same protocol.
Report raw-val alongside for the record. Note the clean subset is score<4-heavy
(the removed docs are precisely val's score>=4 members); since both arms are
evaluated on the identical subset, the comparison stays fair — but state this
composition shift in the writeup.

### F2 — MAJOR: the edu-score4/muon win gates cannot adjudicate a close race as written

`README.md:14-19` pre-registers "val ≤ control − 0.02" (edu-score4, muon). Three
ambiguities, each larger than or comparable to the margin in a close race, and
edu-score4 is one (~3.89 at iter 5000; control final 3.8091; threshold 3.7891;
the cosine tail — LR already down to 5.25e-5 at iter 5000 — typically closes
most of that gap):

1. **Which control number?** `README.md:5-6` publishes two: 3.8091 (in-run
   final) and 3.8219 (probe protocol). They differ by **0.013 — 65% of the
   gate margin**. The gate does not say which.
2. **Which measurement protocol?** The in-run number comes from
   `estimate_loss` (`edu-score4/train.py:61-72`): 20 random 4×1024 batches ≈
   80 windows drawn from the live RNG stream — a stochastic estimator whose
   draw-to-draw variation is plausibly the same order as the 0.02 margin (the
   0.013 protocol gap above is direct evidence that measurement choices move
   the number by comparable amounts). A deterministic evaluation exists in the
   repo (`benchmarks/long_range_probe.py`, 256 fixed windows) but the gate
   doesn't invoke it.
3. **No noise/tie rule.** "≤ control − 0.02" has no stated action for a result
   landing within measurement noise of the threshold.

None of this bit muon or epoch4 (they missed by 0.22 and 0.31 — unambiguous at
any protocol), but edu-score4 may land within ~0.01 of the line.

**Fix (pin BEFORE the run finishes, i.e., now):** adjudicate on a deterministic
full-pass val loss — non-overlapping 1024-token windows over the (decontaminated,
per F1) val set — computed identically for both final checkpoints; that single
number decides. Pre-state the tie rule (e.g., a miss is a miss; no post-hoc
margin softening).

### F3 — MINOR (retrospective): the muon "lever" is a bundle, so the negative result is narrower than the lane name

The single-lever claim is the optimizer, but the arm actually changes three
things at once: the update rule, the hidden-matrix LR (`muon/train.py:15`,
`MUON_LR = 0.02` — 66.7x the AdamW LR), and **weight decay** (control's AdamW
applies wd 0.1 to *all* params, `05-data-frontier/hybrid/train.py:94-96`; the
lane's Muon group has none, `muon/muon.py:31`). The 4.0247-vs-3.8091 result
therefore falsifies "speedrun-standard Muon at LR 0.02, no wd" — not "Muon."
Record the caveat in the README Results section; do not generalize.

(The LR *schedule* itself is clean — see checks: the single multiplicative
factor (`muon/train.py:37-40,185`) reproduces the control's cosine exactly for
the AdamW group and scales Muon's identically; both decay to 0.1x peak.)

### F4 — MINOR: `long_range_probe.py`'s default val path does not resolve for any 11-efficiency lane

`benchmarks/long_range_probe.py:42`: default val bin is
`<experiment-dir>/../data/val.bin` → `experiments/11-efficiency/data/val.bin`,
which does not exist (verified). Every probe invocation for these lanes must
pass an explicit absolute `--val-bin` (the 05 val or the lane's copy — hashes
are identical, either is correct; for F1's sake use the decontaminated subset
where the comparison is edu-score4 vs control). Cost if missed: minutes, not
days — but it is exactly the class of path assumption this repo has been bitten
by, and `11-efficiency-muon-per-position.json` exists while
`11-efficiency-muon.json` (quality eval) does not, so eval runs for this
milestone are still ahead.

### F5 — NOTE: README hygiene

- `README.md:40` still says "*Three runs queued sequentially*" although muon
  (final val **4.0247**, fail vs ≤3.7891) and epoch4 (final val **4.1314**,
  train 3.2334 → **0.90-nat train/val gap = clear 4-epoch overfit**; fail vs
  ≤3.8241) have completed. Record both, including epoch4's train loss — the
  gap is the scientifically useful part of that negative result.
- The epoch4 and edu-score4 gate rows have no explicit failure action (the
  muon/bf16 rows imply theirs). They adjudicated/will adjudicate anyway, but
  pre-registration should state both branches.

## The five audits

### 1. Fairness — PASS with F1/F3 carve-outs

- `config.py` and `model.py`: **byte-identical to control for all four lanes**
  (ran `diff`; all IDENTICAL). TrainConfig equality additionally confirmed
  end-to-end: `muon/out/run_summary.json` and `epoch4/out/run_summary.json`
  embed a `train_config` identical to the control's (seed 1337, 6100 iters,
  eval_iters 20, lr 3e-4/3e-5, warmup 200, wd 0.1, clip 1.0).
- `train.py` diffs vs control, per lane:
  - `edu-score4`: **one line** — `DATA_DIR` (line 15). Only the data lever
    changes. (But see F1: the *content* of that data breaks val hygiene.)
  - `epoch4`: DATA_DIR + `load_data` reads `subpool.bin` and takes val/meta
    from the 05 pool (lines 15, 49-54). Data lever only.
  - `bf16`: DATA_DIR + autocast context in `train_step` (lines 109-110); eval
    stays fp32. Lever only. (Lane cut at probe per its gate — correctly.)
  - `muon`: DATA_DIR + DualOptimizer/param-split/set_lr_factor. Lever plus the
    F3 bundle (wd + LR magnitude ride along undocumented).
- Param counts: 114,114,048 in both completed lanes' run summaries — equals
  the control's documented count. edu-score4 must match (identical model.py,
  config.py, and hash-identical meta.pkl → same vocab).
- Val identity (edu-score4): `data/val.bin` MD5 `de54cf66…` and `meta.pkl` MD5
  `e48cae16…` — **both identical to the 05 originals**; sizes 4,000,000 and
  50 bytes. epoch4 reads the 05 val in place. Same val for every lane. ✓

### 2. Math — PASS

- Muon LR schedule (pure-math check, no torch): factor = get_lr(it)/3e-4
  applied to base LRs. AdamW-group LR equals the control's `get_lr` **exactly**
  at every probed iter (0, 100, 199, 200, 2000, 5000, 6099); Muon group is a
  constant 66.7x scale of it; both warm up linearly over 200 iters and decay
  cosine to 0.1x peak (muon 0.02 → 2.0e-3; control ratio 0.1). Shape preserved
  for both param groups as the docstring claims. ✓
- Muon param split: `p.ndim == 2 and "tok_emb" not in n` with
  `assert len(muon_params) == 7 * n_layer` (`muon/train.py:130-132`) — the
  assert held at runtime (lane completed), so the tied lm_head/embedding and
  1D norm gains stayed on AdamW as intended. ✓
- Newton-Schulz (`muon/muon.py:13-27`): pre-normalization by ‖G‖, quintic
  coefficients and tall-matrix transpose handling match the speedrun
  formulation; aspect-ratio scale `max(1, rows/cols)^0.5` on line 52. Probe
  measured ‖OOᵀ−I‖∞ = 0.19 (README:27). No issues.
- Loss scaling under grad accum (`(loss/accum).backward()`, clip and step once
  per optimizer step) is unchanged from the control in all four lanes. ✓
- No attention/RoPE/mask changes anywhere this milestone (model.py identical),
  so the standing RoPE/mask/pooling test cases don't apply.

### 3. Plumbing — PASS now; the epoch4 launch bug is exactly what this audit exists for

- Path resolution executed for every lane from each train.py's ROOT: muon and
  bf16 → 05 `train.bin` (3,553,800,186 B) and `val.bin` (4,000,000 B) exist;
  epoch4 → `data/subpool.bin` (50,000,000 B) + 05 val/meta exist; edu-score4 →
  `data/train.bin` (593,061,778 B) + local val/meta exist. **All resolve.**
- **Retrospective calibration:** the pre-fix epoch4 path
  (`os.path.join(ROOT, "..", "data")`, i.e. the 05-layout line copied verbatim
  — visible as the control's `train.py:15`) resolves to
  `experiments/11-efficiency/data/train.bin`: **exists=False** (checked). This
  is standing historical bug #4 ("copied train.py whose DATA_DIR resolved one
  directory too high"), and audit 3's mandated one-liner
  (`os.path.exists` from each train.py's ROOT, run pre-launch) catches it in
  seconds. Answer to the calibration question: **yes — audit 3 would have
  caught it before launch**, as a routine check, not a lucky read.
- Results collisions: `benchmarks/results/` naming is
  `<path-under-experiments>` joined with `-` (`run_quality_eval.py:31-36`), so
  these lanes write `11-efficiency-{muon,epoch4,edu-score4}*.json` — no
  collision with anything existing (only `11-efficiency-muon-per-position.json`
  so far). Each lane has its own `out/`. ✓
- Resume machinery present in all lanes (STATE_EVERY=1000, `--resume`, CPU+MPS
  RNG save/restore); the live launch used absolute paths and `caffeinate`
  (verified in the process table). ✓
- F4 (probe default val path) is the one open plumbing item.

### 4. Budget — PASS

- Token math: 4 × 4 × 1024 = 16,384 tokens/step × 6,100 = **99,942,400** —
  matches both completed lanes' `tokens_seen` and the stated 100M budget. ✓
- epoch4 epochs: 99.94M / 25,000,000 = **4.00** ✓ (stochastic sampling → ~4x
  in expectation, as documented). Sub-pool stratification: FWE_TAKE 21,125,000
  = 84.5%, COSMO_TAKE 3,875,000 = 15.5%; control pool is 1,501,640,104 /
  1,776,900,093 = **84.509% FineWeb** — match to 3 decimal places. ✓
- **epoch4 offsets verified against the pool itself**, not just the docs:
  decoding across `COSMO_START` = 1,501,640,104 in the 05 train.bin shows web
  text (river alluvium) ending mid-document immediately before, and
  Cosmopedia-style synthetic-textbook prose immediately after — the boundary
  is where `epoch4/data.py:16` says it is. Byte-identity spot checks:
  `subpool[:64] == pool[:64]`, `subpool[21_125_000:+64] == pool[COSMO_START:+64]`,
  and subpool tail == `pool[COSMO_START+3_875_000−64 : +3_875_000]` — all True.
  The sub-pool is exactly the claimed stratified slice, with no val overlap. ✓
- edu-score4 pool: 296,530,889 tokens ≥ 100M budget → sub-epoch (~0.34
  expected exposures/token); implied blend ≈ 250.6M FWE + 46.0M cosmo = 15.5%
  cosmo — control blend preserved. ✓
- Memory: 4 × 1024 × 50257 × 4B ≈ 823MB logits — the control's own
  established envelope (five completed runs at this exact shape), well clear
  of the batch32×512 pathology. ✓
- Wall-clock: muon 27,408s (7.6h; ~4.49 s/step average vs 4.90 probe — probe
  was conservative), epoch4 24,502s (6.8h ≈ control's ~7h). The README's ~22h
  sequential estimate was honest. ✓

### 5. Gates — F2/F5

Pre-registered before results, measurable, in the README — the discipline is
real, and it correctly killed bf16 at probe time and adjudicates muon/epoch4
without argument. But the win gates lack a pinned measurement protocol and a
noise rule (F2), the exact deficiency the live close race exposes; and the gate
rows lack explicit failure actions (F5). The eval plan otherwise covers the
claims: data-lever lanes claim distribution-level quality on the shared val —
final val loss is the right primary metric, with the per-position probe and
small-scale lm-eval tasks available as secondaries (run with explicit
`--val-bin`, F4).

## Checks run (all CPU-only; training active throughout)

1. `ps aux` — confirmed edu-score4 training live on MPS before any check.
2. `diff` of config.py/model.py, all 4 lanes vs control → all IDENTICAL.
3. `diff` of train.py, all 4 lanes vs control → diffs quoted under audit 1.
4. `md5` of edu-score4 vs 05 `val.bin` / `meta.pkl` → identical.
5. mmap byte-search of val doc windows in `edu-score4/data/train.bin`:
   60-doc sample (4 hits), 200-doc sample (26 hits, all in one cluster), full
   1,668-doc census → 203 hits / 230,086 tokens; split: 200 docs (215,377
   tokens, 10.8% of val) in the shard-001-tail cluster at offsets
   209,891,173–210,121,069; 3 docs elsewhere, all 3 also found in the
   control's train.bin (symmetric duplicates). Cosmo val: 0 hits.
   `epoch4/data/subpool.bin`: 0 hits.
6. tiktoken decode across `COSMO_START` in 05 train.bin + subpool byte-identity
   spot checks (outputs under audit 4).
7. Pure-math LR trajectory table, control vs DualOptimizer factor scheme
   (7 probe iters; outputs under audit 2).
8. `os.path.exists` resolution for every lane's data paths **and** the pre-fix
   buggy epoch4 path (False) — audit 3.
9. `benchmarks/results/` listing + results-naming code inspection — audit 3.

Not run (and why): any model forward (would contend with the live MPS run); a
direct measurement of `estimate_loss` draw noise (same reason — the 3.8091 vs
3.8219 published discrepancy stands in as evidence for F2).

## Required before adjudication (blocker checklist)

1. **F1**: build the decontaminated val doc mask (200 docs); evaluate control
   and edu-score4 final checkpoints on the identical clean subset,
   deterministic full-pass protocol; adjudicate the gate on that number, report
   raw val alongside with the contamination note.
2. **F2**: write the pinned protocol (metric, subset, window scheme, which
   checkpoint, tie rule) into the README *before* the run finishes.

No retraining, no run interruption required. epoch4 and muon verdicts stand as
unambiguous gate failures once recorded (F5).
