# Design review — 12-post-training (pre-launch)

Reviewer: design-reviewer agent. Date: 2026-07-25. Scope: `sft-full/`
(full-SmolTalk SFT of the 09 flagship, 18,300 steps) and `dpo/`
(ultrafeedback_binarized DPO from the Stage-1 checkpoint). No training
was running during review (`ps aux` clean); all checks CPU-only.

## Verdict

**FIX-THEN-LAUNCH** — blockers F1 (before the SFT launch) and F2
(before the DPO sanity/launch). Everything else is advisory. The
training code itself passed every correctness check I could run: the
sft-full arm is a verified minimal diff of the proven 10-sft trainer,
and the new DPO logprob/loss math is numerically exact against a
manual reference on a tiny CPU model.

---

## Findings (ranked by cost-if-missed)

### F1 — BLOCKER (Gates): G1 has no pinned evaluator, and no masked deterministic evaluator exists in the repo

`README.md:13-18` pre-registers G1 as "sft-full final masked val < the
`10-sft` checkpoint's masked val on this val, by >= 0.05 nats",
evaluated on the new val "at gate time". No script is named, and none
exists that can adjudicate it:

- The lab's standard adjudicator, `benchmarks/full_pass_val.py`, was
  built specifically to institutionalize "no verdict from stochastic
  in-run estimates" (its own docstring, line 2-4) — but it computes
  **unmasked** CE over every token and takes no mask bin
  (`grep -n mask benchmarks/full_pass_val.py` → zero hits; verified
  across all `benchmarks/*.py`).
- The only masked estimator anywhere is `sft-full/train.py:63-74
  estimate_loss`, which samples `eval_iters=20 × micro_batch=4 = 80`
  **random** 1024-token windows from the 461,099-token val. Analytic
  noise band: with per-window masked-loss std ~0.2-0.4 at loss ~1.7,
  SE ≈ 0.02-0.045 nats **per checkpoint**; the difference of two such
  estimates has noise comparable to the 0.05-nat threshold. G1 could
  flip on the RNG draw.
- The README also mixes protocols: "final masked val" (a train-log
  number for sft-full) vs "evaluates all three checkpoints ... at gate
  time" (a re-eval for 10-sft/09). These are different estimators.

This is the exact bug class the repo's own history flagged (full-pass
evaluator institutionalized after a prior design-review F2). The 38h
run itself is unaffected, but its headline ship/no-ship question
("10x data must beat 1x data") cannot be answered cleanly as designed.

**Fix**: before launch, pin the protocol in the README and add a
masked full-pass evaluator (deterministic, non-overlapping stride-1024
windows, `sum(CE*mask)/sum(mask)`, one number per checkpoint, applied
identically to `sft-full/out/ckpt.pt`, `10-sft/out/ckpt.pt`, and
`09-flagship-2/out/ckpt.pt` on `sft-full/data/val_{tokens,mask}.bin`).
~30 minutes; e.g. a `--mask-bin` flag on `full_pass_val.py` or a small
sibling script.

### F2 — BLOCKER (DPO gate integrity): sanity mode does not run the pre-registered sanity

`README.md:23-24` pre-registers: "overfit 10 pairs — DPO loss below
0.1 and implicit-reward margin strictly increasing. No launch without
it." The code (`dpo/train.py:148-164`) deviates on all three clauses:

1. **4 pairs, not 10.** `ids = data.train_ids[:10]` (line 149) but the
   loop fetches `ids[:MICRO_PAIRS * 2]` (line 153) = the first **4
   pair ids**, every step. Six of the ten pairs are never touched.
2. **Over-envelope forward.** `fetch(4 pairs)` returns 8 sequences;
   at max pair length that is 8×1023×50257 fp32 = **1.53 GiB** of
   logits per forward, plus a same-size `log_softmax` copy retained
   for backward (`train.py:81`), for policy and (transiently)
   reference. The documented MPS envelope is "well under 1GB"
   (CLAUDE.md); the training path itself is fine (2 pairs = 4 seqs =
   0.77 GiB) — only sanity mode exceeds it.
3. **Weaker margin rule.** Line 161 checks `margins[-1] > margins[0]`
   (first vs last), not "strictly increasing" as registered.
4. **Unverified calibration.** 40 steps at fixed LR 1e-6 (no schedule,
   no clip in this path) must drive loss from ln2=0.693 to <0.1, i.e.
   a DPO logit of ~+2.2 = 22 summed nats of chosen/rejected
   divergence. Plausible with Adam on 4 repeated pairs, but
   unmeasured; if it is unreachable, the pre-registered launch gate
   fails on a healthy implementation and invites an ad-hoc gate edit.

**Fix**: cycle all 10 pairs in chunks of `MICRO_PAIRS` (2 pairs = 4
seqs per forward, inside the envelope), either enforce or re-register
the margin rule, and treat the number of sanity steps / sanity LR as
tunable-before-results (state it in the README when pinned). Then run
the sanity (minutes, and it is required before launch anyway).

### F3 — MAJOR (Budget): DPO wall-clock is unstated; expect ~13-21h, and it is unprobed

The README states no cost for either stage. Recomputed:

- **sft-full**: 18,300 × 16,384 = 299,827,200 tokens vs the
  310,001,500-token pool = **0.9672 epochs** (matches "~0.97"). At the
  10-sft measured rate (27,278.8s / 3,660 iters = **7.454 s/step**,
  `experiments/10-sft/out/run_summary.json`) the projection is
  18,300 × 7.45-7.60 s = **37.9-38.6 h**. Consistent with the ~38h
  planning number; fine, but record it in the README.
- **dpo**: ultrafeedback_binarized `train_prefs` is ~61k pairs; after
  the MAX_LEN=1024 skip (`dpo/data.py:63`) expect roughly 40-60k kept
  (unknowable until the build finishes — pairs.bin was still building
  at review time). At 16 pairs/step that is ~2,500-3,750 steps. Each
  step = 8 micro × (policy fwd+bwd + ref fwd) on 4 seqs × ≤1023; from
  the 10-sft micro rate (~1.86s fwd+bwd, fwd ~0.6s) that is ~19-20
  s/step worst-case, less at the (shorter) average pair length →
  **~13-21 h**. Not a problem on its own, but: state the budget in the
  README before launch, and run the lab-standard ~10-step throughput
  probe with an abort gate (CLAUDE.md requirement) — the DPO step
  shape (two models, padded pairs) has never been timed on this
  machine.

### F4 — MINOR (Gates): G2/G4 metric wording has two small ambiguities

`README.md:16-17`: "within ±2 pts of the 09 base on every task".
(a) The lab reports acc **and** acc_norm (10-sft README table) — pin
which one (or both) gates. (b) n=300 binomial SE is ~2.9 pts
unpaired; the gate is only meaningful as a **paired** comparison on
the same 300 items (which `run_quality_eval.py --limit 300` with the
fixed harness does provide). One sentence in the README pins both.
Evaluator itself is fine: `benchmarks/run_quality_eval.py:12`
MAIN_TASKS = hellaswag/piqa/arc_easy/winogrande, default `--limit
300`, matches the gate as written.

### F5 — MINOR (Plumbing): `dpo/train.py --epochs > 1` crashes instead of reshuffling

`dpo/train.py:168` computes `n_steps` for `epochs>1`, but the cursor
walks straight off `order` (line 193): `fetch([])` then dies on
`max()` of an empty sequence (line 68). Default 1.0 is safe; guard or
delete the flag.

### F6 — MINOR (Cosmetics with confusion risk): DPO accuracy counts ties as losses

`dpo/train.py:99,120`: `acc = (logits > 0)`. At init policy==ref →
logits are exactly 0 → printed accuracy is **0.00**, not 0.5
(verified: tiny-model check printed `loss=0.693147 acc=0.00`). First
log lines of a healthy run will look broken. Conservative for G3
(ties don't inflate it), so acceptable — just expect it.

### F7 — MINOR (Fairness of the val slices): both val sets are positional, not sampled

- SFT val = **first 500 conversations** of SmolTalk `data/all` in
  shard order (`sft-full/data.py:86`); representative only if the
  published `all` mixture is pre-shuffled. Both G1 arms share this
  val, so the comparison is internally consistent regardless — but if
  the head of shard 0 is a single subset, G1 measures "10x data on
  that subset". Cheap check at gate time: decode a sample of the 500
  and eyeball subset diversity (the head decodes as magpie-style math,
  see checks below — inconclusive from one conversation).
- DPO val = **last 1000 pairs** in dataset order (`dpo/data.py:26`,
  `train.py:55-56`). Same caveat for G3.

### F8 — NOTE: dpo/config.py carries a dead copied TrainConfig; DPO hyperparams live as train.py constants

`dpo/config.py:25-43` is the sft-full TrainConfig (max_iters 18300
etc.) verbatim — `dpo/train.py` never imports it; the real
hyperparams are module constants (`train.py:34-39`). Slight deviation
from the "config file per run" convention; mitigated because
`run_summary.json` records beta/lr/steps/pairs_per_step
(`train.py:222-225`). Delete the dead dataclass or move the constants
into it.

### F9 — NOTE: small resume/eval nits in dpo/train.py

- Resume saves `rng_cpu` but not `rng_mps` (`train.py:209-211`);
  benign here (dropout=0.0, data order is numpy-side and
  deterministic from SEED), unlike the sft trainer which saves both.
- `pref_accuracy` default `batch_pairs=4` = 8 seqs/forward = 1.53 GiB
  logits transient under no_grad (twice, policy+ref). Works on 48GB
  but exceeds the documented envelope; `batch_pairs=2` costs ~nothing.
- G3 runs only once, after the last step (`train.py:217`); a crash
  there loses no training (state.pt every 200 steps) but re-running
  the final eval requires re-entering main; acceptable.

---

## What was verified clean (checks run, with outputs)

**Fairness — sft-full vs 10-sft (diff of every file):**
- `model.py`, `sample.py`, `chat.py`: byte-identical.
- `config.py`: single diff — `max_iters` 3660 → 18300 (the documented
  lever, `sft-full/config.py:32`).
- `train.py`: single diff — `DEFAULT_INIT` gains one `".."`
  (`sft-full/train.py:29`), required by the extra directory level;
  resolves to `/Users/anthonytrevino/Desktop/llm-arch-explore/experiments/09-flagship-2/out/ckpt.pt`,
  **exists**, and is the identical absolute path 10-sft trained from
  (its `run_summary.json` `init_ckpt`).
- `data.py`: `format_conversation` is **AST-identical** after
  docstring strip (checked programmatically). Remaining diffs
  enumerated: source shards (2 named → `data/all` × 9, the data
  lever), TARGET_TOKENS 60M → 310M (data lever), handle-dict
  streaming refactor with `.part2` + atomic `os.replace` (inert),
  early-stop at TARGET_TOKENS after a **complete** conversation
  (`data.py:92-96`, no torn tail), `os.remove` of processed shards,
  print/meta cosmetics. All are the data lever or inert plumbing —
  fairness holds. `dpo/model.py` and `dpo/config.py` are identical to
  sft-full's (diff). Param count expected-identical to 10-sft's
  231,852,032 (same GPTConfig).

**Math — DPO logprob/mask alignment (tiny CPU model, 4 ragged
sequences, padded batch, `PairData.fetch` padding replicated
exactly):**
- `response_logprobs` (`dpo/train.py:78-85`) vs manual per-sequence
  `sum_{j=r}^{n-1} log p(t_j | t_<j)`: **max |diff| 7.6e-6** — the
  `mask[:,1:]` slice against `tgt=x[:,1:]` is correctly aligned;
  position r-1 predicts token r, so the **first response token is
  included**, the final eot is included, and both sides are treated
  identically. Padding beyond each length is inert (zeros masked;
  causality keeps pad tokens out of real positions' logits).
- r=0 would silently drop token 0's logprob, but `dpo/data.py`
  guarantees `resp >= len(encode("Assistant:")) >= 1` and rejects
  `resp >= len` (`data.py:63`).
- Loss sign/direction: policy==ref gives loss exactly ln2 = 0.693147
  (observed). `-logsigmoid(beta*((pc-pr)-(rc-rr)))` pushes chosen up /
  rejected down — standard sigmoid DPO. `margin` (`train.py:98`) is
  algebraically the same quantity as the DPO logit (correct: the
  implicit-reward margin), so `acc=(logits>0)` is consistent.
- Model returns full-sequence logits with no last-position shortcut
  (`dpo/model.py:181-195`), so `model(x[:, :-1])` is valid; max
  padded length 1024 → input ≤ 1023 ≤ block_size.
- Template consistency: `encode_side` output == `format_conversation`
  tokens minus the trailing `"\n"` after the final eot (verified on a
  synthetic 5-turn conversation), and the DPO scored region
  `[resp, len)` == the SFT mask-1 region of the final assistant turn
  (body + eot). The missing trailing newline is outside the scored
  region and after eot — inert. The DPO policy is scored on exactly
  the distribution the SFT stage trains.
- LR schedule (`train.py:186-187`): 2.0e-8 at step 0 → 9.99e-7 at end
  of warmup → 5.5e-7 mid → 1.0e-7 floor. Sane cosine-to-10% with
  50-step warmup.

**Plumbing:**
- `sft-full` DATA_DIR resolves; all five data files present.
  tokens.bin = 310,001,500 tokens, mask.bin byte-aligned 1:1; val =
  461,099 tokens, aligned. meta vocab_size 50257 matches GPTConfig.
- Masked fraction recomputed from mask.bin: train **0.7069**, val
  0.7139 — matches the stated 0.707.
- Real val stream spot-check: decodes as `User: ...\nAssistant: ...`;
  mask is 0 on the `Assistant:` prefix tokens, 1 on the full body
  through eot, 0 on the following `\n`, and 0 everywhere before the
  first assistant body. Exactly the 10-sft template.
- `dpo` DEFAULT_INIT resolves to `sft-full/out/ckpt.pt` (correctly
  absent pre-Stage-1). `10-sft/out/ckpt.pt` exists (G1 dependency).
  `GPTConfig(**ckpt["model_cfg"])` round-trips (sft saves
  `as_dict()`). pairs.bin/index.npz were still building; code audited,
  index arithmetic (per-side start/len/resp, sequential offsets,
  int64) is consistent with `fetch`; gpt2 ids fit uint16.
- Results naming: `run_quality_eval.py`/`full_pass_val.py` name by
  path under `experiments/` → `12-post-training-sft-full*.json`,
  `12-post-training-dpo*.json`. Grepped `benchmarks/results/`: no
  existing basename collision (the `12-budget-ablation-*` files are a
  different milestone directory; distinct prefixes).
- Resume: sft-full inherits the proven state-every-500 machinery incl.
  MPS RNG; dpo has state-every-200 with atomic replace and
  deterministic data order on resume (`cursor = start_step*16`,
  permutation regenerated from fixed SEED). Both present.
- Memory (DPO full run): two 232M fp32 models ≈ 1.7 GiB + AdamW
  moments 1.7 GiB + grads 0.86 GiB ≈ 4.3 GiB static; training micro is
  4 seqs → 0.77 GiB logits (+ log_softmax copy) — inside the
  documented envelope; ~48GB machine is comfortable. Sanity/eval paths
  exceed the envelope per F2/F9.

**Budget:** 18,300 × 16,384 = 299,827,200 = 0.9672 epochs of the
310.0M pool; 37.9-38.6 h at the measured 7.45-7.60 s/step. DPO
~13-21 h estimated (F3). Total milestone compute ≈ 2.2-2.5 days.

**Gates coverage:** G1 (blocked by F1), G2/G4 (evaluator exists and
matches; wording nit F4), G3 (implemented at `dpo/train.py:217-218`,
1000 held-out pairs = VAL_PAIRS, threshold >0.60 unambiguous; note
G1's deliberate re-eval of the 10-sft checkpoint on the NEW val is
correctly motivated in the README — the two milestones' val mixes
differ, so cross-val comparison is rightly forbidden). Stage-3 probes
P1/P2 are stated with decision rules and commit no training.

## Pre-launch checklist (after F1/F2)

1. Add + pin the masked full-pass evaluator; name it in the README
   under G1 (F1).
2. Fix sanity mode to the registered 10 pairs at ≤4 seqs/forward;
   pin the sanity step count/LR; run it (F2).
3. Record both stages' wall-clock budgets in the README (F3); pin
   acc/acc_norm for G2/G4 (F4).
4. Wait for the DPO data build; check the `pairs kept/skipped` line —
   if the ≤1024-token filter drops a large fraction, note it (it
   biases the pool toward short responses; acceptable, but record it).
5. Standard launch protocol per CLAUDE.md: contention check, sanity
   probe, ~10-step throughput probe with abort gate, absolute paths,
   sequential runs (SFT fully done before DPO — DPO's init is
   sft-full's final ckpt.pt, which is only written after the last
   step).

Verdict: **FIX-THEN-LAUNCH** (F1, F2).
