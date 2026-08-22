# 14-babylm — does inductive bias pay more when data is scarce?

**Status: PRE-REGISTRATION. No arm has been trained.** Gates below are written
before any training run, per repo convention. Numbers marked `[PIN]` are
deterministic functions of the frozen tokenizer and will be filled in before the
first launch, not after.

## Hypothesis

`12-budget-ablation` measured the hybrid-vs-full attention gap shrinking
monotonically as the training budget grows:

| Tokens | Gap (nats) |
|---|---|
| 50M | −0.072 |
| 100M | −0.057 |
| 200M | −0.041 |

Linear in log₂: **≈ +0.0155 nats per doubling**, i.e.

```
gap(D) = -0.057 + 0.0155 * log2(D / 100M)
```

Read forward, that says restricted attention is a favourable inductive bias whose
advantage *decays with training*. Read backward, it predicts the advantage should
**grow as data shrinks** — and BabyLM's fixed word budgets are the sharpest
data-scarce regime available with a public corpus and a public leaderboard.

**The claim under test: architectural priors matter most exactly where BabyLM
operates.** This milestone tests a prediction the repo already owns, in a regime
it has never entered.

## Verified facts (checked 2026-07-30, sources recorded)

### The round

From `babylm.github.io` and the CFP, arXiv:2602.20092 (*"BabyLM Turns 4 and Goes
Multilingual"*, Choshen, Cotterell, Gul, Jumelet, Linzen, Mueller, Salhan, Shah,
Warstadt, Wilcox; submitted 2026-02-23):

- **Round 4 (2026) is CLOSED.** ARR deadline 2026-05-25; direct submission
  2026-07-20 AoE. Workshop at EMNLP Budapest, 2026-10-24/29.
- **Target is Round 5 (2027).** No time pressure on a ~78h experiment.
- Tracks: Strict (<100M words), Strict-Small (<10M words), Multilingual (new).
- **Hard cap: no more than 10 epochs over the training data.**
- The provided corpus is mandatory. **Teacher-model feedback is permitted**
  (so the milestone-12 distillation line is *not* disqualified — an earlier
  assumption in planning was wrong and is corrected here).
- Models must be public on HuggingFace to submit. The repo's existing `release/`
  export tooling already satisfies this.

### The corpus (downloaded, measured)

`BabyLM-community/BabyLM-2026-Strict` and `-Strict-Small`. Ungated, MIT, plain
`.txt`. Word counts land at **exactly** 10,000,000 and 100,000,000 — the
organisers trimmed to budget, which confirms `text.split()` as the canonical
counting convention.

| Source | Share | tok/word (gpt2 BPE) |
|---|---|---|
| childes | 28.4% | 2.326 |
| gutenberg | 25.6% | 1.437 |
| open_subtitles | 22.8% | 1.574 |
| simple_wiki | 15.3% | 1.397 |
| bnc_spoken | 7.6% | 1.291 |
| switchboard | 0.2% | 1.680 |
| **weighted** | | **1.703** |

Both tracks share these proportions exactly. 5.43 bytes/word.

Under gpt2 BPE that is ~170.3M tokens (strict) and ~17.0M (strict-small) — the
planning estimate of 1.35 tok/word was wrong; CHILDES is the outlier that moves
it. A corpus-trained tokenizer will lower this, which is why the step budgets
are `[PIN]`.

### Pre-launch corpus diagnostic (run before these gates were written)

58.8% of this corpus is short-utterance dialogue, which raised a direct
TinyStories concern: milestones 03/04 found attention comparisons uninformative
on a corpus where *no* mechanism extracts signal past ~128 tokens. That would
make this whole milestone a predictable null.

Tested cheaply with `benchmarks/long_range_probe.py` using the **already-trained
05 models** (114M, FineWeb-Edu) against a 5M-token stratified BabyLM val slice.
This is a diagnostic on the *corpus*, not on the experimental arms, so it does
not compromise pre-registration of the comparison below. Disclosed here so it
cannot read as post-hoc justification.

| Position | sliding-window | full (gqa) | hybrid |
|---|---|---|---|
| 64–127 | 3.9766 | 3.8970 | 3.9048 |
| 960–1023 | 3.9791 | 3.7235 | 3.7199 |
| change | **+0.003 (flat)** | **−0.174** | **−0.185** |

Sliding-window flattens at position 64 and never recovers; full and hybrid keep
improving to 1023. The sliding-window↔hybrid separation grows from 0.072 to
**0.259 nats** — *larger* than the ~0.15 measured on FineWeb-Edu in 05.
**Verdict: the corpus has ample exploitable long-range structure. Not a
TinyStories repeat. Green light.**

**Yellow flag, recorded in advance.** Under this same out-of-distribution
evaluation the hybrid↔full gap is **−0.0017**, essentially zero, versus −0.055
in-distribution on FineWeb. OOD gap comparisons are confounded (both models are
far from their training distribution), so this is weak evidence — but it is a
caution about the *magnitude* to expect, and it is on the record before any arm
runs. Raw: `benchmarks/results/05-data-frontier-*-babylm-per-position.json`.

## The prediction — PINNED 2026-07-30, before any training

Tokenizers are frozen and `D` is measured (`data/*/data_summary.json`), so the
formula now yields fixed numbers. These do not move again.

| Track | Train tokens `D` | tok/word | **Predicted gap** | **Gate window (±0.02)** |
|---|---|---|---|---|
| strict | 164,525,756 | 1.6653 | **−0.046** | **[−0.066, −0.026]** |
| strict-small | 15,825,289 | 1.6325 | **−0.098** | **[−0.118, −0.078]** |

Strict-small is a **3.2× extrapolation below** 12b's smallest measured point
(50M). That is the honest weakness of the prediction, and also why the test is
worth running.

Step budgets at 16,384 tokens/step, one epoch each:
**strict 10,041 steps**, **strict-small 965 steps**.

## Design

Hybrid vs full attention, **n=3 seeds** (1337/1338/1339, matching 08).
**Parameters held identical across both tracks**, as 12b did — 12b varied only
`D` at fixed params, so holding params constant is what makes these points land
on one curve. Only `max_iters` and the corpus differ between tracks.

| Track | Words | Train tokens | Steps | Params | Epochs | Runs | Subtotal |
|---|---|---|---|---|---|---|---|
| strict-small | 10M | 15,825,289 | 965 | 88.1M | 1 | 6 | ~6h |
| strict | 100M | 164,525,756 | 10,041 | 88.1M | 1 | 6 | ~56h |

**~62h total** (probe will firm up s/step). Single epoch each: the tracks then
differ in `D` by 10.4×, which is what the prediction is about. Training both
tracks to equal total tokens would instead test data *repetition* and would not
test the decay curve at two points — a worthwhile separate experiment, not this
one.

Params are **88,099,584** for every arm in both tracks: the 05 non-embedding
stack (75,516,672) plus a 16384×768 embedding. Identical across tracks and arms,
so only `D` varies.

A useful side effect of the smaller vocab: the per-forward logits tensor drops
from 823MB to **268MB** at micro-batch 4 — far inside the ~1GB MPS envelope this
repo's allocator doctrine requires. The throughput probe should therefore test
micro-batch 8 × accum 2 (identical 16,384 tokens/step) and adopt whichever is
faster, applied identically to both arms.

Only the attention module differs between arms. Tokenizer, corpus, token budget,
optimizer, schedule, dropout, seed, and eval protocol are held fixed, per the
fair-comparison rule in `CLAUDE.md`.

## Doctrine changes, and why

**1. Corpus-trained tokenizer.** Every prior run uses tiktoken `gpt2`
(vocab 50257). Against a 17M-token budget that is pathological: the embedding
table alone would be a large fraction of the model. Train an 8k–16k BPE on the
BabyLM corpus, frozen and shared identically across arms.

Consequence: cross-entropy in nats is **not comparable across tokenizers**.
Report **bits-per-byte alongside nats**, and treat 12b as the *motivating prior*
that generated the hypothesis rather than a directly comparable datum. The decay
slope is re-derived within this corpus, so the test is self-contained.

**2. Dropout becomes a real hyperparameter.** `dropout = 0.0` in every config in
this repo, correct for sub-epoch training against a huge pool. A fixed word
budget is a different regime, and 11-efficiency priced the unregularised version:
4× reuse of a 25M-token pool cost **+0.322 nats** with a 0.9-nat train/val gap.
Tune once on the control arm, then **freeze identically across both arms**.

**3. Deliberate overparameterisation at strict-small.** 15.8M tokens against
88.1M params is 0.18 tokens/param — far past where 11's epoch4 lane broke down.
Accepted *on purpose*: holding params fixed across tracks is what makes the two
points comparable to each other, and to 12b's shape. It makes strict-small a poor
competitive entry and a clean scientific one. A competitive Round-5 entry would
tune size and epochs separately.

## Pre-registered gates

Noise yardstick is 08's **0.02 nats** (SD across 3 seeds ≈ 0.018).

- **G1 — presence, strict.** Gap ≤ −0.02.
  *Fail action:* report absence at this budget.
- **G2 — presence, strict-small.** Gap ≤ −0.02.
  *Fail action:* report absence.
- **G3 — the prediction (primary).** Measured gap within **±0.02** of predicted:
  **strict in [−0.066, −0.026]**, **strict-small in [−0.118, −0.078]**.
  *Fail action:* **report the miss and the revised slope. A failed prediction is
  the publishable result. Do not re-fit and re-report as if predicted.**
- **G4 — monotonicity.** Gap magnitude strictly larger at strict-small than at
  strict.
- **G5 — downstream.** Full official suite reported, not just loss: BLiMP, BLiMP
  Supplement, EWoK, Entity Tracking, COMPS, GlobalPIQA, Reading, AoA, and the
  SuperGLUE fine-tuning set. **13 showed loss and downstream capability coming
  apart**, so a loss win with no BLiMP movement is a live outcome and must be
  reported as one, not buried.
- **G6 — three-way.** Params, KV-cache size, and decode tok/s for both arms.
  No verdict on quality alone.

## Protocol

- Deterministic adjudication via `benchmarks/full_pass_val.py`, identical
  invocation across all arms. **In-run numbers decide nothing** — 13 measured
  in-run→full-pass offsets with *opposite signs* between arms.
- Contamination census before adjudication (pattern:
  `experiments/11-efficiency/adjudicate.py`). The corpus is small and this repo
  has **no deduplication anywhere**.
- Budget compliance asserted at data-build time and printed in the run summary:
  word count ≤ track limit. This is the one error that invalidates a submission.
- Per-arm pre-launch: forward shapes/NaN, param count vs analytic, 15-step loss
  decrease, ~10-step throughput probe with a numeric abort threshold.
- Seed spread reported per arm, as 08 did.
- Runs strictly sequential, `nohup` fully detached, `STATE_EVERY = 250`.
- **Closing the chain includes disarming the stall watchdog.**

## Open items before launch

1. Train and freeze the tokenizer; measure `D`; pin every `[PIN]`.
2. Tune dropout once on the control arm; freeze.
3. Stand up the official eval pipeline (`github.com/babylm-org/babylm-eval`);
   confirm the repo's HF export satisfies its model interface.
