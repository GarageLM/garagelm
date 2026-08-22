# 15-harness: an open small model + inference harness vs frontier scores

**Status: PRE-REGISTRATION. No lane has been run.** Gates below are written
before any run. `[PIN]` marks values fixed by the throughput probe (G0) or by
an official source lookup before the lane they belong to is launched. Plan of
record: `~/.claude/plans/vectorized-sleeping-trinket.md` (2026-08-21).

## Question

How much of the gap between a small open model on one consumer machine and
the frontier does inference-time compute close, at what cost in tokens and
wall-clock? This is the lab's frontier-lag question moved from pretraining to
inference: the model is fixed, the harness is the one lever.

## Why this, why now

Pretraining scale is not a lever this lab can pull past ~250M params. The
2025-2026 record (see `docs/literature/test-time-compute.md`) says harness
design is the largest single lever available at small scale: +52 points on
ARC-AGI-1 from harness alone at fixed weights (arXiv 2607.06764); 1B/3B models
passing 8B/70B on MATH-500 with verifier search (HF search-and-learn);
ARC Prize's compute-capped track won by refinement loops. The previously
committed next lever, distillation, is deferred to 16: it needs a synthetic
corpus, and this milestone's verifier-filtered traces are that corpus.

## Model

- **Primary: `mlx-community/Qwen3.5-9B-4bit`** (Apache-2.0; thinking by
  default; 262k context; card: GPQA-D 81.7, HMMT-Feb25 83.2, LCB-v6 65.6,
  IFEval 91.5). ~5.5GB weights. Revision sha `[PIN]`.
- **Dev/iteration: `mlx-community/Qwen3.5-4B-4bit`.** Never in a headline row.
- **Conditional stretch: `mlx-community/Qwen3.6-35B-A3B-4bit`** (3B active,
  ~20GB weights). Runs only if it clears G0 and the 9B lands within ~5 points
  of a lane's frontier target.
- **Quantization gate Q:** 4-bit vs 8-bit, NLL delta <= 0.02 nats on a fixed
  64x2048-token held-out set plus 64 MATH-500 reference solutions, and |k=1
  accuracy delta| <= 3 points (McNemar p > 0.05) on a fixed 100-item MATH-500
  slice. Fail: ship 8-bit for the 9B. The 35B-A3B has no local higher-precision
  reference: it is gated against its card's k=1 number instead.
- **Floor rows:** `hybrid-gpt-232m-chat` v2 on GPQA-D via the existing
  loglikelihood path (`benchmarks/run_quality_eval.py --tasks gpqa_diamond_zeroshot
  --limit 198`, minutes). "n/a (ctx 1024)" for everything else. Never a contender.

## Lanes, in launch order

| Lane | Set | n | Harness | Grader | Frontier reference (quoted, dated) |
|---|---|---|---|---|---|
| M | AIME 2024 (`Maxwell-Jia/AIME_2024`), 2025 (`math-ai/aime25`), 2026 (`[PIN]`) | 90 | cons@k, k from one k=8 run | `\boxed{}` exact match (lm-eval hendrycks_math utils) | AIME-2025 score of the current frontier model `[PIN]`; AIME-2026 Qwen3.5-397B-A17B 91.3 |
| A | ARC-AGI-1 public eval (400); ARC-AGI-2 public eval (120) secondary | 400 / 120 | Explorer-Definer program synthesis, train pairs as verifier, <=2 reflect rounds, pass@2 | official exact match, all test outputs | arXiv 2607.06764: 57.5% ($0.25/task) and 67.25% ($0.62/task); ARC Prize 2025 Kaggle 24.03% ARC-AGI-2 private ($0.20/task) |
| G | GPQA Diamond (`Idavidrein/gpqa`, gated) | 198 | cons@k from one k=8 run | letter regex, last match | Claude Sonnet 4.5 83.4 `[PIN from system card]` |
| C | HumanEval+ (164), MBPP+ (378) | 542 | verified best-of-8 + one repair round; hidden tests never touch selection | EvalPlus hidden tests in the sandbox | none (saturated); corpus lane |

Control for every lane: **the same model, same quantization, same prompt,
same sampler, same max_tokens, k=1.** avg@1 over the k=8 samples is the
primary control; a standalone k=1 run is the plumbing check (must agree with
cons@1-from-k=8 within CI); greedy (temperature 0) is a secondary row on Lane M.

Sampling (Lane M/G): Qwen3.5 card values `[PIN]` (temperature, top_p, top_k,
presence penalty), thinking on, max_tokens 32768 (M) / 16384 (G). A `length`
finish counts as wrong; truncation rate is reported; one 32k->24k sensitivity
row on a 30-item subset.

Lane A per-task budget: explore k=8 (cap 3k tokens incl. thinking; on cap append
`</think>` and finish), define one `transform(grid)` per distinct hypothesis
(thinking off, cap 1.5k), execute on all train pairs, reflect <=2 rounds with the
first failing pair's diff, early exit when >=2 all-train-passing programs agree
on the test output, select top-2 distinct outputs by (train pairs passed, vote
count, program length). Hard cap 72k generated tokens per task. Grids as compact
unspaced digit rows. Baseline: same model, 2 direct-grid samples. Dev set: a
seeded 40-task slice of ARC-AGI-1 *training*; pilot-40 from eval; the eval sets
are never looked at during iteration. Harness frozen (git hash in config)
before the baseline runs; only batch-size and abort mechanics may change after.

## Pre-registered gates

Noise yardstick: paired, item-level. Claims are differences on identical items
with a clustered bootstrap 95% CI (10k resamples, seed 1234) and exact McNemar;
never two independent proportions. Full sets always; no `--limit`.

- **G0, throughput and memory probe (abort gate, before any long run).**
  `mlx_lm` loads the primary under the current transformers pin; aggregate
  decode >= 150 tok/s at concurrency 8 on 12 fixed prompts; planned peak =
  weights + C x (prompt + max_tokens) x KV bytes/token + prompt-cache cap + 4GB
  <= 40GB and measured RSS <= 42GB; mean trace <= 20k tokens extrapolated from
  16 AIME-25 items at max_tokens 4096; grader cross-check vs lm-eval `aime25`
  (same model, greedy) agrees on >= 28/30 items.
  *Fail:* C 8 -> 4, then max_tokens 32k -> 24k, then tier 9B -> 4B; re-probe.
- **G-M1, leverage.** cons@8 - avg@1 >= +5.0 points on the 90 items, CI
  excludes 0. *Fail:* report "harness adds nothing for this model on AIME";
  Lane G runs at k=4 only.
- **G-M2, proximity.** cons@8 >= (pinned frontier AIME-2025 score - 5).
  *Fail:* report the gap; no "frontier-level" wording anywhere.
- **G-A0, baseline.** Recorded, not gated. If >> 15.5%, flag contamination
  before reading G-A2.
- **G-A1, pilot (decides the spend).** harness - baseline >= +15 points on
  pilot-40 (SE ~7 at p~0.3) AND projected 400-task wall-clock <= 36h.
  *Fail:* stop, report the negative, do not tune on eval.
- **G-A2, pass@2.** ARC-AGI-1 >= 57.5% (Explorer-Definer parity); stretch
  >= 67.25%. ARC-AGI-2 >= 24.03% (Kaggle-2025 parity; different split, stated);
  floor >= 10%. *Fail:* the number is still the frontier-lag datum, published
  as-is.
- **G-A3, cost.** Median <= 5 min and <= 72k generated tokens per task;
  $-equivalent (~$0.10/h amortized machine + power) reported beside $0.20,
  $0.25, $0.62.
- **G-A4, contamination control.** 100-task slice re-run under colour
  permutation + transpose; a drop > 5 points makes the augmented accuracy the
  headline.
- **G-G1.** cons@8 - avg@1 >= +3.0 (198 items, CI excludes 0). **G-G2.**
  cons@8 >= 83.4 - 5. *Fail:* report.
- **G-C1.** best-of-8 + repair - avg@1 >= +5.0 pass@1 on hidden tests (542
  items). *Fail:* report.
- **G6, three-way, every table.** accuracy (CI) | k, tokens/item mean and p95,
  tokens per correct | wall-clock/item, aggregate tok/s, peak memory, weights
  on disk | frontier row (model, date, score, source URL, "not re-run locally")
  | gap. No verdict on accuracy alone.

Frontier-lag sentence template: "On <bench>, <frontier model> reported X% on
<date>; a 9B open model at 4-bit on one 48GB consumer machine reached Y%
(cons@k, 95% CI) N months later at T minutes and M tokens per item." Hardware
fraction: ~48/640 = 7.5% of serving memory against the open 397B-A17B; "not
computable" for closed APIs.

## Budget (assumed 150 tok/s aggregate at C=8; G0 replaces the assumption)

Lane M ~11.5M generated tokens (~21h). Lane G ~12.7M (~24h). Lane A 400 tasks
18-32h, 120 tasks 5-10h, pilot 2-3h. Lane C ~10h + ~17h. About 5-6 machine-days.
Throughput gate: any lane projecting > 30h at G0 rates drops k 8 -> 4 before
anything else; items are never cut. Drop order if the budget binds: 35B-A3B arm,
MBPP+, Lane C, ARC-AGI-2 full (keep a 40-task measurement), Lane G k 8 -> 4,
reflect round 2. Never dropped: Lane M, Lane A baseline + pilot, G-A4, three-way
reporting, trace JSONL.

## Protocol

- Substrate: `benchmarks/harness/` (README there). Serving via `mlx_lm.server`
  with continuous batching; requests carry no seed (a seed disables batching),
  so sampling is not bitwise reproducible; `runner.py --regrade` reproduces
  every aggregate bit-exactly from stored traces. One frozen config YAML per
  run in `configs/`; control and harness configs differ in exactly one key.
- Results: `benchmarks/results/harness-<task>-<model>-<quant>-<strategy>.json`;
  traces `benchmarks/results/harness-traces/<run>.jsonl.gz` (git-ignored above
  10MB, shipped to HF as `garagelm/harness-traces-15`).
- Frontier and calibration rows: `benchmarks/harness/references/frontier.json`
  (every open model's card number sits beside our local k=1; a large gap fails
  preflight until explained, the SmolLM2 IFEval precedent).
- Pre-launch, every run: `preflight.py` (server health, thinking flag echoed,
  grader self-test, 5-item smoke, 10-request probe under the G0 abort gate,
  contention check); `design-reviewer` pass before any run > 2h; runs
  strictly sequential, `nohup`, pid file, progress heartbeat; **closing a run
  includes disarming its watchdog.**
- Contamination: prefer the newest sets; record each model's data cutoff next
  to each score; G-A4 for ARC.

## Deferred, explicitly

Kaggle ARC Prize submission (CUDA-only, offline, time-boxed: a framing, not an
objective; a port is a separate decision only if the local number is
competitive). TRM/CompressARC-style tiny models and per-task test-time training
(training milestones). LiveCodeBench and SWE-bench (tooling absent; SWE-bench
rollouts do not fit the budget). Inspect AI (revisit if the harness grows tools
or agents). Learned verifiers/PRMs (a training milestone in disguise).

## Run log

- **2026-08-21, load check.** `mlx-community/Qwen3.5-9B-4bit` loads under
  transformers 5.5.4 / mlx-lm 0.31.3 (the pin holds): `has_thinking=True`,
  chat template emits `<think>` with `enable_thinking=True` and an empty think
  block with `False`; single-stream 43.4 tok/s on a 256-token generation,
  peak 5.23 GB. Config (nested `text_config`): 32 layers, 8 full-attention
  (every 4th), 4 KV heads x 256 head_dim, so KV = 32 KB/token bf16; weights
  5.95 GB on disk. Planned peak at C=8 x 32k tokens = 20.7 GB (G0 memory
  side passes on paper).
- **2026-08-21, G0 probe, first reading (CONTENDED, not valid for G0).**
  `benchmarks/results/harness-probe-qwen3.5-9b-4bit.json`. The machine was
  running a game (three League of Legends processes at 60-67% CPU plus
  WindowServer at 36%) for the whole probe. C=1 40.9 tok/s, C=4 93.9
  aggregate (23.5/stream), then C=8 51.4 and C=16 17.4: a collapse that
  continuous batching cannot produce on its own, attributed to the server's
  Python batching loop being CPU-starved. The C=16 errors were client
  timeouts from queueing 16 requests against decode-concurrency 8 (probe bug:
  the probe now passes a 3600 s timeout and the JSON carries a contention
  snapshot and `valid_for_g0`). **Action: re-run the probe on an idle machine
  before pinning G0 and before any lane launch.** If the collapse survives
  an idle re-run, it is a server-side finding (qwen3_5 hybrid linear-attention
  layers under batched decode) and C=4 becomes the working concurrency
  (budget x2).
- **2026-08-21, preflight smoke, first pass (plumbing OK, gate tripped on
  truncation).** Five AIME-2024 items at k=1 with a 6,144-token cap: 5/5
  requests served, traces/grading/aggregate/RSS monitor all worked
  (`harness-aime-qwen3.5-9b-q4-k1-smoke.json`, 92 tok/s aggregate at 5 in
  flight, 6.4 GB RSS, still under the game's contention), but every trace was
  cut inside `<think>` at 13-22k reasoning characters, so 0/5 and
  truncation 1.00: AIME traces from this model run well past 6k tokens,
  consistent with the 10-30k planning figure and the 32,768 cap in the
  config. Preflight's smoke now defaults to 5 easy MATH-500 items at 8,192
  tokens (that is what "easy items" in the protocol means); AIME smokes only
  test the truncation rule.
- **2026-08-21, preflight smoke, MATH-500 Level 1-2 (gate still tripped on
  truncation; a budget finding).** Five Level 1-2 MATH-500 items, thinking on,
  8,192-token cap: 1/5 correct (`42`, stop at 6,652 tokens), 4/5 cut inside
  `<think>` at the cap. So this model thinks past 8k tokens even on easy
  problems; the 32,768 cap on AIME will bind for some items and G0's
  "mean trace <= 20k" gate is the one to watch. Under the game's contention
  the five items took 56 minutes (~5 tok/s per stream at 5 in flight), which
  is why no lane launches until the machine is idle. Preflight's smoke now
  runs with thinking OFF by default (the correctness/plumbing half; the
  thinking path is covered by the echo check), `--smoke-thinking` restores
  the thinking smoke. Results: `harness-math500-qwen3.5-9b-q4-k1-smoke.json`.
