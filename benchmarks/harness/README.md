# benchmarks/harness: inference-harness substrate (milestone 15)

One runner, one control, one results schema for every harness lane. The model
is served by `mlx_lm.server` (continuous batching); the runner only talks HTTP,
so the same code later points at the lab's own `benchmarks/mlx/server.py` for a
distilled student. Pre-registration and gates: `experiments/15-harness/README.md`.

## Commands

```
benchmarks/harness/serve.sh start mlx-community/Qwen3.5-9B-4bit 8421 8      # detached, pid file, health wait
uv run python benchmarks/harness/selftest.py                                 # graders, stats, config sha
uv run python benchmarks/harness/probe_throughput.py --model <id> --quant q4 # G0: tok/s x concurrency, RSS, planned peak
uv run python benchmarks/harness/preflight.py --config experiments/15-harness/configs/<run>.yaml
uv run python benchmarks/harness/runner.py   --config experiments/15-harness/configs/<run>.yaml
uv run python benchmarks/harness/runner.py   --config ... --regrade           # re-aggregate from stored traces
benchmarks/harness/serve.sh stop 8421
```

Runs are sequential (`nohup`, pid file, `progress.json` heartbeat under
`experiments/15-harness/runs/<run_id>/`); closing a run includes disarming its
stall watchdog.

## Rules baked into the code

- **No per-request seed.** mlx_lm.server excludes seeded requests from the
  continuous batch and reseeds per request anyway. Sampling is therefore not
  bitwise reproducible; `--regrade` reproduces every aggregate bit-exactly
  from the stored traces, which is the reproducibility contract.
- **Control = same config with k=1.** A control YAML and its harness YAML differ
  in exactly one key; the config sha (which ignores notes/concurrency/url) is
  in every results JSON.
- **Every k from one k=K run.** cons@1..K are unbiased subsample averages of the
  K stored samples; pass@k is reported as a diagnostic (an oracle), never as
  the accuracy.
- **`length` finish counts as wrong**; truncation rate is always reported.
- **Full sets only.** `--limit` is accepted only with `--smoke`, and smoke
  outputs are suffixed `-smoke` and never cited.
- **Graders are imported, not rewritten** (lm-eval `hendrycks_math` utils);
  `GRADER_VERSION` is stored per sample. The code executor is best-effort
  isolation on macOS (fresh `python -I -B`, empty env, rlimits, timeout): not a
  security boundary, network not blocked, lab-model outputs only.
- **Frontier rows are quoted, dated, sourced** (`references/frontier.json`,
  `local_rerun: false`, `pinned: false` until re-pinned from the official card
  or system card). Each open model used also gets a calibration row (its card
  number beside our local k=1).

## Results schema

Aggregate: `benchmarks/results/harness-<task>-<model>-<quant>-<strategy>.json`
(`cons8`, `k1`, ...; collision-checked, `--force` to overwrite). Fields:
experiment, run_id, task, task_args, model, quant, strategy, k, sampler,
config_sha, git_hash, date, n_items, n_samples, n_errors, accuracy (+ metric,
item-bootstrap CI, Wilson), avg_at_1, cons_at_k{k}, pass_at_k_diagnostic{k},
truncation_rate, completion_tokens_{total, per_sample_mean/p95,
per_item_mean/p95}, tokens_per_correct, wall_s_per_sample_mean, elapsed_s,
wall_s_per_item, aggregate_tok_s, peak_rss_gb, per_item_cons_correct,
per_item_avg1.

Traces: `benchmarks/results/harness-traces/<run_id>.jsonl.gz` (git-ignored;
shipped to HF when large). One record per sample: run_id, task, item_id,
sample_idx, prompt_tokens, completion_tokens, reasoning_chars, finish_reason,
ttft_s, wall_s, error, extracted, normalized, correct, truncated,
grader_version, text, reasoning, ts. `correct` + the run's selection make
verifier-filtered trace extraction a one-liner (the distillation corpus).

Probe: `benchmarks/results/harness-probe-<model>.json`: aggregate and
per-stream tok/s per concurrency, TTFT, server RSS peak, KV bytes/token from
the model config, planned peak at 32k tokens, contention snapshot, G0 verdict.

## Three-way row (every table)

accuracy (CI) | k, tokens/item mean + p95, tokens per correct | wall-clock per
item, aggregate tok/s, peak memory, weights on disk | frontier row (model,
date, score, URL, not re-run locally) | gap. `report.three_way_row()` renders
the local part from a results JSON.
