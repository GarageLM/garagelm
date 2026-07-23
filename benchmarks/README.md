# Benchmarks

Evaluation methodology for every model produced in `experiments/`, chosen to
be meaningful at the model sizes this repo actually trains (roughly 1M–150M
params — see `CLAUDE.md`) rather than borrowed wholesale from leaderboards
built for much larger models.

## Quality

- **Validation perplexity** — the primary, always-comparable metric across
  every variant and every scale this repo touches.
- **`lm-evaluation-harness`, small-scale-appropriate tasks**: HellaSwag,
  PIQA, ARC-Easy, WinoGrande. These still produce a real signal at tens of
  millions of parameters.
- **MMLU** is included too (0-shot) specifically because it's the most
  recognizable "Top Model Benchmark," with the expectation set up front:
  at 114M params and 0-shot it lands near random chance — see results below.
- **Explicitly deferred**: MMLU-Pro, GPQA, IFEval, BBH, MATH — the standard
  HF Open LLM Leaderboard suite. At this repo's model sizes these sit near
  random chance and aren't discriminating; revisit only once (if) a larger
  checkpoint exists.
- Tooling: real `lm-evaluation-harness` (`benchmarks/lm_eval_adapter.py`
  wraps this repo's raw PyTorch models — not HuggingFace `transformers` —
  via a `TemplateLM` subclass; `benchmarks/run_quality_eval.py` runs it
  against any experiment's checkpoint and writes
  `benchmarks/results/<experiment>.json`).
- **Per-position validation loss** (`benchmarks/long_range_probe.py`) —
  the long-range diagnostic added for milestone 05: mean val loss bucketed
  by token position over a fixed, seeded set of windows. If a long-range
  model's curve keeps dropping past the local window size where a windowed
  model's flattens, long-range attention is measurably earning something.
  (Run against `04`'s full-attention checkpoint on TinyStories it confirms
  the `04` verdict directly: no improvement past position ~128.)

## Results: `02-baseline-small-lm` (114.1M params, 0-shot)

Validation perplexity: **6.47** (`exp(1.8665)`, from the recorded final val
loss in `experiments/02-baseline-small-lm/out/run_summary.json`).

| Task | Metric | Score | Chance level |
|---|---|---|---|
| HellaSwag | acc / acc_norm | 0.287 / 0.277 | 0.25 (4-choice) |
| PIQA | acc / acc_norm | 0.547 / 0.460 | 0.50 (2-choice) |
| ARC-Easy | acc / acc_norm | 0.260 / 0.253 | 0.25 (4-choice) |
| WinoGrande | acc | 0.487 | 0.50 (2-choice) |
| MMLU | acc | 0.270 | 0.25 (4-choice) |

300 examples/task (HellaSwag/PIQA/ARC-Easy/WinoGrande); MMLU sampled 5
examples from each of its 57 subjects (~285 total) since it's a group task
and `--limit` applies per-subject — a full `limit=300` on MMLU would have
meant 17K examples, dwarfing the other four tasks. Tokens/sec (decode,
lightweight PyTorch/MPS measurement, not the rigorous MLX benchmark below):
**68.9 tok/s**.

Reading: HellaSwag and PIQA (narrative/physical commonsense, closest to
TinyStories' domain) show a small but real edge over chance. ARC-Easy
(science facts), WinoGrande (pronoun resolution), and MMLU (broad academic
knowledge) all land at chance — expected, since this model has only ever
seen simple children's stories, not factual or academic text. This is a
coherent, sensible result for the model as trained, not a broken eval.

## Results: attention variants

The same suite was run on all four `03-attention-variants/` checkpoints
(per-variant JSON in `results/`, prefixed `03-`). Full comparison table and
findings live in `experiments/03-attention-variants/README.md`; headline:
benchmark accuracies at n=300 can't separate the variants (differences
within stderr) — validation perplexity is what discriminates, and by it
sliding-window ≥ nsa-lite ≈ GQA baseline > MHA > MLA.

## Results: scaling checks

Both `04-scaling-checks/` checkpoints (231.9M params, 512 context, 49M
tokens of real TinyStories train data) were evaluated with the same suite
(`results/04-*.json`). The `03` ordering held: sliding-window tied full
attention on perplexity (4.10 vs 4.11) and on every benchmark (all within
stderr). Verdict and caveats in `experiments/04-scaling-checks/README.md`.

## Efficiency

The point of this project — quality alone is not the goal, quality relative
to size and latency is. Every run reports:
- **Tokens/sec**, prefill and decode measured and reported separately
- **Time-to-first-token (TTFT)**
- **Peak memory**
- **Parameter count**
- **KV-cache size vs. sequence length** — the metric MLA/GQA/sparse-attention
  variants are specifically trying to win on, so it needs to be measured
  directly rather than inferred from throughput alone

Inference-latency numbers are measured with MLX (see `CLAUDE.md` for why:
it currently outperforms PyTorch+MPS for on-device inference on Apple
Silicon, and that's the deployment-relevant number).

### Results: MLX inference benchmark (M4 Pro, measured)

All five checkpoints (three `05` 114M models at block 1024, two `04` 232M
models at block 512), fp16 and 4-bit, full JSONs in `results/mlx-*.json`.
Headlines:

| Model (fp16) | Decode tok/s | KV @ full ctx | 4-bit decode | 4-bit quality cost |
|---|---|---|---|---|
| 05 gqa (full attn) | 511–535 | 12.58 MB | 688–747 | +0.012 nats |
| 05 sliding-window | 528–531 | **0.77 MB** (constant) | 701–751 | +0.012 nats |
| 05 hybrid | 513–527 | **3.73 MB** (29.7% of full) | 705–709 | +0.012 nats |
| 04 gqa (232M) | 238–312 | 8.39 MB | 485–508 | +0.004 nats |
| 04 sliding-window (232M) | 222–312 | 1.03 MB (constant) | 449–482 | +0.004 nats |

- **MLX vs PyTorch/MPS decode gap: ~10x** (e.g. 05 models: ~53 tok/s in
  the PyTorch harness vs ~530 tok/s MLX fp16) — vindicating the decision
  to measure deployment latency in MLX.
- The hybrid's measured cache is exactly its analytic 30%-of-full, and its
  decode speed matches the other variants (weights-bound at these sizes) —
  its `05` quality win costs nothing at inference.
- 4-bit quantization: +35–60% decode speed for ~0.01 nats — an easy win at
  deployment time.
- TTFT stays under 40ms everywhere (prefill 22–24k tok/s at 114M).

**Local chat server (`benchmarks/mlx/server.py`)**: an OpenAI-compatible
endpoint (`/v1/chat/completions`, streaming + non-streaming) over the MLX
port with the real rotating KV cache — point any OpenAI-client chat UI at
`http://localhost:8080/v1`. Defaults to the `10-sft` chat model and the SFT
training template; `--cpu` runs politely while a training job owns Metal.
Requests serialize behind a lock (MLX single-stream) and each request is
one prefill + token-by-token decode — the exact regime the rotating cache
supports.

**Implementation (milestone 06, `benchmarks/mlx/`)**: `model.py` is an MLX
port of this repo's GPT covering all three attention variants (full /
sliding-window / hybrid) with real per-layer KV caches — append-only for
global layers, bounded rotating (window-1 entries) for local layers, so
KV-cache-vs-length is measured from live cache arrays rather than derived.
`convert.py` dumps a PyTorch checkpoint to safetensors (module trees mirror
1:1) and gates **logit parity**: fp32 prefill + cached-decode logits must
match PyTorch within 5e-3 with 100% top-1 agreement (measured ~1e-5 on both
04 checkpoints). `bench_inference.py` reports TTFT, prefill/decode tok/s,
KV MB after prefill/decode, peak Metal memory, and (with `--bits 4`)
4-bit-quantized numbers plus a val-loss quality delta. Benchmarks must run
on an idle machine — they compete with MPS training.

## Reporting

One results file per run (e.g. `benchmarks/results.csv` or one JSON file per
run under `benchmarks/results/`), with quality, parameter count, and
latency/throughput columns together — never reported in isolation — so every
attention variant is directly comparable against the `02-baseline-small-lm/`
control.
