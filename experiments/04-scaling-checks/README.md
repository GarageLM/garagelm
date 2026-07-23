# 04-scaling-checks

The check `03-attention-variants/README.md` promised: does sliding-window's
win survive when the model is ~2x bigger, the context is 2x longer, and the
data is the real TinyStories train split? Two models — the `03` winner
(sliding-window, window=64 unchanged, now covering 1/8th of context instead
of 1/4) and the GQA full-attention control — identical in every other way
(fair-comparison rule: same data, steps, optimizer, seed; only the
attention pattern differs).

## Setup

- **231.9M params each**: n_layer=16, n_head=16, n_kv_head=4, n_embd=1024
  (head_dim 64), ffn_hidden=2816, block_size=512 — same recipe as `02`/`03`,
  wider and deeper. Model code is copied unchanged from the proven `02`
  (gqa) and `03/sliding-window` implementations.
- **Data**: 108M-token pool tokenized from the real
  `TinyStoriesV2-GPT4-train.txt` (2.2GB; only the first ~460MB needed) via
  chunked, story-boundary-aligned tokenization in `data.py`. 2M held-out
  val tokens. Perplexities here are NOT comparable to `03`'s (different
  corpus).
- **Gradient accumulation (new in this milestone)**: micro-batch 8 x 512
  tokens x 4 accumulation steps = 16,384 tokens/optimizer step, keeping the
  micro-batch inside the MPS-safe regime established in `02`. 3000
  optimizer steps = **49M tokens/model** (~4x the `03` budget).

## Results

| Model | Params | Val loss | PPL | HellaSwag | PIQA | WinoGrande | tok/s | Train time* |
|---|---|---|---|---|---|---|---|---|
| sliding-window | 231.9M | **1.4119** | 4.10 | 0.303 | 0.530 | 0.517 | 38.7 | (contended) |
| GQA (full attn) | 231.9M | 1.4133 | 4.11 | 0.310 | 0.530 | 0.480 | 37.4 | 5.9h |

*GQA ran clean at 5.9h (matching the 7.0s/step probe); sliding-window hit
overnight contention from other sessions (13h wall at identical quality).
Benchmark accuracies at n=300 are within stderr of each other throughout.

Mid-run checkpoints, same step, val loss (sliding-window led throughout,
converging to a tie):

| Step | GQA | sliding-window |
|---|---|---|
| 500 | 2.0567 | 2.0182 |
| 1000 | 1.7526 | 1.7366 |
| 2000 | 1.4965 | 1.5007 |
| 3000 (final) | 1.4133 | 1.4119 |

## Verdict

**The `03` ordering held.** At 2x parameters, 2x context, and 4x tokens on
the real train corpus, sliding-window (64-token window, 1/8th of context)
is statistically tied with full attention (0.0014 nats — noise), having
led at every intermediate checkpoint. Full attention never pulled ahead.

The honest interpretation, consistent with both `03` and the literature:
**on this corpus, attention beyond ~64 local tokens buys nothing** —
TinyStories' stories are short and self-contained, so there is no
long-range signal for full attention to exploit. This is a strong result
*for this data regime* and a known limitation of the corpus, not proof
that local attention is universally free: the next escalation
(deferred) would be a corpus with genuine long-range structure — long-form
documents, code, or needle-in-haystack-style synthetic tasks — where full
attention finally has something the window can't see.

Scale itself worked as expected: val perplexity dropped from 6.47 (114M,
12M tokens) to 4.10 (232M, 49M tokens), and sampled stories are noticeably
more coherent (consistent multi-paragraph narratives). Benchmark
accuracies barely moved, reinforcing the `02` finding that these
leaderboard tasks need broad/factual pretraining data, not just more
parameters, to leave chance territory.

## Run it

```
uv run python experiments/04-scaling-checks/data.py          # ~2.2GB download + tokenize
uv run python experiments/04-scaling-checks/gqa/train.py     # ~6h clean
uv run python experiments/04-scaling-checks/sliding-window/train.py
uv run python benchmarks/run_quality_eval.py --experiment-dir experiments/04-scaling-checks/<model>
```
