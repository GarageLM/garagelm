# sliding-window

Tests whether restricting attention to a fixed local window (`window=64`
of the 256-token context) costs any quality against `02`'s full causal
attention, at the same GQA width (n_kv_head=4) and training budget. Only
the attention *pattern* changes: `SlidingWindowAttention` swaps
`is_causal=True` for a precomputed `(256,256)` boolean mask (causal AND
within `window`), passed via SDPA's `attn_mask=`. Zero parameter
difference from `02` — same `q/k/v/o_proj` shapes.

TinyStories' short-story structure rarely needs dependencies longer than
64 tokens, so this variant's real question is whether the baseline's full
256-token attention span is doing any work at all on this corpus, or
whether a much cheaper (in principle, at longer context lengths) windowed
pattern gets equivalent quality for free.

Data is read directly from `../../02-baseline-small-lm/data/`.

## Run it

```
uv run python experiments/03-attention-variants/sliding-window/train.py
uv run python experiments/03-attention-variants/sliding-window/sample.py --prompt "Once upon a time"
```

## Result

See `out/run_summary.json` after training.
