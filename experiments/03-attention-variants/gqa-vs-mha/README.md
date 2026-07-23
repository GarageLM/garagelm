# gqa-vs-mha

Tests whether `02-baseline-small-lm`'s GQA (12 query heads sharing 4 kv
heads) costs any quality against full multi-head attention (MHA, 12 query
heads = 12 kv heads), at the exact same training budget. Only
`config.py`'s `n_kv_head` differs from `02` (12 instead of 4) — `model.py`
is otherwise byte-identical (`repeat_kv` becomes a no-op when
`n_head == n_kv_head`).

MHA has larger `k_proj`/`v_proj` (768×768 vs. 768×256 each), so this
variant has more parameters than the GQA baseline and a 3x larger
per-layer KV-cache footprint — that's the actual tradeoff GQA exists to
avoid, so it's expected, not a bug.

Data is read directly from `../../02-baseline-small-lm/data/` (same
tokenizer, same corpus, same split) rather than re-downloaded.

## Run it

```
uv run python experiments/03-attention-variants/gqa-vs-mha/train.py
uv run python experiments/03-attention-variants/gqa-vs-mha/sample.py --prompt "Once upon a time"
```

## Result

See `out/run_summary.json` after training.
