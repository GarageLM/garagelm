# nsa-lite

Simplified 2-branch approximation of Native Sparse Attention (DeepSeek's
NSA): a local causal sliding window (`window=64`) plus a compressed/pooled
global-context branch (`compress_block=16`, mean-pooled K/V). NSA's real
third branch — trainable top-k *selective* attention over individual past
tokens — needs differentiable top-k machinery and is out of scope here;
dropping it is the "lite" in the name, not a hidden shortcut. Same GQA KV
width as `02`/`sliding-window` (n_kv_head=4) — only the attention pattern
varies, isolating that as the tested variable, zero parameter difference
from the baseline.

A design review caught two real bugs in the first draft before any code
was written (see the approved plan for the full analysis):
- **Coverage gap.** The original masking rule left a silent gap of up to
  ~30 tokens invisible to both branches, because the window boundary
  slides by 1 each step while the compression grid is fixed in steps of
  16. Fixed by using plain causal `block_last_pos < i` for the compressed
  branch, letting the newest 1-2 compressed blocks overlap the dense
  window — one joint softmax handles that redundancy fine.
- **Pooling K after RoPE was wrong.** Averaging already-rotated vectors
  from different positions has no well-defined position and destroys
  high-frequency RoPE channels. Fixed by pooling the *pre-RoPE* `k_proj`
  output per block, then applying RoPE once per pooled vector at a
  representative position (block start).

The mask and compressed-position tables are recomputed per forward call
from the actual sequence length, not just for `T == block_size` — this
matters because `benchmarks/run_quality_eval.py` calls the model with much
shorter contexts than 256 tokens, and getting that path wrong would
silently corrupt quality numbers rather than crash.

Data is read directly from `../../02-baseline-small-lm/data/`.

## Run it

```
uv run python experiments/03-attention-variants/nsa-lite/train.py
uv run python experiments/03-attention-variants/nsa-lite/sample.py --prompt "Once upon a time"
```

## Result

See `out/run_summary.json` after training.
