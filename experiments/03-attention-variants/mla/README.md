# mla

Multi-head Latent Attention (DeepSeek-V2-style), simplified. Instead of
separate per-head K/V projections, K and V are jointly compressed into a
shared low-rank latent (`kv_latent_dim=128` — smaller than GQA's 256-dim
KV width, well under MHA's 768) and reconstructed per-head. RoPE can't be
applied after this compression (rotation doesn't commute with the
low-rank projection), so a small **decoupled** slice of each head
(`rope_dim=16` out of `head_dim=64`) carries position info separately —
computed from its own projection and concatenated with the compressed
"content" slice (`nope_dim=48`). This is DeepSeek's actual design, not a
simplification; `k_rope_proj` is deliberately shared across all heads (one
small vector per position, broadcast), matching the paper.

Simplification vs. the paper: **query is not compressed** — only K/V would
ever be cached at inference, so compressing Q only saves parameters, not
cache size, and isn't needed to test the technique's actual claim.

A design review (see the approved plan) caught two bugs in the first draft
before any code was written: (1) the rope_dim=16 slice needs its own RoPE
table, not a slice of a head_dim=64 table — `precompute_rope`'s
`cos=cat(freqs,freqs)` layout means a naive slice gives the wrong angle
pattern; (2) `k_rope` (no head axis) must be explicitly `.expand()`ed to
`n_head` before `torch.cat`, not left to implicit broadcasting, which
would silently cross-broadcast the batch dimension against the head
dimension instead.

Non-embedding attention params/layer: **1.46M**, vs. GQA's 1.57M — MLA is
claiming both a smaller KV-cache *and* fewer parameters here.

Data is read directly from `../../02-baseline-small-lm/data/`.

## Run it

```
uv run python experiments/03-attention-variants/mla/train.py
uv run python experiments/03-attention-variants/mla/sample.py --prompt "Once upon a time"
```

## Result

See `out/run_summary.json` after training.
