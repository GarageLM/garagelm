# 03-attention-variants

The attention-architecture comparison this repo exists for. Four variants,
each changing **only** the attention module against
`02-baseline-small-lm`'s control (fair-comparison rule in `CLAUDE.md`:
identical tokenizer, dataset, token/step budget, optimizer, and seed —
data is read directly from `../02-baseline-small-lm/data/`). One variant
per subdirectory; each has its own README with design details:

- `gqa-vs-mha/` — full multi-head attention (n_kv_head=12 vs. baseline's 4)
- `sliding-window/` — 64-token causal window instead of full attention
- `mla/` — Multi-head Latent Attention (DeepSeek-V2-style, decoupled RoPE)
- `nsa-lite/` — 2-branch Native Sparse Attention approximation
  (sliding window + mean-pooled compressed global branch)

## Results (all trained 3000 iters on TinyStories-valid, seed 1337)

Quality benchmarks via `benchmarks/run_quality_eval.py` (0-shot, 300
examples/task; full JSON per variant in `benchmarks/results/`). At n=300,
benchmark accuracies carry roughly ±0.026 stderr — **validation
perplexity is the discriminating metric at this scale**, as expected from
the `02` benchmark analysis.

| Variant | Params | Val loss | PPL | HellaSwag | PIQA | tok/s (decode) | Train time* |
|---|---|---|---|---|---|---|---|
| **sliding-window** | 114.1M | **1.8605** | **6.43** | 0.290 | 0.537 | 66.0 | 55 min |
| nsa-lite | 114.1M | 1.8646 | 6.45 | 0.293 | 0.533 | 54.3 | (contended) |
| GQA baseline (`02`) | 114.1M | 1.8665 | 6.47 | 0.287 | 0.547 | 68.9 | (contended) |
| MHA | 123.6M | 1.8916 | 6.63 | 0.300 | 0.503 | 67.8 | 60 min |
| MLA | 112.8M | 1.9585 | 7.09 | 0.287 | 0.520 | 63.9 | 53 min |

*Clean-machine runs only; contended runs (other sessions competing for
the GPU) ran 6-8x longer at identical quality.

## Findings

1. **Winner: sliding-window.** Best perplexity, zero parameter difference
   from baseline, simplest implementation, and the pattern whose compute
   advantage *grows* with context length. The top three (sliding-window,
   nsa-lite, GQA baseline) are within 0.006 nats — effectively tied on
   quality — which itself is the finding: on this corpus, restricting
   attention to 64 local tokens costs nothing.
2. **NSA-lite's compressed global branch adds nothing over pure
   sliding-window here** — same quality, 18% slower decode (the pooled
   branch overhead in this unoptimized implementation). TinyStories has
   essentially no long-range dependencies for the global branch to
   capture; this variant needs a long-context task to show its value.
3. **GQA beats MHA at matched budget** (1.8665 vs. 1.8916) despite MHA
   having 9.4M more parameters and a 3x larger KV-cache. GQA's cache
   savings are free at this scale — consistent with why it became the
   industry default.
4. **MLA clearly underperforms at this scale/budget** (+0.09 nats over
   baseline). The d_c=128 low-rank KV bottleneck costs real quality in a
   12.3M-token training run. DeepSeek's results suggest this reverses at
   much larger scale/longer training — a scaling-check candidate, not a
   dismissal.

## Caveats

Single-seed, single-run comparisons; the top-3 gap is well within
run-to-run noise. Benchmark accuracies at n=300 cannot separate these
models (all HellaSwag/PIQA differences are within stderr). The honest
summary: **sliding-window is at least tied for best while being the
cheapest — that's the win.** Next step per the project roadmap:
scale the winner up (`04-scaling-checks`) and check whether the ordering
holds with more data and parameters — especially whether full attention
pulls ahead once the corpus contains longer-range structure.
