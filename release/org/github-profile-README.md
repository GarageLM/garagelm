# GarageLM

**The most intelligent model trainable on the cheapest hardware.**

Frontier labs compete at unlimited budget. GarageLM claims the opposite
axis: maximum measured capability from **one consumer machine** — every
run trained, evaluated, and served on a single Apple M4 Pro laptop chip.

## Proof, so far

A 232M-parameter model, 1B refined tokens, 127 hours, ~$3 of electricity:

- **Ties Pythia-160M** (300B training tokens — 300× more) on HellaSwag,
  PIQA, ARC-Easy, and WinoGrande under a matched evaluation harness
- **Beats GPT-2** on ARC-Easy and PIQA with ~5× fewer training FLOPs
- **Hybrid local+global attention beats full attention** at equal
  parameters (replicated, 3/3 seeds) at ~30% of the KV cache
- **530 tok/s on-device decode** (MLX fp16; ~700 at 4-bit) — chat served
  by the same laptop that did the training

## How

Data quality over data volume. Efficient attention over brute force.
Controlled experiments over vibes: fixed seeds, matched baselines,
pre-registered gates, negative results published alongside wins. Every
claim ships with the quality **and** size **and** latency numbers —
never one without the others.

## Artifacts

- [`hybrid-gpt-232m`](https://huggingface.co/garagelm/hybrid-gpt-232m) —
  base model · [`hybrid-gpt-232m-chat`](https://huggingface.co/garagelm/hybrid-gpt-232m-chat) — chat-tuned
- Research repo: training pipeline, MLX inference stack, full write-up

*Small hardware. Real research.*
