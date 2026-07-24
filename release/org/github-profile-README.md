# GarageLM

**Chasing the most intelligent model on the cheapest hardware.**

Frontier labs compete at unlimited budget. GarageLM claims the opposite
axis: maximum measured capability from **one consumer machine** — every
run trained, evaluated, and served on a single Apple M4 Pro Mac mini.

## Evidence so far

A 232M-parameter model, 1B refined tokens, 127 hours, ~$3 of electricity:

- **At-or-above Pythia-160M** (300B training tokens — 300× more) on
  HellaSwag, PIQA, ARC-Easy, and WinoGrande under a matched evaluation
  harness (within ±2.8-pt slice resolution)
- **Ahead of GPT-2 on ARC-Easy** (the one gap beyond 2× the eval SE)
  with ~5× fewer training FLOPs (6ND, one-epoch-WebText assumed)
- **Hybrid local+global attention beats full attention** at equal
  parameters (replicated, 3/3 seeds; under-trained regime) at ~30% of
  the KV cache
- **~310 tok/s fp16 / 530+ at 4-bit on-device decode** (MLX) — chat
  served by the same machine that did the training

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
