# 08-solidify

Two things the `05`/`07` headline needed before scaling further: seed
replication of the hybrid-vs-full-attention gap (the write-up's own
limitations section flagged n=1), and a check that the inherited window=64 /
1-in-4-globals choices weren't lucky. Six runs at the exact `05` recipe
(114M, block 1024, 100M tokens of the shared `05` pool); seed dirs are
byte-identical code to `05` with only the seed changed.

## Seed replication (the load-bearing result)

Val loss (train-time final / 256-window probe mean):

| Seed | GQA (full attn) | hybrid (w64, ge4) | gap (hybrid−GQA) |
|---|---|---|---|
| 1337 (from `05`) | 3.8680 / 3.8764 | 3.8091 / 3.8219 | **−0.059 / −0.055** |
| 1338 | 3.8671 / 3.8524 | 3.8392 / 3.8235 | **−0.028 / −0.029** |
| 1339 | 3.8413 / 3.8519 | 3.8128 / 3.8240 | **−0.028 / −0.028** |

**Hybrid beats full attention in 3 of 3 seeds; mean gap ≈ −0.038 nats
(probe: −0.037), range −0.028 to −0.059.** The original `05` seed was the
most favorable of the three — the honest headline number is the mean.
Notably the hybrid arm is far more seed-stable than the GQA arm (probe-mean
spread 0.002 vs 0.025).

## Sweep (seed 1337, directly comparable to the `05` hybrid at 3.8091/3.8219)

| Variant | Config | Val loss (final / probe) | KV cache @1024 |
|---|---|---|---|
| hybrid (baseline) | w=64, globals every 4th | 3.8091 / 3.8219 | 30% of full |
| hybrid-w128 | w=128, globals every 4th | 3.8357 / 3.8416 | 35% |
| hybrid-ge6 | w=64, globals at 5, 11 | **3.8083 / 3.8191** | **~15%** |

- **Bigger windows hurt** (+0.027 at w=128): more local reach is not better
  — consistent with local layers acting as an inductive bias rather than a
  budget to maximize.
- **ge6 ties-to-edges the baseline at half the cache** — a real efficiency
  finding at 114M/12 layers. It does NOT gate into `09`: its 0.003 edge is
  far below the pre-registered 0.02 switching threshold, and the 12-layer
  ge6 ends on a global top layer while a 16-layer ge6 would leave the top
  4 layers with no global above them — a structurally different, untested
  network. Validating sparser-globals at 16 layers (e.g. globals at 7, 15)
  is flagged as future work.

## Gate decision for `09-flagship-2`

**Keep w=64 / globals-every-4th** (the `07` configuration), per the
pre-registered default. The seed replication upgrades the write-up's claim
from n=1 to n=3.
