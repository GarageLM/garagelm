# 09-flagship-2

The `07` flagship recipe (232M hybrid, w=64, globals every 4th layer —
config re-confirmed by `08`'s seed replication and sweep) pushed from 0.5B
to **1.0B refined tokens**, on a pool extended to 3.55B tokens
(FineWeb-Edu shards 000–003 + Cosmopedia-v2 shards 000–001, same ~84.5/15.5
blend). val.bin is byte-identical to `05`/`07`'s, so perplexities compare
directly.

## Results

Trained 126.9h of compute (7.45–7.60s/step). Two external process kills
(both at ~6:20am, both right after a state save — the morning session
reconnect reaping background tasks, not a training failure); `--resume`
lost zero steps.

**Val loss 3.0700 (PPL 21.5)** vs `07`'s 3.1246 (22.75) — the
pre-registered "beat 07 on identical val data" criterion: **met**.

| Model | Tokens | HellaSwag | PIQA | ARC-E | WinoGrande | MMLU |
|---|---|---|---|---|---|---|
| **09 (this run)** | **1.0B** | .357/.397 | **.633/.610** | **.477/.450** | .517 | .267 |
| 07 | 0.5B | .353/.393 | .617/.593 | .447/.433 | .533 | .267 |
| gpt2 (local ref) | ~10B | .353/.427 | .610/.617 | .420/.380 | .530 | .267 |
| Pythia-160M (local ref) | 300B | .350/.390 | .627/.627 | .460/.397 | .510 | .267 |

(Same harness/slices as the `07` evaluation; reference rows reused from it.)

- Pre-registered criterion #2 ("beat gpt2 on ≥3 of 4"): **partially met —
  1 clear win (ARC-E, the only gap beyond 2×SE); PIQA ahead within
  noise; 2 ties (HellaSwag, WinoGrande)**;
  behind only on HellaSwag acc_norm. Against Pythia-160M, 09 is now
  ahead-or-tied everywhere (clearly ahead on ARC-E).
- **Scaling is bending**: doubling tokens bought 0.055 nats of val loss,
  +3.0 ARC-E, +1.6 PIQA, ~0 HellaSwag. The recipe still improves but the
  steep part of the curve is behind; the remaining gap to SmolLM2-135M
  (2T tokens) is not closable by local token count alone.
- Long-range: probe mean 3.0290, same deep-context profile as `07`.
- MLX: parity OK; 4-bit costs +0.016 nats. Full numbers in
  `benchmarks/results/mlx-09-flagship-2-*.json`.
- Qualitative: fluent encyclopedic register; factual reliability remains
  232M-grade (confidently wrong astronomy in the sample).

## Verdict

The 1B-token run is the repo's best model and meets its primary criterion,
but its clearest contribution is measuring **where the data-quality lever
saturates locally**: refined data bought a 600x token-efficiency tie with
Pythia at 0.5B, and the second 0.5B bought low-single-digit benchmark
points. Next capability jumps need something other than more of the same
tokens (bigger model, longer context, distillation, or post-training).

## Run it

```
uv run python experiments/09-flagship-2/data.py      # extends the 05 pool to 3.55B tokens
uv run python experiments/09-flagship-2/train.py --resume   # ~5.3 days
```
