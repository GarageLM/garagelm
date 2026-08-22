# 13-moe

MoE at the hardware floor. The question frontier papers don't answer: at
a hard memory budget, is a sparse model worth its bytes? Frontier MoE
comparisons match **active** parameters (equal FLOPs, memory treated as
free). At this lab's floor, memory *is* the budget — so we run both
controls.

**Released** (2026-08-21): the moe arm is on the Hub as
[`garagelm/hybrid-gpt-moe-284m-a114m`](https://huggingface.co/garagelm/hybrid-gpt-moe-284m-a114m),
bf16, `trust_remote_code` wrapper `release/modeling_moe_gpt.py`, card
`release/card_moe.md` carrying the three-way tables below. The wrapper
defaults to dropless routing (the training-time fixed capacity never
bound: drop rate 0.0004) and keeps capacity mode switchable; its card
reports the shipped artifact's own full-pass val in both modes. Not an
MLX build: that port stays deferred per the verdict.

## Pre-registered gates (written before any run)

- **Arms** (05 pool, block 1024, 100M tokens, seed 1337, hybrid
  attention everywhere; only the FFN differs):
  - `control-active` — the existing 05 hybrid (114M dense; full-pass
    **3.8398**). No new training.
  - `dense-284m/` — FFN-width-only scale-up (ffn_hidden 8192), ~284M
    dense: the memory-matched control.
  - `moe/` — 8 experts × ffn_hidden 1024, top-2 (renormalized gates),
    Switch aux loss 0.01 (train-time only, never in val CE), fixed
    expert capacity 1.25 (fixed tensor shapes; overflow dropped and
    logged): ~284M total / ~114M active.
- **G-A (vs active-matched)**: MoE full-pass ≤ 3.8398 − 0.03. If 2.5x
  the parameters at equal FLOPs can't clear this, MoE is dead at the
  floor.
- **G-M (vs memory-matched)**: measurement, not adoption. Report the
  MoE−dense284 gap; |gap| < 0.02 (the 08 seed-noise yardstick) is
  declared "parity per byte."
- **G-R (router health)**: min expert EMA load ≥ 2% and mean drop rate
  ≤ 10% at every logged eval; violation = routing collapse → run
  stopped, reported as a negative.
- **Probe gates**: MoE step time ≤ 1.5x the 114M dense step (~4.1s →
  ≤ 6.2s), memory inside the envelope, param counts within 1% of
  analytic (~284M both new arms), router entropy ≈ ln 8 at init,
  15-step loss decrease, init drop-rate < 15%.
- **Adjudication**: `benchmarks/full_pass_val.py`, shared 05 val, all
  three checkpoints. In-run numbers decide nothing.
- **Deferred, explicitly**: LatentMoE / quantile balancing and other K3
  lanes (Attention Residuals, per-head Muon); shared-expert designs;
  router z-loss; MLX MoE inference port (expert paging) — until this
  milestone's verdict justifies them.

## Results

**Verdict: at 100M tokens, 2.5x the parameters bought validation loss and
nothing else measurable. Neither 284M arm separates from the 114M control
on downstream tasks, and both decode slower. MoE clears its pre-registered
loss gate by 0.0004 nats and loses on every other axis.**

| Arm | Total params | Active | s/step | Full-pass val | Decode tok/s | 4-task avg |
|---|---|---|---|---|---|---|
| control-active (05 hybrid) | 114,114,048 | 114M | 4.17 | 3.8398 | **54.3** | **44.08** |
| moe | 284,057,088 | 114M | 5.70 | **3.8094** | 8.9‡ | 42.08 |
| dense-284m | 283,983,360 | 284M | 7.46† | **3.7711** | 41.8 | 43.75 |

† contended; see Run log.  ‡ implementation-bound; see "MoE decode collapse".

Per-task (lm-eval, n=300/task, 0-shot, MMLU 5/subject — the control's exact
protocol, acc / acc_norm):

| Task | control 114M | moe | dense-284m |
|---|---|---|---|
| hellaswag | 32.3 / 33.7 | 34.3 / 33.0 | 33.3 / 33.3 |
| piqa | 58.3 / 56.3 | 56.3 / 52.0 | 56.0 / 53.7 |
| arc_easy | 36.3 / 34.7 | 39.3 / 34.3 | 39.7 / 38.3 |
| winogrande | 51.7 | 49.0 | 49.7 |
| mmlu | 26.7 | 26.0 | 27.0 |

**No downstream difference is resolvable.** The 4-task average spans 2.0
points across all three arms; at n=300 the standard error on a proportion is
~2.9 points, so ~2.05 points on the difference of two 4-task averages. The
whole spread is about one sigma. The correct statement is not "the control
wins downstream" — it is that this suite at this sample size cannot
distinguish a 114M model from a 284M model trained on the same 100M tokens.
Resolving a ~3-point difference would need a much larger `--limit`.

### The result that matters

Validation loss and downstream capability came apart. dense-284m is 0.0687
nats better than the control — a real, deterministic, 2M-token measurement —
and converts none of it into measurable task performance, while giving up
23% of decode throughput. This is exactly what single-axis reporting hides,
and it is why the convention requires quality, size, and latency together.

### MoE decode collapse — 8.9 tok/s, and why

The moe arm decodes **6.1x slower than the 114M control** despite equal
active parameters. The arithmetic identifies the cause exactly: control =
18.4 ms/token, moe = 112 ms/token, gap ≈ 94 ms. Design-review F4 counted
**96 `nonzero()` GPU-to-host syncs per forward** (8 experts × 12 layers).
94 ms / 96 ≈ 1 ms per sync, a routine MPS sync latency. Measurement and
diagnosis agree to rounding.

Why training never saw it: the routing bookkeeping is **per forward, not
per token**. At training, T=4096 per micro-batch amortizes 96 syncs to
nothing, which is why the arm cleared its 5.70 s/step gate comfortably. At
decode, T=1, so a single token pays all 96.

**This is not an architectural verdict on MoE**, and F4 pre-registered that
caveat: *"batch the routing bookkeeping (e.g., argsort-based dispatch)
before concluding MoE is slow."* 8.9 tok/s is an honest number for what is
implemented here — a naive per-expert loop with host syncs — and a batched
dispatch or the deferred MLX port would change it substantially. Cite it as
an implementation measurement, never as "MoE is slow at the floor."

- **G-A (vs active-matched) — PASS, by 0.0004 nats.** Gate was ≤ 3.8398 −
  0.03 = 3.8098; moe landed 3.8094. Honest reading: the margin is 1/50th of
  the 0.02 seed-noise yardstick from 08, at n=1 seed. This is a pass by the
  letter of the pre-registration and is statistically indistinguishable from
  sitting exactly on the threshold. It should not be cited as "MoE beats the
  control" without that qualifier.
- **G-M (vs memory-matched) — not parity; dense wins.** moe − dense-284m =
  **+0.0383**, about 2x the 0.02 yardstick and in dense's favour. Spending
  the same memory on FFN width beat spending it on 8 experts.
- **G-R (router health) — PASS.** See below.

What the numbers say together: extra capacity did buy validation loss —
both 284M arms beat the control, so "100M tokens binds, parameters cannot
help" is refuted *on loss*. But none of that gain reached the downstream
suite, and both arms decode slower. On the project's actual three-way
criterion, neither 284M arm is a good trade at this token budget: the
control is at least as good on every measurable task and is the fastest of
the three.

MoE specifically loses on all three axes against the control — worst
4-task average (42.08 vs 44.08, within noise), 2.5x the memory, 6.1x
slower decode — while winning only the pre-registered loss gate, and that
by 0.0004 nats.

In-run numbers decide nothing (20-batch stochastic estimates) and this run
demonstrated why: the moe arm's in-run → full-pass offset was **−0.019**
while the control's was **+0.031** — opposite signs, so extrapolating one
arm's full-pass from another's offset predicts the wrong verdict. Param
counts land within 0.0000% of analytic for both new arms.

### Deferred lanes — still deferred

The pre-registration gated LatentMoE / quantile balancing, shared-expert
designs, router z-loss, and an MLX MoE inference port (expert paging) on
"until this milestone's verdict justifies them." It does not: the sparse
arm lost the memory-matched comparison, so expert paging would be
optimising the arm that came second. Revisit only if a future run changes
the per-byte answer.

### Probe gates

Both arms cleared their pre-registered probes. moe: 5.70 s/step measured
over the full run = **1.37x** the 114M control, inside the 1.5x gate.
dense-284m (run 2026-07-28, F7's owed probe): params exact, forward
(4, 1024, 50257) loss 10.978 finite, 15-step decrease 10.9614 → 9.5313
(−1.43 nats), **7.46 s/step** (the review projected 10.4 from the FLOPs
ratio; the measurement was taken under Spotlight contention, so it is an
upper bound), 5.06 GB allocated / 21.88 GB driver. Projected wall clock
12.6 h. Verdict GO.

### G-R (router health) — PASS

The moe run's stdout was not captured to a file, so the per-eval
`min_load`/`drop` lines survive only where they happened to be retained.
What is recoverable:

- **End of run, from `ckpt.pt`** (F3's `persistent=True` fix is what made
  this possible): min expert EMA load **0.1147** across all 12 layers
  against a uniform 0.125, mean drop rate **0.0004**, max any-layer drop
  0.0013. Per-layer loads sum to exactly 1.0000 — the F2 scale fix
  confirmed in the trained artifact, not just in review.
- **Logged evals recovered**: iters 0, 4000, 4500, 5000, 5500, 6000, all
  reading min_load 0.115–0.117 and drop ≤ 0.001.
- **Not retained**: iters 500–3500. The gate reads "at every logged eval,"
  so this is stated as a partial record rather than a clean sweep. The EMA
  half-life is ~69 steps, so a collapse-and-recovery inside the unobserved
  window would have to be both fast and self-healing to escape the iter-4000
  reading; nothing in the loss curve suggests it. Recorded as PASS with the
  gap disclosed.

Capacity 1.25 was ample — essentially nothing was dropped — which is
itself a finding: the fixed-capacity ceiling never bound, so the drop
policy did not shape this result.

### Run log

The dense-284m arm took three launches:

1. **05:29–06:58** — killed 89 min in by a machine reboot (no panic report
   on disk; cause unattributed). `STATE_EVERY` was 1000 (~3.1 h at the
   measured step time), so nothing was resumable and the arm lost the lot.
2. **07:25–~09:03** — reached iter 750 and wrote a complete `state.pt`,
   then the process was reaped. No OOM/jetsam event, 13.7 GB free, swap
   barely touched; the supervising background task was reported killed, so
   the cause looks like the launcher's lifecycle rather than the run.
3. **09:08–20:26** — relaunched fully detached (`nohup`), `--resume` from
   iter 750. Cost of the interruption: ~6 min, versus the 89 min the first
   one cost. That is the `STATE_EVERY` → 250 change paying for itself.
   Completed 6100 iters / 99,942,400 tokens in 12.91 h.

Changes made across the relaunches, none of which touch training math:
`STATE_EVERY` 1000 → 250 for this arm, stdout tee'd to `out/train.log`
so the eval record survives the process, and a pid recorded in
`out/train.pid` for the watchdog.

Caveat on the dense arm's step time: iters 0–750 ran at 7.66 s/step, but
the machine was concurrently running an unrelated training, an MLX export
at 100% CPU, and a third job. The loss trajectory is unaffected — it is
deterministic given seed 1337 and the restored RNG state — but the dense
`s/step` figure is measured under contention and is not a clean
throughput comparison against the moe arm's 5.70.
