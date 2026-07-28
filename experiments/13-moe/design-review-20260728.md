# Design review: 13-moe (pre-launch)

Reviewer: design-reviewer agent. Status at review: PRE-LAUNCH (no out/ dirs,
no probes yet). All executed checks were CPU-only (an MLX eval server was
active on the GPU during review). Check script: parity/gradient/aux/EMA
verification vs a naive per-token MoE reference, param counts, path
resolution, capacity/window/budget arithmetic; outputs quoted inline below.

## Verdict

**FIX-THEN-LAUNCH** — blockers F1, F2, F3. Each is a minutes-scale change to
`moe/model.py` (or a README re-registration); the architecture, fairness
setup, plumbing, and budget are otherwise sound. Forward/backward math of the
fixed-capacity MoE was verified exactly against a reference implementation.

---

## Findings (ranked by cost-if-missed)

### F1 — BLOCKER: aux loss is 8x the pre-registered "Switch aux 0.01"

`experiments/13-moe/moe/model.py:156-160`

```python
f = assign.mean(dim=0) * (self.n_expert / self.top_k)
...
self.last_aux = (f.detach() * P).sum() * self.n_expert
```

The `E/k` factor on `f` makes `sum_e f_e = E` (not 1), so the loss double
counts the `* n_expert` that Switch eq. 4 applies outside the sum.
**Measured (CPU, router zeroed so P is exactly uniform): `last_aux = 8.0000`.
Switch's eq. 4 evaluates to 1.0 at uniform.** The gradient pressure toward
uniform routing is therefore 8x the paper definition at the same coefficient:
`aux_loss_coeff: 0.01` behaves like Switch alpha ~ 0.08 (also 8x the
HF-Mixtral top-k convention, which likewise normalizes to 1 at uniform).

Cost if missed: G-A's adoption margin is only 0.03 nats. An 8x-stronger
balancing tax on CE is exactly the kind of effect that can flip G-A to a
false "MoE is dead at the floor" after ~28h of training, with the run
recorded as having used the pre-registered recipe when it did not. Fix
either way, but pick one **before** launch:
- code: `f = assign.mean(dim=0) / self.top_k` (then uniform value = 1.0,
  matching Switch and the README), or
- README: re-register the actual definition and effective scale.

### F2 — BLOCKER: `ema_load` update disagrees with its init by a factor of k=2; G-R scale is ambiguous

`experiments/13-moe/moe/model.py:142-143` (init `1/E = 0.125`) vs `:184`
(update `0.01 * f / self.n_expert * self.top_k`, which algebraically equals
`0.01 * assign.mean(0)` — fixed point `k/E = 0.25` at balance).

**Measured: after 1200 perfectly balanced CPU steps, `ema_load` sum = 2.0000
(= k), per-expert ~0.196-0.28; init is 0.125.** Two consequences:
1. G-R ("min expert EMA load >= 2%") is calibrated against an undefined
   scale: 2% of a uniform-0.25 scale is 2x more lenient than 2% of a
   uniform-0.125 scale.
2. Readings drift upward from 0.125 toward ~0.25 over the first ~500 steps
   even at perfect balance (EMA half-life ~69 steps) — the sanity "min load
   11.8%" is still mostly the init value, not measured load.

Fix: delete `* self.top_k` at `:184` (fixed point becomes `1/E`, matching
the init and making uniform = 12.5%), or re-register G-R's threshold on the
`k/E` scale. One-token change; do it together with F1.

### F3 — BLOCKER: EMA buffers are `persistent=False`, so `--resume` silently resets the G-R monitor

`experiments/13-moe/moe/model.py:142-144`. **Verified: `ema_load`/`ema_drop`
absent from `model.state_dict()`**, hence absent from `state.pt`
(`moe/train.py:29-38` saves `model.state_dict()`). The train loop evaluates
BEFORE stepping at each eval iter (`moe/train.py:136-145`), so the first
post-resume eval prints `min_load=0.125 drop=0.000` from freshly
re-initialized buffers — fabricated-healthy numbers that can mask an
in-progress routing collapse for up to ~500 steps (~50 min at the 6.2s gate
step time). G-R is adjudicated from exactly these logged lines ("violation
at every logged eval = run stopped"), and this repo's history (05 gqa
allocator pathology, the reason STATE_EVERY exists) makes a mid-run resume
likely, not hypothetical.

Fix: make the buffers `persistent=True` (they are 9 floats/layer; checkpoint
cost nil). Note `rope_cos`/`rope_sin`/`window_mask` are also
`persistent=False` but are deterministically rebuilt in `__init__` — those
are fine; the EMAs are the only stateful non-persistent buffers.

### F4 — NOTE: "identical tensor shapes" holds for compute kernels, not for routing bookkeeping; `nonzero()` syncs are the real probe risk

`experiments/13-moe/moe/model.py:168-179`. Verified by shape-tracing: the
variable-length intermediates (`routed` from `nonzero()`, `gate[routed]`,
`idx[:n_routed] = ...`, `g[:n_routed, 0] = ...`, and the overflow-branch
`topk`) never reach a large kernel — `index_select`, both expert matmul
stacks, the gate multiply, and `index_add` all run at fixed
`capacity = 1280` (train shape). The variable pieces are 1-D tensors of at
most N = 4096 elements (<= 32KB), far below the DPO-episode pathology's
large-activation regime. However, `nonzero()` forces a GPU-to-host sync per
expert per layer: 8 x 12 = 96 per forward, 384 across the 4 micro-batch
forwards per optimizer step (plus backward graph overhead). This, not FLOPs
(expert FLOPs = 1.25x the control FFN by construction), is the most likely
cause if the <= 6.2s probe gate fails. No change required pre-probe; the
probe gate is the right adjudicator. If it fails, batch the routing
bookkeeping (e.g., argsort-based dispatch) before concluding MoE is slow.

### F5 — NOTE: capacity floors to 1 (or 0) at tiny N; sample.py's default prompt starts with heavy drops

`experiments/13-moe/moe/model.py:162`. Verified: `int(1.25*N*2/8)` gives
capacity 1 at N=6 (sample.py's default 6-token prompt) and 0 at N=1 — with
capacity 0 the FFN contribution is silently zero (residual only; no crash;
verified index paths handle empty tensors). Affects only qualitative
sampling: `generate` feeds the growing full context, so capacity recovers
within a few dozen tokens; train/eval (N=4096) and full-pass (N>=1024) are
unaffected. Optional: `capacity = max(capacity, self.top_k)`.

### F6 — NOTE: full-pass tail group runs one (1,1024) forward (capacity 320)

Verified: val.bin = 2,000,000 tokens, n_win = 1953, 1953 % 4 = 1, so
`full_pass_val.py` (micro=4) ends with a single (1,1024) window; capacity
becomes `int(1.25*1024*2/8) = 320` for that one forward. Capacity scales
proportionally with N, so drop policy is statistically identical; one extra
fixed shape, allocated once, eval-only. All three arms see the identical
evaluator and windows, so the comparison stays fair. No action.

### F7 — NOTE: dense-284m init/LR at the new width

`dense-284m/config.py:12` sets ffn_hidden 8192 (10.7x n_embd) under the
recipe's fan-in-independent init (std 0.02; down_proj 0.02/sqrt(24),
`model.py:241-243`), giving the FFN residual branch ~4x the control's output
variance at init. Precedent says this is survivable: 04 trained 231,852,032
params at the same LR 3e-4 stably (both 04 run summaries read). Watch the
iter-0/500 evals; no change required. Also: the README's own sequence
("sanity -> design review -> probes -> runs") still owes BOTH new arms their
probe gates — the reported sanity numbers (loss 12.1->7.1, drop EMA 2.1%,
min load 11.8%, entropy 2.079) cover only the moe arm; dense-284m has no
reported 15-step sanity yet. Run both probes before launch as pre-registered.

---

## Audit 1: Fairness — every diff enumerated

`diff` run against `experiments/05-data-frontier/hybrid/` (the arm whose
checkpoint produced the 3.8398 control number):

- **dense-284m/**: `model.py` and `sample.py` byte-identical (diff exit 0).
  `config.py`: ffn_hidden 2048 -> 8192 only (the lever). `train.py`:
  DATA_DIR `../data` -> `../../05-data-frontier/data` only (path shift
  because the milestone lives outside 05; verified to resolve to the same
  directory). **Exactly the two stated deltas; nothing unlisted.**
- **moe/**: `config.py`: ffn_hidden 2048 -> 1024 + {n_expert 8, top_k 2,
  capacity_factor 1.25, aux_loss_coeff 0.01} (the lever). `model.py`: SwiGLU
  -> ExpertSwiGLU + MoE router/dispatch; `GPT.forward` adds
  `aux_loss_coeff * sum(last_aux)` guarded by `if self.training`
  (`moe/model.py:267-270`). `train.py`: DATA_DIR (same as above) + the
  router-health print at evals (`moe/train.py:141-145`) — read-only
  monitoring, declared in the task. `sample.py` byte-identical.
- **moe/train.py vs dense-284m/train.py** differ ONLY in the router-health
  print. TrainConfig identical across both new arms and the 05 control
  (seed 1337, eval_iters 20, 6100 iters, LR schedule, clip, decay — diffed).
- Attention stack byte-identical in all arms (hybrid mask, RoPE, GQA);
  the FFN is the only lever, as pre-registered.

Param counts (CPU-instantiated, vocab 50257 from the 05 meta.pkl):
dense-284m **283,983,360** (= stated), moe **284,057,088** (= stated),
moe active **114,187,776** ~ control's 114,114,048 (delta = 12 routers x
6,144 + the 73,728 expert-split bookkeeping; active FLOPs match by
construction: 2 x (3 x 768 x 1024) = 3 x 768 x 2048 exactly). Total-param
gap moe vs dense-284m = 73,728 (0.026%) — memory-matched as claimed.

## Audit 2: Math — checks run and results

Reference checks (CPU, tiny configs, `moe`'s actual SwiGLU class vs a naive
per-token no-capacity loop):

- **Forward parity (drop-free capacity)**: max |diff| 8.9e-08 — PASS.
- **Input-gradient parity**: max |diff| 8.9e-08 — PASS.
- **Router + all-expert parameter-gradient parity**: max |diff| 4.8e-07 — PASS.
- **Padded index-0 trick**: pad rows carry gate exactly 0 in a freshly
  zeroed `g`, so token 0's output matched naive exactly, and with a loss
  touching every token EXCEPT token 0, `x.grad[token 0]` = 0.000e+00 —
  no forward corruption, no spurious gradient through the zero-gate multiply
  (grad into `expert(flat[0])` is `grad_out[0] * 0`; grad into `g`'s pad
  rows is discarded because those entries are constants, not graph nodes).
  Duplicate index-0 entries in `index_add` accumulate zeros — safe. PASS.
  (Residual caveat: if activations ever reach inf/NaN, `0 * inf = NaN`
  would poison token 0 via pad rows — only relevant in an already-diverged
  run.)
- **Experts receiving only pad rows**: parameter grads exactly 0 — PASS.
- **Overflow path**: at capacity 1, output matched a keep-top-gate reference
  exactly (drops select lowest gates; dropped tokens get zero FFN, residual
  carries them) — PASS.
- **Gate renormalization**: top-2 softmax probs renormalized to sum 1
  (`:153`); denominator strictly positive (softmax top-k), no div-by-zero.
  If one of a token's two experts drops its assignment, the kept gate is not
  re-renormalized — standard fixed-capacity behavior.
- **Aux differentiability split**: `f` computed under `no_grad` and
  `.detach()`ed; `P = probs.mean(0)` differentiable — correct Switch
  structure (scale is F1).
- **Grad accumulation**: aux is inside `loss`, which is divided by
  `grad_accum_steps` (`train.py:106`) — consistent scaling; clip and LR per
  optimizer step unchanged from control.
- **Eval purity**: `model.eval()` forward with targets returned exactly
  manual CE (4.615841 = 4.615841, no aux); train mode returned exactly
  CE + 0.01 * sum(aux) — PASS. EMA updates are also guarded by
  `self.training` (`:182`), so estimate_loss does not pollute the monitor.

## Audit 3: Plumbing — verified, not assumed

- DATA_DIR from both train.py ROOTs resolves to
  `/Users/anthonytrevino/Desktop/llm-arch-explore/experiments/05-data-frontier/data`;
  meta.pkl (50 B), train.bin (3,553,800,186 B = 1,776,900,093 tokens),
  val.bin (4,000,000 B = 2,000,000 tokens) all exist — the historical
  one-directory-too-high failure mode does not recur here.
- No `out/` exists in either arm; no stale state.pt to mis-resume from.
- Results naming: `full_pass_val.py::result_name` yields `13-moe-moe` and
  `13-moe-dense-284m`; grep of `benchmarks/results/` shows no existing
  `13-moe*` files — no collision with any prior milestone.
- `full_pass_val.py:80-85` pops cached `config`/`model` modules before
  import — the two same-named arm modules cannot cross-contaminate when
  evaluated back-to-back.
- Adjudication path confirmed aux-free twice over: `full_pass_val.py:52`
  calls `model(x)` with NO targets (loss branch never runs) and computes CE
  externally from logits at `:53-54`, under `model.eval()` (`:92`) and
  `@torch.no_grad()`. `ckpt["model_cfg"]` round-trips the MoE fields through
  `GPTConfig(**...)`.
- Resume machinery present and structurally sound (STATE_EVERY=1000, atomic
  tmp+rename, CPU+MPS RNG save/restore, `--resume` flag) — except F3.
- Control number confirmed: `benchmarks/results/05-data-frontier-hybrid-fullpass.json`
  = 3.8398, block 1024, same val.bin — G-A's target is real and comparable.

## Audit 4: Budget

- Token math: 6100 x (4 x 4 x 1024) = **99,942,400** ~ stated 100M. Pool =
  1,776,900,093 tokens = **17.8x** budget (config comments say ~16x — minor
  doc drift, favorable direction, no repeats either way).
- Memory: logits envelope unchanged for all arms — 4 x 1024 x 50257 fp32 =
  **823 MB** < 1 GB documented ceiling; batch shape 4x1024 is nowhere near
  the 32x512 pathology. MoE expert-loop activation overhead ~ 8 experts x
  1280 x (768 + 3x1024 + 768) x 4 B x 12 layers ~ **2.3 GB** saved for
  backward (~1.25x the control FFN's, per the capacity factor). dense-284m
  FFN activations ~ 4096 x (768 + 3x8192) x 4 B x 12 ~ **5.0 GB**. Both
  comfortable on 48 GB.
- Wall clock: control measured 25,449 s / 6,100 = **4.17 s/step** (05 hybrid
  run_summary). dense-284m FLOPs ratio 284.0M/114.1M = 2.49x -> ~10.4 s/step
  -> **~17.6 h** (04 precedent: 232M full-attention at the same 16,384
  tokens/step ran 7.08 s/step, so ~10 s at 284M/block-1024 is plausible).
  moe at the probe gate <= 6.2 s -> **<= 10.5 h**. Total ~28 h, sequential
  runs only, contention check before each (an lm_eval + MLX server was
  running at review time — confirm it has finished before launch).
- Capacity arithmetic at every shape that matters: train and estimate_loss
  both use (4,1024) -> N=4096 -> capacity **1280** (identical fixed shapes
  in-run); full-pass groups of 4 windows -> same 1280, one tail forward at
  (1,1024) -> 320 (F6); sampling small-N floor (F5).

## Audit 5: Gates

- G-A: measurable, unambiguous (full-pass <= 3.8098 vs verified 3.8398
  control, shared evaluator/val/block), action stated. Sound — contingent on
  F1 so the trained model actually matches the registered recipe.
- G-M: explicitly a measurement with the 08 seed-noise yardstick (0.02) and
  a declared non-adoption role. Sound.
- G-R: numeric thresholds and a stop action, but the load metric's scale is
  self-inconsistent (F2) and resets on resume (F3) — fix both so the gate
  means one thing for the whole run.
- Probe gates: all numeric (step time <= 6.2 s, params within 1% — both new
  arms already measured exact, entropy ln 8 — reported 2.079, 15-step loss
  decrease, drop < 15%); dense-284m's probes still owed (F7).
- Deferrals (MLX MoE port, shared experts, z-loss, LatentMoE) are explicit,
  which keeps this milestone's claim honest: quality-per-byte + router
  health + train-time step cost, with inference latency deferred and named.
  The eval plan covers the claim being made.

## Checks run (reproducible)

Script: scratchpad `moe_checks.py` (CPU-only), `uv run python` from repo
root; plus `diff` of all six arm files against `experiments/05-data-frontier/hybrid/`,
`ls`/`getsize` of the 05 data pool and `benchmarks/results/`, and reads of
the 04/05/12 run summaries for step-time precedent. Raw outputs:

```
[dense-284m] params = 283,983,360 (stated 283,983,360) match=True
[moe] params = 284,057,088 (stated 284,057,088) match=True
[moe] active params = 114,187,776 (per-expert 2,359,296)
[moe] ema buffers in state_dict (survive --resume): False
[dense-284m] DATA_DIR -> .../experiments/05-data-frontier/data  exists=True
[moe]        DATA_DIR -> .../experiments/05-data-frontier/data  exists=True
capacity: N=4096 -> 1280 | N=1024 -> 320 | N=6 -> 1 | N=1 -> 0
forward matches naive (drop-free): True  max|diff|=8.94e-08
input-grad matches: True  max|diff|=8.94e-08
param-grad matches (router + all experts): True  max|diff|=4.77e-07
token-0 spurious grad (should be 0): 0.000e+00
token-0 output matches naive: True
unused experts [3, 6, 7]: max param-grad (should be 0): 0.000e+00
overflow path matches keep-top-gate reference: True
aux at uniform P: 8.0000  (Switch eq.4 uniform value = 1.0)
ema_load init = 0.1250 (1/E)
ema_load after 1200 balanced steps: min=0.1959 max=0.2802 sum=2.0000 (k/E=0.25)
eval loss == pure CE (no aux): True (4.615841 vs 4.615841)
train loss == CE + 0.01*aux: True (aux_sum=16.0536)
val tokens=2,000,000  n_win=1953  last group size=1
pool=1,776,900,093  budget=99,942,400  pool/budget=17.8x
```

## Verdict

**FIX-THEN-LAUNCH.** Blockers, each minutes-scale:
1. **F1** — make the aux loss match the pre-registered Switch definition
   (`f = assign.mean(dim=0) / self.top_k`) or re-register the actual 8x
   scale in the README before any training step.
2. **F2** — remove `* self.top_k` from the `ema_load` update (or re-register
   G-R's threshold on the k/E scale) so the gate has one defined meaning.
3. **F3** — make `ema_load`/`ema_drop` persistent so `--resume` does not
   reset the G-R monitor.

Then run the still-owed probe gates for BOTH arms (dense-284m has no sanity
numbers yet) and launch sequentially.
