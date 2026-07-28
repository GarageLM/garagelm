# Kimi K3: what survives miniaturization

Notes on the Kimi K3 technical report (Kimi Team, 2026; `k3_tech_report.pdf`
in this directory), read as a candidate recipe book for the frontier-compression
thesis. K3 is a 2.8T-parameter MoE (104B active, 1M-token context) that claims
~2.5x scaling-efficiency over K2 from a bundle of architecture, data, and
training changes. Each technique below is triaged on the standard three axes
(what it solves, what it costs, where it breaks down) plus a fourth: whether it
can be replicated on one M4 Pro at 1M-250M params, and what we would learn.

One meta-observation before the list: Moonshot themselves miniaturized this
architecture. The chip-design case study runs a "nano" model with the same
recipe (hybrid KDA + NoPE-MLA, Block AttnRes, sigmoid MoE routing, INT4). The
recipe is explicitly scale-portable in the authors' own hands, which is the
strongest external signal yet for this repo's thesis. The 2.5x headline,
however, is an aggregate from their internal scaling-law fits at 1e20-1e21
FLOPs; nothing at our compute validates it, and decomposing that bundle into
techniques that do or don't survive miniaturization is exactly the job.

## The K3 recipe in one paragraph

Sequence mixing: each block is 3 Kimi Delta Attention layers (gated delta-rule
linear attention with channel-wise decay; fixed-size recurrent state instead of
a KV cache) plus 1 Gated MLA layer with no positional encoding; a final global
layer closes the stack. Depth mixing: Attention Residuals, where each layer
softmax-attends over all preceding layer outputs instead of accumulating one
residual stream. Width mixing: LatentMoE with 896 routed experts (16 active)
operating at half width, stabilized by RMSNorm before the up-projection,
softcapped SiTU-GLU activations, and Quantile Balancing for aux-free load
balance. Optimizer: Muon with per-head Newton-Schulz orthogonalization, cosine
schedule (they find cosine beats WSD once each schedule gets its own HP
search). Post-training: SFT, RL across domains and effort levels, multi-teacher
on-policy distillation, MXFP4 quantization-aware training from SFT onward, and
an MTP layer fine-tuned into an EAGLE-3 draft for speculative decoding.

## Tier 1: replicable as-is, high expected signal

**Hybrid linear+global attention (3:1 KDA:MLA)** (Kimi Linear, arXiv
2510.26692; Gated DeltaNet, Yang et al., 2025). *Solves*: long-context compute
and cache. The linear layers carry a fixed-size state (dk x dv per head), so
the per-layer cache is O(1) in sequence length; only every 4th layer pays a
growing KV cache. This is our 05/08 hybrid conclusion (local layers + sparse
global layers, exactly 3:1) with the local mechanism upgraded from
sliding-window to a learned recurrence. *Costs*: implementation. FLA's kernels
are Triton/CUDA; on MPS this needs a pure-PyTorch chunkwise scan. K3's
lower-bounded decay (scaled sigmoid, g_min = -5, replacing unbounded
negative-Softplus) exists to keep the chunkwise rescaling inside bf16 range and
is directly portable, arguably more valuable at low precision than at fp32.
*Breaks down*: at our 512-1024 training contexts a window-64 cache is already
tiny; the payoff only shows at 4k+ context, which small models can afford to
train at. Short-context throughput may regress vs SDPA sliding-window since we
have no fused kernel. *Status here*: the natural phase-two attention
milestone. Fair comparison against the 05 hybrid, swap only the local layers,
train at 4k context, report long-range probe + state size vs KV size.

**NoPE on the global layers.** K3 puts no positional encoding on MLA layers;
the recurrent layers carry position implicitly, and context extension then
needs no RoPE rescaling or YaRN. *Solves*: context extension surgery.
*Costs*: none at train time; one config flag. *Breaks down*: unknown whether
window-64 sliding-window layers encode position as well as a decay recurrence
does; that is the experiment. *Status here*: cheapest ablation on the existing
hybrid, even before any KDA work; `benchmarks/long_range_probe.py` is the
instrument.

**Full-rank sigmoid output gate on attention** (Qiu et al., 2025). Per-token,
per-channel sigmoid gate on the attention output, claimed to remove attention
sinks and add useful non-linearity; K3 applies it to both KDA and MLA.
*Costs*: ~d^2 params per attention layer, so the control must be
param-matched per the fair-comparison rule. *Status here*: cheap bolt-on arm
for the next attention experiment.

**Attention Residuals, full form** (Kimi Team preprint, 2026). Each layer
learns a pseudo-query and softmax-attends over RMSNorm'd outputs of all
preceding layers, replacing the single accumulated residual stream. At K3's
depth this needs a block-sparse variant; at our depth 16 the full O(L^2 d)
form is trivial and the O(Ld) memory of keeping layer outputs alive is
nothing. *Breaks down*: entirely unvalidated at small scale and small depth;
16 layers may not be deep enough for selective depth-mixing to beat a plain
residual. That makes it a genuine, publishable small-scale question. *Status
here*: standalone variant arm, one new module, no other changes.

**Per-schedule/per-optimizer HP search discipline.** K3 found cosine beats WSD
only after giving each schedule its own peak-LR and batch search; a shared
config unfairly favors whichever schedule it was tuned for. This directly
re-opens our 11-efficiency Muon negative, which ran default LR with no weight
decay and was explicitly scoped as such. *Status here*: any optimizer rematch
(below) must budget a small LR sweep per arm, or it measures nothing.

## Tier 2: replicable with adaptation

**Per-Head Muon.** Orthogonalize each head's momentum block separately instead
of the full Q/K/V matrix; equalizes update scale across heads and is cheaper
per Newton-Schulz step. *Status here*: the right shape for a Muon rematch at
20-50M params: per-head NS, per-arm LR sweep, cosine for both arms. Our prior
negative stands only for Muon-at-default.

**LatentMoE + Quantile Balancing at garage scale.** MoE is this repo's biggest
literature gap (no note, no experiment). K3's two portable ideas: (1) routed
experts operate at half width behind a shared full-width path, cutting the
param and compute cost of expert multiplicity; (2) QB sets each expert's
routing bias from the score quantile matching its target load, a closed-form,
hyperparameter-free replacement for the sign-update of aux-loss-free
balancing, derived from optimal assignment. QB is scale-agnostic and exactly
computable at our batch sizes (the histogram estimator exists only because
their batch spans millions of tokens). Unified memory is also unusually
MoE-friendly at inference: total params sit in the same 48GB the GPU reads,
so we pay flops for active params while holding many experts resident.
*Breaks down*: 896 experts is not miniaturizable, and whether expert
specialization emerges at all at <250M total params and ~1B tokens is an open
question, which is the point. Total params inflate against the 250M target, so
a first run looks like ~60M-active / ~180M-total, 8-16 latent experts.
*Status here*: candidate milestone after the attention phase; needs its own
literature note first (MoE fundamentals: Switch, DeepSeekMoE, aux-free
balancing).

**SiTU-GLU.** Softcap both SwiGLU branches (beta1=4 gate, beta2=25 up), giving
a bounded output (|y| <= 100) while matching SwiGLU near the origin.
*Solves*: activation outliers that break low-precision training. *Breaks
down*: our training is fp32 on MPS (bf16 was a validated null on throughput),
so the motivation is weak today; it matters for the 4-bit MLX deployment path
and any future GPU tier with fp8. *Status here*: trivial param-matched
ablation arm; low priority; more interesting jointly with QAT below.

**MTP layer -> EAGLE-style draft + speculative decoding.** K3 pre-trains one
MTP layer and fine-tunes it into an EAGLE-3 draft with a loss that directly
maximizes acceptance rate (LK loss: -log sum min(p,q)). *Solves*: decode
latency, the deployment number this repo cares most about. *Status here*: a
232M target with a single-block draft is small enough to train both on the
M4; MLX supports speculative decoding; the win lands directly in
`benchmarks/mlx/bench_inference.py` as tokens/sec at matched quality.

**Quantization-aware training for the 4-bit path.** K3 runs QAT (MXFP4
experts, MXFP8 activations) through all of post-training so serving precision
equals training precision. We already measured post-training 4-bit MLX costs
of 0.01-0.016 nats for +35-75% decode speed; QAT during SFT is the obvious
attempt to push that toward zero. *Status here*: bolt onto the next SFT run
with int4 fake-quant on expert/FFN weights; report the standard three axes.

**Rephrased-corpus recipe.** K3 rephrases knowledge and math corpora with
style-diverse prompting, chunkwise generation, and fidelity verification
against the source. We already ride the open version of this bet
(Cosmopedia-v2), and 09 showed the data lever saturating at our scale (0.055
nats for 2x tokens). The incremental delta needs a teacher model and real
spend. *Status here*: noted, not actionable locally; revisit if a GPU tier or
API budget appears.

## Tier 3: observe only

The 1M-token context ladder, native vision (though the from-scratch-beats-
SigLIP-init stability result is a good prior for "train it yourself" at any
scale), agentic RL environments, MoonEP expert parallelism, microVM sandbox
fleets, and fleet-level scheduling all live above our hardware floor and off
the current thesis. One conceptual import: their KDA-state prefix caching
(fixed-size state checkpointed at hash boundaries, jointly evicted with KV)
is the shape any hybrid-model serving cache takes, ours included, if serve.sh
ever grows one. MLA itself remains cautionary here: it lost to GQA at our
scale in 03 (+0.09 nats), and K3 uses it gated and NoPE'd inside a hybrid,
so "MLA at small scale" stays an open scale-check, not a settled loss.

## What a 13-milestone does with this

Sequenced smallest-risk-first, every run behind the usual design review,
sanity checks, and throughput probe with abort gate:

1. **13a, bolt-on ablations at ~50M, n=3 seeds**: NoPE-globals, attention
   output gate, AttnRes-full, SiTU-GLU; one variable per arm, param-matched
   controls, existing refined corpus and eval harness.
2. **13b, KDA-lite hybrid at 4k context**: pure-PyTorch chunkwise gated
   delta rule with lower-bounded decay replacing sliding-window layers in
   the 05 hybrid; gates on long-range probe and state-vs-KV size, honest
   tokens/sec including the no-fused-kernel penalty.
3. **13c, Muon rematch**: per-head variant, per-arm LR sweep, 20-50M scale.
4. **14, latent MoE**: after its own literature note; ~180M total / ~60M
   active, QB balancing, specialization diagnostics.
5. **Deployment track alongside**: MTP draft + QAT-int4 through the MLX
   bench; these attack the inference numbers the flagship already reports.

The honest framing for any write-up: K3 bundles its 2.5x; we can only ever
measure per-technique deltas at 1/10,000th the compute. Both negative and
positive results are informative, since "does not survive miniaturization"
is half of the frontier-lag map this repo exists to draw.
