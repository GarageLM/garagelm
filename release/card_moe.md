---
license: apache-2.0
language: [en]
datasets: [HuggingFaceFW/fineweb-edu, HuggingFaceTB/smollm-corpus]
pipeline_tag: text-generation
library_name: transformers
tags: [mixture-of-experts, moe, hybrid-attention, sliding-window, small-model, research]
model-index:
- name: hybrid-gpt-moe-284m-a114m
  results:
  - task:
      type: text-generation
      name: Commonsense inference
    dataset:
      type: hellaswag
      name: HellaSwag (0-shot, n=300, local lm-eval)
    metrics:
    - type: acc
      value: 34.3
    - type: acc_norm
      value: 33.0
  - task:
      type: text-generation
      name: Physical commonsense
    dataset:
      type: piqa
      name: PIQA (0-shot, n=300, local lm-eval)
    metrics:
    - type: acc
      value: 56.3
    - type: acc_norm
      value: 52.0
  - task:
      type: text-generation
      name: Science QA
    dataset:
      type: ai2_arc
      name: ARC-Easy (0-shot, n=300, local lm-eval)
    metrics:
    - type: acc
      value: 39.3
    - type: acc_norm
      value: 34.3
  - task:
      type: text-generation
      name: Coreference reasoning
    dataset:
      type: winogrande
      name: WinoGrande (0-shot, n=300, local lm-eval)
    metrics:
    - type: acc
      value: 49.0
  - task:
      type: text-generation
      name: Multitask knowledge
    dataset:
      type: cais/mmlu
      name: MMLU (5-shot, 5 questions per subject, local lm-eval)
    metrics:
    - type: acc
      value: 26.0
---

# hybrid-gpt-moe-284m-a114m

A 284M-parameter (114M active) research model: the lab's hybrid
local+global attention stack with every FFN replaced by a **top-2
mixture of 8 SwiGLU experts**. Trained on one Apple M4 Pro Mac mini on
100M tokens of refined data, as one arm of a controlled three-way
comparison (milestone 13 of github.com/GarageLM/garagelm). It is released
for reproducibility of that comparison, not as a model to use.

**The finding it documents.** At a hard memory budget, is a sparse model
worth its bytes? Frontier MoE papers match *active* parameters (equal
FLOPs, memory free). At this lab's floor memory is the budget, so both
controls were run: the 114M dense model it shares active parameters with,
and a 284M dense model it shares memory with. Result: **MoE bought
validation loss and nothing else measurable.** It clears the
pre-registered loss gate against the active-matched control by 0.0004
nats, loses to the memory-matched dense model by 0.038 nats, separates
from neither on downstream tasks, and decodes slowest of the three under
the research implementation.

| Arm | Total params | Active | Full-pass val | Decode tok/s | 4-task avg |
|---|---|---|---|---|---|
| control: hybrid 114M dense | 114,114,048 | 114M | 3.8398 | **54.3** | **44.08** |
| **this model (moe)** | 284,057,088 | 114M | **3.8094** | 8.9 (a) | 42.08 |
| dense-284m (FFN 8192) | 283,983,360 | 284M | **3.7711** | 41.8 | 43.75 |

(a) An implementation number, not an architectural one: 96 GPU-to-host
syncs per forward (8 experts x 12 layers) in the research dispatch loop,
which training amortizes and single-token decode does not. See the
milestone README for the arithmetic.

Per task (lm-evaluation-harness, 0-shot, n=300 per task, identical slices
for every row, acc / acc_norm):

| Task | control 114M | **moe** | dense-284m |
|---|---|---|---|
| HellaSwag | 32.3 / 33.7 | 34.3 / 33.0 | 33.3 / 33.3 |
| PIQA | 58.3 / 56.3 | 56.3 / 52.0 | 56.0 / 53.7 |
| ARC-Easy | 36.3 / 34.7 | 39.3 / 34.3 | 39.7 / 38.3 |
| WinoGrande | 51.7 | 49.0 | 49.7 |
| MMLU (5/subject) | 26.7 | 26.0 | 27.0 |

The whole 4-task spread across the three arms is 2.0 points. At n=300 the
standard error on a proportion is ~2.9 points, so this suite cannot
separate a 114M model from a 284M one trained on the same 100M tokens. The
honest statement is "no downstream difference is resolvable", not "the
control wins".

## Architecture

Decoder-only transformer: 12 layers, d_model 768, 12 heads / 4 KV heads
(GQA), RoPE (theta 10000), RMSNorm, weight tying, context 1024, gpt2 BPE
(vocab 50257). Attention is a sliding window of 64 tokens except layers
3/7/11, which are fully causal: the KV cache at 1024 context is 3.7MB fp16
against 12.6MB for an all-global equivalent (~30%), identical to the 114M
control since only the FFN changed.

FFN: 8 experts per layer, each a SwiGLU with hidden 1024; a linear router
(no bias) over d_model; softmax, top-2, gates renormalized over the two
selected experts. Two experts x 1024 hidden is the same active FFN width
as the control's dense 2048, so active parameters match (114.2M vs
114.1M). Expert parameters: 226.5M of the 284.1M total.

### Routing in this wrapper

Training used a **fixed per-expert capacity** (1.25 x tokens x 2 / 8 per
forward, overflow dropped, lowest gates first) because the MPS allocator
needs static shapes. It never bound: end-of-run drop rate 0.0004, minimum
expert load 0.1147 against a uniform 0.125 (router-health gate passed; the
per-layer EMA buffers ship in the weights as `blocks.N.mlp.ema_load` /
`ema_drop`). At inference the rule is a liability, since capacity scales
with the number of tokens in the forward and a short prompt would drop
most tokens from expert compute. **This wrapper defaults to dropless
routing** (`config.dropless = True`: every token reaches both of its
experts). `config.dropless = False` reproduces the research behaviour
exactly; the release smoke test checks structural parity in that mode
(max |logit diff| 1.3e-05 against the fp32 research model).

Measured on the shipped artifact (bf16 weights, fp32 compute, same 2M-token
held-out split and protocol as the table above):

| Routing | Full-pass val |
|---|---|
| research model (fp32, capacity) | 3.8094 |
| this artifact, capacity mode | 3.8094 |
| **this artifact, dropless (default)** | **3.8086** |

bf16 rounding flips ~0.07% of top-2 routing decisions on natural text and
costs nothing at four decimals; the dropless default recovers 0.0008 nats
by serving the tokens the capacity rule dropped.

## Training

- **Data**: 100M tokens (6,100 steps x 16,384) from a 1.78B-token pool,
  84.5% FineWeb-Edu (`sample-10BT`) + 15.5% Cosmopedia-v2. Sub-epoch.
- **Recipe**: AdamW, cosine 3e-4 to 3e-5 (warmup 200), weight decay 0.1,
  grad clip 1.0, dropout 0, seed 1337, fp32 on PyTorch/MPS. Switch-style
  load-balance auxiliary loss, coefficient 0.01, train-time only (never in
  any reported validation number). 5.70 s/step, 9.7h wall-clock on one M4
  Pro (48GB).
- **Fair comparison**: tokenizer, data, token budget, optimizer, schedule,
  seed, and attention identical to both control arms. Only the FFN differs.
- **Validation**: full-pass 3.8094 nats (PPL 45.1) on the 2M-token
  held-out split, non-overlapping 1024-token windows, fp32.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("garagelm/hybrid-gpt-moe-284m-a114m")
model = AutoModelForCausalLM.from_pretrained("garagelm/hybrid-gpt-moe-284m-a114m", trust_remote_code=True)
out = model.generate(**tok("The water cycle begins when", return_tensors="pt"), max_new_tokens=100)
print(tok.decode(out[0]))
```

No KV cache in this wrapper (research release; generation re-runs the
prefix each step), and the expert dispatch is a plain per-expert gather
loop written for clarity. There is no MLX build: the milestone deferred the
MoE inference port because the sparse arm lost the memory-matched
comparison.

## Limitations

- **A research artifact, not an assistant.** 100M training tokens, 1024
  context, English only, unreliable facts, no alignment or safety tuning.
  Its purpose is to let the milestone-13 comparison be re-run.
- Trained on FineWeb-Edu and Cosmopedia-v2 (both ODC-BY); inherits their
  biases and educational-web distribution.
- MMLU-class benchmarks are at chance, as for all models this size.
- For a model to actually use, see
  [`garagelm/hybrid-gpt-232m`](https://huggingface.co/garagelm/hybrid-gpt-232m)
  (1B tokens) and its chat variant.
