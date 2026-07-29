---
license: apache-2.0
language: [en]
datasets: [HuggingFaceFW/fineweb-edu, HuggingFaceTB/smollm-corpus]
pipeline_tag: text-generation
library_name: transformers
tags: [hybrid-attention, sliding-window, small-model, research]
model-index:
- name: hybrid-gpt-232m
  results:
  - task:
      type: text-generation
      name: Commonsense inference
    dataset:
      type: hellaswag
      name: HellaSwag (0-shot, n=300, local lm-eval)
    metrics:
    - type: acc
      value: 35.7
    - type: acc_norm
      value: 39.7
  - task:
      type: text-generation
      name: Physical commonsense
    dataset:
      type: piqa
      name: PIQA (0-shot, n=300, local lm-eval)
    metrics:
    - type: acc
      value: 63.3
    - type: acc_norm
      value: 61.0
  - task:
      type: text-generation
      name: Science QA
    dataset:
      type: ai2_arc
      name: ARC-Easy (0-shot, n=300, local lm-eval)
    metrics:
    - type: acc
      value: 47.7
    - type: acc_norm
      value: 45.0
  - task:
      type: text-generation
      name: Coreference reasoning
    dataset:
      type: winogrande
      name: WinoGrande (0-shot, n=300, local lm-eval)
    metrics:
    - type: acc
      value: 51.7
  - task:
      type: text-generation
      name: Yes/no reading comprehension
    dataset:
      type: boolq
      name: BoolQ (0-shot, n=300, local lm-eval)
    metrics:
    - type: acc
      value: 62.3
  - task:
      type: text-generation
      name: Truthfulness
    dataset:
      type: truthful_qa
      name: TruthfulQA-mc2 (0-shot, n=300, local lm-eval)
    metrics:
    - type: acc
      value: 39.3
---

# hybrid-gpt-232m

A 232M-parameter research base model with **hybrid local+global attention**
(sliding window w=64 on 12 of 16 layers, full causal attention every 4th
layer), trained **entirely on one Apple M4 Pro Mac mini** on 1.0B tokens of
refined data. This card is self-contained — the training code lives in a
research repo (github.com/GarageLM/garagelm) that may be private;
everything needed to understand and evaluate the model is below.

**Why it's interesting:**
- The hybrid attention pattern **beat full attention at equal parameters**
  in controlled experiments (3/3 random seeds, mean gap −0.038 nats), at
  ~30% of full attention's KV cache (measured: 4.97MB at 1024 context).
- At matched, locally re-run evaluation it is **ahead-or-tied vs
  Pythia-160M (300B training tokens) on all four tasks below with 300x
  fewer tokens** (within n=300 eval resolution), and ahead of gpt2 on
  ARC-Easy — the one gap beyond 2× the slice standard error.

| Model | Train tokens | HellaSwag | PIQA | ARC-E | WinoGrande |
|---|---|---|---|---|---|
| **hybrid-gpt-232m** | 1B | .357/.397 | .633/.610 | .477/.450 | .517 |
| gpt2 (124M) | ~10B | .353/.427 | .610/.617 | .420/.380 | .530 |
| Pythia-160M | 300B | .350/.390 | .627/.627 | .460/.397 | .510 |
| SmolLM2-135M | 2T | .403/.553 | .653/.680 | .583/.437 | .490 |

(0-shot acc/acc_norm via lm-evaluation-harness, n=300/task, **identical
harness and example slices for every row** — the reference models were
re-run locally rather than quoting published full-set numbers, which use
different settings and are not comparable.)

## Architecture

Decoder-only transformer: 16 layers, d_model 1024, 16 heads / 4 KV heads
(GQA), SwiGLU FFN (hidden 2816), RoPE (θ=10000), RMSNorm, weight tying,
context 1024, gpt2 BPE (vocab 50257). Attention is a sliding window of 64
tokens except layers 3/7/11/15, which are fully causal — the KV cache at
full context is ~30% of an all-global equivalent, decoding at ~310 tok/s fp16 / ~530 tok/s 4-bit in MLX on M4 Pro
(TTFT ≤75ms).

## Training

- **Data**: 1.0B tokens sampled from a 3.55B-token pool — 84.5%
  FineWeb-Edu (`sample-10BT` shards) + 15.5% Cosmopedia-v2 synthetic
  textbooks. Sub-epoch (each token seen <1x on average).
- **Recipe**: AdamW, cosine LR 3e-4→3e-5 (warmup 200), weight decay 0.1,
  grad clip 1.0, 61,000 steps × 16,384 tokens/step, dropout 0.0, seed
  1337. ~127h wall-clock on a single M4 Pro (48GB, PyTorch/MPS), fp32.
- **Validation**: final val loss 3.070 (PPL 21.5) on a 2M-token held-out
  split of the training distribution.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("garagelm/hybrid-gpt-232m")
model = AutoModelForCausalLM.from_pretrained("garagelm/hybrid-gpt-232m", trust_remote_code=True)
out = model.generate(**tok("The water cycle begins when", return_tensors="pt"), max_new_tokens=100)
print(tok.decode(out[0]))
```

No KV cache in this wrapper (research release; generation re-runs the
prefix each step).

### Apple Silicon (MLX)

The MLX conversions are the runtime the throughput numbers above were measured
under, and unlike this wrapper they carry real per-layer KV caches (bounded on
the 12 sliding-window layers):

- [`hybrid-gpt-232m-mlx`](https://huggingface.co/garagelm/hybrid-gpt-232m-mlx) — float16, 464MB
- [`hybrid-gpt-232m-mlx-4bit`](https://huggingface.co/garagelm/hybrid-gpt-232m-mlx-4bit) — 4-bit, 130MB, at +0.018 nats validation loss

Both are logit-parity-gated against this PyTorch implementation.

## Limitations

- **A 232M-parameter research artifact, not an assistant.** Fluent text,
  unreliable facts, 1024-token context, English only, no alignment or
  safety tuning of any kind.
- Trained on FineWeb-Edu and Cosmopedia-v2 (both ODC-BY); inherits their
  biases and educational-web distribution.
- MMLU-class benchmarks are at chance, as for all models this size.
- The chat-tuned variant is `garagelm/hybrid-gpt-232m-chat` (v2: 310M-token
  SmolTalk SFT + UltraFeedback DPO; IFEval 13.9% prompt-strict, at a
  disclosed ~2-point full-set ARC-Easy cost vs this base — details on its
  card).
