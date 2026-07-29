---
license: apache-2.0
language: [en]
datasets: [HuggingFaceFW/fineweb-edu, HuggingFaceTB/smollm-corpus, HuggingFaceTB/smoltalk, HuggingFaceH4/ultrafeedback_binarized]
pipeline_tag: text-generation
library_name: transformers
base_model: garagelm/hybrid-gpt-232m
base_model_relation: finetune
tags: [hybrid-attention, sliding-window, small-model, research, chat, dpo]
model-index:
- name: hybrid-gpt-232m-chat
  results:
  - task:
      type: text-generation
      name: Instruction following
    dataset:
      type: google/IFEval
      name: IFEval (0-shot, full 541-prompt set, local lm-eval)
    metrics:
    - type: acc
      name: prompt-level strict accuracy
      value: 13.9
    - type: acc
      name: instruction-level strict accuracy
      value: 25.1
  - task:
      type: text-generation
      name: Yes/no reading comprehension
    dataset:
      type: boolq
      name: BoolQ (0-shot, n=300, local lm-eval)
    metrics:
    - type: acc
      value: 64.3
  - task:
      type: text-generation
      name: Truthfulness
    dataset:
      type: truthful_qa
      name: TruthfulQA-mc2 (0-shot, n=300, local lm-eval)
    metrics:
    - type: acc
      value: 41.6
---

# hybrid-gpt-232m-chat

**v2 — the full post-training pipeline (SFT + DPO).** The chat-tuned
variant of
[hybrid-gpt-232m](https://huggingface.co/garagelm/hybrid-gpt-232m) — see
that card for the architecture, pretraining, and base evaluations (both
cards are self-contained; the research code repo may be private).
Trained end-to-end — pretraining included — on one Apple M4 Pro.

**Recipe** (initialized from the 1B-token base):

1. **SFT**: ~310M formatted tokens of SmolTalk (`all` mix), ~1 epoch,
   **assistant-only loss masking**, LR 2e-5 cosine.
2. **DPO**: UltraFeedback-binarized `train_prefs`, β=0.1, one epoch over
   53,952 pairs, LR 1e-6, frozen SFT reference. Held-out preference
   accuracy **65.2%** (1k pairs).

The previous release (60M-token SFT, no DPO) remains available at
revision [`v1`](https://huggingface.co/garagelm/hybrid-gpt-232m-chat/tree/v1).

## Measured behavior

All numbers from the same local lm-evaluation-harness setup used for
every claim this lab publishes — reference models re-run locally on
identical settings, never quoted from their cards.

**IFEval** (rule-verifiable instruction following; full 541-prompt set,
chat template; ours served via MLX, reference via the HF backend,
identical harness):

| Model | prompt-strict | inst-strict |
|---|---|---|
| SmolLM2-135M-Instruct (2T pretrain + full post-train stack) | 21.4% | 35.6% |
| **hybrid-gpt-232m-chat (this model)** | **13.9%** | **25.1%** |

Real instruction-following exists at this scale — roughly a quarter of
atomic constraints satisfied, ~65% of the reference's prompt-strict
score with ~2000x fewer pretraining tokens.

**Knowledge/comprehension** (0-shot, n=300, acc): BoolQ **.643** vs the
base model's .623; TruthfulQA-mc2 **.416** vs .393 — both slightly favor
this chat model over its base.

**Alignment tax, disclosed**: the SFT stage costs measurable science-QA
capability — full-set ARC-Easy (2,376 items) drops from .520 (base) to
.498 after 310M SFT tokens; the DPO stage adds no further regression
(HellaSwag/PIQA/ARC-Easy/WinoGrande all within one point of the SFT
checkpoint). If you want maximum raw-benchmark capability, use the base
model; this variant trades ~2 ARC-Easy points for chat quality and
preference alignment.

## Chat template

Plain text (no special tokens beyond gpt2's `<|endoftext|>`):

```
User: {message}
Assistant: {reply}<|endoftext|>
```

Multi-turn conversations repeat the pair. Generation should stop at
`<|endoftext|>` (id 50256), which is the model's assistant-turn terminator.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("garagelm/hybrid-gpt-232m-chat")
model = AutoModelForCausalLM.from_pretrained("garagelm/hybrid-gpt-232m-chat", trust_remote_code=True)
prompt = "User: Why is the sky blue?\nAssistant:"
out = model.generate(**tok(prompt, return_tensors="pt"), max_new_tokens=200,
                     eos_token_id=50256, do_sample=True, temperature=0.7, top_k=50)
print(tok.decode(out[0]))
```

### Apple Silicon (MLX)

MLX conversions with real per-layer KV caches (bounded on the 12
sliding-window layers), decoding ~310 tok/s at float16 and ~530 tok/s at
4-bit on an M4 Pro:

- [`hybrid-gpt-232m-chat-mlx`](https://huggingface.co/garagelm/hybrid-gpt-232m-chat-mlx) — float16, 464MB
- [`hybrid-gpt-232m-chat-mlx-4bit`](https://huggingface.co/garagelm/hybrid-gpt-232m-chat-mlx-4bit) — 4-bit, 130MB, at +0.018 nats validation loss

Both are logit-parity-gated against this PyTorch implementation.

## Limitations

**A 232M-parameter research artifact, not a product assistant.** Fluent,
follows the conversational format and simple instructions, and stays on
topic — but facts are unreliable, reasoning is shallow, context is 1024
tokens, English only. Preference tuning is UltraFeedback DPO only —
**no dedicated safety/alignment tuning**. Do not deploy for anything
where correctness matters. The measured capability cost of the chat
tuning is stated above, not hidden.
