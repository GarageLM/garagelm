---
license: apache-2.0
language: [en]
datasets: [HuggingFaceFW/fineweb-edu, HuggingFaceTB/smollm-corpus, HuggingFaceTB/smoltalk]
pipeline_tag: text-generation
library_name: transformers
tags: [hybrid-attention, sliding-window, small-model, research, chat]
---

# hybrid-gpt-232m-chat

The chat-tuned variant of
[hybrid-gpt-232m](https://huggingface.co/garagelm/hybrid-gpt-232m) — see
that card for the full architecture, pretraining, and evaluation details
(both cards are self-contained; the research code repo may be private).
SFT: one epoch over 60M tokens of SmolTalk conversations
(everyday-conversations + smol-magpie-ultra), **assistant-only loss
masking**, LR 2e-5 cosine, initialized from the 1B-token base checkpoint.
Trained end-to-end — pretraining included — on one Apple M4 Pro.
Post-SFT benchmark deltas vs the base are all within noise (no capability
regression).

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

## Limitations

**A 232M-parameter research artifact, not a product assistant.** Fluent,
follows the conversational format, and stays on topic — but facts are
unreliable, reasoning is shallow, context is 1024 tokens, English only,
and it has had **no safety/alignment tuning beyond SmolTalk SFT**. Do not
deploy for anything where correctness matters. Base-model benchmark
scores (HellaSwag/PIQA/ARC-E) are within noise of the base model after
SFT — see the GitHub repo for the before/after table.
