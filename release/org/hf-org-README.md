---
title: README
emoji: 🔧
colorFrom: gray
colorTo: blue
sdk: static
pinned: false
---

<!-- Deploys to the garagelm/README *Space*, whose README.md renders as
the org profile at huggingface.co/garagelm. IMPORTANT: keep this Space
README-only. Adding an app file (e.g. index.html for the static sdk)
makes the org page embed the running app INSTEAD of this markdown -- the
"configuration error" banner on the Space's own page is cosmetic and must
be left alone. -->


# GarageLM

**Chasing the most intelligent model on the cheapest hardware.**

Every model here was pretrained, fine-tuned, evaluated, and benchmarked on
a single Apple M4 Pro Mac mini — no cluster, no cloud, ~$3 of electricity
per flagship run. The research axis: how much measured capability can one
consumer machine produce?

## Models

| Model | What it is |
|---|---|
| [hybrid-gpt-232m](https://huggingface.co/garagelm/hybrid-gpt-232m) | 232M base · hybrid local+global attention · 1B refined tokens |
| [hybrid-gpt-232m-chat](https://huggingface.co/garagelm/hybrid-gpt-232m-chat) | SmolTalk SFT of the base — zero benchmark regression |

**Headline result**: at matched, locally re-run evaluation the base model
is at-or-above Pythia-160M (trained on 300× more tokens) across
HellaSwag / PIQA / ARC-Easy / WinoGrande within ±2.8-pt slice
resolution, and ahead of GPT-2 on ARC-Easy — the one gap beyond 2×SE —
with ~5× fewer training FLOPs (6ND, one-epoch-WebText assumed). The
hybrid attention runs at ~30% of full attention's KV cache and decodes
on-device at ~310 tok/s fp16 (530+ at 4-bit, MLX).

## Quick start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("garagelm/hybrid-gpt-232m-chat")
model = AutoModelForCausalLM.from_pretrained(
    "garagelm/hybrid-gpt-232m-chat", trust_remote_code=True)
prompt = "User: Why is the sky blue?\nAssistant:"
out = model.generate(**tok(prompt, return_tensors="pt"),
                     max_new_tokens=200, eos_token_id=50256)
print(tok.decode(out[0]))
```

These are honest research artifacts, not products: fluent, format-reliable,
and factually unreliable in the way every ~200M model is — limitations are
stated prominently on each card.

*Small hardware. Real research.*
