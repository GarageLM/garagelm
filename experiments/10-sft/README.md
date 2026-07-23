# 10-sft

SFT of the `09` flagship on SmolTalk conversations, plus the Hugging Face
release tooling (`release/`).

## Training

- **Data**: 60.0M tokens / 40k conversations (everyday-conversations +
  smol-magpie-ultra shard 0), formatted with a plain `User:`/`Assistant:`
  template (gpt2 BPE unchanged, `<|endoftext|>` terminates assistant
  turns). **Assistant-only loss masking** (uint8 mask bin parallel to the
  token bin; 83.7% of tokens carry loss); 500 conversations held out.
- **Run**: init from `09`'s checkpoint, masked CE, LR 2e-5 cosine, one
  epoch (3,660 steps), 7.6h. Masked val loss 2.54 → **1.70**.

## Regression check (n=300, same harness as everything else)

| | HellaSwag | PIQA | ARC-E | WinoGrande |
|---|---|---|---|---|
| base (09) | .357/.397 | .633/.610 | .477/.450 | .517 |
| after SFT | .360/.410 | .623/.613 | .460/.407 | .527 |

No capability collapse — all deltas within noise. Qualitatively the chat
model follows the format reliably and stays on topic; factual reliability
remains 232M-grade (disclosed prominently in the model card).

## Release (`release/`)

- `configuration_hybrid_gpt.py` / `modeling_hybrid_gpt.py` — self-contained
  `trust_remote_code` transformers wrapper (weight names match the research
  repo 1:1; no KV cache — documented). Two hard-won transformers-5 gotchas
  encoded in comments: RoPE/mask buffers must be `persistent=True` (meta-
  device init never materializes computed buffers), and `attribute_map`
  must alias the standard config names for generation's cache setup.
- `export.py` → bf16 safetensors + config + tokenizer + card;
  `smoke_test.py` → logit parity vs the research checkpoint (bf16
  tolerance) + generation; `upload.py` → `huggingface_hub` push.
- Both exports pass smoke: base max|Δlogit| 0.087 / top1 1.000; chat 0.073
  / top1 0.992 (bf16 rounding).

```
uv run python experiments/10-sft/data.py
uv run python experiments/10-sft/train.py --resume          # ~7.6h
uv run python release/export.py --ckpt experiments/09-flagship-2/out/ckpt.pt --out release/hf/hybrid-gpt-232m --card base
uv run python release/smoke_test.py --hf-dir release/hf/hybrid-gpt-232m --ckpt ... --experiment-dir ...
uv run python release/upload.py --hf-dir release/hf/hybrid-gpt-232m --repo <user>/hybrid-gpt-232m
```
