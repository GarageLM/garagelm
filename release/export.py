"""Export a repo checkpoint to a Hugging Face model directory.

  uv run python release/export.py --ckpt experiments/09-flagship-2/out/ckpt.pt \
      --out release/hf/hybrid-gpt-232m --card base
  uv run python release/export.py --ckpt experiments/10-sft/out/ckpt.pt \
      --out release/hf/hybrid-gpt-232m-chat --card chat
  uv run python release/export.py --ckpt experiments/13-moe/moe/out/ckpt.pt \
      --out release/hf/hybrid-gpt-moe-284m-a114m --card moe --arch moe

Produces: model.safetensors (bf16), config.json (with auto_map for
trust_remote_code), the modeling/configuration .py files, gpt2 tokenizer
files, generation_config.json, and README.md (model card). Then smoke-test
with release/smoke_test.py before uploading with release/upload.py.
"""

import argparse
import importlib
import json
import os
import shutil
import sys

import torch

RELEASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(RELEASE_DIR))

CARDS = {"base": "card_base.md", "chat": "card_chat.md", "moe": "card_moe.md"}

# arch -> (config module, config class, modeling module, model class)
ARCHS = {
    "hybrid": ("configuration_hybrid_gpt", "HybridGPTConfig",
               "modeling_hybrid_gpt", "HybridGPTForCausalLM"),
    "moe": ("configuration_moe_gpt", "MoEGPTConfig",
            "modeling_moe_gpt", "MoEGPTForCausalLM"),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--card", choices=sorted(CARDS), required=True)
    p.add_argument("--arch", choices=sorted(ARCHS), default="hybrid")
    args = p.parse_args()

    cfg_mod, cfg_cls, mdl_mod, mdl_cls = ARCHS[args.arch]
    ConfigCls = getattr(importlib.import_module(f"release.{cfg_mod}"), cfg_cls)
    ModelCls = getattr(importlib.import_module(f"release.{mdl_mod}"), mdl_cls)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ConfigCls(**ckpt["model_cfg"])
    cfg.auto_map = {
        "AutoConfig": f"{cfg_mod}.{cfg_cls}",
        "AutoModelForCausalLM": f"{mdl_mod}.{mdl_cls}",
    }
    cfg.architectures = [mdl_cls]
    cfg.dtype = "bfloat16"

    model = ModelCls(cfg)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    # lm_head.weight is tied to tok_emb.weight; the rope/mask buffers are
    # persistent in the HF wrapper (absent from research checkpoints) and
    # keep their correct __init__-computed values. Anything else is an error.
    allowed_missing = {"lm_head.weight", "rope_cos", "rope_sin", "window_mask"}
    assert not unexpected and set(missing) <= allowed_missing, (missing, unexpected)
    model = model.to(torch.bfloat16)

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    for f in (f"{cfg_mod}.py", f"{mdl_mod}.py"):
        shutil.copyfile(os.path.join(RELEASE_DIR, f), os.path.join(args.out, f))

    from transformers import GPT2TokenizerFast
    GPT2TokenizerFast.from_pretrained("gpt2").save_pretrained(args.out)

    with open(os.path.join(args.out, "generation_config.json"), "w") as f:
        json.dump({"max_length": cfg.block_size, "eos_token_id": 50256,
                   "bos_token_id": 50256, "do_sample": True,
                   "temperature": 0.7, "top_k": 50}, f, indent=2)

    shutil.copyfile(os.path.join(RELEASE_DIR, CARDS[args.card]),
                    os.path.join(args.out, "README.md"))

    n = sum(p.numel() for p in model.parameters())
    print(f"exported {n:,} params (bf16, arch={args.arch}) -> {args.out}")


if __name__ == "__main__":
    main()
