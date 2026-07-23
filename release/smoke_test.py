"""Smoke-test an exported HF model dir: trust_remote_code load, logit
parity vs the research checkpoint, and generation.

  uv run python release/smoke_test.py --hf-dir release/hf/hybrid-gpt-232m \
      --ckpt experiments/09-flagship-2/out/ckpt.pt \
      --experiment-dir experiments/09-flagship-2
"""

import argparse
import os
import sys

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hf-dir", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--experiment-dir", required=True)
    p.add_argument("--prompt", default="The history of astronomy begins")
    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.hf_dir)
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.hf_dir, trust_remote_code=True, torch_dtype=torch.float32
    )
    hf_model.eval()

    # reference: the research implementation
    exp = os.path.abspath(args.experiment_dir)
    sys.path.insert(0, exp)
    for m in ("config", "model"):
        sys.modules.pop(m, None)
    from config import GPTConfig
    from model import GPT
    ckpt = torch.load(args.ckpt, map_location="cpu")
    ref = GPT(GPTConfig(**ckpt["model_cfg"]))
    ref.load_state_dict(ckpt["model"])
    ref.eval()

    rng = np.random.default_rng(7)
    x = torch.tensor(rng.integers(0, 50257, size=(1, 128)))
    with torch.no_grad():
        hf_logits = hf_model(input_ids=x).logits
        ref_logits, _ = ref(x)
    diff = (hf_logits - ref_logits).abs().max().item()
    top1 = (hf_logits.argmax(-1) == ref_logits.argmax(-1)).float().mean().item()
    print(f"logit parity: max|diff|={diff:.2e} top1={top1:.3f}")
    # the release is bf16: rounding on ~|15|-magnitude logits gives ~0.1 max
    # diff and can flip argmax on near-ties; structural bugs give NaN/garbage
    assert diff < 0.25 and top1 >= 0.99, "HF wrapper does not match research model"

    ids = tok(args.prompt, return_tensors="pt")
    with torch.no_grad():
        out = hf_model.generate(**ids, max_new_tokens=60, do_sample=True,
                                temperature=0.7, top_k=50)
    print("--- generation:", tok.decode(out[0]))
    print("SMOKE OK")


if __name__ == "__main__":
    main()
