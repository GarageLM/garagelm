"""Smoke-test an exported HF model dir: trust_remote_code load, logit
parity vs the research checkpoint, and generation.

  uv run python release/smoke_test.py --hf-dir release/hf/hybrid-gpt-232m \
      --ckpt experiments/09-flagship-2/out/ckpt.pt \
      --experiment-dir experiments/09-flagship-2

For MoE exports (config has n_expert) the parity gate runs with the
wrapper in research-faithful capacity mode (config.dropless=False), then
the default dropless mode is compared against the same reference and the
delta reported: that is the size of the inference-time routing change.
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
    is_moe = getattr(hf_model.config, "n_expert", None) is not None
    with torch.no_grad():
        ref_logits, _ = ref(x)

    if not is_moe:
        with torch.no_grad():
            hf_logits = hf_model(input_ids=x).logits
        diff = (hf_logits - ref_logits).abs().max().item()
        top1 = (hf_logits.argmax(-1) == ref_logits.argmax(-1)).float().mean().item()
        print(f"logit parity: max|diff|={diff:.2e} top1={top1:.3f}")
        # the release is bf16: rounding on ~|15|-magnitude logits gives ~0.1 max
        # diff and can flip argmax on near-ties; structural bugs give NaN/garbage
        assert diff < 0.25 and top1 >= 0.99, "HF wrapper does not match research model"
    else:
        # MoE routing is discrete, so bf16 weight rounding does not give the
        # smooth ~0.1 drift above: a flipped top-2 choice on a near-tie token
        # is a large local change (measured: ~0.07% of token-layer decisions
        # flip, max|diff| ~1.4). The structural gate therefore runs the
        # wrapper CLASS on the research fp32 weights, where parity must be
        # exact; the bf16 artifact and the dropless default are then
        # reported against the same reference with sanity bounds only.
        import copy
        cfg32 = copy.deepcopy(hf_model.config)
        cfg32.dropless = False
        w32 = type(hf_model)(cfg32)
        w32.load_state_dict(ckpt["model"], strict=False)
        w32.eval()
        with torch.no_grad():
            l32 = w32(input_ids=x).logits
        d32 = (l32 - ref_logits).abs().max().item()
        t32 = (l32.argmax(-1) == ref_logits.argmax(-1)).float().mean().item()
        print(f"structural parity (fp32 weights, capacity mode): max|diff|={d32:.2e} top1={t32:.3f}")
        assert d32 < 1e-3 and t32 == 1.0, "MoE wrapper does not match research model"
        del w32
        for dropless in (False, True):
            hf_model.config.dropless = dropless
            with torch.no_grad():
                hl = hf_model(input_ids=x).logits
            d = (hl - ref_logits).abs().max().item()
            t1 = (hl.argmax(-1) == ref_logits.argmax(-1)).float().mean().item()
            print(f"bf16 artifact, {'dropless' if dropless else 'capacity'} mode, vs research: "
                  f"max|diff|={d:.2e} top1={t1:.3f}")
            assert torch.isfinite(hl).all(), "non-finite logits"
        # leave the shipped default in place for generation
        hf_model.config.dropless = True

    ids = tok(args.prompt, return_tensors="pt")
    with torch.no_grad():
        out = hf_model.generate(**ids, max_new_tokens=60, do_sample=True,
                                temperature=0.7, top_k=50)
    print("--- generation:", tok.decode(out[0]))
    print("SMOKE OK")


if __name__ == "__main__":
    main()
