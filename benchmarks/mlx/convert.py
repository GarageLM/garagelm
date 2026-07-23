"""Convert a PyTorch checkpoint (any 02/04/05/07 variant) to MLX safetensors
and optionally gate logit parity between the two implementations.

The MLX module tree mirrors PyTorch's names 1:1, so conversion is just:
drop lm_head.weight (weight-tied duplicate of tok_emb.weight; MLX side uses
Embedding.as_linear), dump the rest. RoPE/mask buffers are non-persistent in
PyTorch and never enter the state_dict.

Parity (--parity) runs on the MLX CPU device so a training run on the GPU
is not disturbed: full-forward logits at T=128 and 4 cached decode steps
must match PyTorch fp32 within tolerance and agree on every top-1 token.
"""

import argparse
import json
import os
import sys

import mlx.core as mx
import numpy as np

MLX_DIR = os.path.dirname(os.path.abspath(__file__))


def result_name(exp_dir):
    parts = os.path.abspath(exp_dir).rstrip("/").split(os.sep)
    if "experiments" in parts:
        parts = parts[parts.index("experiments") + 1:]
    else:
        parts = parts[-1:]
    return "-".join(parts)


def convert(exp_dir, out_dir):
    import torch

    ckpt = torch.load(os.path.join(exp_dir, "out", "ckpt.pt"), map_location="cpu")
    cfg = ckpt["model_cfg"]
    weights = {}
    for k, v in ckpt["model"].items():
        if k == "lm_head.weight":
            continue  # tied to tok_emb.weight
        weights[k] = mx.array(v.numpy())

    os.makedirs(out_dir, exist_ok=True)
    mx.save_safetensors(os.path.join(out_dir, "weights.safetensors"), weights)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"converted {len(weights)} tensors -> {out_dir} (cfg: {cfg})")
    return cfg


def parity(exp_dir, out_dir, T=128, decode_steps=4, tol=5e-3):
    import torch

    mx.set_default_device(mx.cpu)  # do not disturb a GPU training run

    sys.path.insert(0, exp_dir)
    for m in ("config", "model"):
        sys.modules.pop(m, None)
    from config import GPTConfig
    from model import GPT as TorchGPT
    sys.path.pop(0)

    sys.path.insert(0, MLX_DIR)
    sys.modules.pop("model", None)
    import model as mlx_model
    sys.path.pop(0)
    sys.modules.pop("model", None)

    ckpt = torch.load(os.path.join(exp_dir, "out", "ckpt.pt"), map_location="cpu")
    tcfg = GPTConfig(**ckpt["model_cfg"])
    tmodel = TorchGPT(tcfg)
    tmodel.load_state_dict(ckpt["model"])
    tmodel.eval()

    with open(os.path.join(out_dir, "config.json")) as f:
        cfg = json.load(f)
    mmodel = mlx_model.GPT(cfg)
    mmodel.load_weights(os.path.join(out_dir, "weights.safetensors"))

    rng = np.random.default_rng(1234)
    total = T + decode_steps
    seq = rng.integers(0, cfg["vocab_size"], size=(total,), dtype=np.int64)

    # prefill parity: all T positions
    with torch.no_grad():
        t_logits, _ = tmodel(torch.tensor(seq[None, :T]))
    caches = mmodel.make_cache()
    m_logits = mmodel(mx.array(seq[None, :T].astype(np.int32)), caches)
    mx.eval(m_logits)
    diff = float(np.abs(t_logits.numpy() - np.array(m_logits)).max())
    top1 = (t_logits.numpy().argmax(-1) == np.array(m_logits).argmax(-1)).mean()
    print(f"prefill T={T}: max|diff|={diff:.2e} top1 agreement={top1:.3f}")
    assert diff < tol and top1 == 1.0, "prefill parity FAILED"

    # decode parity: cached single-token steps vs full PyTorch re-forward
    for s in range(decode_steps):
        pos = T + s
        with torch.no_grad():
            t_last, _ = tmodel(torch.tensor(seq[None, :pos + 1]))
        m_last = mmodel(mx.array(seq[None, pos:pos + 1].astype(np.int32)), caches)
        mx.eval(m_last)
        d = float(np.abs(t_last[0, -1].numpy() - np.array(m_last[0, -1])).max())
        agree = int(t_last[0, -1].argmax()) == int(np.array(m_last[0, -1]).argmax())
        print(f"decode pos={pos}: max|diff|={d:.2e} top1 match={agree}")
        assert d < tol and agree, "decode parity FAILED"
    print("PARITY OK")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--out", default=None,
                        help="output dir; default benchmarks/mlx/converted/<name>")
    parser.add_argument("--parity", action="store_true")
    args = parser.parse_args()

    exp_dir = os.path.abspath(args.experiment_dir)
    out_dir = args.out or os.path.join(MLX_DIR, "converted", result_name(exp_dir))
    convert(exp_dir, out_dir)
    if args.parity:
        parity(exp_dir, out_dir)


if __name__ == "__main__":
    main()
