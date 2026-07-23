import argparse
import os
import pickle

import torch

from config import GPTConfig
from model import GPT

ROOT = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "out")


def get_device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="\n")
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    args = parser.parse_args()

    device = get_device()
    with open(os.path.join(DATA_DIR, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    stoi, itos = meta["stoi"], meta["itos"]

    ckpt = torch.load(os.path.join(OUT_DIR, "ckpt.pt"), map_location=device)
    model_cfg = GPTConfig(**ckpt["model_cfg"])
    model = GPT(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    def encode(s):
        return [stoi[c] for c in s]

    def decode(ids):
        return "".join(itos[i] for i in ids)

    idx = torch.tensor([encode(args.prompt)], dtype=torch.long, device=device)
    out = model.generate(idx, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
    print(decode(out[0].tolist()))


if __name__ == "__main__":
    main()
