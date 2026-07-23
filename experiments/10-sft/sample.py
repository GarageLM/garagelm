"""Chat with the SFT model using the exact training template
(User:/Assistant:, generation stops at <|endoftext|>)."""

import argparse
import os

import tiktoken
import torch

from config import GPTConfig
from model import GPT

OUT_DIR = os.path.join(os.path.dirname(__file__), "out")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="What is the difference between a comet and an asteroid?")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_k", type=int, default=50)
    args = p.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    enc = tiktoken.get_encoding("gpt2")
    ckpt = torch.load(os.path.join(OUT_DIR, "ckpt.pt"), map_location=device)
    model = GPT(GPTConfig(**ckpt["model_cfg"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    text = f"User: {args.prompt}\nAssistant:"
    idx = torch.tensor([enc.encode_ordinary(text)], dtype=torch.long, device=device)
    out = model.generate(idx, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
    reply_ids = out[0, idx.shape[1]:].tolist()
    if enc.eot_token in reply_ids:
        reply_ids = reply_ids[:reply_ids.index(enc.eot_token)]
    print(f"{text}{enc.decode(reply_ids)}")


if __name__ == "__main__":
    main()
