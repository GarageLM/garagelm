"""Interactive multi-turn chat with the SFT model, using the exact training
template (User:/Assistant:, <|endoftext|> terminates assistant turns).

  uv run python experiments/10-sft/chat.py            # MPS (contends with a training run)
  uv run python experiments/10-sft/chat.py --device cpu

History is kept in-template and left-trimmed to fit the 1024-token context.
Generation stops at <|endoftext|> (early-stop loop, not fixed-length).
"""

import argparse
import os

import tiktoken
import torch
import torch.nn.functional as F

from config import GPTConfig
from model import GPT

OUT_DIR = os.path.join(os.path.dirname(__file__), "out")


@torch.no_grad()
def generate_reply(model, idx, max_new, eot, temperature, top_k):
    for _ in range(max_new):
        idx_cond = idx[:, -model.config.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        if top_k:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")
        nxt = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
        if nxt.item() == eot:
            break
        idx = torch.cat((idx, nxt), dim=1)
    return idx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--max-new", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=50)
    args = p.parse_args()

    enc = tiktoken.get_encoding("gpt2")
    ckpt = torch.load(os.path.join(OUT_DIR, "ckpt.pt"), map_location=args.device)
    model = GPT(GPTConfig(**ckpt["model_cfg"])).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"GarageLM chat -- 232M SFT model on {args.device}. "
          f"Ctrl-C or empty line to quit.\n")

    history = []  # token ids of the conversation so far, in-template
    budget = model.config.block_size - args.max_new - 8
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            break
        history += enc.encode_ordinary(f"User: {user}\nAssistant:")
        history = history[-budget:]
        idx = torch.tensor([history], dtype=torch.long, device=args.device)
        out = generate_reply(model, idx, args.max_new, enc.eot_token,
                             args.temperature, args.top_k)
        reply_ids = out[0, idx.shape[1]:].tolist()
        reply = enc.decode(reply_ids).strip()
        print(f"lm > {reply}\n")
        history = out[0].tolist() + enc.encode_ordinary("\n")


if __name__ == "__main__":
    main()
