import os
import pickle
import urllib.request

import numpy as np
import tiktoken

DATA_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def download():
    os.makedirs(DATA_DIR, exist_ok=True)
    raw_path = os.path.join(DATA_DIR, "input.txt")
    if not os.path.exists(raw_path):
        print(f"downloading TinyStories (valid split) from {DATA_URL}")
        urllib.request.urlretrieve(DATA_URL, raw_path)
    with open(raw_path, "r") as f:
        return f.read()


def prepare():
    text = download()
    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab

    n = len(text)
    train_text = text[: int(n * 0.95)]
    val_text = text[int(n * 0.95):]

    # TinyStories separates stories with "<|endoftext|>", which is GPT-2's
    # actual EOT token — encode it as that special token, not literal text.
    train_ids = np.array(
        enc.encode(train_text, allowed_special={"<|endoftext|>"}), dtype=np.uint16
    )
    val_ids = np.array(
        enc.encode(val_text, allowed_special={"<|endoftext|>"}), dtype=np.uint16
    )

    train_ids.tofile(os.path.join(DATA_DIR, "train.bin"))
    val_ids.tofile(os.path.join(DATA_DIR, "val.bin"))
    with open(os.path.join(DATA_DIR, "meta.pkl"), "wb") as f:
        pickle.dump({"vocab_size": vocab_size, "encoding": "gpt2"}, f)

    print(f"vocab_size={vocab_size} train_tokens={len(train_ids)} val_tokens={len(val_ids)}")
    return vocab_size


if __name__ == "__main__":
    prepare()
