import os
import pickle
import urllib.request

import numpy as np

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def download():
    os.makedirs(DATA_DIR, exist_ok=True)
    raw_path = os.path.join(DATA_DIR, "input.txt")
    if not os.path.exists(raw_path):
        print(f"downloading TinyShakespeare from {DATA_URL}")
        urllib.request.urlretrieve(DATA_URL, raw_path)
    with open(raw_path, "r") as f:
        return f.read()


def prepare():
    text = download()
    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    n = len(text)
    train_text = text[: int(n * 0.9)]
    val_text = text[int(n * 0.9):]

    train_ids = np.array([stoi[c] for c in train_text], dtype=np.uint16)
    val_ids = np.array([stoi[c] for c in val_text], dtype=np.uint16)

    train_ids.tofile(os.path.join(DATA_DIR, "train.bin"))
    val_ids.tofile(os.path.join(DATA_DIR, "val.bin"))
    with open(os.path.join(DATA_DIR, "meta.pkl"), "wb") as f:
        pickle.dump({"vocab_size": vocab_size, "stoi": stoi, "itos": itos}, f)

    print(f"vocab_size={vocab_size} train_tokens={len(train_ids)} val_tokens={len(val_ids)}")
    return vocab_size


if __name__ == "__main__":
    prepare()
