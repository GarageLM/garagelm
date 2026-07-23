import os
import pickle
import urllib.request

import numpy as np
import tiktoken

DATA_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

TARGET_TOKENS = 110_000_000  # pool for random-window sampling; training budget is 49M/model
VAL_TOKENS = 2_000_000
CHUNK_BYTES = 50 * 1024 * 1024
SEPARATOR = "<|endoftext|>"


def download():
    os.makedirs(DATA_DIR, exist_ok=True)
    raw_path = os.path.join(DATA_DIR, "train_full.txt")
    if not os.path.exists(raw_path):
        print(f"downloading TinyStories train split (2.2GB) from {DATA_URL}")
        urllib.request.urlretrieve(DATA_URL, raw_path)
        print("download complete")
    return raw_path


def prepare():
    raw_path = download()
    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab

    # Tokenize the file in chunks aligned to story boundaries, stopping once
    # TARGET_TOKENS is reached -- the full 530M-token file isn't needed for a
    # 49M-token/model training budget, and holding 2.2GB of text plus its
    # tokens in memory at once is pointless.
    all_ids = []
    total = 0
    leftover = ""
    with open(raw_path, "r") as f:
        while total < TARGET_TOKENS:
            chunk = f.read(CHUNK_BYTES)
            if not chunk:
                break
            text = leftover + chunk
            cut = text.rfind(SEPARATOR)
            if cut == -1:
                leftover = text
                continue
            cut += len(SEPARATOR)
            usable, leftover = text[:cut], text[cut:]
            ids = enc.encode(usable, allowed_special={SEPARATOR})
            all_ids.append(np.array(ids, dtype=np.uint16))
            total += len(ids)
            print(f"  tokenized {total:,} tokens...", flush=True)

    ids = np.concatenate(all_ids)[:TARGET_TOKENS]
    train_ids = ids[:-VAL_TOKENS]
    val_ids = ids[-VAL_TOKENS:]

    train_ids.tofile(os.path.join(DATA_DIR, "train.bin"))
    val_ids.tofile(os.path.join(DATA_DIR, "val.bin"))
    with open(os.path.join(DATA_DIR, "meta.pkl"), "wb") as f:
        pickle.dump({"vocab_size": vocab_size, "encoding": "gpt2"}, f)

    print(f"vocab_size={vocab_size} train_tokens={len(train_ids):,} val_tokens={len(val_ids):,}")
    return vocab_size


if __name__ == "__main__":
    prepare()
