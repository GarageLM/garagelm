"""Build the 05 refined-corpus pool: FineWeb-Edu (edu-filtered web) blended
with Cosmopedia-v2 (synthetic textbooks distilled from a large model).

Downloads plain parquet shards over HTTPS (no `datasets` library, no auth),
streams row-group batches through tiktoken, and writes uint16 token bins in
the same memmap format as 02/04. Val is held out from the tail of EACH
source (stratified), so the val mix matches the train mix.

Pool sizing: ~1.65B tokens ≈ 3x the 07 flagship budget (500M) — tokens are
seen roughly once (sub-epoch), which is also why dropout is 0.0 in 05/07.
"""

import os
import pickle
import urllib.request

import numpy as np
import pyarrow.parquet as pq
import tiktoken

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# (url, local filename, val tokens held out of this file's tail)
# Val quotas are proportional to each source's share of the pool (~84/16).
SHARDS = [
    ("https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/000_00000.parquet",
     "fineweb-edu-000.parquet", 0),
    ("https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/001_00000.parquet",
     "fineweb-edu-001.parquet", 1_680_000),
    ("https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/resolve/main/cosmopedia-v2/train-00000-of-00104.parquet",
     "cosmopedia-v2-000.parquet", 320_000),
]

BATCH_DOCS = 2048


def download(url, path):
    if os.path.exists(path):
        print(f"already downloaded: {os.path.basename(path)}")
        return
    print(f"downloading {os.path.basename(path)} from {url}")
    last_reported = [0]

    def hook(blocks, block_size, total_size):
        done = blocks * block_size
        if done - last_reported[0] >= 200 * 1024 * 1024:
            last_reported[0] = done
            print(f"  {done / 1e9:.1f}GB / {total_size / 1e9:.1f}GB", flush=True)

    tmp = path + ".part"
    urllib.request.urlretrieve(url, tmp, reporthook=hook)
    os.rename(tmp, path)
    print(f"  done: {os.path.getsize(path) / 1e9:.2f}GB")


def tokenize_shard(enc, parquet_path, bin_path):
    """Stream one parquet's `text` column into a uint16 token bin.

    Every document gets a trailing <|endoftext|> token, matching the
    separator convention of the 02/04 TinyStories bins.
    """
    if os.path.exists(bin_path):
        n = os.path.getsize(bin_path) // 2
        print(f"already tokenized: {os.path.basename(bin_path)} ({n:,} tokens)")
        return n
    eot = enc.eot_token
    total = 0
    pf = pq.ParquetFile(parquet_path)
    tmp = bin_path + ".part"
    with open(tmp, "wb") as f:
        for batch in pf.iter_batches(batch_size=BATCH_DOCS, columns=["text"]):
            texts = batch.column("text").to_pylist()
            ids_list = enc.encode_ordinary_batch(texts)
            arrs = [np.array(ids + [eot], dtype=np.uint16) for ids in ids_list]
            out = np.concatenate(arrs)
            f.write(out.tobytes())
            total += len(out)
            if total % 50_000_000 < len(out):
                print(f"  {os.path.basename(bin_path)}: {total:,} tokens...", flush=True)
    os.rename(tmp, bin_path)
    print(f"tokenized {os.path.basename(bin_path)}: {total:,} tokens")
    return total


def concat_bins(shard_bins, train_path, val_path):
    """train.bin = each shard minus its val tail; val.bin = the tails.

    Pure disk-to-disk chunked copies -- nothing large is held in memory.
    """
    copy_chunk = 64 * 1024 * 1024  # tokens per copy chunk
    with open(train_path, "wb") as ftrain, open(val_path, "wb") as fval:
        for bin_path, val_tokens in shard_bins:
            data = np.memmap(bin_path, dtype=np.uint16, mode="r")
            split = len(data) - val_tokens
            for start in range(0, split, copy_chunk):
                ftrain.write(data[start:min(start + copy_chunk, split)].tobytes())
            if val_tokens:
                fval.write(data[split:].tobytes())
            del data


def prepare():
    os.makedirs(DATA_DIR, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")

    shard_bins = []
    for url, fname, val_tokens in SHARDS:
        parquet_path = os.path.join(DATA_DIR, fname)
        bin_path = parquet_path.replace(".parquet", ".bin")
        download(url, parquet_path)
        tokenize_shard(enc, parquet_path, bin_path)
        shard_bins.append((bin_path, val_tokens))

    train_path = os.path.join(DATA_DIR, "train.bin")
    val_path = os.path.join(DATA_DIR, "val.bin")
    concat_bins(shard_bins, train_path, val_path)

    with open(os.path.join(DATA_DIR, "meta.pkl"), "wb") as f:
        pickle.dump({"vocab_size": enc.n_vocab, "encoding": "gpt2"}, f)

    n_train = os.path.getsize(train_path) // 2
    n_val = os.path.getsize(val_path) // 2
    print(f"vocab_size={enc.n_vocab} train_tokens={n_train:,} val_tokens={n_val:,}")

    # Spot-check: decode a slice from each end of train.bin and from val.bin.
    data = np.memmap(train_path, dtype=np.uint16, mode="r")
    val = np.memmap(val_path, dtype=np.uint16, mode="r")
    for label, arr, off in (("train[fineweb]", data, 1000),
                            ("train[cosmo]", data, len(data) - 5000),
                            ("val[head]", val, 0)):
        snippet = enc.decode(arr[off:off + 60].astype(np.int64).tolist())
        print(f"--- {label}: {snippet[:200]!r}")


if __name__ == "__main__":
    prepare()
