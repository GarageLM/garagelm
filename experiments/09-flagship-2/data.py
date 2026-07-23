"""Build the 09 pool: the 05 pool extended with three fresh shards
(FineWeb-Edu 002+003, Cosmopedia-v2 001) to ~3.5B tokens at the same
~84.5/15.5 blend.

val.bin is a byte-identical copy of 05's val set -- the pre-registered 09
success criterion (val PPL < 22.7, the 07 number) is only meaningful on
the same validation data. The new shards are disjoint source files from
the ones 05's val tails were carved from, so there is no leakage.

Requires experiments/05-data-frontier/data/{train,val}.bin to exist.
"""

import os
import pickle
import shutil
import urllib.request

import numpy as np
import pyarrow.parquet as pq
import tiktoken

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
POOL_05 = os.path.join(os.path.dirname(__file__), "..", "05-data-frontier", "data")

NEW_SHARDS = [
    ("https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/002_00000.parquet",
     "fineweb-edu-002.parquet"),
    ("https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/003_00000.parquet",
     "fineweb-edu-003.parquet"),
    ("https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/resolve/main/cosmopedia-v2/train-00001-of-00104.parquet",
     "cosmopedia-v2-001.parquet"),
]

BATCH_DOCS = 2048
COPY_CHUNK = 128 * 1024 * 1024  # bytes


def download(url, path):
    if os.path.exists(path):
        print(f"already downloaded: {os.path.basename(path)}")
        return
    print(f"downloading {os.path.basename(path)}")
    last = [0]

    def hook(blocks, bs, total):
        done = blocks * bs
        if done - last[0] >= 500 * 1024 * 1024:
            last[0] = done
            print(f"  {done / 1e9:.1f}GB / {total / 1e9:.1f}GB", flush=True)

    tmp = path + ".part"
    urllib.request.urlretrieve(url, tmp, reporthook=hook)
    os.rename(tmp, path)


def tokenize_shard(enc, parquet_path, bin_path):
    if os.path.exists(bin_path):
        print(f"already tokenized: {os.path.basename(bin_path)}")
        return
    eot = enc.eot_token
    total = 0
    tmp = bin_path + ".part"
    with open(tmp, "wb") as f:
        for batch in pq.ParquetFile(parquet_path).iter_batches(batch_size=BATCH_DOCS, columns=["text"]):
            ids_list = enc.encode_ordinary_batch(batch.column("text").to_pylist())
            out = np.concatenate([np.array(ids + [eot], dtype=np.uint16) for ids in ids_list])
            f.write(out.tobytes())
            total += len(out)
            if total % 100_000_000 < len(out):
                print(f"  {os.path.basename(bin_path)}: {total:,} tokens...", flush=True)
    os.rename(tmp, bin_path)
    print(f"tokenized {os.path.basename(bin_path)}: {total:,} tokens")


def append_file(dst_handle, src_path):
    with open(src_path, "rb") as src:
        while True:
            chunk = src.read(COPY_CHUNK)
            if not chunk:
                break
            dst_handle.write(chunk)


def prepare():
    os.makedirs(DATA_DIR, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")

    src_train = os.path.join(POOL_05, "train.bin")
    src_val = os.path.join(POOL_05, "val.bin")
    assert os.path.exists(src_train) and os.path.exists(src_val), "run 05 data.py first"

    shard_bins = []
    for url, fname in NEW_SHARDS:
        pq_path = os.path.join(DATA_DIR, fname)
        bin_path = pq_path.replace(".parquet", ".bin")
        download(url, pq_path)
        tokenize_shard(enc, pq_path, bin_path)
        shard_bins.append(bin_path)

    train_path = os.path.join(DATA_DIR, "train.bin")
    with open(train_path + ".part", "wb") as f:
        append_file(f, src_train)
        for b in shard_bins:
            append_file(f, b)
    os.replace(train_path + ".part", train_path)
    shutil.copyfile(src_val, os.path.join(DATA_DIR, "val.bin"))
    with open(os.path.join(DATA_DIR, "meta.pkl"), "wb") as f:
        pickle.dump({"vocab_size": enc.n_vocab, "encoding": "gpt2"}, f)

    for url, fname in NEW_SHARDS:  # reclaim ~9GB of intermediates
        os.remove(os.path.join(DATA_DIR, fname))
        os.remove(os.path.join(DATA_DIR, fname.replace(".parquet", ".bin")))

    n_train = os.path.getsize(train_path) // 2
    n_val = os.path.getsize(os.path.join(DATA_DIR, "val.bin")) // 2
    print(f"train_tokens={n_train:,} val_tokens={n_val:,} (val is byte-identical to 05's)")

    data = np.memmap(train_path, dtype=np.uint16, mode="r")
    for label, off in (("old pool head", 1000), ("new fineweb", 1_800_000_000), ("new cosmo tail", len(data) - 5000)):
        print(f"--- {label}: {enc.decode(data[off:off + 50].astype(np.int64).tolist())[:160]!r}")


if __name__ == "__main__":
    prepare()
