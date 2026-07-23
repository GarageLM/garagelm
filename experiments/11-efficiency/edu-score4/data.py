"""Build the harder-filtered pool: FineWeb-Edu rows with int_score >= 4
(sample-10BT itself is score >= 3, so this keeps only its top tier),
blended with Cosmopedia at the same ~84.5/15.5 ratio as the 05 pool.
val.bin is the 05 val set (copied), so val losses compare directly with
the control -- the question is whether higher-scored training data models
the SAME validation distribution better.

Downloads shards until >= 250M filtered FineWeb tokens (pool stays
sub-epoch for the 100M budget); reports the score>=4 survival rate.
"""

import os
import shutil
import urllib.request

import numpy as np
import pyarrow.parquet as pq
import tiktoken

ROOT = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT, "data")
POOL_05 = os.path.join(ROOT, "..", "..", "05-data-frontier", "data")

FWE_BASE = "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT"
FWE_SHARDS = [f"{i:03d}_00000.parquet" for i in range(4)]
COSMO_URL = ("https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/resolve/main/"
             "cosmopedia-v2/train-00000-of-00104.parquet")

FWE_TARGET = 250_000_000
COSMO_RATIO = 0.155 / 0.845
BATCH_DOCS = 2048


def download(url, path):
    if os.path.exists(path):
        return
    print(f"downloading {os.path.basename(path)}", flush=True)
    urllib.request.urlretrieve(url, path + ".part")
    os.rename(path + ".part", path)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token

    train_path = os.path.join(DATA_DIR, "train.bin")
    if os.path.exists(train_path):
        print(f"already built: {os.path.getsize(train_path) // 2:,} tokens")
        return

    fwe_tokens = 0
    rows_seen = rows_kept = 0
    parquets = []
    with open(train_path + ".part", "wb") as f:
        for shard in FWE_SHARDS:
            if fwe_tokens >= FWE_TARGET:
                break
            path = os.path.join(DATA_DIR, shard)
            download(f"{FWE_BASE}/{shard}", path)
            parquets.append(path)
            for batch in pq.ParquetFile(path).iter_batches(
                    batch_size=BATCH_DOCS, columns=["text", "int_score"]):
                scores = batch.column("int_score").to_pylist()
                texts = [t for t, s in zip(batch.column("text").to_pylist(), scores) if s >= 4]
                rows_seen += len(scores)
                rows_kept += len(texts)
                if not texts:
                    continue
                ids_list = enc.encode_ordinary_batch(texts)
                out = np.concatenate([np.array(ids + [eot], dtype=np.uint16) for ids in ids_list])
                f.write(out.tobytes())
                fwe_tokens += len(out)
                if fwe_tokens >= FWE_TARGET:
                    break
            print(f"  after {shard}: {fwe_tokens:,} filtered tokens "
                  f"(kept {rows_kept:,}/{rows_seen:,} rows)", flush=True)

        cosmo_target = int(fwe_tokens * COSMO_RATIO)
        cosmo_path = os.path.join(DATA_DIR, "cosmo.parquet")
        download(COSMO_URL, cosmo_path)
        parquets.append(cosmo_path)
        cosmo_tokens = 0
        for batch in pq.ParquetFile(cosmo_path).iter_batches(batch_size=BATCH_DOCS, columns=["text"]):
            ids_list = enc.encode_ordinary_batch(batch.column("text").to_pylist())
            out = np.concatenate([np.array(ids + [eot], dtype=np.uint16) for ids in ids_list])
            f.write(out.tobytes())
            cosmo_tokens += len(out)
            if cosmo_tokens >= cosmo_target:
                break

    os.replace(train_path + ".part", train_path)
    shutil.copyfile(os.path.join(POOL_05, "val.bin"), os.path.join(DATA_DIR, "val.bin"))
    shutil.copyfile(os.path.join(POOL_05, "meta.pkl"), os.path.join(DATA_DIR, "meta.pkl"))
    for p in parquets:
        os.remove(p)

    total = os.path.getsize(train_path) // 2
    print(f"pool: {total:,} tokens ({fwe_tokens:,} fineweb score>=4 + {cosmo_tokens:,} cosmo); "
          f"score>=4 survival: {rows_kept / max(rows_seen, 1):.1%}; val = 05's (copied)")


if __name__ == "__main__":
    main()
