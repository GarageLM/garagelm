"""Full-SmolTalk SFT dataset: the `all` config (curated mixture across
every SmolTalk subset), formatted with the exact 10-sft template and
assistant-only loss masking.

Same output format as 10-sft (tokens.bin uint16 + mask.bin uint8 +
val_* + meta.pkl); downloads `data/all` shards until TARGET_TOKENS of
formatted train tokens exist. First VAL_CONVERSATIONS conversations are
held out as val — same convention as 10-sft, so masked-val numbers are
comparable in spirit (different underlying mix, stated in the README).
"""

import os
import pickle
import urllib.request

import numpy as np
import pyarrow.parquet as pq
import tiktoken

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
BASE = "https://huggingface.co/datasets/HuggingFaceTB/smoltalk/resolve/main/data/all"
N_SHARDS = 9
TARGET_TOKENS = 310_000_000  # ~1 epoch at 18,300 steps x 16,384 tokens
VAL_CONVERSATIONS = 500
BATCH_DOCS = 512


def download(url, path):
    if os.path.exists(path):
        return
    print(f"downloading {os.path.basename(path)}", flush=True)
    urllib.request.urlretrieve(url, path + ".part")
    os.rename(path + ".part", path)


def format_conversation(enc, messages):
    eot = enc.eot_token
    toks, mask = [], []
    saw_assistant = False
    for m in messages:
        role, content = m.get("role"), (m.get("content") or "").strip()
        if not content:
            continue
        if role == "assistant":
            prefix = enc.encode_ordinary("Assistant:")
            body = enc.encode_ordinary(" " + content)
            toks += prefix + body + [eot] + enc.encode_ordinary("\n")
            mask += [0] * len(prefix) + [1] * len(body) + [1] + [0]
            saw_assistant = True
        elif role in ("user", "system"):
            tag = "User:" if role == "user" else "System:"
            seg = enc.encode_ordinary(f"{tag} {content}\n")
            toks += seg
            mask += [0] * len(seg)
        else:
            return None
    if not saw_assistant:
        return None
    return toks, mask


def prepare():
    os.makedirs(DATA_DIR, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")

    paths = {name: os.path.join(DATA_DIR, name + ".part2") for name in
             ("tokens.bin", "mask.bin", "val_tokens.bin", "val_mask.bin")}
    handles = {k: open(v, "wb") for k, v in paths.items()}
    total = {"train": 0, "val": 0}
    n_convs = 0
    done = False

    for i in range(N_SHARDS):
        if done:
            break
        fname = f"train-{i:05d}-of-{N_SHARDS:05d}.parquet"
        local = os.path.join(DATA_DIR, fname)
        download(f"{BASE}/{fname}", local)
        for batch in pq.ParquetFile(local).iter_batches(batch_size=BATCH_DOCS, columns=["messages"]):
            for messages in batch.column("messages").to_pylist():
                out = format_conversation(enc, messages)
                if out is None:
                    continue
                toks, mask = out
                n_convs += 1
                split = "val" if n_convs <= VAL_CONVERSATIONS else "train"
                tkey = "val_tokens.bin" if split == "val" else "tokens.bin"
                mkey = "val_mask.bin" if split == "val" else "mask.bin"
                handles[tkey].write(np.array(toks, dtype=np.uint16).tobytes())
                handles[mkey].write(np.array(mask, dtype=np.uint8).tobytes())
                total[split] += len(toks)
                if total["train"] >= TARGET_TOKENS:
                    done = True
                    break
            if done:
                break
        os.remove(local)
        print(f"  shard {i}: cumulative train tokens {total['train']:,}", flush=True)

    for k, h in handles.items():
        h.close()
        os.replace(paths[k], os.path.join(DATA_DIR, k))
    with open(os.path.join(DATA_DIR, "meta.pkl"), "wb") as f:
        pickle.dump({"vocab_size": enc.n_vocab, "encoding": "gpt2",
                     "template": "User:/Assistant:, eot after assistant",
                     "source": "smoltalk data/all"}, f)

    mask = np.memmap(os.path.join(DATA_DIR, "mask.bin"), dtype=np.uint8, mode="r")
    print(f"conversations={n_convs:,} train_tokens={total['train']:,} "
          f"val_tokens={total['val']:,} masked(loss) fraction={mask.mean():.3f}")


if __name__ == "__main__":
    prepare()
