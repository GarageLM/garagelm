"""DPO pair dataset from HuggingFaceH4/ultrafeedback_binarized
(train_prefs split), formatted with the lab's SFT template.

Each pair yields two token sequences (prompt+chosen, prompt+rejected)
sharing a prompt prefix; the DPO loss reads only the response region.
Storage: one uint16 token stream (pairs.bin) plus an index (index.npz)
with, per pair and per side: start offset, total length, and
response-start offset (loss region = [resp, start+len)). Sequences
longer than MAX_LEN tokens are skipped. Last VAL_PAIRS pairs are held
out for the G3 preference-accuracy gate.
"""

import os
import pickle
import urllib.request

import numpy as np
import pyarrow.parquet as pq
import tiktoken

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
BASE = ("https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized/"
        "resolve/main/data")
SHARDS = ["train_prefs-00000-of-00001.parquet"]
MAX_LEN = 1024
VAL_PAIRS = 1000


def download(url, path):
    if os.path.exists(path):
        return
    print(f"downloading {os.path.basename(path)}", flush=True)
    urllib.request.urlretrieve(url, path + ".part")
    os.rename(path + ".part", path)


def encode_side(enc, messages):
    """Template-encode a full conversation whose LAST message is the
    assistant response being scored. Returns (tokens, resp_offset) or None."""
    eot = enc.eot_token
    if not messages or messages[-1].get("role") != "assistant":
        return None
    toks = []
    for m in messages[:-1]:
        role, content = m.get("role"), (m.get("content") or "").strip()
        if not content:
            continue
        if role == "assistant":
            toks += (enc.encode_ordinary("Assistant:")
                     + enc.encode_ordinary(" " + content) + [eot]
                     + enc.encode_ordinary("\n"))
        elif role in ("user", "system"):
            tag = "User:" if role == "user" else "System:"
            toks += enc.encode_ordinary(f"{tag} {content}\n")
        else:
            return None
    final = (messages[-1].get("content") or "").strip()
    if not final:
        return None
    toks += enc.encode_ordinary("Assistant:")
    resp = len(toks)
    toks += enc.encode_ordinary(" " + final) + [eot]
    if len(toks) > MAX_LEN or resp >= len(toks):
        return None
    return toks, resp


def prepare():
    os.makedirs(DATA_DIR, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")

    stream = open(os.path.join(DATA_DIR, "pairs.bin.part"), "wb")
    idx = {k: [] for k in ("c_start", "c_len", "c_resp", "r_start", "r_len", "r_resp")}
    offset = 0
    kept = skipped = 0

    for shard in SHARDS:
        local = os.path.join(DATA_DIR, shard)
        download(f"{BASE}/{shard}", local)
        for batch in pq.ParquetFile(local).iter_batches(
                batch_size=256, columns=["chosen", "rejected"]):
            for chosen, rejected in zip(batch.column("chosen").to_pylist(),
                                        batch.column("rejected").to_pylist()):
                c = encode_side(enc, chosen)
                r = encode_side(enc, rejected)
                if c is None or r is None:
                    skipped += 1
                    continue
                for (toks, resp), pre in ((c, "c"), (r, "r")):
                    stream.write(np.array(toks, dtype=np.uint16).tobytes())
                    idx[pre + "_start"].append(offset)
                    idx[pre + "_len"].append(len(toks))
                    idx[pre + "_resp"].append(resp)
                    offset += len(toks)
                kept += 1
        os.remove(local)

    stream.close()
    os.replace(os.path.join(DATA_DIR, "pairs.bin.part"),
               os.path.join(DATA_DIR, "pairs.bin"))
    arrays = {k: np.array(v, dtype=np.int64) for k, v in idx.items()}
    arrays["n_val"] = np.array(VAL_PAIRS)
    np.savez(os.path.join(DATA_DIR, "index.npz"), **arrays)
    with open(os.path.join(DATA_DIR, "meta.pkl"), "wb") as f:
        pickle.dump({"vocab_size": enc.n_vocab, "max_len": MAX_LEN,
                     "source": "ultrafeedback_binarized/train_prefs"}, f)
    print(f"pairs kept={kept:,} skipped={skipped:,} "
          f"stream_tokens={offset:,} val_pairs={VAL_PAIRS}")


if __name__ == "__main__":
    prepare()
