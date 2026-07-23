"""Build the SFT dataset: SmolTalk conversations formatted as plain text
with an assistant-only loss mask.

Template (plain gpt2 BPE, no new vocab -- documented here and used
identically by sample.py):

    User: {user content}\n
    Assistant: {assistant content}{<|endoftext|>}\n

Multi-turn conversations repeat the pair; system messages are prepended as
"System: {content}\n" (masked). The loss mask (uint8, parallel to the
token stream) is 1 ONLY on assistant content tokens and each assistant
turn's trailing <|endoftext|> -- prompts, role tags, and user/system text
are context, not targets.

Outputs (memmap format like every other milestone): tokens.bin (uint16),
mask.bin (uint8), meta.pkl. Conversations are packed back-to-back;
training samples random 1024-token windows and multiplies CE by the mask.
"""

import os
import pickle

import numpy as np
import pyarrow.parquet as pq
import tiktoken
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

SHARDS = [
    ("https://huggingface.co/datasets/HuggingFaceTB/smoltalk/resolve/main/data/everyday-conversations/train-00000-of-00001.parquet",
     "everyday-conversations.parquet"),
    ("https://huggingface.co/datasets/HuggingFaceTB/smoltalk/resolve/main/data/smol-magpie-ultra/train-00000-of-00006.parquet",
     "magpie-ultra-000.parquet"),
]

TARGET_TOKENS = 60_000_000  # ~3,700 steps at 16,384 tokens/step: an overnight SFT
VAL_CONVERSATIONS = 500


def download(url, path):
    if os.path.exists(path):
        return
    print(f"downloading {os.path.basename(path)}")
    urllib.request.urlretrieve(url, path + ".part")
    os.rename(path + ".part", path)


def format_conversation(enc, messages):
    """Returns (tokens, mask) for one conversation, or None if malformed."""
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

    streams = {"train": ([], []), "val": ([], [])}
    total = {"train": 0, "val": 0}
    n_convs = 0

    tok_path = os.path.join(DATA_DIR, "tokens.bin")
    mask_path = os.path.join(DATA_DIR, "mask.bin")
    ftok = open(tok_path + ".part", "wb")
    fmask = open(mask_path + ".part", "wb")
    vtok = open(os.path.join(DATA_DIR, "val_tokens.bin.part"), "wb")
    vmask = open(os.path.join(DATA_DIR, "val_mask.bin.part"), "wb")

    done = False
    for url, fname in SHARDS:
        if done:
            break
        path = os.path.join(DATA_DIR, fname)
        download(url, path)
        pf = pq.ParquetFile(path)
        assert "messages" in [f.name for f in pf.schema_arrow], pf.schema_arrow
        for batch in pf.iter_batches(batch_size=512, columns=["messages"]):
            for messages in batch.column("messages").to_pylist():
                out = format_conversation(enc, messages)
                if out is None:
                    continue
                toks, mask = out
                n_convs += 1
                split = "val" if n_convs <= VAL_CONVERSATIONS else "train"
                (vtok if split == "val" else ftok).write(np.array(toks, dtype=np.uint16).tobytes())
                (vmask if split == "val" else fmask).write(np.array(mask, dtype=np.uint8).tobytes())
                total[split] += len(toks)
                if total["train"] >= TARGET_TOKENS:
                    done = True
                    break
            if done:
                break

    for f in (ftok, fmask, vtok, vmask):
        f.close()
    os.replace(tok_path + ".part", tok_path)
    os.replace(mask_path + ".part", mask_path)
    os.replace(os.path.join(DATA_DIR, "val_tokens.bin.part"), os.path.join(DATA_DIR, "val_tokens.bin"))
    os.replace(os.path.join(DATA_DIR, "val_mask.bin.part"), os.path.join(DATA_DIR, "val_mask.bin"))

    with open(os.path.join(DATA_DIR, "meta.pkl"), "wb") as f:
        pickle.dump({"vocab_size": enc.n_vocab, "encoding": "gpt2",
                     "template": "User:/Assistant:, eot after assistant"}, f)

    toks = np.memmap(tok_path, dtype=np.uint16, mode="r")
    mask = np.memmap(mask_path, dtype=np.uint8, mode="r")
    frac = mask.mean()
    print(f"conversations={n_convs:,} train_tokens={total['train']:,} val_tokens={total['val']:,} "
          f"masked(loss) fraction={frac:.3f}")
    # spot check: decode a window and show which tokens carry loss
    off = 5000
    seg = toks[off:off + 80].astype(np.int64).tolist()
    print("--- sample:", enc.decode(seg)[:300].replace("\n", "\\n"))
    print("--- mask   :", "".join(str(int(b)) for b in mask[off:off + 80]))


if __name__ == "__main__":
    prepare()
