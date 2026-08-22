"""Build train.bin / val.bin for one BabyLM track.

Differences from 05-data-frontier/data.py, all forced by the task:
- Source is six plain .txt files per track (no parquet, no pyarrow).
- Tokenizer is the track's own frozen BPE from tokenizer.py, NOT tiktoken gpt2
  (vocab 16384, so uint16 still holds).
- WORD counting is a first-class output and a hard gate. Nothing else in this
  repo counts words; BabyLM's budget is denominated in them, and exceeding it
  invalidates a submission outright.

Val holdout: BabyLM ships no dev split (the repos contain only *.train.txt), so
we carve one with 05's convention -- a deterministic tail slice from EACH source,
sized to that source's share, so the val mix matches the train mix. Nothing is
shuffled and no RNG is involved, so the split is exactly reproducible.

Document separator: one EOT between sources, not per line. These files are
line-oriented and CHILDES averages ~5 words per line; an EOT per line would add
~17% pure separator tokens to the budget. FineWeb-Edu documents in 05 averaged
~1000 tokens, where per-document EOT cost ~0.1% -- the same rule does not
transfer to a dialogue corpus.

  uv run python experiments/14-babylm/data.py --track strict-small
  uv run python experiments/14-babylm/data.py --track strict
"""

import argparse
import json
import os
import pickle
import urllib.request

import numpy as np
from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, "data", "raw")

SOURCES = ["bnc_spoken", "childes", "gutenberg", "open_subtitles",
           "simple_wiki", "switchboard"]

TRACKS = {
    "strict":       {"repo": "BabyLM-2026-Strict",       "budget": 100_000_000, "val_tokens": 2_000_000},
    "strict-small": {"repo": "BabyLM-2026-Strict-Small", "budget":  10_000_000, "val_tokens":   500_000},
}


def download(track):
    d = os.path.join(RAW, track)
    os.makedirs(d, exist_ok=True)
    repo = TRACKS[track]["repo"]
    for s in SOURCES:
        out = os.path.join(d, f"{s}.train.txt")
        if os.path.exists(out) and os.path.getsize(out) > 0:
            continue
        url = f"https://huggingface.co/datasets/BabyLM-community/{repo}/resolve/main/{s}.train.txt"
        print(f"  downloading {s} ...", flush=True)
        urllib.request.urlretrieve(url, out + ".part")
        os.replace(out + ".part", out)


def prepare(track):
    cfg = TRACKS[track]
    download(track)

    tok_path = os.path.join(ROOT, "data", track, "tokenizer.json")
    if not os.path.exists(tok_path):
        raise SystemExit(f"missing {tok_path}\nrun: tokenizer.py --track {track}")
    tok = Tokenizer.from_file(tok_path)
    with open(os.path.join(ROOT, "data", track, "tokenizer_meta.json")) as f:
        tmeta = json.load(f)
    eot = tmeta["eot_id"]
    vocab_size = tmeta["vocab_size"]
    assert vocab_size < 2 ** 16, "uint16 storage requires vocab < 65536"

    # Pass 1: tokenize each source, count words, split off its val tail.
    total_words = 0
    per_source = {}
    encoded = {}
    for s in SOURCES:
        p = os.path.join(RAW, track, f"{s}.train.txt")
        with open(p, encoding="utf-8", errors="replace") as f:
            text = f.read()
        words = len(text.split())
        total_words += words
        ids = tok.encode(text).ids
        encoded[s] = ids
        per_source[s] = {"words": words, "tokens": len(ids),
                         "bytes": len(text.encode("utf-8"))}
        print(f"  {s:16s} words={words:11,}  tokens={len(ids):11,}  "
              f"tok/word={len(ids) / max(1, words):.3f}", flush=True)
        del text

    # HARD GATE: the one error that invalidates a submission.
    budget = cfg["budget"]
    assert total_words <= budget, \
        f"BUDGET VIOLATION: {track} has {total_words:,} words, budget is {budget:,}"

    total_tokens = sum(v["tokens"] for v in per_source.values())

    # Val = tail of each source, sized to that source's token share.
    val_target = cfg["val_tokens"]
    train_parts, val_parts = [], []
    for s in SOURCES:
        ids = encoded[s]
        share = len(ids) / total_tokens
        n_val = int(round(val_target * share))
        n_val = min(n_val, len(ids) // 4)  # never take more than a quarter of a source
        if n_val > 0:
            val_parts.append(np.array(ids[-n_val:], dtype=np.uint16))
            head = ids[:-n_val]
        else:
            head = ids
        train_parts.append(np.array(head + [eot], dtype=np.uint16))
        per_source[s]["val_tokens"] = n_val

    out_dir = os.path.join(ROOT, "data", track)
    train = np.concatenate(train_parts)
    val = np.concatenate(val_parts)

    for name, arr in (("train.bin", train), ("val.bin", val)):
        path = os.path.join(out_dir, name)
        arr.tofile(path + ".part")
        os.replace(path + ".part", path)

    with open(os.path.join(out_dir, "meta.pkl"), "wb") as f:
        pickle.dump({"vocab_size": vocab_size, "encoding": tmeta["encoding"],
                     "eot_id": eot}, f)

    summary = {
        "track": track,
        "words": total_words,
        "word_budget": budget,
        "budget_used_pct": round(100 * total_words / budget, 3),
        "tokens_total": int(total_tokens),
        "tokens_per_word": round(total_tokens / total_words, 4),
        "train_tokens": int(len(train)),
        "val_tokens": int(len(val)),
        "vocab_size": vocab_size,
        "per_source": per_source,
        "val_protocol": "deterministic tail slice per source, sized to source token share",
    }
    with open(os.path.join(out_dir, "data_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  WORDS {total_words:,} / {budget:,} = {100 * total_words / budget:.1f}%  OK")
    print(f"  tokens {total_tokens:,} (tok/word {total_tokens / total_words:.4f})")
    print(f"  train {len(train):,}  val {len(val):,}  vocab {vocab_size}")

    # Eyeball check, as 05 does: decode a snippet from three offsets.
    for frac in (0.1, 0.5, 0.9):
        i = int(len(train) * frac)
        snippet = tok.decode([int(x) for x in train[i:i + 40]])
        print(f"  [{frac:.0%}] {snippet[:110]!r}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--track", choices=list(TRACKS), required=True)
    prepare(p.parse_args().track)
