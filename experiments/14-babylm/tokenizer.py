"""Train and freeze one byte-level BPE tokenizer per BabyLM track.

Why not tiktoken gpt2 (as every prior milestone uses): at vocab 50257 and
n_embd 768 the embedding table is 38.6M parameters -- 33.8% of the 114M model
-- for a tokenizer trained on 2019 WebText. Measured on a held-out,
mix-weighted slice of the BabyLM corpus, a corpus-trained vocab-16384 BPE
matches gpt2's token efficiency (1.594 vs 1.607 tokens/word, -0.8%) while
cutting the model to 88.1M parameters. Same sequence lengths, 26M fewer
parameters.

Vocab sweep that produced that choice (identical held-out sample, tokens/word
then total params): 8192 -> 1.686 / 81.8M; 16384 -> 1.594 / 88.1M;
32768 -> 1.535 / 100.7M; gpt2 50257 -> 1.607 / 114.1M.

ONE TOKENIZER PER TRACK, trained only on that track's own corpus. Training the
strict-small tokenizer on the 100M-word strict corpus would leak data past the
10M-word budget and invalidate the entry. Both tracks use the SAME vocab size,
so parameter counts stay identical across tracks and only the learned merges
differ -- which is what keeps the two budget points comparable to each other.

Deterministic: the trainer is seeded by input order, and the input file list is
sorted, so re-running reproduces byte-identical tokenizer.json.

  uv run python experiments/14-babylm/tokenizer.py --track strict-small
  uv run python experiments/14-babylm/tokenizer.py --track strict
"""

import argparse
import json
import os

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, "data", "raw")

# Fixed across both tracks on purpose -- see module docstring.
VOCAB_SIZE = 16384
EOT = "<|endoftext|>"

SOURCES = ["bnc_spoken", "childes", "gutenberg", "open_subtitles",
           "simple_wiki", "switchboard"]

WORD_BUDGET = {"strict": 100_000_000, "strict-small": 10_000_000}


def corpus_files(track):
    d = os.path.join(RAW, track)
    files = [os.path.join(d, f"{s}.train.txt") for s in sorted(SOURCES)]
    missing = [p for p in files if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"missing corpus files: {missing}\nrun data.py's download first")
    return files


def train(track):
    files = corpus_files(track)

    # Budget compliance is asserted here as well as in data.py: the tokenizer
    # is the first thing that touches the corpus, so a wrong-track mixup should
    # fail loudly before anything downstream is built.
    words = 0
    for p in files:
        with open(p, encoding="utf-8", errors="replace") as f:
            words += len(f.read().split())
    budget = WORD_BUDGET[track]
    assert words <= budget, f"{track}: {words:,} words exceeds budget {budget:,}"
    print(f"{track}: {words:,} words / {budget:,} budget ({100 * words / budget:.1f}%)")

    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=[EOT],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train(files, trainer)

    out_dir = os.path.join(ROOT, "data", track)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "tokenizer.json")
    tmp = path + ".part"
    tok.save(tmp)
    os.replace(tmp, path)

    eot_id = tok.token_to_id(EOT)
    assert tok.get_vocab_size() == VOCAB_SIZE, tok.get_vocab_size()

    # Round-trip check on real corpus text -- a byte-level BPE with no unk
    # token must reproduce its input exactly.
    with open(files[0], encoding="utf-8", errors="replace") as f:
        probe = f.read(200_000)
    assert tok.decode(tok.encode(probe).ids) == probe, "tokenizer round-trip failed"

    meta = {"track": track, "vocab_size": VOCAB_SIZE, "eot_id": eot_id,
            "words": words, "word_budget": budget, "encoding": "babylm-bpe-16384"}
    with open(os.path.join(out_dir, "tokenizer_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  vocab={VOCAB_SIZE} eot_id={eot_id} round-trip OK -> {path}")
    return tok


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--track", choices=["strict", "strict-small"], required=True)
    train(p.parse_args().track)
