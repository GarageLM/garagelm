# 01-baseline-smoke

Proves the full local pipeline works end-to-end — data prep, tokenizer,
training, checkpointing, sampling — before investing in the larger
`02-baseline-small-lm`. Not an architecture experiment: the model is a
classic nanoGPT-lineage decoder-only transformer (learned position
embeddings, pre-LN blocks, causal self-attention, GELU MLP), char-level
tokenized on TinyShakespeare. See `docs/literature/README.md` entry 5 for
why nanoGPT is the reference here.

- `data.py` — downloads TinyShakespeare, builds a char-level vocab, writes
  `data/train.bin`, `data/val.bin`, `data/meta.pkl` (gitignored).
- `model.py` — the GPT model definition.
- `config.py` — model/training hyperparameters, sized to train in a few
  minutes on an Apple M4 Pro (MPS): 4 layers, 4 heads, 256-dim, 2000 iters.
- `train.py` — training loop; writes `out/ckpt.pt` and `out/run_summary.json`
  (params, final train/val loss, wall-clock time — gitignored).
- `sample.py` — loads the checkpoint and generates text from a prompt.

## Run it

```
uv run python experiments/01-baseline-smoke/data.py
uv run python experiments/01-baseline-smoke/train.py
uv run python experiments/01-baseline-smoke/sample.py --prompt "ROMEO:"
```

## Result

See `out/run_summary.json` after training for this run's actual params,
final train/val loss, and wall-clock time.
