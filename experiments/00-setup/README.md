# 00-setup

Environment used by every experiment milestone in this repo. Shared once at
the repo root rather than duplicated per experiment, since `02` and `03`
need the same base dependencies as `01`.

## What's installed

- **`uv`** (via `brew install uv`) manages the Python version and the venv.
- **Python 3.12**, pinned via `uv venv --python 3.12` into a repo-root
  `.venv/`. System Python is 3.14.3, newer than most ML package wheels
  currently target — see `CLAUDE.md`.
- **`torch`** and **`numpy`**, declared in the repo-root `pyproject.toml`,
  installed with `uv pip install torch numpy`. The PyPI `torch` wheel for
  macOS arm64 ships with MPS (Metal) support built in — no special index
  URL needed. Verified locally: `torch.backends.mps.is_available()` is
  `True` on this machine (Apple M4 Pro).

Later milestones will add to `pyproject.toml` as needed:
`lm-evaluation-harness` (quality benchmarking, see `benchmarks/README.md`)
and `mlx` (inference-latency benchmarking) — not installed yet, only pulled
in when `04-benchmark` work actually starts.

## Reproducing this environment

```
brew install uv
uv venv --python 3.12
uv pip install torch numpy
```

Run any experiment script with `uv run python experiments/<milestone>/<script>.py`
— `uv run` uses the repo's `.venv` automatically, no manual activation needed.
