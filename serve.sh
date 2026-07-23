#!/bin/bash
# GarageLM one-command chat: repo -> browser.
#
#   ./serve.sh                     # chat model, auto device, opens browser
#   PORT=9000 ./serve.sh           # custom port
#   MODEL=09-flagship-2 ./serve.sh # serve the base model instead
#
# What it does, in order:
#   1. sync the uv environment (no-op when already synced)
#   2. convert the checkpoint to MLX if the converted weights are missing
#      (with the logit-parity gate)
#   3. detect a running training job -> start on CPU so Metal stays free;
#      otherwise run at full Metal speed
#   4. start the OpenAI-compatible server + web UI and open the browser
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8080}"
MODEL="${MODEL:-10-sft}"
MODEL_DIR="benchmarks/mlx/converted/$MODEL"
CKPT_DIR="experiments/$MODEL"

uv sync -q

if [ ! -f "$MODEL_DIR/weights.safetensors" ]; then
  if [ ! -f "$CKPT_DIR/out/ckpt.pt" ]; then
    echo "error: no converted weights at $MODEL_DIR and no checkpoint at $CKPT_DIR/out/ckpt.pt" >&2
    exit 1
  fi
  echo "== converting $CKPT_DIR -> $MODEL_DIR (one-time, with parity gate)"
  uv run python benchmarks/mlx/convert.py --experiment-dir "$CKPT_DIR" --parity
fi

DEVICE_FLAG=""
if pgrep -f "experiments/.*/train\.py" >/dev/null 2>&1; then
  echo "== training run detected: starting on CPU so Metal stays free (slower chat)"
  echo "   rerun after training finishes for ~500 tok/s"
  DEVICE_FLAG="--cpu"
fi

( for _ in $(seq 1 60); do
    sleep 1
    curl -s -o /dev/null "http://127.0.0.1:$PORT/v1/models" && { open "http://127.0.0.1:$PORT"; exit 0; }
  done ) &

exec uv run python benchmarks/mlx/server.py $DEVICE_FLAG --port "$PORT" --model-dir "$MODEL_DIR"
