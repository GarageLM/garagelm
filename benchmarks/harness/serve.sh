#!/bin/bash
# Launch (or stop) mlx_lm.server for the harness, detached, with a pid file.
#
#   benchmarks/harness/serve.sh start mlx-community/Qwen3.5-9B-4bit [PORT=8421] [CONCURRENCY=8]
#   benchmarks/harness/serve.sh stop  [PORT]
#   benchmarks/harness/serve.sh status [PORT]
#
# Rules (CLAUDE.md): contention check before launching, absolute paths, never
# concurrently with a training run. The server owns the GPU; the runner only
# talks HTTP. Logs/pids live under benchmarks/harness/.server/ (gitignored).
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT="$(pwd)"
STATE="$ROOT/benchmarks/harness/.server"
mkdir -p "$STATE"
CMD="${1:-status}"; shift || true

case "$CMD" in
  start)
    MODEL="${1:?model id}"; PORT="${2:-8421}"; CONC="${3:-8}"
    if pgrep -f "train.py" >/dev/null; then echo "refusing: a train.py is running (sequential rule)"; exit 2; fi
    BUSY=$(ps aux | awk 'NR>1 && $3>50 {print $3, $11}' | grep -v mlx_lm.server | head -3 || true)
    [ -n "$BUSY" ] && echo "warning: >50% CPU processes present:" && echo "$BUSY"
    if [ -f "$STATE/server-$PORT.pid" ] && kill -0 "$(cat "$STATE/server-$PORT.pid")" 2>/dev/null; then
      echo "already running on :$PORT (pid $(cat "$STATE/server-$PORT.pid"))"; exit 0; fi
    LOG="$STATE/server-$PORT.log"
    nohup "$ROOT/.venv/bin/python" -m mlx_lm.server --model "$MODEL" --port "$PORT" \
      --decode-concurrency "$CONC" --prompt-concurrency 4 --prompt-cache-bytes 2000000000 \
      --max-tokens 32768 --chat-template-args '{"enable_thinking": true}' --log-level WARNING \
      > "$LOG" 2>&1 &
    echo $! > "$STATE/server-$PORT.pid"
    echo "$MODEL" > "$STATE/server-$PORT.model"
    echo "started pid $! on :$PORT, log $LOG"
    for i in $(seq 1 120); do
      if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then echo "healthy after ${i}s"; exit 0; fi
      if ! kill -0 "$(cat "$STATE/server-$PORT.pid")" 2>/dev/null; then echo "server died; tail of log:"; tail -20 "$LOG"; exit 1; fi
      sleep 1
    done
    echo "not healthy after 120s; tail of log:"; tail -20 "$LOG"; exit 1;;
  stop)
    PORT="${1:-8421}"
    if [ -f "$STATE/server-$PORT.pid" ]; then PID=$(cat "$STATE/server-$PORT.pid"); kill "$PID" 2>/dev/null && echo "stopped $PID" || echo "not running"; rm -f "$STATE/server-$PORT.pid"; else echo "no pid file"; fi;;
  status)
    PORT="${1:-8421}"
    if [ -f "$STATE/server-$PORT.pid" ] && kill -0 "$(cat "$STATE/server-$PORT.pid")" 2>/dev/null; then
      echo "running pid $(cat "$STATE/server-$PORT.pid") model $(cat "$STATE/server-$PORT.model" 2>/dev/null) :$PORT"
      curl -s "http://127.0.0.1:$PORT/v1/models" | head -c 300; echo
    else echo "not running on :$PORT"; fi;;
  *) echo "usage: serve.sh start <model> [port] [concurrency] | stop [port] | status [port]"; exit 1;;
esac
