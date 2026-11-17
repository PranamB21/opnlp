#!/usr/bin/env bash
# Start the whole stack: WhisperLiveKit ASR + diarization (8000) and the
# NLP service that serves the dashboard (8001). Ctrl+C stops both.
#
#   ./start.sh
#   WLK_MODEL=small ./start.sh          # more accurate, needs more VRAM
#   ANTHROPIC_API_KEY=sk-ant-... ./start.sh
#
# Optional .env in this directory is loaded first (see env.example).
set -euo pipefail
cd "$(dirname "$0")"

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

ASR_PORT="${WLK_PORT:-8000}"
NLP_PORT="${NLP_PORT:-8001}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
ASR_LOG="$LOG_DIR/asr.log"
NLP_LOG="$LOG_DIR/nlp.log"

for port in "$ASR_PORT" "$NLP_PORT"; do
  if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
    echo "Port $port is already in use. Stop the running service first:" >&2
    echo "  ./stop.sh" >&2
    exit 1
  fi
done

cleanup() {
  echo
  echo "Stopping services..."
  [ -n "${NLP_PID:-}" ] && kill "$NLP_PID" 2>/dev/null || true
  [ -n "${ASR_PID:-}" ] && kill "$ASR_PID" 2>/dev/null || true
  wait 2>/dev/null || true
  echo "Stopped."
}
trap cleanup EXIT INT TERM

echo "Starting WhisperLiveKit ASR + diarization on :$ASR_PORT ..."
./run_asr.sh >"$ASR_LOG" 2>&1 &
ASR_PID=$!

# Model load takes a few seconds on a warm cache, much longer on first download.
for _ in $(seq 1 150); do
  if curl -fsS "http://localhost:$ASR_PORT/health" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$ASR_PID" 2>/dev/null; then
    echo "ASR server exited. Last lines of $ASR_LOG:" >&2
    tail -20 "$ASR_LOG" >&2
    exit 1
  fi
  sleep 2
done
echo "  ASR ready (log: $ASR_LOG)"

echo "Starting NLP service + dashboard on :$NLP_PORT ..."
./run_nlp.sh >"$NLP_LOG" 2>&1 &
NLP_PID=$!

for _ in $(seq 1 60); do
  if curl -fsS "http://localhost:$NLP_PORT/health" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$NLP_PID" 2>/dev/null; then
    echo "NLP service exited. Last lines of $NLP_LOG:" >&2
    tail -20 "$NLP_LOG" >&2
    exit 1
  fi
  sleep 1
done

echo
echo "Dashboard:   http://localhost:$NLP_PORT/     <- use this one"
echo "ASR-only UI: http://localhost:$ASR_PORT/"
echo "Logs:        $LOG_DIR/asr.log  $LOG_DIR/nlp.log"
echo
echo "Press Ctrl+C to stop both services."
wait