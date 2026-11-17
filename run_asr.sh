#!/usr/bin/env bash
# Start the WhisperLiveKit ASR + diarization server (stage 1).
set -euo pipefail
cd "$(dirname "$0")/WhisperLiveKit"

PORT="${WLK_PORT:-8000}"
MODEL="${WLK_MODEL:-base}"

exec .venv/bin/wlk serve \
  --model "$MODEL" \
  --language "${WLK_LANGUAGE:-en}" \
  --diarization \
  --pcm-input \
  --port "$PORT" \
  --session-idle-timeout "${WLK_SESSION_IDLE_TIMEOUT:-0}" \
  -l "${WLK_LOG_LEVEL:-INFO}"