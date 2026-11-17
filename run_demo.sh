#!/usr/bin/env bash
# Replay the pre-recorded two-speaker demo meeting through the full pipeline.
# Requires run_asr.sh and run_nlp.sh to be running.
#
#   ./run_demo.sh                 # real time
#   ./run_demo.sh --speed 2 -v    # faster, verbose insights
set -euo pipefail
cd "$(dirname "$0")"

AUDIO="demo_audio/meeting.wav"
PY=./WhisperLiveKit/.venv/bin/python

if [ ! -f "$AUDIO" ]; then
  echo "Building demo audio (needs network for edge-tts)..."
  "$PY" demo_audio/build_demo_audio.py
fi

exec "$PY" demo_audio/feed_demo.py "$AUDIO" --end "$@"