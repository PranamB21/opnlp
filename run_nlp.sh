#!/usr/bin/env bash
# Start the NLP microservice (stage 2) — dashboard, transcript bridge, LLM extraction.
set -euo pipefail
cd "$(dirname "$0")"

export NLP_PORT="${NLP_PORT:-8001}"
exec ./WhisperLiveKit/.venv/bin/python -m nlp_service