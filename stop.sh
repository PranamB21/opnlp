#!/usr/bin/env bash
# Stop the stack started by start.sh (or the individual run_*.sh scripts).
# Targets whatever is listening on the ASR and NLP ports, so it never matches
# unrelated shells or editors.
set -uo pipefail

ASR_PORT="${WLK_PORT:-8000}"
NLP_PORT="${NLP_PORT:-8001}"

port_pids() {
  if command -v ss >/dev/null 2>&1; then
    ss -ltnpH "sport = :$1" 2>/dev/null | grep -o 'pid=[0-9]*' | cut -d= -f2
  elif command -v lsof >/dev/null 2>&1; then
    lsof -tiTCP:"$1" -sTCP:LISTEN 2>/dev/null
  fi
}

stopped=0
for port in "$NLP_PORT" "$ASR_PORT"; do
  pids=$(port_pids "$port" | sort -u)
  if [ -z "$pids" ]; then
    echo "Port $port: nothing running"
    continue
  fi
  echo "Port $port: stopping pid(s) $(echo "$pids" | tr '\n' ' ')"
  # shellcheck disable=SC2086
  kill $pids 2>/dev/null || true
  stopped=1
done

if [ "$stopped" -eq 1 ]; then
  for _ in $(seq 1 20); do
    if [ -z "$( { port_pids "$ASR_PORT"; port_pids "$NLP_PORT"; } | sort -u)" ]; then
      echo "All services stopped."
      exit 0
    fi
    sleep 0.5
  done
  echo "Some processes did not exit; force-killing."
  for port in "$NLP_PORT" "$ASR_PORT"; do
    pids=$(port_pids "$port" | sort -u)
    [ -n "$pids" ] && kill -9 $pids 2>/dev/null || true
  done
fi