#!/usr/bin/env bash
# Start songloc. Override host/port in .env (SONGLOC_HOST / SONGLOC_PORT).
set -euo pipefail
command -v ffmpeg >/dev/null || { echo "ffmpeg not found on PATH — install it first."; exit 1; }
HOST="${SONGLOC_HOST:-127.0.0.1}"
PORT="${SONGLOC_PORT:-8000}"
echo "songloc → http://${HOST}:${PORT}/"
exec uvicorn app.main:app --host "$HOST" --port "$PORT" "$@"
