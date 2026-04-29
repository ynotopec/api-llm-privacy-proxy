#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_NAME="$(basename "$APP_DIR")"
VENV_DIR="${VENV_DIR:-$HOME/venv/$APP_NAME}"

cd "$APP_DIR"

if [[ -f ".env" ]]; then
  set -a
  source .env
  set +a
fi

HOST="${1:-${HOST:-0.0.0.0}}"
PORT="${2:-${PORT:-8088}}"

source "$VENV_DIR/bin/activate"

export HOST="$HOST"
export PORT="$PORT"

exec uvicorn app:app \
  --host "$HOST" \
  --port "$PORT" \
  --proxy-headers \
  --forwarded-allow-ips='*' \
  --log-level "${LOG_LEVEL:-info}"
