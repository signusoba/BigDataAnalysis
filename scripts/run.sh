#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PWD/artifacts/cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/artifacts/cache/matplotlib}"
export TORCH_HOME="${TORCH_HOME:-$PWD/artifacts/cache/torch}"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$TORCH_HOME"
exec python -m uvicorn persian_ocr.api:app --host "${NOOR_HOST:-127.0.0.1}" --port "${NOOR_PORT:-8000}"
