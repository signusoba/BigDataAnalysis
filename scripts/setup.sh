#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -c 'import sys; assert sys.version_info[:2] == (3, 12), "Use Python 3.12"'
if [ ! -e .venv ]; then
  python3 -m venv .venv
elif [ ! -f .venv/pyvenv.cfg ]; then
  echo 'Existing .venv is not a virtual environment; preserving it.' >&2
  exit 1
fi
.venv/bin/python -m pip install --require-hashes -r requirements.lock
.venv/bin/python -m pip check
echo 'Setup complete. Activate with: source .venv/bin/activate'
