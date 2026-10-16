#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -e .venv ]; then
  python3 -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else "Use Python 3.12 to create .venv")'
  python3 -m venv .venv
elif [ ! -f .venv/pyvenv.cfg ]; then
  echo 'Existing .venv is not a virtual environment; preserving it.' >&2
  exit 1
fi
if [ ! -x .venv/bin/python ]; then
  echo 'Existing .venv has no usable Linux Python; preserve/rename it and rerun setup.' >&2
  exit 1
fi
.venv/bin/python -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else "Existing .venv must use Python 3.12; preserve/rename it and rerun setup")'
.venv/bin/python -m pip install --require-hashes -r requirements.lock
.venv/bin/python -m pip check
echo 'Setup complete. Activate with: source .venv/bin/activate'
