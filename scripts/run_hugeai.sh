#!/usr/bin/env bash
# Sets up the venv (if needed) and runs routerbench against the real
# hugeai.sa router. Run this from a machine that can actually reach
# hugeai.sa -- it won't work from a network-restricted sandbox.
#
# Usage:
#   export HUGEAI_API_KEY="sk-..."   # your real key -- never hardcode it here
#   ./scripts/run_hugeai.sh [path/to/config.yaml]   # defaults to configs/hugeai.yaml
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

CONFIG="${1:-configs/hugeai.yaml}"

if [ -z "${HUGEAI_API_KEY:-}" ]; then
  echo "Error: HUGEAI_API_KEY is not set." >&2
  echo "Export your real API key first, e.g.:" >&2
  echo '  export HUGEAI_API_KEY="sk-..."' >&2
  exit 1
fi

if [ ! -d .venv ]; then
  echo "Creating virtualenv in .venv ..."
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "Installing/updating dependencies ..."
pip install -q -e ".[dev]"

echo "Running routerbench against $CONFIG ..."
routerbench run --config "$CONFIG"
