#!/bin/bash
# [DO NOT REMOVE] run.sh — Thin wrapper for python3 run.py
# Usage: ./run.sh [--no-gui] [--port PORT] [--host HOST]
# All arguments are forwarded to run.py.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/python:${PYTHONPATH:-}"
mkdir -p .bin
cat << 'RANLIB' > .bin/ranlib
#!/bin/bash
ARGS=()
for arg in "$@"; do
  if [[ "$arg" != "-c" && "$arg" != "c" ]]; then
    ARGS+=("$arg")
  fi
done
/usr/bin/ranlib "${ARGS[@]}"
RANLIB
chmod +x .bin/ranlib
export PATH="$PWD/.bin:$PATH"

exec python3 "$SCRIPT_DIR/run.py" "$@"
