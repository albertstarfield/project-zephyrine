#!/bin/bash
# [DO NOT REMOVE] run.sh — Thin wrapper for python3 run.py
# Usage: ./run.sh [--no-gui] [--port PORT] [--host HOST]
# All arguments are forwarded to run.py.

set -euo pipefail

# macOS OpenMP fix: PyTorch and numpy BLAS both load libomp.dylib — they clash
export KMP_DUPLICATE_LIB_OK=TRUE

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/src/python:${PYTHONPATH:-}"

if [[ "$(uname -s)" == "Darwin" ]]; then
  export SDKROOT="$(xcrun --show-sdk-path)"
  export CPATH="${SDKROOT}/usr/include"
  export C_INCLUDE_PATH="${SDKROOT}/usr/include"
  export LIBRARY_PATH="${SDKROOT}/usr/lib"
fi

mkdir -p "$SCRIPT_DIR/.bin"
cat << 'RANLIB' > "$SCRIPT_DIR/.bin/ranlib"
#!/bin/bash
ARGS=()
for arg in "$@"; do
  if [[ "$arg" != "-c" && "$arg" != "c" ]]; then
    ARGS+=("$arg")
  fi
done
/usr/bin/ranlib "${ARGS[@]}"
RANLIB
chmod +x "$SCRIPT_DIR/.bin/ranlib"
export PATH="$SCRIPT_DIR/.bin:$PATH"

exec python3 -u "$SCRIPT_DIR/run.py" "$@"
