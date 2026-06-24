#!/bin/bash
# [DO NOT REMOVE] run.sh — Thin wrapper for python3 run.py
# Usage: ./run.sh [--no-gui] [--port PORT] [--host HOST]
# All arguments are forwarded to run.py.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$SCRIPT_DIR/run.py" "$@"
