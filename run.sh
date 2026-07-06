#!/bin/bash
# Universal Shim for AdelaideZephyrineSystem
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$BASE_DIR/AdelaideZephyrineSystem"
python3 run.py "$@"
