#!/bin/bash
# Universal Shim for Adelaide_Lite
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$BASE_DIR/Adelaide_Lite"
python3 run.py "$@"
