#!/bin/bash
set -e

# Delegate everything to the unified run.sh
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
exec "$BASE_DIR/run.sh" "$@"
