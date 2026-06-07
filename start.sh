#!/bin/bash
# start.sh - Entry point for Adelaide Intelligence Platform
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "[*] Starting Adelaide Intelligence Platform..."

# Ensure we are in the project root
cd "$BASE_DIR"

# Check if dependencies are met (optional, but good practice)
if ! command -v python3 &> /dev/null; then
    echo "[!] Error: python3 is not installed."
    exit 1
fi

if ! command -v alr &> /dev/null; then
    echo "[!] Warning: alr (Alire) not found. Build might fail if not already built."
fi

# Launch the main run script
# Default to --no-gui if no arguments are provided and we are not in a terminal
if [ -z "$1" ] && [ ! -t 0 ]; then
    ./run.sh --no-gui
else
    ./run.sh "$@"
fi
