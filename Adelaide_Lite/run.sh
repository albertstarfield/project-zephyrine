#!/usr/bin/env bash
set -e

echo "[*] Setting up Adelaide-Lite environment..."

# Check and clone llama.cpp
if [ ! -d "../llama.cpp" ]; then
    echo "[*] Cloning llama.cpp..."
    git clone https://github.com/ggerganov/llama.cpp.git ../llama.cpp
else
    echo "[*] llama.cpp already exists, skipping clone."
fi

# Check and clone supertonic
if [ ! -d "../supertonic" ]; then
    echo "[*] Cloning supertonic..."
    git clone https://github.com/supertone-inc/supertonic.git ../supertonic
else
    echo "[*] supertonic already exists, skipping clone."
fi

# Ensure Deno Playwright Chromium is installed
echo "[*] Installing Playwright Chromium binary for Deno crawler..."
deno run -A npm:playwright install chromium

echo "[*] Resolving Ada dependencies and building project..."
alr build

echo "[*] Booting StellaIcarus Ada Daemon Manager..."
python3 python/stellaicarus_daemon_runner.py &
DAEMON_PID=$!

cleanup() {
    echo "[*] Shutting down daemon..."
    kill $DAEMON_PID 2>/dev/null || true
    wait $DAEMON_PID 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[*] Booting Adelaide Intelligence Server..."
alr run
