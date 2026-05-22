#!/usr/bin/env bash
set -e

echo "[*] Setting up Adelaide-Lite environment..."

# Check and clone llama.cpp
if [ ! -d "llama.cpp" ]; then
    echo "[*] Cloning llama.cpp..."
    git clone https://github.com/ggerganov/llama.cpp.git
else
    echo "[*] llama.cpp already exists, skipping clone."
fi

# Check and clone supertonic
if [ ! -d "supertonic" ]; then
    echo "[*] Cloning supertonic..."
    git clone https://github.com/supertone-inc/supertonic.git
else
    echo "[*] supertonic already exists, skipping clone."
fi

echo "[*] Resolving Ada dependencies and building project..."
alr build

echo "[*] Booting Adelaide Intelligence Server..."
alr run
