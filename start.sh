#!/bin/bash
set -e

# Base directory
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$BASE_DIR"

# 1. Environment Setup
export SDKROOT=$(xcrun --show-sdk-path)
export CPATH=$SDKROOT/usr/include
export LIBRARY_PATH="/opt/homebrew/lib"

# 2. Check for llama.cpp libraries
if [ ! -f "llama.cpp/build/src/libllama.a" ]; then
    echo "[!] llama.cpp libraries not found. Building..."
    cd llama.cpp
    mkdir -p build && cd build
    cmake .. -DBUILD_SHARED_LIBS=OFF
    cmake --build . --config Release --target llama llama-cli llama-server
    cd ../..
fi

# 3. Start Adelaide-Lite (unified Ada server)
echo "[*] Starting Adelaide-Lite on port 11420..."
cd Adelaide_Lite
nice -n -20 alr run adelaide_server
