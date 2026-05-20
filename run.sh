#!/bin/bash
set -e

# Adelaide-Lite Universal Runner (Final Fix Attempt)
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$BASE_DIR"

# 1. Environment Lockdown for macOS
export SDKROOT=$(xcrun --show-sdk-path)
export C_INCLUDE_PATH="$SDKROOT/usr/include:/opt/homebrew/include"
export CPLUS_INCLUDE_PATH="$SDKROOT/usr/include:/opt/homebrew/include"
export LIBRARY_PATH="$SDKROOT/usr/lib:/usr/local/lib:/opt/homebrew/lib"
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"

echo "[*] Initializing Adelaide-Lite Environment..."

# 2. Build llama.cpp if needed
if [ ! -f "llama.cpp/build/src/libllama.a" ]; then
    echo "[!] llama.cpp libraries not found. Building..."
    cd llama.cpp
    mkdir -p build && cd build
    cmake .. -DBUILD_SHARED_LIBS=OFF -DCMAKE_OSX_SYSROOT="$SDKROOT"
    cmake --build . --config Release --target llama llama-cli llama-server
    cd ../..
fi

# 3. Fix Library Indices (Force permissions)
echo "[*] Ensuring library indices are correct..."
find "$HOME/.local/share/alire/builds" -name "*.a" -exec chmod +w {} \; 2>/dev/null || true
find "$HOME/.local/share/alire/builds" -name "*.a" -exec /usr/bin/ranlib {} \; 2>/dev/null || true

# 4. Build Ada Server
echo "[*] Building Adelaide-Lite Server..."
cd Adelaide_Lite
alr -n build

# 5. Run Server
if [ -f "./bin/adelaide_server" ]; then
    echo "[*] Starting Adelaide-Lite Server on port 11420..."
    nice -n -20 ./bin/adelaide_server
else
    echo "[!] Build failed."
    exit 1
fi
