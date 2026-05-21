#!/bin/bash
set -e

# Adelaide-Lite Universal Runner (Compiler-Safe Fix)
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$BASE_DIR"

# 1. Conservative Environment Setup
# We use 16.0 as the floor for Alire compatibility.
export MACOSX_DEPLOYMENT_TARGET="16.0"
export SDKROOT=$(xcrun --show-sdk-path)

# DO NOT set C_INCLUDE_PATH or CPLUS_INCLUDE_PATH as they break C++ stdlib lookups on macOS.
# Instead, let CMake/Clang find the headers using the SDKROOT.
unset C_INCLUDE_PATH
unset CPLUS_INCLUDE_PATH
export LIBRARY_PATH="/usr/lib:/usr/local/lib:/opt/homebrew/lib"
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"

echo "[*] Initializing Adelaide-Lite (Target: macOS $MACOSX_DEPLOYMENT_TARGET)..."

# 2. Rebuild llama.cpp if needed
if [ ! -f "llama.cpp/build/src/libllama.a" ]; then
    echo "[!] Building llama.cpp..."
    cd llama.cpp
    rm -rf build && mkdir -p build && cd build
    cmake .. -DBUILD_SHARED_LIBS=OFF \
             -DCMAKE_OSX_DEPLOYMENT_TARGET="$MACOSX_DEPLOYMENT_TARGET" \
             -DCMAKE_OSX_SYSROOT="$SDKROOT"
    cmake --build . --config Release --target llama llama-cli llama-server
    cd ../..
fi

# 3. Run Test Suite
echo "[*] Running Adelaide_Lite Test Suite..."
./TestSuite.sh

# 4. Clean and Fix Dependencies
echo "[*] Normalizing library indices..."
find "$HOME/.local/share/alire/builds" -name "*.a" -exec chmod +w {} \; 2>/dev/null || true
find "$HOME/.local/share/alire/builds" -name "*.a" -exec /usr/bin/ranlib {} \; 2>/dev/null || true
find "llama.cpp/build" -name "*.a" -exec /usr/bin/ranlib {} \; 2>/dev/null || true

# 5. Run Server
cd Adelaide_Lite
if [ -f "./bin/adelaide_server" ]; then
    echo "[*] Starting Adelaide-Lite Server on port 11420..."
    nice -n -20 ./bin/adelaide_server || ./bin/adelaide_server
else
    echo "[!] Build failed."
    exit 1
fi
