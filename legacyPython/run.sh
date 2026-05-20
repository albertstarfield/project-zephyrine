#!/bin/bash
set -e

# Base directory
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$BASE_DIR"

export ALIRE_SETTINGS_DIR="$BASE_DIR/alirevenv/settings"
export ALIRE_CACHE_DIR="$BASE_DIR/alirevenv/cache"

OS_TYPE=$(uname -s)

# Setup Environment Quirks
if [ "$OS_TYPE" = "Darwin" ]; then
    echo "[*] macOS detected. Configuring SDK paths and PATH order..."
    
    # Prepend /usr/bin to override Homebrew GNU binutils ranlib/ar
    export PATH="/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
    
    # Find macOS SDK
    if [ -z "$SDKROOT" ]; then
        if command -v xcrun &>/dev/null; then
            export SDKROOT=$(xcrun --show-sdk-path)
        else
            export SDKROOT="/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk"
        fi
    fi
    export CFLAGS="-isysroot $SDKROOT"
    export C_INCLUDE_PATH="$SDKROOT/usr/include"
    export LIBRARY_PATH="$SDKROOT/usr/lib"
    export LDFLAGS=""
    
    echo "[*] SDKROOT is set to: $SDKROOT"
else
    echo "[*] Linux/non-macOS detected. Standard GNU environment will be used."
fi

# Auto-configure toolchain if not already done
if [ ! -d "$ALIRE_SETTINGS_DIR" ] || [ ! -d "$ALIRE_CACHE_DIR/toolchains" ]; then
    echo "[*] Initializing isolated Alire toolchain in alirevenv..."
    mkdir -p "$ALIRE_SETTINGS_DIR" "$ALIRE_CACHE_DIR"
    alr -n toolchain --select
fi

# 1. Compile C real-time helper
echo "[*] Compiling Scheduling Helper..."
mkdir -p Adelaide_Lite/obj/development
if [ "$OS_TYPE" = "Darwin" ]; then
    clang -c -isysroot "$SDKROOT" Adelaide_Lite/src/scheduling.c -o Adelaide_Lite/obj/development/scheduling.o
else
    gcc -c Adelaide_Lite/src/scheduling.c -o Adelaide_Lite/obj/development/scheduling.o
fi

# 2. Build Ada/SPARK binary
echo "[*] Building Adelaide_Lite Ada binary..."
cd Adelaide_Lite
alr build
cd ..

# 3. Check arguments
if [ "$1" = "--test" ]; then
    echo "[*] Running test suite..."
    ./TestSuite.sh
    exit 0
elif [ "$1" = "--verify" ]; then
    echo "[*] Running SPARK formal proofs..."
    cd Adelaide_Lite
    alr exec gnatprove -- -P adelaide_lite.gpr --level=2
    cd ..
    exit 0
fi

# 4. Launch both services
echo "=========================================================="
echo "      Starting Adelaide-Lite Dual-Proxy Stack             "
echo "=========================================================="

# Pre-launch cleanup
echo "[*] Cleaning up existing processes..."
pkill -9 -f adelaide_server 2>/dev/null || true
pkill -9 -f ollamaCallModifier.py 2>/dev/null || true
sleep 2

# Trap Ctrl+C for graceful cleanup of both background processes
cleanup() {
    echo -e "\n[*] Terminating services..."
    if [ ! -z "$PY_PID" ]; then
        kill -9 "$PY_PID" 2>/dev/null || true
    fi
    if [ ! -z "$ADA_PID" ]; then
        kill -9 "$ADA_PID" 2>/dev/null || true
    fi
    pkill -9 -f adelaide_server 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM

# Start Python service (Flask agentic backend) on port 11436
echo "[*] Launching Python Agentic Backend (Port 11436)..."
if [ -f "Adelaide_Lite/pyvenv/bin/python" ]; then
    nice -n 0 Adelaide_Lite/pyvenv/bin/python Adelaide_Lite/python/ollamaCallModifier.py --port 11436 "$@" &
    PY_PID=$!
else
    nice -n 0 python3 Adelaide_Lite/python/ollamaCallModifier.py --port 11436 "$@" &
    PY_PID=$!
fi

# Give Python backend a moment, then start Ada server
sleep 5

# Ensure port 11435 is REALLY clear
if lsof -i :11435 >/dev/null; then
    echo "[!] Port 11435 still occupied. Forcing cleanup..."
    pkill -9 -f adelaide_server 2>/dev/null || true
    sleep 2
fi

# Start Ada server (main proxy and semantic cache) on port 11435
echo "[*] Launching Ada AWS Server (Port 11435)..."
./Adelaide_Lite/bin/adelaide_server &
ADA_PID=$!

echo "[+] Both services are running. Press Ctrl+C to stop."
echo "[+] Main endpoint: http://localhost:11435"

# Keep script running to monitor background processes
wait "$ADA_PID" "$PY_PID"
