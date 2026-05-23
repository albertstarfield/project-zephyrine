#!/bin/bash
set -e

# Adelaide-Lite Universal Runner & Test Suite Integration
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$BASE_DIR"

export ALIRE_SETTINGS_DIR="$BASE_DIR/alirevenv/settings"
export ALIRE_CACHE_DIR="$BASE_DIR/alirevenv/cache"

# 1. Kill any existing Adelaide-Lite Server running on port 11420
if command -v lsof &>/dev/null; then
    PID=$(lsof -t -i :11420 || true)
    if [ ! -z "$PID" ]; then
        echo "[*] Terminating existing adelaide_server on port 11420 (PID: $PID)..."
        kill -9 $PID || true
        sleep 2
    fi
fi

# 2. Conservative Environment Setup
export MACOSX_DEPLOYMENT_TARGET="16.0"
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
OS_TYPE=$(uname -s)

if [ "$OS_TYPE" = "Darwin" ]; then
    echo "[*] macOS detected. Configuring SDK paths and PATH order..."
    if [ -z "$SDKROOT" ]; then
        if command -v xcrun &>/dev/null; then
            export SDKROOT=$(xcrun --show-sdk-path)
        else
            export SDKROOT="/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk"
        fi
    fi
    export CFLAGS="-isysroot $SDKROOT"
    export C_INCLUDE_PATH="$SDKROOT/usr/include"
    export LIBRARY_PATH="$SDKROOT/usr/lib:/usr/local/lib:/opt/homebrew/lib"
    echo "[*] SDKROOT is set to: $SDKROOT"
else
    echo "[*] Linux/non-macOS detected. Standard environment will be used."
fi

echo "=========================================================="
echo "      Adelaide_Lite Formal Test & Verification Suite      "
echo "=========================================================="

# 3. Auto-configure toolchain if not already done
if [ ! -d "$ALIRE_SETTINGS_DIR" ] || [ ! -d "$ALIRE_CACHE_DIR/toolchains" ]; then
    echo "[*] Initializing isolated Alire toolchain in alirevenv..."
    mkdir -p "$ALIRE_SETTINGS_DIR" "$ALIRE_CACHE_DIR"
    alr -n toolchain --select
fi

# 4. Ensure Alire dependencies (AUnit, strategy)
echo "[*] Ensuring Alire dependencies (AUnit, strategy)..."
cd Adelaide_Lite
alr get aunit --build || true
alr get strategy --build || true
cd ..

# 5. Rebuild llama.cpp if needed
if [ ! -f "llama.cpp/build/src/libllama.a" ]; then
    echo "[!] Building llama.cpp..."
    cd llama.cpp
    rm -rf build && mkdir -p build && cd build
    (
        unset C_INCLUDE_PATH
        unset CPLUS_INCLUDE_PATH
        cmake .. -DBUILD_SHARED_LIBS=OFF \
                 -DCMAKE_OSX_DEPLOYMENT_TARGET="$MACOSX_DEPLOYMENT_TARGET" \
                 -DCMAKE_OSX_SYSROOT="$SDKROOT"
        cmake --build . --config Release --target llama llama-cli llama-server
    )
    cd ../..
fi

# 6. Compile C Helper
echo "[*] Compiling Mach Real-time Scheduling Helper..."
mkdir -p Adelaide_Lite/obj/development
if [ "$OS_TYPE" = "Darwin" ]; then
    clang -c -isysroot "$SDKROOT" Adelaide_Lite/src/scheduling.c -o Adelaide_Lite/obj/development/scheduling.o
else
    gcc -c Adelaide_Lite/src/scheduling.c -o Adelaide_Lite/obj/development/scheduling.o
fi

# 7. Build Adelaide_Lite Test Binary
echo "[*] Building Adelaide_Lite Ada binary (Test target)..."
cd Adelaide_Lite
alr build adelaide_lite
cd ..

# 8. Run Math and Parity Unit Tests
echo "[*] Running Adelaide_Lite Unit Tests..."
if [ -f "Adelaide_Lite/pyvenv/bin/python" ]; then
    echo "[*] Ensuring test dependencies (numpy, requests) are installed..."
    Adelaide_Lite/pyvenv/bin/pip install -q requests numpy sentence-transformers PyMuPDF networkx python-multipart
    Adelaide_Lite/pyvenv/bin/python Adelaide_Lite/python/test_adelaide.py
else
    python3 Adelaide_Lite/python/test_adelaide.py
fi

# 9. Run Alire Tests and Coverage
echo "[*] Running alr test and gnatcov..."
cd Adelaide_Lite
alr test
if command -v gnatcov &>/dev/null; then
    echo "[*] Generating coverage report with gnatcov..."
    gnatcov run --annotate=xcov ./bin/adelaide_lite
else
    echo "[!] gnatcov not found in PATH, skipping coverage."
fi
cd ..

# 10. Run SPARK Formal Proofs (GNATprove)
echo "[*] Running SPARK Formal Verification (Level 2) on Core Units..."
cd Adelaide_Lite
alr exec gnatprove -- -P adelaide_lite.gpr --level=2 -u src/integrity_utils.ads src/math_utils.ads
cd ..

# 11. Check for AFL++ Fuzzing
echo "[*] Checking for AFL++..."
if command -v afl-fuzz &>/dev/null; then
    echo "[+] AFL++ found. You can run fuzzing with:"
    echo "    mkdir -p afl_in afl_out"
    echo "    echo 'similarity 2 0.1 0.2 0.3 0.4' > afl_in/test1"
    echo "    afl-fuzz -i afl_in -o afl_out -- ./Adelaide_Lite/bin/adelaide_lite"
else
    echo "[!] AFL++ not found. Fuzz testing skipped."
fi

# 12. Build Adelaide-Lite Server Binary (Only if tests/proofs passed)
echo "[*] Building Adelaide-Lite Server..."
cd Adelaide_Lite
alr build adelaide_server
cd ..

# 13. Normalizing library indices
echo "[*] Normalizing library indices..."
find "$HOME/.local/share/alire/builds" -name "*.a" -exec chmod +w {} \; 2>/dev/null || true
find "$HOME/.local/share/alire/builds" -name "*.a" -exec /usr/bin/ranlib {} \; 2>/dev/null || true
find "llama.cpp/build" -name "*.a" -exec /usr/bin/ranlib {} \; 2>/dev/null || true

# 14. Launch Adelaide-Lite GUI & Daemon
cd Adelaide_Lite
if [ -f "./bin/adelaide_server" ]; then
    echo "[*] Starting Adelaide-Lite Subsystem (Daemon, Server, UI)..."
    ./run.sh "$@"
else
    echo "[!] Build failed."
    exit 1
fi
