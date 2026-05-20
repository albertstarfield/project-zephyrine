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

echo "=========================================================="
echo "      Adelaide_Lite Formal Test & Verification Suite      "
echo "=========================================================="

# 0. Dependencies check
echo "[*] Ensuring Alire dependencies (AUnit, strategy)..."
cd Adelaide_Lite
alr get aunit --build || true
alr get strategy --build || true
cd ..

# 1. Compile C Helper
echo "[*] Compiling Mach Real-time Scheduling Helper..."
mkdir -p Adelaide_Lite/obj/development
if [ "$OS_TYPE" = "Darwin" ]; then
    clang -c -isysroot "$SDKROOT" Adelaide_Lite/src/scheduling.c -o Adelaide_Lite/obj/development/scheduling.o
else
    gcc -c Adelaide_Lite/src/scheduling.c -o Adelaide_Lite/obj/development/scheduling.o
fi

# 2. Build Ada Binary
echo "[*] Building Adelaide_Lite Ada binary..."
cd Adelaide_Lite
alr build
cd ..

# 3. Run Math and Parity Unit Tests
echo "[*] Running Adelaide_Lite Unit Tests..."
if [ -f "Adelaide_Lite/pyvenv/bin/python" ]; then
    echo "[*] Ensuring test dependencies (numpy, requests) are installed in pyvenv..."
    Adelaide_Lite/pyvenv/bin/pip install -q requests numpy
    Adelaide_Lite/pyvenv/bin/python Adelaide_Lite/python/test_adelaide.py
else
    python3 Adelaide_Lite/python/test_adelaide.py
fi

# 4. Run Alire Tests and Coverage
echo "[*] Running alr test and gnatcov..."
cd Adelaide_Lite
alr test
# Ensure gnatcov is available or handle its absence
if command -v gnatcov &>/dev/null; then
    echo "[*] Generating coverage report with gnatcov..."
    gnatcov run --annotate=xcov ./bin/adelaide_lite
else
    echo "[!] gnatcov not found in PATH, skipping coverage."
fi
cd ..

# 5. Run SPARK Formal Proofs (GNATprove)
echo "[*] Running SPARK Formal Verification (Level 2) on Core Units..."
cd Adelaide_Lite
# Use -u to analyze only the specified units and their dependencies (without analyzing libraries)
alr exec gnatprove -- -P adelaide_lite.gpr --level=2 -u src/integrity_utils.ads src/math_utils.ads
cd ..

# 6. AFL++ Fuzzing placeholder
echo "[*] Checking for AFL++..."
if command -v afl-fuzz &>/dev/null; then
    echo "[+] AFL++ found. You can run fuzzing with:"
    echo "    mkdir -p afl_in afl_out"
    echo "    echo 'similarity 2 0.1 0.2 0.3 0.4' > afl_in/test1"
    echo "    afl-fuzz -i afl_in -o afl_out -- ./Adelaide_Lite/bin/adelaide_lite"
else
    echo "[!] AFL++ not found. Fuzz testing skipped."
fi

echo ""
echo "=========================================================="
echo "  [+] All Tests and Formal Proofs Completed Successfully! "
echo "=========================================================="
