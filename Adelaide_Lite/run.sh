#!/usr/bin/env bash
set -e

BASE_DIR=$(pwd)
echo "[*] Setting up Adelaide-Lite environment in $BASE_DIR..."

# Record start time for WCET
START_TIME=$(python3 -c 'import time; print(int(time.time() * 1000))')

# Generate MD5 of the Ada sources, config, and UI
CURRENT_HASH=$(find src config adelaide_lite.gpr ui/frontend/src ui/frontend/index.html ui/frontend/package.json -type f 2>/dev/null | sort | xargs md5 -q | md5 -q)

DAEMON_BUILD_FLAG=""

if [ ! -f .build_hash ] || [ "$CURRENT_HASH" != "$(cat .build_hash)" ]; then
    echo "[*] Changes detected, checking downloads and rebuilding..."
    
    # Check and clone llama.cpp
    if [ ! -d "$BASE_DIR/../llama.cpp" ]; then
        echo "[*] Cloning llama.cpp..."
        git clone https://github.com/ggerganov/llama.cpp.git "$BASE_DIR/../llama.cpp"
    else
        echo "[*] llama.cpp already exists, skipping clone."
    fi

    # Check and clone supertonic
    if [ ! -d "$BASE_DIR/../supertonic" ]; then
        echo "[*] Cloning supertonic..."
        git clone https://github.com/supertone-inc/supertonic.git "$BASE_DIR/../supertonic"
    else
        echo "[*] supertonic already exists, skipping clone."
    fi

    # Ensure Deno Playwright Chromium is installed
    echo "[*] Installing Playwright Chromium binary for Deno crawler..."
    deno run -A npm:playwright install chromium

    echo "[*] Resolving Ada dependencies and building project..."
    cd "$BASE_DIR"
    export SDKROOT="$(xcrun --show-sdk-path)"
    export CPATH="$SDKROOT/usr/include"
    export LIBRARY_PATH="$SDKROOT/usr/lib"
    
    # Note for future agents: The user strictly wants Alire to use the local alirevenv
    # rather than the global ~/.local/share/alire for caching dependency builds.
    # We can enforce this in the future by setting XDG_DATA_HOME or similar ALIRE env vars.
    alr build
    
    echo "[*] Building Vite Frontend for Sidecar UI..."
    if [ -d "ui/frontend" ]; then
        (cd ui/frontend && npm install && npm run build)
    fi
    
    echo "$CURRENT_HASH" > .build_hash
else
    echo "[*] No changes detected, skipping build."
    DAEMON_BUILD_FLAG="--skip-build"
fi

# Parse arguments
LAUNCH_GUI=true
for arg in "$@"; do
    if [ "$arg" == "--no-gui" ]; then
        LAUNCH_GUI=false
    fi
done

echo "[*] Booting StellaIcarus Ada Daemon Manager..."
python3 python/stellaicarus_daemon_runner.py $DAEMON_BUILD_FLAG &
DAEMON_PID=$!

cleanup() {
    echo "[*] Shutting down background processes..."
    kill $DAEMON_PID 2>/dev/null || true
    if [ -n "$SERVER_PID" ]; then
        kill $SERVER_PID 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo "[*] Booting Adelaide Intelligence Server..."
END_TIME=$(python3 -c 'import time; print(int(time.time() * 1000))')
echo "[*] Startup completed in $((END_TIME - START_TIME))ms (WCET)"

if [ "$LAUNCH_GUI" = true ]; then
    ./bin/adelaide_server &
    SERVER_PID=$!
    echo "[*] Booting Python Sidecar UI..."
    cd ui && python3 sidecar_ui.py
else
    ./bin/adelaide_server
fi
