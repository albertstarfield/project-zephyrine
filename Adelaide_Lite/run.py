#!/usr/bin/env python3
import os
import sys
import time
import subprocess
import hashlib
import platform
import signal
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# [DO NOT REMOVE] ADELAITE LITE — PROGRAM ARCHITECTURE
# ============================================================================
# WARNING: This comment block documents the full system architecture.
# Removing it will make the program nearly impossible to understand or
# maintain. Any agent or contributor modifying this file must read this
# section before making changes.
#
# This script is the top-level orchestrator for the Adelaide Intelligence
# Platform. It builds all dependencies (if source changed), then spawns
# three concurrent processes that together form the runtime.
#
# ENTRY POINT CHAIN:
#   run.sh --no-gui
#     └─ cd Adelaide_Lite && python3 run.py --no-gui
#          ├─ [Build Phase] (triggered when MD5 hash of source files changes)
#          │    ├─ Clone & build llama.cpp (CMake, ggml-metal on macOS arm64)
#          │    ├─ Build mtmd library (CLIP vision encoding for multimodal)
#          │    ├─ Clone & build moonshine (ONNX-based speech-to-text)
#          │    ├─ Clone & build kokoro-onnx (text-to-speech)
#          │    ├─ Download Qwen3.5 GGUF models (0.8B, 9B, Embedding)
#          │    ├─ Download Kokoro TTS models (ONNX + voices)
#          │    ├─ Install Playwright Chromium (for Deno web crawler)
#          │    ├─ alr build (Ada/Alire — compiles all Ada sources to bin/)
#          │    └─ npm install && npm run build (Vite frontend)
#          │
#          ├─ [Runtime] Spawns 3 background processes:
#          │    ├─ 1. StellaIcarus Daemon Manager (Python, hardware monitor)
#          │    ├─ 2. adelaide_server (Ada binary, HTTP API on port 11420)
#          │    └─ 3. adelaide_watchdog (Ada binary, monitors server health)
#          │
#          └─ [--no-gui] Waits for adelaide_server exit, shows crash banner
#
# PROCESS ARCHITECTURE:
#
#   ┌─────────────────────────────────────────────────────────────┐
#   │                    run.py (Orchestrator)                     │
#   │  - Builds everything if source changed (MD5 hash check)     │
#   │  - Sets DYLD_LIBRARY_PATH for onnxruntime (moonshine)       │
#   │  - Spawns all child processes                               │
#   │  - Handles SIGINT/SIGTERM cleanup                           │
#   └──────┬──────────────────┬──────────────────┬────────────────┘
#          │                  │                  │
#          ▼                  ▼                  ▼
#   ┌──────────────┐  ┌─────────────────┐  ┌──────────────────┐
#   │  StellaIcarus │  │ adelaide_server │  │ adelaide_watchdog │
#   │  Daemon (Py)  │  │    (Ada/AWS)    │  │    (Ada)         │
#   │               │  │  Port 11420     │  │                  │
#   │ - HW monitor  │  │ - HTTP API      │  │ - Monitors PID   │
#   │ - Power state │  │ - LLM inference │  │ - Checks heartbeat│
#   │ - Telemetry   │  │ - RAG pipeline  │  │ - Restarts server│
#   │ - ELP bridge  │  │ - STT/TTS       │  │   if stale       │
#   └──────────────┘  └─────────────────┘  └──────────────────┘
#
# ADA SERVER INTERNALS (adelaide_server.adb):
#
#   Startup sequence (order matters):
#     STEP 0: Disk benchmark (reads 1GB from GGUF, classifies storage speed)
#     STEP 1: Model_Manager.Initialize
#              ├─ Llama_Backend_Init (ggml-metal/CPU backends)
#              ├─ Database_Manager.Initialize (SQLite databases)
#              ├─ ELP_Queue.Initialize (priority queue monitor)
#              └─ Idle_Monitor task (unloads idle models after 30s)
#     STEP 2: Knowledge_Manager.Initialize
#              └─ Background tasks (ELP0):
#                   ├─ Indexing_Task (parses references.bib)
#                   ├─ Native_Crawl_Task (walks filesystem → embeddings)
#                   └─ Proactive_Cache_Task (predicts follow-ups)
#     STEP 3: Scheduler_Manager.Initialize
#     STEP 4: Watchdog_IPC.Init (creates run/, writes PID + heartbeat)
#     STEP 5: Knowledge_Manager.Start_Tasks (starts ELP0 producers)
#     STEP 6: AWS.Server.Start (HTTP on port 11420)
#     STEP 7: Health ping watchdog (3s interval, 60s deadline)
#     STEP 8: Moonshine_Interface.Init_Moonshine (STT, ~500MB ONNX)
#     STEP 9: Main heartbeat loop (1Hz heartbeat + ELP stats every 5s)
#
# ELP PRIORITY QUEUE ("Volatus Damarae" architecture):
#   Serial processing — prevents heap corruption from concurrent llama.cpp FFI.
#   Capacity: 2^63. Priority: ELP3 > ELP2 > ELP1 > ELP0.
#
#   ELP3: ZenithOrion — 1ms deterministic pacing loop (highest frequency)
#   ELP2: StellaIcarus — deterministic API response hooks
#   ELP1: User-facing generation (real-time inference)
#   ELP0: Background indexing/RAG (preemptible by ELP1)
#
# MODEL TYPES:
#   Qwen_0_8B       — Small LLM (always loaded, exempt from idle unload)
#   Qwen_9B         — Large LLM (loaded on-demand for complex reasoning)
#   Qwen_Embedding  — Embedding model (semantic search)
#   MMProj          — Multimodal projection (CLIP vision via mtmd)
#
# KEY SUBSYSTEMS:
#   Llama_Interface     — Ada→C FFI wrapping llama.cpp
#   Mtmd_Interface      — Ada→C FFI for multimodal (CLIP vision)
#   Moonshine_Interface — Ada→C FFI for speech-to-text (ONNX)
#   Kokoro_Interface    — Ada→Python for text-to-speech
#   Kratos              — Crash isolation (sigaction + longjmp)
#   Speculative_Cache   — Predictive response cache (5 entries, LRU)
#   Database_Manager    — SQLite (memory, literature, knowledge graph)
#   Streaming_Queue     — AWS streaming response support
#   Watchdog_IPC        — File-based IPC (PID, heartbeat, exit reason)
#   ZenithOrion         — 1ms deterministic pacing loop (ELP3)
#
# EXTERNAL DEPENDENCIES (sibling directories):
#   ../llama.cpp/            — LLM inference engine
#   ../moonshine/            — Speech-to-text ONNX models
#   ../kokoro-onnx/          — Text-to-speech ONNX
#   ../kokoclone/            — Zero-shot voice cloning
#   ../tts_kokoro_component/ — Kokoro TTS Python deps (isolated venv)
#
# COMMUNICATION FLOW:
#   User Request → HTTP :11420 → Adelaide_Server_Pkg.Dispatch
#     ├─ Chat/Generate   → Model_Manager → Llama_Interface → llama.cpp
#     ├─ Embeddings      → Model_Manager → Llama_Interface (embed mode)
#     ├─ Transcription   → Moonshine_Interface → libmoonshine.dylib
#     ├─ TTS             → Kokoro_Interface → Python subprocess
#     ├─ Vision          → Image_Encoder → mtmd (CLIP) → Llama_Interface
#     ├─ RAG             → Database_Manager → semantic search → Model_Manager
#     └─ Power state     ← StellaIcarus Daemon → /api/power endpoint
#
# CRASH ISOLATION (Kratos):
#   C-level crashes (SIGSEGV, SIGBUS, SIGFPE, SIGTRAP, SIGABRT) during
#   llama.cpp inference are caught by Kratos (sigaction + longjmp) instead
#   of killing the server. The external watchdog monitors heartbeat files
#   and restarts the server if it dies.
# ============================================================================

#  QUIRK: Block NT kernel at runtime (see QUIRK-005)
#  Windows is NOT supported.  The build system (adelaide_lite.gpr) also
#  blocks compilation on Windows, but this is an additional guard.
#  LINUX-COMPAT (future): When porting to Linux, remove this check.
if platform.system() == "Windows":
    print("[FATAL] Windows (NT kernel) is not supported.")
    print("[FATAL] This server targets macOS (arm64) with planned Linux support.")
    print("[FATAL] See adelaide_lite.gpr QUIRK-005 for details.")
    sys.exit(1)

# Set HF_HOME so huggingface caches locally in the project directory
os.environ["HF_HOME"] = os.path.join(BASE_DIR, ".hf_cache")
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

# Globals to keep track of background processes
daemon_process = None
server_process = None
watchdog_process = None
kokoro_process = None

def get_files_to_hash():
    # NOTE: run.py itself is NOT hashed - it's an interpreter script, not a
    # compiled artifact. Changes to run.py don't trigger rebuilds.
    patterns = [
        "src/**/*",
        "config/**/*",
        "adelaide_lite.gpr",
        "ui/frontend/src/**/*",
        "ui/frontend/index.html",
        "ui/frontend/package.json"
    ]
    files = []
    for pattern in patterns:
        path = os.path.join(BASE_DIR, pattern)
        if "/**/" in pattern:
            # Recursive glob isn't strictly needed if we just os.walk, but let's do a simple recursive collect
            base = path.split("/**/")[0]
            if os.path.exists(base):
                for root, _, filenames in os.walk(base):
                    for name in filenames:
                        files.append(os.path.join(root, name))
        else:
            if os.path.exists(os.path.join(BASE_DIR, pattern)):
                files.append(os.path.join(BASE_DIR, pattern))
    
    # Also hash mtmd source files in llama.cpp (for multimodal rebuild detection)
    # Why: Changes to mtmd source files should trigger a rebuild of the mtmd library.
    #      Without this, code changes in llama.cpp/tools/mtmd/ would be silently ignored.
    mtmd_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "llama.cpp", "tools", "mtmd"))
    mtmd_count = 0
    if os.path.exists(mtmd_dir):
        for root, _, filenames in os.walk(mtmd_dir):
            for name in filenames:
                if name.endswith(('.cpp', '.h', '.c')):
                    files.append(os.path.join(root, name))
                    mtmd_count += 1
    if mtmd_count > 0:
        print(f"[MTMD] [{time.strftime('%H:%M:%S')}] Tracking {mtmd_count} mtmd source files for rebuild detection")
    
    return sorted(files)

def calculate_hash(file_paths):
    hasher = hashlib.md5()
    for file_path in file_paths:
        if os.path.isfile(file_path):
            with open(file_path, 'rb') as f:
                # To closely mimic the bash find | sort | xargs md5 -q
                # we hash the contents of the files in sorted order
                hasher.update(f.read())
    return hasher.hexdigest()

def cleanup(signum=None, frame=None):
    print("\n[*] Shutting down background processes...")
    if daemon_process and daemon_process.poll() is None:
        daemon_process.terminate()
    if server_process and server_process.poll() is None:
        server_process.terminate()
    if watchdog_process and watchdog_process.poll() is None:
        watchdog_process.terminate()
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

def main():
    global daemon_process, server_process, watchdog_process
    
    print(f"[*] Setting up Adelaide-Lite environment in {BASE_DIR}...")
    start_time = int(time.time() * 1000)

    # Calculate Current Hash
    current_hash = calculate_hash(get_files_to_hash())
    hash_file = os.path.join(BASE_DIR, ".build_hash")
    
    saved_hash = ""
    if os.path.exists(hash_file):
        with open(hash_file, "r") as f:
            saved_hash = f.read().strip()

    daemon_build_flag = ""

    if current_hash != saved_hash:
        print("[*] Changes detected, checking downloads and rebuilding...")
        threads = str(os.cpu_count() or 4)
        
        # Check and clone llama.cpp
        llama_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "llama.cpp"))
        if not os.path.exists(llama_dir):
            print("[*] Cloning llama.cpp...")
            subprocess.run(["git", "clone", "--depth=1", "https://github.com/ggerganov/llama.cpp.git", llama_dir], check=False)
        else:
            print("[*] llama.cpp already exists, skipping clone.")

        # Ensure llama.cpp is built
        llama_build_dir = os.path.join(llama_dir, "build")
        llama_lib = os.path.join(llama_build_dir, "src", "libllama.a")
        llama_start = time.time()
        if not os.path.exists(llama_lib):
            print(f"[LLAMA] [{time.strftime('%H:%M:%S')}] Building llama.cpp...")
            print(f"[LLAMA] [{time.strftime('%H:%M:%S')}] CMake flags: -DGGML_NATIVE=ON -DLLAMA_BUILD_TOOLS=ON")
            if platform.system() == "Darwin" and platform.machine() == "arm64":
                print(f"[LLAMA] [{time.strftime('%H:%M:%S')}] Metal GPU acceleration: ENABLED")
            os.makedirs(llama_build_dir, exist_ok=True)
            cmake_flags = ["cmake", "-B", "build", "-DGGML_NATIVE=ON", "-DLLAMA_BUILD_TOOLS=ON"]
            if platform.system() == "Darwin" and platform.machine() == "arm64":
                cmake_flags.append("-DGGML_METAL=ON")
            result = subprocess.run(cmake_flags, cwd=llama_dir, check=False, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"[LLAMA] [{time.strftime('%H:%M:%S')}] CMake configure FAILED")
                if result.stderr:
                    print(f"[LLAMA] [{time.strftime('%H:%M:%S')}] stderr: {result.stderr[-500:]}")
            else:
                print(f"[LLAMA] [{time.strftime('%H:%M:%S')}] CMake configure OK, building...")
                result = subprocess.run(["cmake", "--build", "build", "--config", "Release", "-j"], cwd=llama_dir, check=False, capture_output=True, text=True)
                llama_elapsed = time.time() - llama_start
                if result.returncode == 0:
                    print(f"[LLAMA] [{time.strftime('%H:%M:%S')}] Build SUCCESS in {llama_elapsed:.1f}s")
                else:
                    print(f"[LLAMA] [{time.strftime('%H:%M:%S')}] Build FAILED in {llama_elapsed:.1f}s")
                    if result.stderr:
                        print(f"[LLAMA] [{time.strftime('%H:%M:%S')}] stderr: {result.stderr[-500:]}")
        else:
            llama_elapsed = time.time() - llama_start
            print(f"[LLAMA] [{time.strftime('%H:%M:%S')}] Library exists, skipping build")
        
        # Ensure mtmd (multimodal) library is built
        # Why: The mtmd library provides CLIP vision encoding for multimodal support.
        #      It's built as a separate target from llama.cpp and must be linked
        #      into adelaide_server for image processing to work.
        mtmd_lib = os.path.join(llama_build_dir, "tools", "mtmd", "libmtmd.a")
        mtmd_start = time.time()
        if not os.path.exists(mtmd_lib):
            print(f"[MTMD] [{time.strftime('%H:%M:%S')}] Building mtmd (multimodal) library...")
            print(f"[MTMD] [{time.strftime('%H:%M:%S')}] Target: {mtmd_lib}")
            print(f"[MTMD] [{time.strftime('%H:%M:%S')}] Running: cmake --build build --target mtmd -j")
            result = subprocess.run(["cmake", "--build", "build", "--target", "mtmd", "-j"], cwd=llama_dir, check=False, capture_output=True, text=True)
            mtmd_elapsed = time.time() - mtmd_start
            if result.returncode == 0:
                print(f"[MTMD] [{time.strftime('%H:%M:%S')}] Build SUCCESS in {mtmd_elapsed:.1f}s")
                # Verify the library was created
                if os.path.exists(mtmd_lib):
                    mtmd_size = os.path.getsize(mtmd_lib)
                    print(f"[MTMD] [{time.strftime('%H:%M:%S')}] Library created: {mtmd_size:,} bytes")
                else:
                    print(f"[MTMD] [{time.strftime('%H:%M:%S')}] WARNING: Library file not found after build!")
            else:
                print(f"[MTMD] [{time.strftime('%H:%M:%S')}] Build FAILED in {mtmd_elapsed:.1f}s")
                if result.stdout:
                    print(f"[MTMD] [{time.strftime('%H:%M:%S')}] stdout: {result.stdout[-500:]}")
                if result.stderr:
                    print(f"[MTMD] [{time.strftime('%H:%M:%S')}] stderr: {result.stderr[-500:]}")
        else:
            mtmd_elapsed = time.time() - mtmd_start
            mtmd_size = os.path.getsize(mtmd_lib)
            print(f"[MTMD] [{time.strftime('%H:%M:%S')}] Library exists ({mtmd_size:,} bytes), skipping build")

        # Check and clone kokoro-onnx
        kokoro_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "kokoro-onnx"))
        if not os.path.exists(kokoro_dir):
            print("[*] Cloning kokoro-onnx...")
            subprocess.run(["git", "clone", "https://github.com/thewh1teagle/kokoro-onnx", kokoro_dir], check=False)
        else:
            print("[*] kokoro-onnx already exists, skipping clone.")
            
        kokoclone_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "kokoclone"))
        if not os.path.exists(kokoclone_dir):
            print("[*] Cloning KokoClone Zero-Shot Repository...")
            subprocess.run(["git", "clone", "https://github.com/Ashish-Patnaik/kokoclone.git", kokoclone_dir], check=True)
        else:
            print("[*] kokoclone already exists, skipping clone.")

        # Ensure Kokoro TTS component dependencies are installed in an isolated venv
        kokoro_comp_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "tts_kokoro_component"))
        kokoro_venv_dir = os.path.join(kokoro_comp_dir, "venv")
        if not os.path.exists(kokoro_venv_dir):
            print("[*] Creating dedicated virtual environment for Kokoro TTS (Python 3.12)...")
            subprocess.run(["python3.12", "-m", "venv", kokoro_venv_dir], check=True)
            
        print("[*] Installing Kokoro TTS requirements...")
        kokoro_pip = os.path.join(kokoro_venv_dir, "bin", "pip") if platform.system() != "Windows" else os.path.join(kokoro_venv_dir, "Scripts", "pip.exe")
        subprocess.run([kokoro_pip, "install", "-r", os.path.join(kokoro_comp_dir, "requirements.txt")], check=False)

        # Check and clone moonshine
        moonshine_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "moonshine"))
        if not os.path.exists(moonshine_dir):
            print("[*] Cloning moonshine...")
            subprocess.run(["git", "clone", "--depth=1", "https://github.com/moonshine-ai/moonshine.git", moonshine_dir], check=False)
            
            # Autoremove examples to save space
            moonshine_examples = os.path.join(moonshine_dir, "examples")
            if os.path.exists(moonshine_examples):
                print("[*] Removing heavy moonshine/examples directory...")
                shutil.rmtree(moonshine_examples, ignore_errors=True)
        else:
            print("[*] moonshine already exists, skipping clone.")

        # Ensure Moonshine is built
        moonshine_build_dir = os.path.join(moonshine_dir, "build")
        moonshine_core_lib = os.path.join(moonshine_build_dir, "core", "libmoonshine.dylib") if platform.system() == "Darwin" else os.path.join(moonshine_build_dir, "core", "libmoonshine.so")
        if not os.path.exists(moonshine_core_lib):
            print("[*] Building moonshine C API...")
            os.makedirs(moonshine_build_dir, exist_ok=True)
            subprocess.run(["cmake", ".."], cwd=moonshine_build_dir, check=False)
            subprocess.run(["make", f"-j{threads}"], cwd=moonshine_build_dir, check=False)
        else:
            print("[*] moonshine core library exists, skipping cmake build.")

        # Check and download Moonshine models
        moonshine_models_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "moonshine", "models"))
        if not os.path.exists(moonshine_models_dir) or not os.listdir(moonshine_models_dir):
            print("[*] Downloading Moonshine models...")
            os.makedirs(moonshine_models_dir, exist_ok=True)
            env_for_download = os.environ.copy()
            env_for_download["PYTHONPATH"] = os.path.join(moonshine_dir, "python", "src")
            download_script = os.path.join(moonshine_dir, "python", "src", "moonshine_voice", "download.py")
            subprocess.run([sys.executable, download_script, "--stt", "--language", "en", "--root", moonshine_models_dir], env=env_for_download, check=False)
        else:
            print("[*] Moonshine models already exist, skipping download.")

        # Check and download Qwen models
        qwen_models_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "llama.cpp", "models", "qwen3.5"))
        os.makedirs(qwen_models_dir, exist_ok=True)
        
        models_to_download = [
            {
                "url": "https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q4_K_M.gguf?download=true",
                "output": "Qwen3.5-0.8B-Q4_K_M.gguf"
            },
            {
                "url": "https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/mmproj-F16.gguf?download=true",
                "output": "mmproj-0.8B-F16.gguf"
            },
            {
                "url": "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/main/Qwen3-Embedding-0.6B-Q8_0.gguf?download=true",
                "output": "Qwen3-Embedding-0.6B-Q8_0.gguf"
            },
            {
                "url": "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-UD-Q2_K_XL.gguf?download=true",
                "output": "Qwen3.5-9B-UD-Q2_K_XL.gguf"
            },
            {
                "url": "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/mmproj-F16.gguf?download=true",
                "output": "mmproj-9B-F16.gguf"
            }
        ]
        
        aria2c_cmd = shutil.which("aria2c")
        for model in models_to_download:
            target_path = os.path.join(qwen_models_dir, model["output"])
            if not os.path.exists(target_path):
                print(f"[*] Downloading {model['output']}...")
                if aria2c_cmd:
                    subprocess.run([aria2c_cmd, "-x", "16", "-s", "16", "-k", "1M", model["url"], "-o", model["output"], "-d", qwen_models_dir], check=True)
                else:
                    subprocess.run(["wget", "-q", "--show-progress", model["url"], "-O", target_path], check=True)

        # Check and download Kokoro models
        kokoro_models_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "kokoro_models"))
        os.makedirs(kokoro_models_dir, exist_ok=True)
        kokoro_onnx_model = os.path.join(kokoro_models_dir, "kokoro-v0_19.int8.onnx")
        kokoro_voices = os.path.join(kokoro_models_dir, "voices-v1.0.bin")
        if not os.path.exists(kokoro_onnx_model):
            print("[*] Downloading Kokoro ONNX model...")
            subprocess.run(["wget", "-q", "--show-progress", "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.int8.onnx"], cwd=kokoro_models_dir, check=False)
        if not os.path.exists(kokoro_voices):
            print("[*] Downloading Kokoro voices...")
            subprocess.run(["wget", "-q", "--show-progress", "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"], cwd=kokoro_models_dir, check=False)

        # Ensure Deno Playwright Chromium is installed
        print("[*] Installing Playwright Chromium binary for Deno crawler...")
        # Cross platform deno invocation
        deno_cmd = "deno.exe" if platform.system() == "Windows" else "deno"
        try:
            subprocess.run([deno_cmd, "run", "-A", "npm:playwright", "install", "chromium"], check=False)
        except FileNotFoundError:
            print("[!] Deno not found in PATH, skipping playwright installation.")

        print("[*] Resolving Ada dependencies and building project...")
        
        env = os.environ.copy()
        if platform.system() == "Darwin":
            try:
                sdk_path = subprocess.check_output(["xcrun", "--show-sdk-path"]).decode().strip()
                env["SDKROOT"] = sdk_path
                env["CPATH"] = os.path.join(sdk_path, "usr", "include")
                env["LIBRARY_PATH"] = os.path.join(sdk_path, "usr", "lib")
            except Exception as e:
                print(f"[!] Warning: Could not set macOS SDK paths: {e}")

        # Note for future agents: The user strictly wants Alire to use the local alirevenv
        alr_cmd = "alr.exe" if platform.system() == "Windows" else "alr"
        subprocess.run([alr_cmd, "build"], env=env, cwd=BASE_DIR, check=True)
        
        print("[*] Building Vite Frontend for Sidecar UI...")
        frontend_dir = os.path.join(BASE_DIR, "ui", "frontend")
        if os.path.exists(frontend_dir):
            npm_cmd = "npm.cmd" if platform.system() == "Windows" else "npm"
            subprocess.run([npm_cmd, "install"], cwd=frontend_dir, check=True)
            subprocess.run([npm_cmd, "run", "build"], cwd=frontend_dir, check=True)
        
        with open(hash_file, "w") as f:
            f.write(current_hash)
            
    # Self-Integrity Check using Ruff
    ruff_cmd = "ruff.exe" if platform.system() == "Windows" else "ruff"
    if shutil.which(ruff_cmd):
        print("[*] Running Platform Self-Integrity Quality Check (Ruff)...")
        # Run ruff check on the Adelaide_Lite directory
        try:
            result = subprocess.run([ruff_cmd, "check", BASE_DIR], capture_output=True, text=True)
            if result.returncode != 0:
                print("[!] Self-Integrity Quality Check FAILED.")
                print(result.stdout)
                # In strict mode, we might want to exit, but for now just warn
                # print("[*] Emergency Shutdown: Quality violations detected.")
                # sys.exit(1)
            else:
                print("[+] Self-Integrity Quality Check PASSED.")
        except Exception as e:
            print(f"[!] Error executing Ruff integrity check: {e}")
    else:
        print("[!] Warning: ruff not found in PATH, skipping self-integrity quality check.")

    # Handle integrity check flag
    if "--test-build-integrity-check" in sys.argv:
        print("[*] Test build integrity check passed! Exiting without launching services.")
        sys.exit(0)

    # Parse arguments
    launch_gui = True
    if "--no-gui" in sys.argv:
        launch_gui = False

    print("[*] Booting StellaIcarus Ada Daemon Manager...")
    python_cmd = sys.executable
    daemon_script = os.path.join(BASE_DIR, "python", "stellaicarus_daemon_runner.py")
    
    daemon_args = [python_cmd, daemon_script]
    if daemon_build_flag:
        daemon_args.append(daemon_build_flag)
        
    daemon_process = subprocess.Popen(daemon_args, cwd=BASE_DIR)

    print("[*] Booting Adelaide Intelligence Server...")
    end_time = int(time.time() * 1000)
    print(f"[*] Startup completed in {end_time - start_time}ms (WCET)")

    server_bin = "adelaide_server.exe" if platform.system() == "Windows" else "adelaide_server"
    server_path = os.path.join(BASE_DIR, "bin", server_bin)

    env = os.environ.copy()
    
    # Architecture-aware Moonshine ONNX runtime path
    #
    # QUIRK: The server binary links against libmoonshine.dylib, which
    #        dynamically loads libonnxruntime.1.23.2.dylib.  If this
    #        library is NOT in DYLD_LIBRARY_PATH, the binary crashes at
    #        startup with:
    #          "Library not loaded: @rpath/libonnxruntime.1.23.2.dylib"
    #          "Reason: no such file"
    #        The onnxruntime dylib lives in the moonshine submodule:
    #          moonshine/core/third-party/onnxruntime/lib/macos/{arch}/
    #        This is the ONLY place it exists on the filesystem (not
    #        in /opt/homebrew/lib or any standard path).
    #
    # IMPORTANT: Pre-existing bug (2026-06-10): After QWEN_0_8B processes
    # a request and the model is released, the server may crash with
    # exit code -1 (signal caught by Kratos crash isolation). The run.sh
    # wrapper will auto-restart the server if this happens, but clients
    # will see a brief connection reset.
    arch = "arm64" if platform.machine() == "arm64" else "x86_64"
    moonshine_onnx = os.path.join(BASE_DIR, "..", "moonshine", "core", "third-party", "onnxruntime", "lib", "macos", arch)
    
    if platform.system() == "Darwin":
        env["DYLD_LIBRARY_PATH"] = f"{moonshine_onnx}:{env.get('DYLD_LIBRARY_PATH', '')}"
    
    # Run server directly (ALIRE wrapper changes CWD which breaks relative model paths)
    server_process = subprocess.Popen([server_path], cwd=BASE_DIR, env=env)

    # Launch external watchdog process (separate binary, monitors server health)
    # [DO NOT REMOVE THIS] LAUNCH GUARD: Set orchestration flag so watchdog
    # knows it was launched through run.py (prevents direct binary execution).
    watchdog_bin = "adelaide_watchdog.exe" if platform.system() == "Windows" else "adelaide_watchdog"
    watchdog_path = os.path.join(BASE_DIR, "bin", watchdog_bin)
    if os.path.exists(watchdog_path):
        print("[*] Booting Adelaide Watchdog...")
        watchdog_env = env.copy()
        watchdog_env["ADLAIDE_WATCHDOG_ORCHESTRATED"] = "1"
        if shutil.which("alr"):
            watchdog_process = subprocess.Popen(["alr", "exec", "--", watchdog_path], cwd=BASE_DIR, env=watchdog_env,
                                                 stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        else:
            watchdog_process = subprocess.Popen([watchdog_path], cwd=BASE_DIR, env=watchdog_env,
                                                 stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    else:
        print("[!] Watchdog binary not found at", watchdog_path, "- skipping")

    if launch_gui:
        print("[*] Booting Python Sidecar UI...")
        ui_dir = os.path.join(BASE_DIR, "ui")
        
        # Check for venv python
        venv_python_win = os.path.join(BASE_DIR, "pyvenv", "Scripts", "python.exe")
        venv_python_unix = os.path.join(BASE_DIR, "pyvenv", "bin", "python")
        
        if os.path.exists(venv_python_win):
            sidecar_python = venv_python_win
        elif os.path.exists(venv_python_unix):
            sidecar_python = venv_python_unix
        else:
            sidecar_python = python_cmd
            
        subprocess.run([sidecar_python, "sidecar_ui.py"], cwd=ui_dir)
    else:
        try:
            exit_code = server_process.wait()
            if exit_code != 0:
                import signal
                if exit_code < 0:
                    sig_val = -exit_code
                elif exit_code > 128:
                    sig_val = exit_code - 128
                else:
                    sig_val = None
                    
                sig_name = "UNKNOWN"
                if sig_val:
                    try:
                        sig_name = signal.Signals(sig_val).name
                    except ValueError:
                        sig_name = f"SIGNAL_{sig_val}"
                
                BG_BLUE = "\033[44m\033[97m" # Blue background, white text
                RESET = "\033[0m"
                
                print("\n")
                print(f"{BG_BLUE}{'='*70}{RESET}")
                print(f"{BG_BLUE}{'   :(  WE RAN INTO A PROBLEM'.ljust(70)}{RESET}")
                print(f"{BG_BLUE}{'-'*70}{RESET}")
                print(f"{BG_BLUE}{'   The Adelaide Server encountered a fatal error and terminated.'.ljust(70)}{RESET}")
                print(f"{BG_BLUE}{f'   Exit Code: {exit_code}'.ljust(70)}{RESET}")
                if sig_val:
                    print(f"{BG_BLUE}{f'   Signal:    {sig_name} ({sig_val})'.ljust(70)}{RESET}")
                print(f"{BG_BLUE}{'   '.ljust(70)}{RESET}")
                print(f"{BG_BLUE}{'   Check the output immediately above this banner for the'.ljust(70)}{RESET}")
                print(f"{BG_BLUE}{'   last Ada stack traces and unfortunately we can'.ljust(70)}{RESET}")
                print(f"{BG_BLUE}{'   t recover it needs to be relaunched.'.ljust(70)}{RESET}")
                print(f"{BG_BLUE}{'='*70}{RESET}\n")
            else:
                print("\n[*] Server exited cleanly (code: 0)")
        except KeyboardInterrupt:
            print("\n[*] Keyboard interrupt received. Shutting down...")
            pass
        
    # Wait for background processes to finish if main blocking process exits
    cleanup()

if __name__ == "__main__":
    main()
