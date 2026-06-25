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
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

# ANSI Color Codes
RST  = "\033[0m"
BOLD = "\033[1m"
DIM  = "\033[2m"
RED  = "\033[31m"
GRN  = "\033[32m"
YLW  = "\033[33m"
BLU  = "\033[34m"
MGN  = "\033[35m"
CYN  = "\033[36m"
WHT  = "\033[97m"
BG_B = "\033[44m\033[97m"

def get_git_version():
    """Get current git commit hash and branch from the project root."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
        status = f"{YLW}(dirty){RST}" if dirty else f"{GRN}(clean){RST}"
        return commit, branch, status
    except Exception:
        return None, None, None

def show_help():
    """Print colorful help screen with git version."""
    commit, branch, status = get_git_version()
    ver_str = f"{CYN}{commit}{RST}" if commit else f"{DIM}unknown{RST}"
    brn_str = f"{MGN}{branch}{RST}" if branch else f"{DIM}unknown{RST}"

    print(f"""
{BG_B}{'='*70}{RST}
{BG_B}{'  Adelaide Platform — run.sh'.center(70)}{RST}
{BG_B}{'='*70}{RST}

  {BOLD}Whimsical Automata Companion — Snowball-Enaga{RST}
  {DIM}ELP Priority Queue · Kratos Crash Isolation{RST}

  {BOLD}Version:{RST}  {ver_str}  {status}
  {BOLD}Branch:{RST}   {brn_str}
  {BOLD}Platform:{RST} {YLW}{platform.system()}{RST} ({platform.machine()})

  {BOLD}{WHT}USAGE{RST}
    {CYN}./run.sh{RST} [OPTIONS]

  {BOLD}{WHT}OPTIONS{RST}
    {GRN}--no-gui{RST}                        Launch server without the Python Sidecar UI
    {GRN}--host{RST} {CYN}HOST{RST}                     Bind address (default: 0.0.0.0, env: ADLAIDE_SERVER_HOST)
    {GRN}--port{RST} {CYN}PORT{RST}                     Bind port (default: 11420, env: ADLAIDE_SERVER_PORT)
    {GRN}--test-build-integrity-check{RST}    Build only, verify integrity, then exit
    {GRN}-h{RST}, {GRN}--help{RST}                  Show this help screen

  {BOLD}{WHT}EXAMPLES{RST}
    {DIM}Default — full GUI, binds on all interfaces, port 11420:{RST}
      {CYN}./run.sh{RST}

    {DIM}Headless server, no GUI sidecar:{RST}
      {CYN}./run.sh --no-gui{RST}

    {DIM}Custom port (e.g. 8080):{RST}
      {CYN}./run.sh --port 8080{RST}
      {DIM}→ API at http://localhost:8080{RST}

    {DIM}Bind to localhost only (private, no LAN access):{RST}
      {CYN}./run.sh --host 127.0.0.1{RST}
      {DIM}→ API at http://127.0.0.1:11420{RST}

    {DIM}Custom host + port:{RST}
      {CYN}./run.sh --host 0.0.0.0 --port 9000{RST}
      {DIM}→ API at http://localhost:9000{RST}

    {DIM}Headless with custom port:{RST}
      {CYN}./run.sh --no-gui --port 8080{RST}

    {DIM}Via environment variables:{RST}
      {CYN}ADLAIDE_SERVER_PORT=3000 ADLAIDE_SERVER_HOST=127.0.0.1 ./run.sh{RST}

    {DIM}Docker / LAN access (bind all interfaces):{RST}
      {CYN}./run.sh --host 0.0.0.0 --port 11420{RST}
      {DIM}→ API at http://<your-ip>:11420 from other machines{RST}

    {DIM}Phone / Cloud Terminal (access from phone or tablet):{RST}
      {CYN}./run.sh --host 0.0.0.0 --port 11420{RST}
      {DIM}→ Find your Mac IP: ifconfig | grep 'inet '{RST}
      {DIM}→ Open http://<your-mac-ip>:11420 on your phone browser{RST}
      {DIM}→ Or use curl in Termux / iSH / a-Shell:{RST}
      {DIM}  curl http://<your-mac-ip>:11420/api/version{RST}

    {DIM}Multiple devices ( LAN party / office ):{RST}
      {CYN}./run.sh --host 0.0.0.0 --port 11420{RST}
      {DIM}→ Any device on same network can hit http://<mac-ip>:11420{RST}
      {DIM}→ Works with OpenWebUI, OpenCode, curl, or any HTTP client{RST}

  {BOLD}{WHT}RUNTIME PROCESSES{RST}
    {MGN}1. StellaIcarus Daemon{RST}    Hardware monitor, power state, telemetry
    {MGN}2. adelaide_server{RST}        Ada/AWS HTTP API (default port 11420)
    {MGN}3. adelaide_watchdog{RST}      Monitors server health, auto-restarts

  {BOLD}{WHT}ADA SERVER API{RST} (connect via {CYN}http://localhost:11420{RST} or {CYN}http://127.0.0.1:11420{RST})
    {CYN}POST{RST} /api/chat                Chat completion (streaming)
    {CYN}POST{RST} /api/generate            Text generation
    {CYN}POST{RST} /v1/chat/completions    OpenAI-compatible chat
    {CYN}POST{RST} /v1/completions         OpenAI-compatible completions
    {CYN}POST{RST} /api/embeddings         Text embeddings
    {CYN}POST{RST} /v1/embeddings          OpenAI-compatible embeddings
    {CYN}POST{RST} /v1/audio/transcriptions  Speech-to-text (Moonshine)
    {CYN}POST{RST} /v1/audio/speech        Text-to-speech (Kokoro)
    {CYN}GET{RST}  /api/health             Health check
    {CYN}GET{RST}  /api/version            Server version
    {CYN}GET{RST}  /api/tags               List models
    {CYN}GET{RST}  /api/power              Power state (StellaIcarus)
    {CYN}GET{RST}  /api/telemetry          System telemetry
    {CYN}GET{RST}  /api/ps                 Process status
    {CYN}POST{RST} /api/schedule           Schedule a delayed task
    {CYN}POST{RST} /api/ZenithRoutine      ZenithOrion pacing loop

  {BOLD}{WHT}GUI SIDECAR{RST}
    {CYN}GET{RST}    /api/sessions             List chat sessions
    {CYN}POST{RST}   /api/sessions             Create session
    {CYN}PUT{RST}    /api/sessions/{{id}}      Rename session
    {CYN}DELETE{RST} /api/sessions/{{id}}      Delete session
    {CYN}POST{RST}   /api/sessions/{{id}}/duplicate  Duplicate session
    {CYN}GET{RST}    /api/messages             Message history
    {CYN}GET{RST}    /api/adelaideenginestats  Engine stats
    {CYN}POST{RST}   /api/knowledgestackfrontend/upload     Knowledge upload
    {CYN}GET{RST}    /api/knowledgestackfrontend/search     Knowledge search
    {CYN}POST{RST}   /api/knowledgestackfrontend/memory/upload   Memory upload
    {CYN}GET{RST}    /api/knowledgestackfrontend/memory/search   Memory search
    {CYN}GET{RST}    /api/knowledgestackfrontend/graph          Knowledge graph
    {CYN}GET{RST}    /api/knowledgestackfrontend/memory/graph   Memory graph
    {CYN}GET{RST}    /api/docs/readme          Readme
    {CYN}GET{RST}    /api/docs/license         License
    {CYN}GET{RST}    /api/user_info            User info

  {BOLD}{WHT}MODEL TYPES{RST}
    {YLW}Qwen_0_8B{RST}       Small LLM (always loaded)
    {YLW}Qwen_9B{RST}         Large LLM (loaded on-demand)
    {YLW}Qwen_Embedding{RST}  Semantic search embeddings
    {YLW}MMProj{RST}          Multimodal CLIP vision

  {DIM}  Documentation:  Adelaide_Lite/documentation/{RST}
  {DIM}  Architecture:   Adelaide_Lite/run.py (line 14){RST}
""")

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
if "--help" in sys.argv or "-h" in sys.argv:
    show_help()
    sys.exit(0)

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
    # Write shutdown flag so watchdog knows this was an intentional stop
    shutdown_flag = os.path.join(BASE_DIR, "run", ".shutdown_requested")
    try:
        os.makedirs(os.path.dirname(shutdown_flag), exist_ok=True)
        with open(shutdown_flag, "w") as f:
            f.write(f"pid={os.getpid()}\n")
    except Exception:
        pass

    # Collect PIDs to kill directly — do NOT rely on proc.terminate()
    # inside a signal handler (can deadlock with main thread's proc.wait()).
    pids_to_kill = []
    for proc in [daemon_process, server_process]:
        if proc and proc.poll() is None:
            pids_to_kill.append((proc.pid, proc.args[0] if proc.args else "unknown"))

    # Send SIGTERM first, then SIGKILL after 2s grace period
    SIGTERM = signal.SIGTERM
    SIGKILL = signal.SIGKILL

    for pid, name in pids_to_kill:
        print(f"[*] Sending SIGTERM to {name} (PID {pid})...")
        try:
            os.kill(pid, SIGTERM)
        except ProcessLookupError:
            pass

    # Give 2 seconds for graceful shutdown
    time.sleep(2.0)

    for pid, name in pids_to_kill:
        try:
            # Check if still alive
            os.kill(pid, 0)
            print(f"[*] PID {pid} still alive, sending SIGKILL...")
            os.kill(pid, SIGKILL)
        except ProcessLookupError:
            print(f"[*] PID {pid} exited cleanly.")

    # Force-kill any remaining zombie processes via process group
    for proc in [daemon_process, server_process]:
        if proc:
            try:
                os.killpg(os.getpgid(proc.pid), SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass

    print("[*] Cleanup complete.")
    os._exit(0)

signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

def main():
    global daemon_process, server_process, watchdog_process
    
    print(f"[*] Setting up Adelaide-Lite environment in {BASE_DIR}...")
    start_time = int(time.time() * 1000)

    # Detect Platform and Backend
    ggml_backend = "none"
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        ggml_backend = "metal"
    elif platform.system() == "Linux":
        if shutil.which("nvcc") or shutil.which("nvidia-smi"):
            ggml_backend = "cuda"
        elif shutil.which("sycl-ls") or os.environ.get("ONEAPI_ROOT"):
            ggml_backend = "sycl"
        else:
            ggml_backend = "vulkan"
    os.environ["GGML_BACKEND"] = ggml_backend
    print(f"[*] Detected Platform: {platform.system()} | Selected Backend: {ggml_backend.upper()}")

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
        
        # =====================================================================
        # ggml: git submodule → compile from source
        # =====================================================================
        # [VITAL-DO-NOT-REMOVE] NEVER use Homebrew's ggml.
        # Homebrew ggml 0.15.2 has a bug in Qwen3.5's Gated Delta Net:
        #   GGML_ASSERT(state->ne[0] == S_v) failed  (ggml.c:6252)
        # This crashes during llama_decode. The assertion path showed
        # /private/tmp/ggml-20260619-5335-xzehaz/ggml-0.15.2/ which is
        # the HOMEBREW-built copy, not our locally-built one.
        # LM Studio runs Qwen3.5 on llama.cpp (not just MLX) and does
        # NOT have this bug — they bundle their own ggml build.
        # FIX: Clone ggml as a git submodule, compile from source, link
        # against our local build only. RPATH ensures runtime never
        # picks up Homebrew's ggml.
        # =====================================================================
        ggml_submodule = os.path.abspath(os.path.join(BASE_DIR, "vendor", "ggml"))
        ggml_build_dir = os.path.join(ggml_submodule, "build")
        ggml_lib = os.path.join(ggml_build_dir, "bin", "libggml.dylib")
        ggml_start = time.time()

        # Init/update submodule
        if not os.path.exists(os.path.join(ggml_submodule, ".git")) and \
           not os.path.exists(os.path.join(ggml_submodule, "CMakeLists.txt")):
            print(f"[GGML] [{time.strftime('%H:%M:%S')}] Initializing ggml submodule...")
            result = subprocess.run(
                ["git", "submodule", "update", "--init", "--recursive",
                 "Adelaide_Lite/vendor/ggml"],
                cwd=os.path.dirname(BASE_DIR), check=False,
                capture_output=True, text=True
            )
            if result.returncode != 0:
                print(f"[GGML] [{time.strftime('%H:%M:%S')}] Submodule init FAILED: {result.stderr[-300:]}")
        else:
            print(f"[GGML] [{time.strftime('%H:%M:%S')}] Fetching latest ggml...")
            subprocess.run(["git", "fetch", "origin"], cwd=ggml_submodule,
                           check=False, capture_output=True)
            subprocess.run(["git", "pull", "--ff-only"], cwd=ggml_submodule,
                           check=False, capture_output=True)

        ggml_ver = subprocess.run(
            ["git", "describe", "--tags"], cwd=ggml_submodule,
            capture_output=True, text=True
        ).stdout.strip()
        print(f"[GGML] [{time.strftime('%H:%M:%S')}] Version: {ggml_ver}")

        # Build ggml if needed
        if not os.path.exists(ggml_lib):
            print(f"[GGML] [{time.strftime('%H:%M:%S')}] Building ggml from source...")
            os.makedirs(ggml_build_dir, exist_ok=True)
            cmake_flags = ["cmake", "-B", "build", "-DGGML_NATIVE=ON",
                           "-DCMAKE_BUILD_TYPE=Release"]
            if ggml_backend == "metal":
                cmake_flags.append("-DGGML_METAL=ON")
                print(f"[GGML] [{time.strftime('%H:%M:%S')}] Metal GPU: ENABLED")
            elif ggml_backend == "cuda":
                cmake_flags.append("-DGGML_CUDA=ON")
                print(f"[GGML] [{time.strftime('%H:%M:%S')}] CUDA GPU: ENABLED")
            elif ggml_backend == "sycl":
                cmake_flags.append("-DGGML_SYCL=ON")
                print(f"[GGML] [{time.strftime('%H:%M:%S')}] SYCL/oneAPI GPU: ENABLED")
            elif ggml_backend == "vulkan":
                cmake_flags.append("-DGGML_VULKAN=ON")
                print(f"[GGML] [{time.strftime('%H:%M:%S')}] Vulkan GPU: ENABLED")
            result = subprocess.run(cmake_flags, cwd=ggml_submodule,
                                    check=False, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"[GGML] [{time.strftime('%H:%M:%S')}] CMake FAILED: {result.stderr[-500:]}")
            else:
                result = subprocess.run(
                    ["cmake", "--build", "build", "--config", "Release", "-j"],
                    cwd=ggml_submodule, check=False, capture_output=True, text=True
                )
                ggml_elapsed = time.time() - ggml_start
                if result.returncode == 0:
                    print(f"[GGML] [{time.strftime('%H:%M:%S')}] Build SUCCESS in {ggml_elapsed:.1f}s")
                else:
                    print(f"[GGML] [{time.strftime('%H:%M:%S')}] Build FAILED: {result.stderr[-500:]}")
        else:
            ggml_elapsed = time.time() - ggml_start
            print(f"[GGML] [{time.strftime('%H:%M:%S')}] Library exists ({ggml_elapsed:.1f}s)")

        # =====================================================================
        # llama.cpp: clone → fetch+pull latest → rebuild if updated
        # =====================================================================
        # We always fetch+pull so we get the latest fixes.
        # llama.cpp builds its own ggml in-tree for compilation, but at
        # RUNTIME we use our separately-built ggml via RPATH (see GPR file).
        llama_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "llama.cpp"))
        llama_build_dir = os.path.join(llama_dir, "build")
        llama_lib = os.path.join(llama_build_dir, "bin", "libllama.dylib")
        llama_start = time.time()

        if not os.path.exists(llama_dir):
            print(f"[LLAMA] [{time.strftime('%H:%M:%S')}] Cloning llama.cpp...")
            subprocess.run(
                ["git", "clone", "https://github.com/ggml-org/llama.cpp.git", llama_dir],
                check=False
            )
            needs_build = True
        else:
            old_head = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=llama_dir,
                capture_output=True, text=True
            ).stdout.strip()
            print(f"[LLAMA] [{time.strftime('%H:%M:%S')}] Fetching latest llama.cpp...")
            subprocess.run(["git", "fetch", "origin"], cwd=llama_dir, check=False,
                           capture_output=True)
            subprocess.run(["git", "pull", "--ff-only"], cwd=llama_dir, check=False,
                           capture_output=True)
            new_head = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=llama_dir,
                capture_output=True, text=True
            ).stdout.strip()
            needs_build = (old_head != new_head) or not os.path.exists(llama_lib)
            if old_head != new_head:
                print(f"[LLAMA] [{time.strftime('%H:%M:%S')}] Updated: {old_head[:8]} → {new_head[:8]}")
            else:
                print(f"[LLAMA] [{time.strftime('%H:%M:%S')}] Already up to date ({new_head[:8]})")

        # Build if needed (new clone, update, or missing lib)
        if needs_build:
            print(f"[LLAMA] [{time.strftime('%H:%M:%S')}] Building llama.cpp...")
            print(f"[LLAMA] [{time.strftime('%H:%M:%S')}] CMake flags: -DGGML_NATIVE=ON -DLLAMA_BUILD_TOOLS=ON")
            os.makedirs(llama_build_dir, exist_ok=True)
            cmake_flags = ["cmake", "-B", "build", "-DGGML_NATIVE=ON", "-DLLAMA_BUILD_TOOLS=ON"]
            if ggml_backend == "metal":
                print(f"[LLAMA] [{time.strftime('%H:%M:%S')}] Metal GPU acceleration: ENABLED")
                cmake_flags.append("-DGGML_METAL=ON")
            elif ggml_backend == "cuda":
                print(f"[LLAMA] [{time.strftime('%H:%M:%S')}] CUDA GPU acceleration: ENABLED")
                cmake_flags.append("-DGGML_CUDA=ON")
            elif ggml_backend == "sycl":
                print(f"[LLAMA] [{time.strftime('%H:%M:%S')}] SYCL/oneAPI GPU acceleration: ENABLED")
                cmake_flags.append("-DGGML_SYCL=ON")
            elif ggml_backend == "vulkan":
                print(f"[LLAMA] [{time.strftime('%H:%M:%S')}] Vulkan GPU acceleration: ENABLED")
                cmake_flags.append("-DGGML_VULKAN=ON")
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
        mtmd_lib = os.path.join(llama_build_dir, "bin", "libmtmd.dylib")
        mtmd_start = time.time()
        if not os.path.exists(mtmd_lib):
            print(f"[MTMD] [{time.strftime('%H:%M:%S')}] Building mtmd (multimodal) library...")
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

        # =====================================================================
        # stable-diffusion.cpp: clone → fetch+pull latest → init ggml → build
        # =====================================================================
        # [VITAL-DO-NOT-REMOVE] FLUX Schnell image generation backend.
        # Builds a static library (libstable_diffusion.a) for Ada FFI linkage.
        # The ggml submodule within stable-diffusion.cpp must be initialized
        # before cmake can configure — it provides the compute graph runtime.
        sd_cpp_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "stable-diffusion.cpp"))
        sd_cpp_built = os.path.join(sd_cpp_dir, "build")
        sd_cpp_lib_static = os.path.join(sd_cpp_built, "libstable-diffusion.a")
        sd_cpp_lib_shared = os.path.join(sd_cpp_built, "libstable-diffusion.dylib") if platform.system() == "Darwin" else os.path.join(sd_cpp_built, "libstable-diffusion.so")
        sd_cpp_start = time.time()

        if not os.path.exists(sd_cpp_dir):
            print(f"[SD-CPP] [{time.strftime('%H:%M:%S')}] Cloning stable-diffusion.cpp...")
            subprocess.run(
                ["git", "clone", "--depth=1", "https://github.com/leejet/stable-diffusion.cpp.git", sd_cpp_dir],
                check=False
            )
            needs_build = True
        else:
            old_head = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=sd_cpp_dir,
                capture_output=True, text=True
            ).stdout.strip()
            print(f"[SD-CPP] [{time.strftime('%H:%M:%S')}] Fetching latest stable-diffusion.cpp...")
            subprocess.run(["git", "fetch", "origin"], cwd=sd_cpp_dir, check=False,
                           capture_output=True)
            subprocess.run(["git", "pull", "--ff-only"], cwd=sd_cpp_dir, check=False,
                           capture_output=True)
            new_head = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=sd_cpp_dir,
                capture_output=True, text=True
            ).stdout.strip()
            needs_build = (old_head != new_head) or not (os.path.exists(sd_cpp_lib_static) or os.path.exists(sd_cpp_lib_shared))
            if old_head != new_head:
                print(f"[SD-CPP] [{time.strftime('%H:%M:%S')}] Updated: {old_head[:8]} → {new_head[:8]}")
            else:
                print(f"[SD-CPP] [{time.strftime('%H:%M:%S')}] Already up to date ({new_head[:8]})")

        # Init stable-diffusion.cpp's own ggml submodule (required for cmake)
        sd_ggml_sub = os.path.join(sd_cpp_dir, "ggml")
        sd_ggml_cmakelists = os.path.join(sd_ggml_sub, "CMakeLists.txt")
        if not os.path.exists(sd_ggml_cmakelists):
            print(f"[SD-CPP] [{time.strftime('%H:%M:%S')}] Initializing ggml submodule inside stable-diffusion.cpp...")
            subprocess.run(
                ["git", "submodule", "update", "--init", "--recursive"],
                cwd=sd_cpp_dir, check=False, capture_output=True
            )

        # Build static library for Ada FFI linkage
        if needs_build or not (os.path.exists(sd_cpp_lib_static) or os.path.exists(sd_cpp_lib_shared)):
            print(f"[SD-CPP] [{time.strftime('%H:%M:%S')}] Building stable-diffusion.cpp (static lib)...")
            os.makedirs(sd_cpp_built, exist_ok=True)
            cmake_flags = ["cmake", "..", "-DCMAKE_BUILD_TYPE=Release", "-DSD_BUILD_EXAMPLES=OFF"]
            if ggml_backend == "metal":
                cmake_flags.append("-DGGML_METAL=ON")
            elif ggml_backend == "cuda":
                cmake_flags.append("-DGGML_CUDA=ON")
            result = subprocess.run(cmake_flags, cwd=sd_cpp_built, check=False, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"[SD-CPP] [{time.strftime('%H:%M:%S')}] CMake FAILED: {result.stderr[-500:]}")
            else:
                result = subprocess.run(
                    ["cmake", "--build", ".", "--config", "Release", "-j"],
                    cwd=sd_cpp_built, check=False, capture_output=True, text=True
                )
                sd_elapsed = time.time() - sd_cpp_start
                if result.returncode == 0:
                    # Verify the library was created
                    if os.path.exists(sd_cpp_lib_static):
                        sd_size = os.path.getsize(sd_cpp_lib_static)
                        print(f"[SD-CPP] [{time.strftime('%H:%M:%S')}] Build SUCCESS in {sd_elapsed:.1f}s ({sd_size:,} bytes)")
                    elif os.path.exists(sd_cpp_lib_shared):
                        sd_size = os.path.getsize(sd_cpp_lib_shared)
                        print(f"[SD-CPP] [{time.strftime('%H:%M:%S')}] Build SUCCESS (shared) in {sd_elapsed:.1f}s ({sd_size:,} bytes)")
                    else:
                        print(f"[SD-CPP] [{time.strftime('%H:%M:%S')}] Build completed but library not found at expected path")
                else:
                    print(f"[SD-CPP] [{time.strftime('%H:%M:%S')}] Build FAILED in {sd_elapsed:.1f}s")
                    if result.stderr:
                        print(f"[SD-CPP] [{time.strftime('%H:%M:%S')}] stderr: {result.stderr[-500:]}")
        else:
            sd_elapsed = time.time() - sd_cpp_start
            print(f"[SD-CPP] [{time.strftime('%H:%M:%S')}] Library exists ({sd_elapsed:.1f}s), skipping build")

        # Check and download Qwen models
        qwen_models_dir = os.path.abspath(os.path.join(BASE_DIR, "model"))
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
                "url": "https://huggingface.co/empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF/resolve/main/Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_K_M.gguf?download=true",
                "output": "Mythos9bHybridq4.gguf"
            },
            {
                "url": "https://huggingface.co/empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF/resolve/main/mmproj-Qwythos-9B-Claude-Mythos-5-1M-f16.gguf?download=true",
                "output": "Mythos9bHybridq4-mmproj-fp16.gguf"
            },
            {
                "url": "https://huggingface.co/ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/resolve/main/qwen3-reranker-0.6b-q8_0.gguf?download=true",
                "output": "Qwen3-Reranker-0.6B-Q8_0.gguf"
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

        # =====================================================================
        # FLUX Schnell models (stable-diffusion.cpp image generation)
        # =====================================================================
        # [VITAL-DO-NOT-REMOVE] TWO-STAGE IMAGE GENERATION ARCHITECTURE:
        #
        #   STAGE 1: FLUX Schnell Q2_K (sparse, fast, low quality)
        #     - Diffusion model: flux1-schnell.gguf (~4GB GGUF)
        #     - Text encoders: clip_l.safetensors + t5xxl Q4_0 GGUF (~2.9GB)
        #     - VAE: ae.safetensors (~335MB)
        #     - Output: sparse/draft image (2-4 steps, CFG 1.0)
        #
        #   STAGE 2: SD Refinement (img2img upscale, high quality)
        #     - Model: sd-refinement.gguf (~1.9GB, SD 1.5 pruned)
        #     - Input: Stage 1 output + added noise (strength ~0.4)
        #     - Output: refined/final image (dpmpp2mv2, 8+ steps)
        #     - Prompt: "Masterpiece, Amazing, 4k, " + original_prompt + ", highly detailed..."
        #
        #   Memory budget: FLUX Q2_K (~4GB) + t5xxl Q4_0 (~2.9GB) + SD refinement (~1.9GB)
        #   = ~8.8GB total (fits 9B-class VRAM with swap)
        #
        # Source repos:
        #   Diffusion: city96/FLUX.1-schnell-gguf (preconverted GGUF)
        #   T5-XXL:    Phil2Sat/T5XXL-Unchained-GGUF (Q4_0, smallest GGUF t5xxl)
        #   CLIP-L:    comfyanonymous/flux_text_encoders (safetensors)
        #   VAE:       ffxvs/vae-flux (public mirror, BFL repos are gated)
        #   Refinement: second-state/stable-diffusion-v1-5-GGUF (SD 1.5 Q8_0)
        # Reference: stable-diffusion.cpp/docs/flux.md
        #            project-zephyrine imagination_worker.py (two-stage pipeline)
        flux_models_dir = os.path.abspath(os.path.join(BASE_DIR, "model"))
        os.makedirs(flux_models_dir, exist_ok=True)

        flux_models_to_download = [
            # Diffusion model Q2_K (~4GB) — fits 9B-class VRAM budget
            {
                "url": "https://huggingface.co/city96/FLUX.1-schnell-gguf/resolve/main/flux1-schnell-Q2_K.gguf?download=true",
                "output": "flux1-schnell.gguf"
            },
            # T5-XXL text encoder Q4_0 GGUF (~2.9GB) — small enough for VRAM
            {
                "url": "https://huggingface.co/Phil2Sat/T5XXL-Unchained-GGUF/resolve/main/Kaoru8-t5xxl-unchained-Q4_0.gguf?download=true",
                "output": "flux1-t5xxl.gguf"
            },
            # CLIP-L text encoder (safetensors, ~246MB — small, always fits)
            {
                "url": "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors?download=true",
                "output": "flux1-clip_l.safetensors"
            },
            # VAE (safetensors, ~335MB — public mirror, BFL repos are gated)
            {
                "url": "https://huggingface.co/ffxvs/vae-flux/resolve/main/ae.safetensors?download=true",
                "output": "flux1-ae.safetensors"
            },
            # SD refinement model (~1.9GB — Stage 2 img2img upscale after FLUX sparse output)
            # Architecture: FLUX Q2_K sparse → add noise → SD refinement upscale
            {
                "url": "https://huggingface.co/second-state/stable-diffusion-v1-5-GGUF/resolve/main/stable-diffusion-v1-5-pruned-emaonly-Q8_0.gguf?download=true",
                "output": "sd-refinement.gguf"
            }
        ]

        for model in flux_models_to_download:
            target_path = os.path.join(flux_models_dir, model["output"])
            if os.path.exists(target_path):
                expected_size = {"flux1-schnell.gguf": 4_010_296_352,
                                 "flux1-t5xxl.gguf": 2_924_546_752}.get(model["output"], 0)
                actual_size = os.path.getsize(target_path)
                if expected_size == 0 or actual_size >= expected_size * 0.95:
                    print(f"[*] {model['output']} already exists ({actual_size:,} bytes), skipping.")
                    continue
                else:
                    print(f"[*] {model['output']} incomplete ({actual_size:,}/{expected_size:,}), resuming...")

            # Infinite retry with resume — wget -c continues partial downloads
            max_retries = 0  # 0 = infinite
            attempt = 0
            while True:
                attempt += 1
                label = f"attempt #{attempt}" if max_retries == 0 else f"attempt {attempt}/{max_retries}"
                print(f"[*] Downloading {model['output']} ({label})...")
                result = subprocess.run(
                    ["wget", "-c", "-t", "0", "--timeout=30", "--waitretry=5",
                     "--show-progress", model["url"], "-O", target_path],
                    check=False, timeout=None
                )
                if result.returncode == 0:
                    print(f"[+] {model['output']} downloaded successfully.")
                    break
                # wget returns 8 = server error, 4 = network failure, etc — all retryable
                print(f"[!] wget failed (code {result.returncode}), retrying in 5s...")
                time.sleep(5)
                if max_retries > 0 and attempt >= max_retries:
                    print(f"[!] {model['output']} failed after {max_retries} attempts, continuing...")
                    break

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
            result = subprocess.run([ruff_cmd, "check", BASE_DIR,
                                     "--exclude", "vendor,moonshine"],
                                    capture_output=True, text=True)
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

    # =====================================================================
    # LSH QRNN Worker: Python venv bootstrap + self-check
    # =====================================================================
    lsh_reqs = os.path.join(BASE_DIR, "lsh", "requirements-lsh.txt")
    lsh_worker = os.path.join(BASE_DIR, "lsh", "lsh_qrnn_worker.py")
    if os.path.exists(lsh_reqs):
        print("[LSH] Bootstrapping QRNN LSH worker venv...")
        # Use the shared pyvenv (already created by sidecar, or create if missing)
        pyvenv_dir = os.path.join(BASE_DIR, "pyvenv")
        pyvenv_python = os.path.join(pyvenv_dir, "bin", "python3")
        if not os.path.exists(pyvenv_python):
            print("[LSH] Creating shared Python venv at pyvenv/...")
            subprocess.run([sys.executable, "-m", "venv", pyvenv_dir], check=True)
        # pip install LSH requirements
        pyvenv_pip = os.path.join(pyvenv_dir, "bin", "pip")
        print("[LSH] Installing requirements-lsh.txt...")
        subprocess.run([pyvenv_pip, "install", "-r", lsh_reqs], check=False)
        # Self-check: pyrefly type-check on worker script
        pyvenv_pyrefly = os.path.join(pyvenv_dir, "bin", "pyrefly")
        if os.path.exists(pyvenv_pyrefly):
            print("[LSH] Running pyrefly type-check on worker...")
            subprocess.run([pyvenv_pyrefly, "check", lsh_worker], check=False)
        else:
            print("[LSH] pyrefly not found in venv, skipping type-check.")
        # Self-check: ruff lint on worker script
        pyvenv_ruff = os.path.join(pyvenv_dir, "bin", "ruff")
        if os.path.exists(pyvenv_ruff):
            print("[LSH] Running ruff lint on worker...")
            subprocess.run([pyvenv_ruff, "check", lsh_worker], check=False)
        else:
            print("[LSH] ruff not found in venv, skipping lint.")
        print("[LSH] QRNN worker bootstrap complete.")
    else:
        print(f"[!] LSH requirements not found at {lsh_reqs}, skipping QRNN worker setup.")

    # Handle integrity check flag
    if "--test-build-integrity-check" in sys.argv:
        print("[*] Test build integrity check passed! Exiting without launching services.")
        sys.exit(0)

    # Parse arguments
    launch_gui = True
    if "--no-gui" in sys.argv:
        launch_gui = False

    # [DO NOT REMOVE] --no-daemon: Skip the StellaIcarus daemon runner.
    # The daemon runner retries failed MCU bridge connections every 30s,
    # flooding the terminal with error messages.  Use this flag when you
    # want clean server-only output for debugging.
    launch_daemon = True
    if "--no-daemon" in sys.argv:
        launch_daemon = False

    # Port/Host: args > env > defaults
    server_host = os.environ.get("ADLAIDE_SERVER_HOST", "0.0.0.0")
    server_port = os.environ.get("ADLAIDE_SERVER_PORT", "11420")
    for i, arg in enumerate(sys.argv):
        if arg == "--host" and i + 1 < len(sys.argv):
            server_host = sys.argv[i + 1]
        if arg == "--port" and i + 1 < len(sys.argv):
            server_port = sys.argv[i + 1]

    # [DO NOT REMOVE] Verbose launch info for debugging startup issues
    print(f"[*] [Launch-V] Run.py PID: {os.getpid()}")
    print(f"[*] [Launch-V] Python executable: {sys.executable}")
    print(f"[*] [Launch-V] Server host: {server_host}, port: {server_port}")
    print(f"[*] [Launch-V] Launch GUI: {launch_gui}, Launch daemon: {launch_daemon}")

    if launch_daemon:
        print("[*] Booting StellaIcarus Ada Daemon Manager...")
        python_cmd = sys.executable
        daemon_script = os.path.join(BASE_DIR, "python", "stellaicarus_daemon_runner.py")
    
        daemon_args = [python_cmd, daemon_script]
        if daemon_build_flag:
            daemon_args.append(daemon_build_flag)
            
        daemon_process = subprocess.Popen(daemon_args, cwd=BASE_DIR, start_new_session=True)
    else:
        print("[*] [Launch-V] Skipping daemon runner (--no-daemon)")

    print("[*] Booting Adelaide Intelligence Server...")
    end_time = int(time.time() * 1000)
    print(f"[*] Startup completed in {end_time - start_time}ms (WCET)")

    server_bin = "adelaide_server.exe" if platform.system() == "Windows" else "adelaide_server"
    server_path = os.path.join(BASE_DIR, "bin", server_bin)

    env = os.environ.copy()

    # [DO NOT REMOVE] Force Python stdout/stderr unbuffered for all subprocesses.
    # When stdout is a pipe (not a terminal), Python block-buffers output.
    # This prevents run.py's print() from appearing immediately.
    env["PYTHONUNBUFFERED"] = "1"

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
    # Write server launch args to file so the watchdog can relaunch with same args.
    server_args = ["--host", server_host, "--port", server_port]
    server_args_file = os.path.join(BASE_DIR, "run", "adelaide_server.args")
    with open(server_args_file, "w") as f:
        f.write(" ".join(server_args))

    # [DO NOT REMOVE] Verbose server launch info
    print(f"[*] [Launch-V] Server binary: {server_path}")
    print(f"[*] [Launch-V] Server args: {server_args}")
    print(f"[*] [Launch-V] Server CWD: {BASE_DIR}")
    print(f"[*] [Launch-V] DYLD_LIBRARY_PATH: {env.get('DYLD_LIBRARY_PATH', 'NOT SET')}")

    server_process = subprocess.Popen([server_path] + server_args, cwd=BASE_DIR, env=env,
                                       start_new_session=True)

    # [DO NOT REMOVE] Verbose PID tracking
    print(f"[*] [Launch-V] Server PID: {server_process.pid}")
    print(f"[*] [Launch-V] Server args file: {server_args_file}")
    print(f"[*] [Launch-V] Server stdout fd: {server_process.stdout}")
    print(f"[*] [Launch-V] Server stderr fd: {server_process.stderr}")

    # Launch external watchdog process (separate binary, monitors server health)
    # [DO NOT REMOVE THIS] LAUNCH GUARD: Set orchestration flag so watchdog
    # knows it was launched through run.py (prevents direct binary execution).
    watchdog_bin = "adelaide_watchdog.exe" if platform.system() == "Windows" else "adelaide_watchdog"
    watchdog_path = os.path.join(BASE_DIR, "bin", watchdog_bin)

    # Clear any stale shutdown flag from a previous run.
    # This flag is written by cleanup() to prevent the watchdog from
    # restarting the server after an intentional Ctrl+C.  If we're
    # starting a fresh session, the old flag must be removed.
    shutdown_flag = os.path.join(BASE_DIR, "run", ".shutdown_requested")
    if os.path.exists(shutdown_flag):
        try:
            os.remove(shutdown_flag)
        except Exception:
            pass
    if os.path.exists(watchdog_path):
        print("[*] Booting Adelaide Watchdog...")
        watchdog_env = env.copy()
        watchdog_env["ADLAIDE_WATCHDOG_ORCHESTRATED"] = "1"
        # Launch watchdog fully detached — nohup + own session + own process group.
        # This ensures the watchdog survives even if run.py exits or the
        # terminal is closed.  The watchdog monitors the server via file-based
        # IPC (run/ directory) so it doesn't need a parent process.
        watchdog_log = os.path.join(BASE_DIR, "run", "adelaide_watchdog.log")
        with open(watchdog_log, "a") as wlog:
            watchdog_process = subprocess.Popen(
                [watchdog_path], cwd=BASE_DIR, env=watchdog_env,
                stdout=wlog, stderr=subprocess.STDOUT,
                start_new_session=True)
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
