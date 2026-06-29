#!/usr/bin/env python3
import os
import fcntl
import sys
import time
import subprocess
import hashlib
import platform
import signal
import shutil
import threading

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
MAX_LOG_BYTES = 10 * 1024 * 1024  # 10 MB total cap

try:
    _lock_fd = open(os.path.join(BASE_DIR, ".adelaide.lock"), "w")
    fcntl.flock(_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    print("[!] FATAL: Another instance of Adelaide is already running.")
    print("    Singleton lock enforced. Aborting startup.")
    sys.exit(1)


# Enforce Huggingface cache location
os.environ["HF_HOME"] = os.path.join(BASE_DIR, "model")
os.environ["HF_HUB_CACHE"] = os.path.join(BASE_DIR, "model")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(BASE_DIR, "model")


# ---------------------------------------------------------------------------
#  Logging: tee stdout+stderr to logs/ with 10 MB rollover
# ---------------------------------------------------------------------------
class _TeeWriter:
    """Write to an original stream AND append to a log file simultaneously."""

    def __init__(self, original, log_file):
        self._orig = original
        self._log = log_file

    def write(self, data):
        self._orig.write(data)
        try:
            self._log.write(data)
            self._log.flush()
        except Exception:
            pass

    def flush(self):
        self._orig.flush()
        try:
            self._log.flush()
        except Exception:
            pass

    def __getattr__(self, attr):
        return getattr(self._orig, attr)


class _PipeReader(threading.Thread):
    """Daemon thread that reads a subprocess pipe and tees it to a writer."""

    def __init__(self, pipe, writer, label=""):
        super().__init__(daemon=True)
        self._pipe = pipe
        self._writer = writer
        self._label = label

    def run(self):
        try:
            for line in iter(self._pipe.readline, b""):
                self._writer.write(line)
        except Exception:
            pass
        finally:
            self._pipe.close()


def _rotate_logs():
    """Delete oldest log files until total size <= MAX_LOG_BYTES."""
    if not os.path.isdir(LOGS_DIR):
        return
    entries = []
    for name in os.listdir(LOGS_DIR):
        if name.endswith(".log"):
            path = os.path.join(LOGS_DIR, name)
            try:
                entries.append((os.path.getmtime(path), os.path.getsize(path), path))
            except OSError:
                pass
    entries.sort(key=lambda e: e[0])  # oldest first
    total = sum(sz for _, sz, _ in entries)
    for _mtime, sz, path in entries:
        if total <= MAX_LOG_BYTES:
            break
        try:
            os.remove(path)
            total -= sz
        except OSError:
            pass


def setup_logging():
    """Create logs/ dir, rotate old logs, redirect stdout/stderr to tee.
    Returns the path of the current log file."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    _rotate_logs()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOGS_DIR, f"run_{timestamp}.log")
    log_fp = open(log_path, "a", encoding="utf-8", buffering=1)  # line-buffered
    sys.stdout = _TeeWriter(sys.__stdout__, log_fp)
    sys.stderr = _TeeWriter(sys.__stderr__, log_fp)
    print(f"[*] Logging to {log_path}")
    return log_path

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
BG_RED = "\033[41m\033[97m"

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

def verify_environment():
    """Check for all required tools and libraries before proceeding."""
    print(f"\n{BOLD}{WHT}[*] Verifying Environment Prerequisites...{RST}")
    
    critical_tools = {
        "alr": "Alire (Ada Package Manager) - install via 'brew install alire'",
        "python3": "Python 3.12+ - install via 'brew install python'",
        "cmake": "CMake - install via 'brew install cmake'",
        "git": "Git - install via 'brew install git'",
        "wget": "wget - install via 'brew install wget'",
        "npm": "Node.js/npm - install via 'brew install node'",
        "deno": "Deno - install via 'curl -fsSL https://deno.land/install.sh | sh'",
        "ruff": "Ruff (Linter) - install via 'pip install ruff'",
    }
    
    missing = []
    for tool, desc in critical_tools.items():
        if shutil.which(tool):
            print(f"  {GRN}[ok]{RST} {tool}")
        else:
            print(f"  {RED}[!!]{RST} {tool} is missing: {desc}")
            missing.append(tool)
    
    # macOS specific SDK check
    if platform.system() == "Darwin":
        # Check for full Xcode.app installation (not just Command Line Tools)
        xcode_path = "/Applications/Xcode.app"
        if os.path.exists(xcode_path):
            print(f"  {GRN}[ok]{RST} Full Xcode.app found")
        else:
            print(f"  {RED}[!!]{RST} Full Xcode.app NOT found at {xcode_path}")
            print("    Prerequisite: Install full Xcode from the App Store")
            missing.append("xcode-app")

        try:
            subprocess.check_output(["xcrun", "--show-sdk-path"], stderr=subprocess.DEVNULL)
            print(f"  {GRN}[ok]{RST} macOS SDK path available")
        except Exception:
            print(f"  {RED}[!!]{RST} macOS SDK path not found: run 'xcode-select --install'")
            missing.append("macos-sdk")

    if missing:
        print(f"\n{BG_RED}[BUGCHECK] [FATAL] Environment check failed. Please install the missing tools listed above.{RST}")
        sys.exit(1)
    else:
        print(f"{GRN}[+] Environment verified. All prerequisites met.{RST}\n")

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
#   vendor/llama.cpp/            — LLM inference engine
#   vendor/moonshine/            — Speech-to-text ONNX models
#   vendor/kokoro-onnx/          — Text-to-speech ONNX
#   vendor/kokoclone/            — Zero-shot voice cloning
#   vendor/tts_kokoro_component/ — Kokoro TTS Python deps (isolated venv)
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
os.environ["HF_HOME"] = os.path.join(BASE_DIR, "model")
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

# Kill any stale processes from previous runs before starting
print("[*] Cleaning up any stale processes from previous runs...")
try:
    subprocess.run(["pkill", "-9", "-f", "adelaide_server"], stderr=subprocess.DEVNULL)
    subprocess.run(["pkill", "-9", "-f", "adelaide_watchdog"], stderr=subprocess.DEVNULL)
    subprocess.run(["pkill", "-9", "-f", "vad_worker.py"], stderr=subprocess.DEVNULL)
except Exception:
    pass

# Globals to keep track of background processes
daemon_process = None
server_process = None
vad_process = None
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
        mtmd_dir = os.path.abspath(os.path.join(BASE_DIR, "vendor", "llama.cpp", "tools", "mtmd"))
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
    for proc in [daemon_process, server_process, watchdog_process, vad_process]:
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
    for proc in [daemon_process, server_process, watchdog_process, vad_process]:
        if proc:
            try:
                os.killpg(os.getpgid(proc.pid), SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass

    print("[*] Cleanup complete.")
    os._exit(0)

signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

def checkout_latest_release(repo_dir, module_name):
    """Fetches the latest release tag and checks it out for stability."""
    try:
        # Fetch tags
        subprocess.run(["git", "fetch", "--tags", "origin"], cwd=repo_dir, check=False, capture_output=True)
        # Find latest tag
        result = subprocess.run(["git", "describe", "--tags", "--abbrev=0"], cwd=repo_dir, capture_output=True, text=True)
        latest_tag = result.stdout.strip()
        if latest_tag:
            # Checkout tag
            checkout_res = subprocess.run(["git", "checkout", latest_tag], cwd=repo_dir, check=False, capture_output=True, text=True)
            if checkout_res.returncode == 0:
                print(f"[{module_name}] Checked out latest release: {latest_tag}")
            else:
                print(f"[{module_name}] Failed to checkout {latest_tag}: {checkout_res.stderr}")
            return latest_tag
    except Exception as e:
        print(f"[{module_name}] Error checking out latest tag: {e}")
    return None

def safe_cmake_configure(cmake_flags, cwd, build_dir, module_name):
    """Robust CMake configure that detects cache corruption and retries cleanly."""
    result = subprocess.run(cmake_flags, cwd=cwd, check=False, capture_output=True, text=True)
    if result.returncode != 0 and ("CMakeCache.txt" in result.stderr or "CMake Error" in result.stderr):
        print(f"{BG_RED}[BUGCHECK] [{module_name}] Corrupted CMakeCache detected. Clearing build dir and retrying...{RST}")
        shutil.rmtree(build_dir, ignore_errors=True)
        os.makedirs(build_dir, exist_ok=True)
        # Re-run from scratch
        result = subprocess.run(cmake_flags, cwd=cwd, check=False, capture_output=True, text=True)
    return result

def main():
    global daemon_process, server_process, watchdog_process, vad_process
    
    current_log_path = setup_logging()
    
    # 0. Verify all critical prerequisites are installed
    verify_environment()
    
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
        # GGML: Built in-tree by llama.cpp (vendor/llama.cpp/build/ggml/)
        # =====================================================================
        # The GPR links against vendor/llama.cpp/build/ggml/src/libggml*.a
        # No separate ggml build needed — llama.cpp compiles its own ggml
        # as part of its cmake build. This ensures version consistency.
        # [VITAL-DO-NOT-REMOVE] Never use Homebrew's ggml.

        # =====================================================================
        # llama.cpp: clone → fetch+pull latest → rebuild if updated
        # =====================================================================
        # We always fetch+pull so we get the latest fixes.
        # llama.cpp builds ggml in-tree. The GPR links the in-tree build.
        llama_dir = os.path.abspath(os.path.join(BASE_DIR, "vendor", "llama.cpp"))
        llama_build_dir = os.path.join(llama_dir, "build")
        llama_lib = os.path.join(llama_build_dir, "src", "libllama.a")
        llama_start = time.time()

        if not os.path.exists(llama_dir):
            print(f"[LLAMA] [{time.strftime('%H:%M:%S')}] Cloning llama.cpp...")
            subprocess.run(
                ["git", "clone", "https://github.com/ggml-org/llama.cpp.git", llama_dir],
                check=False
            )
            checkout_latest_release(llama_dir, "LLAMA")
            needs_build = True
        else:
            old_head = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=llama_dir,
                capture_output=True, text=True
            ).stdout.strip()
            print(f"[LLAMA] [{time.strftime('%H:%M:%S')}] Fetching latest llama.cpp release...")
            checkout_latest_release(llama_dir, "LLAMA")
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
            print(f"[LLAMA] [{time.strftime('%H:%M:%S')}] CMake flags: -DGGML_NATIVE=ON -DLLAMA_BUILD_TOOLS=ON -DBUILD_SHARED_LIBS=OFF")
            os.makedirs(llama_build_dir, exist_ok=True)
            cmake_flags = ["cmake", "-B", "build", "-DGGML_NATIVE=ON", "-DLLAMA_BUILD_TOOLS=ON", "-DBUILD_SHARED_LIBS=OFF"]
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
            result = safe_cmake_configure(cmake_flags, cwd=llama_dir, build_dir=llama_build_dir, module_name="LLAMA")
            if result.returncode != 0:
                print(f"{BG_RED}[BUGCHECK] [LLAMA] [{time.strftime('%H:%M:%S')}] CMake configure FAILED{RST}")
                if result.stderr:
                    print(f"[LLAMA] [{time.strftime('%H:%M:%S')}] stderr: {result.stderr[-500:]}")
            else:
                print(f"[LLAMA] [{time.strftime('%H:%M:%S')}] CMake configure OK, building...")
                # DO NOT SUPPRESS VERBOSITY IF YOU ARE NOT OVERCONFIDENT
                result = subprocess.run(["cmake", "--build", "build", "--config", "Release", "-j", "--verbose"], cwd=llama_dir, check=False, capture_output=True, text=True)
                llama_elapsed = time.time() - llama_start
                if result.returncode == 0:
                    print(f"[LLAMA] [{time.strftime('%H:%M:%S')}] Build SUCCESS in {llama_elapsed:.1f}s")
                else:
                    print(f"{BG_RED}[BUGCHECK] [LLAMA] [{time.strftime('%H:%M:%S')}] Build FAILED in {llama_elapsed:.1f}s{RST}")
                    if result.stderr:
                        print(f"[LLAMA] [{time.strftime('%H:%M:%S')}] stderr: {result.stderr[-500:]}")
        else:
            llama_elapsed = time.time() - llama_start
            print(f"[LLAMA] [{time.strftime('%H:%M:%S')}] Library exists, skipping build")
        
        # Ensure mtmd (multimodal) library is built
        mtmd_lib = os.path.join(llama_build_dir, "tools", "mtmd", "libmtmd.a")
        mtmd_start = time.time()
        if not os.path.exists(mtmd_lib):
            print(f"[MTMD] [{time.strftime('%H:%M:%S')}] Building mtmd (multimodal) library...")
            # DO NOT SUPPRESS VERBOSITY IF YOU ARE NOT OVERCONFIDENT
            result = subprocess.run(["cmake", "--build", "build", "--target", "mtmd", "-j", "--verbose"], cwd=llama_dir, check=False, capture_output=True, text=True)
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
                print(f"{BG_RED}[BUGCHECK] [MTMD] [{time.strftime('%H:%M:%S')}] Build FAILED in {mtmd_elapsed:.1f}s{RST}")
                if result.stdout:
                    print(f"[MTMD] [{time.strftime('%H:%M:%S')}] stdout: {result.stdout[-500:]}")
                if result.stderr:
                    print(f"[MTMD] [{time.strftime('%H:%M:%S')}] stderr: {result.stderr[-500:]}")
        else:
            mtmd_elapsed = time.time() - mtmd_start
            mtmd_size = os.path.getsize(mtmd_lib)
            print(f"[MTMD] [{time.strftime('%H:%M:%S')}] Library exists ({mtmd_size:,} bytes), skipping build")

        # Check and clone kokoro-onnx
        kokoro_dir = os.path.abspath(os.path.join(BASE_DIR, "vendor", "kokoro-onnx"))
        if not os.path.exists(kokoro_dir):
            print("[*] Cloning kokoro-onnx...")
            subprocess.run(["git", "clone", "https://github.com/thewh1teagle/kokoro-onnx", kokoro_dir], check=False)
            checkout_latest_release(kokoro_dir, "KOKORO-ONNX")
        else:
            print("[*] kokoro-onnx already exists, skipping clone.")
            
        kokoclone_dir = os.path.abspath(os.path.join(BASE_DIR, "vendor", "kokoclone"))
        if not os.path.exists(kokoclone_dir):
            print("[*] Cloning KokoClone Zero-Shot Repository...")
            subprocess.run(["git", "clone", "https://github.com/Ashish-Patnaik/kokoclone.git", kokoclone_dir], check=True)
            checkout_latest_release(kokoclone_dir, "KOKOCLONE")
        else:
            print("[*] kokoclone already exists, skipping clone.")

        # Ensure Kokoro TTS component dependencies are installed in an isolated venv
        kokoro_comp_dir = os.path.abspath(os.path.join(BASE_DIR, "vendor", "tts_kokoro_component"))
        kokoro_venv_dir = os.path.join(kokoro_comp_dir, "venv")
        if not os.path.exists(kokoro_venv_dir):
            print("[*] Creating dedicated virtual environment for Kokoro TTS (Python 3.12)...")
            subprocess.run(["python3.12", "-m", "venv", kokoro_venv_dir], check=True)
            
        print("[*] Installing Kokoro TTS requirements...")
        kokoro_pip = os.path.join(kokoro_venv_dir, "bin", "pip") if platform.system() != "Windows" else os.path.join(kokoro_venv_dir, "Scripts", "pip.exe")
        subprocess.run([kokoro_pip, "install", "-r", os.path.join(kokoro_comp_dir, "requirements.txt")], check=False)

        # Check and clone moonshine
        moonshine_dir = os.path.abspath(os.path.join(BASE_DIR, "vendor", "moonshine"))
        if not os.path.exists(moonshine_dir):
            print("[*] Cloning moonshine...")
            subprocess.run(["git", "clone", "https://github.com/moonshine-ai/moonshine.git", moonshine_dir], check=False)
            checkout_latest_release(moonshine_dir, "MOONSHINE")
            
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
            result = safe_cmake_configure(["cmake", ".."], cwd=moonshine_build_dir, build_dir=moonshine_build_dir, module_name="MOONSHINE")
            subprocess.run(["make", f"-j{threads}"], cwd=moonshine_build_dir, check=False)
        else:
            print("[*] moonshine core library exists, skipping cmake build.")

        # Check and download Moonshine models
        moonshine_models_dir = os.path.abspath(os.path.join(BASE_DIR, "vendor", "moonshine", "models"))
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
        sd_cpp_dir = os.path.abspath(os.path.join(BASE_DIR, "vendor", "stable-diffusion.cpp"))
        sd_cpp_built = os.path.join(sd_cpp_dir, "build")
        sd_cpp_lib_static = os.path.join(sd_cpp_built, "libstable-diffusion.a")
        sd_cpp_lib_shared = os.path.join(sd_cpp_built, "libstable-diffusion.dylib") if platform.system() == "Darwin" else os.path.join(sd_cpp_built, "libstable-diffusion.so")
        sd_cpp_start = time.time()

        if not os.path.exists(sd_cpp_dir):
            print(f"[SD-CPP] [{time.strftime('%H:%M:%S')}] Cloning stable-diffusion.cpp...")
            subprocess.run(
                ["git", "clone", "https://github.com/leejet/stable-diffusion.cpp.git", sd_cpp_dir],
                check=False
            )
            checkout_latest_release(sd_cpp_dir, "SD-CPP")
            needs_build = True
        else:
            old_head = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=sd_cpp_dir,
                capture_output=True, text=True
            ).stdout.strip()
            print(f"[SD-CPP] [{time.strftime('%H:%M:%S')}] Fetching latest stable-diffusion.cpp release...")
            checkout_latest_release(sd_cpp_dir, "SD-CPP")
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
            result = safe_cmake_configure(cmake_flags, cwd=sd_cpp_built, build_dir=sd_cpp_built, module_name="SD-CPP")
            if result.returncode != 0:
                print(f"{BG_RED}[BUGCHECK] [SD-CPP] [{time.strftime('%H:%M:%S')}] CMake FAILED: {result.stderr[-500:]}{RST}")
            else:
                # DO NOT SUPPRESS VERBOSITY IF YOU ARE NOT OVERCONFIDENT
                result = subprocess.run(
                    ["cmake", "--build", ".", "--config", "Release", "-j", "--verbose"],
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
                    print(f"{BG_RED}[BUGCHECK] [SD-CPP] [{time.strftime('%H:%M:%S')}] Build FAILED in {sd_elapsed:.1f}s{RST}")
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
        kokoro_models_dir = os.path.abspath(os.path.join(BASE_DIR, "vendor", "kokoro_models"))
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

        #  SHA256 hashes verified from HuggingFace repo metadata.
        #  None = no hash available, skip verification.
        flux_models_to_download = [
            # Diffusion model Q2_K (~4GB) — fits 9B-class VRAM budget
            {
                "url": "https://huggingface.co/city96/FLUX.1-schnell-gguf/resolve/main/flux1-schnell-Q2_K.gguf?download=true",
                "output": "flux1-schnell.gguf",
                "sha256": None  # ~4GB, too large to pre-verify
            },
            # T5-XXL text encoder Q4_0 GGUF (~2.9GB) — small enough for VRAM
            {
                "url": "https://huggingface.co/Phil2Sat/T5XXL-Unchained-GGUF/resolve/main/Kaoru8-t5xxl-unchained-Q4_0.gguf?download=true",
                "output": "flux1-t5xxl.gguf",
                "sha256": None
            },
            # CLIP-L text encoder (safetensors, ~246MB — small, always fits)
            {
                "url": "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors?download=true",
                "output": "clip_l.safetensors",
                "sha256": "660c6f5b1abae9dc498ac2d21e1347d2abdb0cf6c0c0c8576cd796491d9a6cdd"
            },
            # VAE (safetensors, ~335MB — public mirror, BFL repos are gated)
            {
                "url": "https://huggingface.co/ffxvs/vae-flux/resolve/main/ae.safetensors?download=true",
                "output": "ae.safetensors",
                "sha256": "afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38"
            },
            # SD refinement model (~1.9GB — Stage 2 img2img upscale after FLUX sparse output)
            # Architecture: FLUX Q2_K sparse → add noise → SD refinement upscale
            {
                "url": "https://huggingface.co/second-state/stable-diffusion-v1-5-GGUF/resolve/main/stable-diffusion-v1-5-pruned-emaonly-Q8_0.gguf?download=true",
                "output": "sd-refinement.gguf",
                "sha256": None
            }
        ]

        def sha256_file(filepath):
            """Compute SHA256 of a file, streaming in chunks for large files."""
            h = hashlib.sha256()
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(8192 * 1024), b""):
                    h.update(chunk)
            return h.hexdigest()

        def download_with_retry(url, output_path, expected_sha256=None):
            """Download a file with infinite retry, resume, and SHA256 verification."""
            attempt = 0
            while True:
                attempt += 1
                print(f"[*] Downloading {os.path.basename(output_path)} (attempt #{attempt})...")
                result = subprocess.run(
                    ["wget", "-c", "-t", "0", "--timeout=30", "--waitretry=5",
                     "--show-progress", url, "-O", output_path],
                    check=False, timeout=None
                )
                if result.returncode != 0:
                    print(f"{BG_RED}[BUGCHECK] [!] wget failed (code {result.returncode}), retrying in 5s...{RST}")
                    time.sleep(5)
                    continue

                # wget succeeded — verify SHA256 if provided
                if expected_sha256:
                    print(f"[*] Verifying SHA256 for {os.path.basename(output_path)}...")
                    actual_sha256 = sha256_file(output_path)
                    if actual_sha256 == expected_sha256:
                        print(f"[+] {os.path.basename(output_path)} OK (hash verified)")
                        return True
                    else:
                        print(f"[!] SHA256 MISMATCH: expected={expected_sha256} actual={actual_sha256}")
                        print("[!] Corrupted download, deleting and retrying...")
                        os.remove(output_path)
                        time.sleep(5)
                        continue
                else:
                    print(f"[+] {os.path.basename(output_path)} downloaded ({os.path.getsize(output_path):,} bytes)")
                    return True

        for model in flux_models_to_download:
            target_path = os.path.join(flux_models_dir, model["output"])
            expected_sha256 = model.get("sha256")

            if os.path.exists(target_path):
                if expected_sha256:
                    actual_sha256 = sha256_file(target_path)
                    if actual_sha256 == expected_sha256:
                        print(f"[SKIP] {model['output']} exists and hash verified ({os.path.getsize(target_path):,} bytes)")
                        continue
                    else:
                        print(f"[REHASH] {model['output']} hash mismatch, re-downloading...")
                        os.remove(target_path)
                else:
                    print(f"[SKIP] {model['output']} exists ({os.path.getsize(target_path):,} bytes)")
                    continue

            download_with_retry(model["url"], target_path, expected_sha256)

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
        # Update version.ads with current git hash before building
        version_script = os.path.join(BASE_DIR, "scripts", "update_version.sh")
        if os.path.exists(version_script):
            subprocess.run(["bash", version_script], cwd=BASE_DIR, check=False)
        subprocess.run([alr_cmd, "build"], env=env, cwd=BASE_DIR, check=True)
        
        # =====================================================================
        # VERIFICATION STAGE: Formal Proofs & Fuzzing
        # =====================================================================
        if "--verify" in sys.argv or "--test-build-integrity-check" in sys.argv:
            print("\n" + "="*70)
            print("  RUNNING FORMAL VERIFICATION & STABILITY ANALYSIS")
            print("="*70)
            
            # 1. GNATprove Formal Verification
            print("\n[*] Stage: GNATprove SPARK Static Analysis...")
            prove_cmd = [alr_cmd, "exec", "--", "gnatprove", "-P", "adelaide_spark.gpr", 
                         "--level=4", "--prover=cvc5,z3,altergo", "--timeout=60", 
                         "--memlimit=2000", "--steps=0", "--counterexamples=on", 
                         "--report=fail", "--warnings=error", "-j0"]
            
            try:
                subprocess.run(prove_cmd, cwd=BASE_DIR, env=env, check=True)
                print("[+] GNATprove: Formal verification PASSED.")
            except subprocess.CalledProcessError:
                print(f"{BG_RED}[BUGCHECK] [!] GNATprove: Formal verification FAILED. Check obj/spark/gnatprove/gnatprove.out{RST}")
                if "--strict-verify" in sys.argv:
                    sys.exit(1)
            
            # 2. AFL++ Fuzzing Environment Check
            print("\n[*] Stage: AFL++ Fuzzing Readiness Check...")
            fuzz_ready = False
            for compiler in ["afl-clang-fast", "afl-gcc-fast", "afl-clang-lto"]:
                if shutil.which(compiler):
                    print(f"[+] AFL++ compiler found: {compiler}")
                    fuzz_ready = True
                    break
            
            if fuzz_ready and shutil.which("afl-fuzz"):
                print("[+] AFL++ environment is fully ready for binary torture.")
            else:
                print("[!] AFL++ environment incomplete. Fuzzing skipped.")
            
            print("\n" + "="*70)
            print("  VERIFICATION STAGE COMPLETE")
            print("="*70 + "\n")

        print("[*] Building Vite Frontend for Sidecar UI...")
        frontend_dir = os.path.join(BASE_DIR, "ui", "frontend")
        if os.path.exists(frontend_dir):
            npm_cmd = "npm.cmd" if platform.system() == "Windows" else "npm"
            subprocess.run([npm_cmd, "install"], cwd=frontend_dir, check=True)
            print("[*] Running auto npm audit fix to resolve vulnerabilities...")
            subprocess.run([npm_cmd, "audit", "fix"], cwd=frontend_dir, check=False)
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
                print(f"{BG_RED}[BUGCHECK] [!] Self-Integrity Quality Check FAILED.{RST}")
                print(result.stdout)
                print("[!] Emergency Shutdown: Ruff quality violations detected.")
                sys.exit(1)
            else:
                print("[+] Self-Integrity Quality Check PASSED.")
        except Exception as e:
            print(f"{BG_RED}[BUGCHECK] [!] Error executing Ruff integrity check: {e}{RST}")
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
            result = subprocess.run([pyvenv_pyrefly, "check", lsh_worker], capture_output=True, text=True)
            if result.returncode != 0:
                print(f"{BG_RED}[BUGCHECK] [!] LSH pyrefly type-check FAILED.{RST}")
                print(result.stdout)
                print(result.stderr)
                print("[!] Emergency Shutdown: pyrefly violations detected.")
                sys.exit(1)
        else:
            print("[LSH] pyrefly not found in venv, skipping type-check.")
        # Self-check: ruff lint on worker script
        pyvenv_ruff = os.path.join(pyvenv_dir, "bin", "ruff")
        if os.path.exists(pyvenv_ruff):
            print("[LSH] Running ruff lint on worker...")
            result = subprocess.run([pyvenv_ruff, "check", lsh_worker], capture_output=True, text=True)
            if result.returncode != 0:
                print(f"{BG_RED}[BUGCHECK] [!] LSH ruff lint FAILED.{RST}")
                print(result.stdout)
                print(result.stderr)
                print("[!] Emergency Shutdown: ruff violations detected.")
                sys.exit(1)
        else:
            print("[LSH] ruff not found in venv, skipping lint.")
        print("[LSH] QRNN worker bootstrap complete.")
    else:
        print(f"[!] LSH requirements not found at {lsh_reqs}, skipping QRNN worker setup.")

    # =====================================================================
    # VAD ONNX Sidecar Worker: Python venv bootstrap
    # =====================================================================
    vad_worker_script = os.path.join(BASE_DIR, "vad_component", "vad_worker.py")
    if os.path.exists(vad_worker_script):
        print("[VAD] Bootstrapping ONNX VAD worker...")
        pyvenv_dir = os.path.join(BASE_DIR, "pyvenv")
        pyvenv_python = os.path.join(pyvenv_dir, "bin", "python3") if platform.system() != "Windows" else os.path.join(pyvenv_dir, "Scripts", "python.exe")
        if not os.path.exists(pyvenv_python):
            subprocess.run([sys.executable, "-m", "venv", pyvenv_dir], check=True)
        pyvenv_pip = os.path.join(pyvenv_dir, "bin", "pip") if platform.system() != "Windows" else os.path.join(pyvenv_dir, "Scripts", "pip.exe")
        
        print("[VAD] Installing onnxruntime...")
        subprocess.run([pyvenv_pip, "install", "onnxruntime", "numpy"], check=False)
        print("[VAD] VAD worker bootstrap complete.")

    # Handle integrity check flag
    test_build_integrity = False
    if "--test-build-integrity-check" in sys.argv:
        print("[*] Test build integrity check: will launch server and invoke benchmark.")
        test_build_integrity = True

    # Parse arguments
    launch_gui = True
    if "--no-gui" in sys.argv or test_build_integrity:
        launch_gui = False
    
    run_benchmark = False
    if "--benchmark" in sys.argv or test_build_integrity:
        run_benchmark = True

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
    moonshine_onnx = os.path.join(BASE_DIR, "vendor", "moonshine", "core", "third-party", "onnxruntime", "lib", "macos", arch)
    
    if platform.system() == "Darwin":
        env["DYLD_LIBRARY_PATH"] = f"{moonshine_onnx}:{env.get('DYLD_LIBRARY_PATH', '')}"
    
    # Run server directly (ALIRE wrapper changes CWD which breaks relative model paths)
    # Write server launch args to file so the watchdog can relaunch with same args.
    server_args = ["--host", server_host, "--port", server_port]
    server_args_file = os.path.join(BASE_DIR, "run", "adelaide_server.args")
    with open(server_args_file, "w") as f:
        f.write(" ".join(server_args))

    # [DO NOT REMOVE] Generate SSL certificate if not exists
    # This enables HTTPS for secure communication between frontend and backend.
    cert_script = os.path.join(BASE_DIR, "scripts", "generate_cert.py")
    if os.path.exists(cert_script):
        print("[*] Checking SSL certificate...")
        cert_result = subprocess.run(
            [sys.executable, cert_script],
            cwd=BASE_DIR,
            capture_output=True,
            text=True
        )
        if cert_result.returncode == 0:
            print("[*] SSL certificate ready")
        else:
            print(f"{BG_RED}[BUGCHECK] [!] SSL certificate generation failed: {cert_result.stderr}{RST}")
            print("[!] Falling back to HTTP mode")

    # [DO NOT REMOVE] Verbose server launch info
    print(f"[*] [Launch-V] Server binary: {server_path}")
    print(f"[*] [Launch-V] Server args: {server_args}")
    print(f"[*] [Launch-V] Server CWD: {BASE_DIR}")
    print(f"[*] [Launch-V] DYLD_LIBRARY_PATH: {env.get('DYLD_LIBRARY_PATH', 'NOT SET')}")

    # Inject log file path so the Ada server can tail it for SSE benchmarking
    env["ADELAIDE_LOG_FILE"] = current_log_path

    # Launch server through tee so its output goes to terminal + log file
    tee_process = subprocess.Popen(
        ["tee", "-a", current_log_path],
        stdin=subprocess.PIPE,
        start_new_session=True
    )
    server_process = subprocess.Popen([server_path] + server_args, cwd=BASE_DIR, env=env,
                                       stdout=tee_process.stdin, stderr=subprocess.STDOUT,
                                       start_new_session=True)

    # [DO NOT REMOVE] Verbose PID tracking
    print(f"[*] [Launch-V] Server PID: {server_process.pid}")
    print(f"[*] [Launch-V] Server args file: {server_args_file}")

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
        
        def watchdog_monitor(path, w_env, log_path):
            global watchdog_process
            while True:
                w_exit = watchdog_process.wait()
                if os.path.exists(os.path.join(BASE_DIR, "run", ".shutdown_requested")):
                    break
                if w_exit in (0, 9, -9):
                    break
                print(f"\n[*] Watchdog crashed (code {w_exit})! Relaunching instantly...")
                with open(log_path, "a") as wlog2:
                    watchdog_process = subprocess.Popen(
                        [path], cwd=BASE_DIR, env=w_env,
                        stdout=wlog2, stderr=subprocess.STDOUT,
                        start_new_session=True)
        
        import threading
        t = threading.Thread(target=watchdog_monitor, args=(watchdog_path, watchdog_env, watchdog_log), daemon=True)
        t.start()
    else:
        print("[!] Watchdog binary not found at", watchdog_path, "- skipping")

    # Launch VAD ONNX Sidecar
    if os.path.exists(vad_worker_script):
        print("[*] Booting VAD ONNX Sidecar...")
        vad_log = os.path.join(BASE_DIR, "run", "vad_worker.log")
        with open(vad_log, "a") as vlog:
            vad_process = subprocess.Popen(
                [pyvenv_python, vad_worker_script], cwd=BASE_DIR, env=env,
                stdout=vlog, stderr=subprocess.STDOUT,
                start_new_session=True)

    if run_benchmark:
        print("[*] Booting benchmark runner thread...")
        def benchmark_runner():
            import time, urllib.request, json
            print("[Benchmark] Waiting 15s for server to settle...")
            time.sleep(15)
            url = f"http://{server_host}:{server_port}/api/snowballEnagaValidationBenchmark"
            print(f"[Benchmark] Invoking {url} (Performance)...")
            success = False
            try:
                data = json.dumps({"benchmark_type": "performance"}).encode('utf-8')
                req = urllib.request.Request(url, data=data, headers={
                    'Content-Type': 'application/json',
                    'x-api-key': 'IknowtheConsequencesAndWouldLockupTheServerForHours'
                }, method='POST')
                start_t = time.time()
                with urllib.request.urlopen(req, timeout=300) as res:
                    status = res.getcode()
                    print(f"[Benchmark] Connected. HTTP {status}")
                    
                    while True:
                        line = res.readline().decode('utf-8')
                        if not line:
                            break
                        line = line.strip()
                        if line.startswith("data: "):
                            payload = line[6:]
                            if payload == "[DONE]":
                                success = True
                                break
                            
                            try:
                                parsed = json.loads(payload)
                                if "type" in parsed and parsed["type"] == "log":
                                    print(f"[Ada-Log] {parsed.get('line', '')}")
                                elif "type" in parsed and parsed["type"] == "progress":
                                    print(f"[Benchmark Progress] {payload}")
                                elif "performance" in parsed:
                                    print("[Benchmark] Scoring Report:")
                                    print(json.dumps(parsed, indent=2))
                                    success = True
                            except json.JSONDecodeError:
                                print(f"[SSE Raw] {payload}")

                    elapsed = time.time() - start_t
                    print(f"[Benchmark] Completed in {elapsed:.2f}s")
            except Exception as e:
                print(f"[!] Benchmark failed: {e}")

                print("[*] Running comprehensive loopback API tests...")
                # We will test all endpoints from the API reference
                tests = [
                    ("Server Root (GET)", f"http://{server_host}:{server_port}/", None, "GET"),
                    ("Server Root (HEAD)", f"http://{server_host}:{server_port}/", None, "HEAD"),
                    ("Health / Power", f"http://{server_host}:{server_port}/api/power", None, "GET"),
                    ("Telemetry", f"http://{server_host}:{server_port}/api/telemetry", None, "GET"),
                    ("Version", f"http://{server_host}:{server_port}/api/version", None, "GET"),
                    ("Process Status", f"http://{server_host}:{server_port}/api/ps", None, "GET"),
                    ("Zenith Routine", f"http://{server_host}:{server_port}/api/ZenithRoutine", None, "GET"),
                    ("List Models (v1)", f"http://{server_host}:{server_port}/v1/models", None, "GET"),
                    ("Ollama Tags", f"http://{server_host}:{server_port}/api/tags", None, "GET"),
                    
                    # POST requests
                    ("OpenAI Chat", f"http://{server_host}:{server_port}/v1/chat/completions", {"model": "Snowball-Enaga", "messages": [{"role": "user", "content": "ping"}]}, "POST"),
                    ("OpenAI Completions", f"http://{server_host}:{server_port}/v1/completions", {"model": "Snowball-Enaga", "prompt": "ping"}, "POST"),
                    ("OpenAI Embeddings", f"http://{server_host}:{server_port}/v1/embeddings", {"model": "Snowball-Enaga", "input": "ping"}, "POST"),
                    ("Claude Messages", f"http://{server_host}:{server_port}/v1/messages", {"model": "Snowball-Enaga", "messages": [{"role": "user", "content": "ping"}], "max_tokens": 10}, "POST"),
                    ("Ollama Chat", f"http://{server_host}:{server_port}/api/chat", {"model": "Snowball-Enaga", "messages": [{"role": "user", "content": "ping"}], "stream": False}, "POST"),
                    ("Ollama Generate", f"http://{server_host}:{server_port}/api/generate", {"model": "Snowball-Enaga", "prompt": "ping", "stream": False}, "POST"),
                    ("Ollama Embeddings", f"http://{server_host}:{server_port}/api/embeddings", {"model": "Snowball-Enaga", "prompt": "ping"}, "POST"),
                    ("Ollama Show", f"http://{server_host}:{server_port}/api/show", {"name": "Snowball-Enaga"}, "POST"),
                    ("AGC/ACP", f"http://{server_host}:{server_port}/api/acp", {"jsonrpc": "2.0", "method": "chat/completion", "params": {"prompt": "ping"}, "id": 1}, "POST"),
                    
                    # Media / specialized APIs
                    ("TTS Kokoro", f"http://{server_host}:{server_port}/v1/audio/speech", {"input": "ping", "voice": "default", "response_format": "wav"}, "POST"),
                    ("Image Gen (FLUX)", f"http://{server_host}:{server_port}/v1/images/generations", {"prompt": "ping", "n": 1, "size": "1024x1024"}, "POST"),
                ]
                
                all_passed = True
                for name, endpoint, payload, method in tests:
                    try:
                        req_data = json.dumps(payload).encode('utf-8') if payload else None
                        headers = {'Content-Type': 'application/json'} if payload else {}
                        req = urllib.request.Request(endpoint, data=req_data, headers=headers, method=method)
                        with urllib.request.urlopen(req, timeout=30) as res:
                            code = res.getcode()
                            if code in (200, 201, 204):
                                print(f"[+] {name} Test: PASSED (HTTP {code})")
                            else:
                                print(f"[-] {name} Test: FAILED (HTTP {code})")
                                all_passed = False
                    except urllib.error.HTTPError as e:
                        # Some endpoints might correctly return 400 or 401 if not fully configured,
                        # but ideally they shouldn't crash. 404 is a missing endpoint.
                        print(f"[-] {name} Test: HTTP ERROR {e.code}")
                        all_passed = False
                    except Exception as e:
                        print(f"[-] {name} Test: EXCEPTION ({e})")
                        all_passed = False
                
                if not all_passed:
                    success = False

            if test_build_integrity:
                if success:
                    print("[*] Test build integrity check passed! Exiting successfully.")
                    cleanup()
                else:
                    print("[!] Test build integrity check FAILED!")
                    os._exit(1)

        import threading
        b_thread = threading.Thread(target=benchmark_runner, daemon=True)
        b_thread.start()

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
            
        # [DO NOT REMOVE] macOS .app bundle for microphone/camera/screen capture permissions
        # On Darwin, create a proper .app bundle with Info.plist containing
        # NSMicrophoneUsageDescription, NSCameraUsageDescription, and
        # NSScreenCaptureUsageDescription for hardware access permissions.
        # The .app launches Terminal and runs the server with GUI.
        #
        # IMPORTANT: Only launch .app if NOT already running in Terminal.
        # If launched from Terminal, just run sidecar_ui.py directly to avoid bootloop.
        if sys.platform == "darwin":
            # Check if we're already in a Terminal session or launched from .app
            # ADELAIDE_LAUNCHED_FROM_APP is set by .app launcher script
            # TERM_SESSION_ID is set by bash/zsh when in terminal
            launched_from_app = os.environ.get("ADELAIDE_LAUNCHED_FROM_APP") == "1"
            in_terminal = os.environ.get("TERM_SESSION_ID") is not None
            
            # [DO NOT REMOVE] Clear stale flag after reading
            # Prevents false positives if flag persists in shell environment
            if launched_from_app:
                os.environ.pop("ADELAIDE_LAUNCHED_FROM_APP", None)
            
            if launched_from_app or in_terminal:
                # Already in Terminal or launched from .app - launch sidecar directly (no .app)
                print("[*] Running in Terminal - launching sidecar directly...")
                subprocess.run([sidecar_python, "sidecar_ui.py"], cwd=ui_dir)
            else:
                # Not in Terminal (e.g., launched from Finder) - use .app
                app_bundle_path = os.path.join(BASE_DIR, "run", "Adelaide Zephyrine Assistant.app")
                create_app_script = os.path.join(ui_dir, "create_macos_app.py")
                
                # Create .app bundle if it doesn't exist
                if not os.path.exists(app_bundle_path):
                    print("[*] Creating macOS .app bundle for microphone/camera permissions...")
                    subprocess.run([sidecar_python, create_app_script, "--output", app_bundle_path], cwd=ui_dir)
                
                # Launch via .app bundle for proper permissions
                print("[*] Launching Adelaide Zephyrine Assistant.app for hardware access...")
                subprocess.run(["open", app_bundle_path])
        else:
            # Non-Darwin: launch directly
            subprocess.run([sidecar_python, "sidecar_ui.py"], cwd=ui_dir)
    else:
        try:
            while True:
                exit_code = server_process.wait()
                shutdown_flag = os.path.join(BASE_DIR, "run", ".shutdown_requested")
                if exit_code == 0 or os.path.exists(shutdown_flag):
                    print(f"\n[*] Server exited cleanly or shutdown requested (code: {exit_code})")
                    # Clean shutdown — remove SIGKILL context cap so next boot starts fresh
                    cap_file = os.path.join(BASE_DIR, "run", ".oom_kill_ctx_cap")
                    if os.path.exists(cap_file):
                        os.remove(cap_file)
                        print(f"[*] Removed SIGKILL context cap: {cap_file}")
                    break

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

                # === PANIC RECOVERY: Generate plot + dump CSV/logs ===
                import time as _time
                epoch_s = int(_time.time())
                panic_log_path = os.path.join(LOGS_DIR, f"I_am_incompetent_Panicked_and_Never_Enough_PANIC_{epoch_s}.log")
                wcet_csv = os.path.join(BASE_DIR, "run", "wcet.csv")
                accel_csv = os.path.join(BASE_DIR, "run", "acceleration.csv")

                # Find latest run log
                latest_log = None
                if os.path.isdir(LOGS_DIR):
                    log_files = sorted(
                        [f for f in os.listdir(LOGS_DIR) if f.startswith("run_") and f.endswith(".log")],
                        reverse=True,
                    )
                    if log_files:
                        latest_log = os.path.join(LOGS_DIR, log_files[0])

                # Write panic log with full CSV + logs
                try:
                    with open(panic_log_path, "w") as pf:
                        pf.write("=== INCOMPETENT PANIC LOG ===\n")
                        pf.write(f"Epoch: {epoch_s}\n")
                        pf.write(f"Exit Code: {exit_code}\n")
                        pf.write(f"Signal: {sig_name} ({sig_val})\n\n")

                        pf.write("=== WCET CSV (run/wcet.csv) ===\n")
                        if os.path.exists(wcet_csv):
                            with open(wcet_csv) as f:
                                pf.write(f.read())
                        else:
                            pf.write("(no wcet.csv found)\n")

                        pf.write("=== ACCELERATION CSV (run/acceleration.csv) ===\n")
                        if os.path.exists(accel_csv):
                            with open(accel_csv) as f:
                                pf.write(f.read())
                        else:
                            pf.write("(no gpu.csv found)\n")

                        pf.write(f"\n=== RUN LOG ({latest_log or 'none'}) ===\n")
                        if latest_log and os.path.exists(latest_log):
                            with open(latest_log) as f:
                                pf.write(f.read())
                        else:
                            pf.write("(no run log found)\n")

                    print(f"[*] Panic log written: {panic_log_path}")
                except Exception as e:
                    print(f"[!] Failed to write panic log: {e}")

                # === SIGKILL CONTEXT CAP: Save the ctx size that OOM'd ===
                if sig_val == 9:
                    try:
                        import re as _re
                        cap_file = os.path.join(BASE_DIR, "run", ".oom_kill_ctx_cap")
                        cap_val = None
                        if latest_log and os.path.exists(latest_log):
                            with open(latest_log) as lf:
                                for line in lf:
                                    # Match: [CtxMonitor] LLM CTX:  7950 /  16384 tokens
                                    m = _re.search(r'LLM CTX:\s*\d+\s*/\s*(\d+)\s*tokens', line)
                                    if m:
                                        cap_val = int(m.group(1))
                        if cap_val:
                            with open(cap_file, "w") as cf:
                                cf.write(str(cap_val))
                            print(f"[*] SIGKILL context cap saved: {cap_val} tokens → {cap_file}")
                        else:
                            print("[*] SIGKILL detected but could not parse context size from log")
                    except Exception as e:
                        print(f"[!] Failed to write SIGKILL context cap: {e}")

                # Generate plot from CSVs
                try:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt
                    import csv

                    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
                    fig.suptitle(f"Adelaide Crash Report — Epoch {epoch_s} — Exit {exit_code}", fontsize=13)

                    # WCET plot
                    if os.path.exists(wcet_csv):
                        times, pipeline, elp0, elp1, elp2, elp3 = [], [], [], [], [], []
                        with open(wcet_csv) as f:
                            reader = csv.DictReader(f)
                            for row in reader:
                                try:
                                    t = int(row["uptime_s"].strip())
                                    p = int(row["pipeline_ns"].strip())
                                    e0 = int(row["elp0_ns"].strip())
                                    e1 = int(row["elp1_ns"].strip())
                                    e2 = int(row["elp2_ns"].strip())
                                    e3 = int(row["elp3_ns"].strip())
                                    times.append(t)
                                    pipeline.append(p)
                                    elp0.append(e0)
                                    elp1.append(e1)
                                    elp2.append(e2)
                                    elp3.append(e3)
                                except (ValueError, KeyError, AttributeError):
                                    continue
                        if times:
                            axes[0].plot(times, pipeline, label="Pipeline", linewidth=0.8)
                            axes[0].plot(times, elp0, label="ELP0", linewidth=0.5, alpha=0.7)
                            axes[0].plot(times, elp1, label="ELP1", linewidth=0.5, alpha=0.7)
                            axes[0].plot(times, elp2, label="ELP2", linewidth=0.5, alpha=0.7)
                            axes[0].plot(times, elp3, label="ELP3", linewidth=0.5, alpha=0.7)
                            axes[0].set_ylabel("WCET (ns)")
                            axes[0].legend(fontsize=7)
                            axes[0].set_title("WCET Timing")

                    # Acceleration plot
                    if os.path.exists(accel_csv):
                        times, free, total, pct, metal_broken = [], [], [], [], []
                        with open(accel_csv) as f:
                            reader = csv.DictReader(f)
                            for row in reader:
                                try:
                                    t = int(row["uptime_s"].strip())
                                    f_mb = int(row["free_mb"].strip())
                                    t_mb = int(row["total_mb"].strip())
                                    p = int(row["percent"].strip())
                                    mb = int(row["metal_broken"].strip())
                                    times.append(t)
                                    free.append(f_mb)
                                    total.append(t_mb)
                                    pct.append(p)
                                    metal_broken.append(mb)
                                except (ValueError, KeyError, AttributeError):
                                    continue
                        if times:
                            ax1 = axes[1]
                            ax2 = ax1.twinx()
                            ax1.plot(times, free, color="green", label="Free MB", linewidth=0.8)
                            ax1.plot(times, total, color="blue", label="Total MB", linewidth=0.5, alpha=0.7)
                            ax2.plot(times, pct, color="red", label="Free %", linewidth=0.5, alpha=0.7)
                            ax1.set_ylabel("Memory (MB)")
                            ax2.set_ylabel("Free %")
                            ax1.legend(fontsize=7, loc="upper left")
                            ax2.legend(fontsize=7, loc="upper right")
                            ax1.set_title("GPU Memory")
                            # Mark OOM events
                            for i, mb in enumerate(metal_broken):
                                if mb:
                                    axes[1].axvline(x=times[i], color="red", linestyle="--", alpha=0.3)

                    # Acceleration free % as heatmap-style fill
                    if os.path.exists(accel_csv):
                        times_pct, pcts = [], []
                        with open(accel_csv) as f:
                            reader = csv.DictReader(f)
                            for row in reader:
                                try:
                                    t = int(row["uptime_s"].strip())
                                    p = int(row["percent"].strip())
                                    times_pct.append(t)
                                    pcts.append(p)
                                except (ValueError, KeyError, AttributeError):
                                    continue
                        if times_pct:
                            axes[2].fill_between(times_pct, pcts, alpha=0.4, color="cyan")
                            axes[2].plot(times_pct, pcts, color="darkcyan", linewidth=0.6)
                            axes[2].set_ylabel("Free %")
                            axes[2].set_xlabel("Uptime (s)")
                            axes[2].set_title("GPU Free Memory %")
                            axes[2].set_ylim(0, 100)

                    plt.tight_layout()
                    plot_path = os.path.join(LOGS_DIR, f"I_am_incompetent_Panicked_and_Never_Enough_PANIC_{epoch_s}.png")
                    plt.savefig(plot_path, dpi=120)
                    plt.close()
                    print(f"[*] Crash plot saved: {plot_path}")
                except ImportError:
                    print("[!] matplotlib not installed — skipping crash plot (pip install matplotlib)")
                except Exception as e:
                    print(f"[!] Failed to generate crash plot: {e}")

                print("\n[*] Relaunching server instantly (JMP back Rebounce back)...")
                # Kill any lingering old daemon to prevent CSV write races
                subprocess.run(["pkill", "-9", "-f", "adelaide_server"],
                               stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
                import time as _kill_wait
                _kill_wait.sleep(0.5)  # Give OS time to release file handles
                tee_process = subprocess.Popen(
                    ["tee", "-a", current_log_path],
                    stdin=subprocess.PIPE,
                    start_new_session=True
                )
                server_process = subprocess.Popen([server_path] + server_args, cwd=BASE_DIR, env=env,
                                                  stdout=tee_process.stdin, stderr=subprocess.STDOUT,
                                                  start_new_session=True)
                print(f"[*] [Launch-V] Server PID (relaunch): {server_process.pid}")
        except KeyboardInterrupt:
            print("\n[*] Keyboard interrupt received. Shutting down...")
            pass
        
    # Wait for background processes to finish if main blocking process exits
    cleanup()

if __name__ == "__main__":
    main()
