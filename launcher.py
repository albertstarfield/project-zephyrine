#!/usr/bin/env python3
import os
import sys
import subprocess
import threading
import time
import signal
import atexit
import shutil
import platform
from datetime import datetime, date
import json  # For parsing conda info
import traceback
import compileall
#import requests #(Don't request at the top) But on the main after conda relaunch to make sure it's installed
from typing import Optional, Dict
try:
    from loguru import logger
    # Ensure default configuration for this module if needed
    logger.remove()
    logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>", level="DEBUG")
    logger.info("📝 Loguru logger configured successfully.")
except ImportError:
    import logging
    # Configure a basic logger that mimics loguru's method names
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s',
        stream=sys.stderr
    )
    logger = logging.getLogger(__name__)
    # Add a 'success' method for compatibility
    def success(msg, *args, **kwargs):
        logger.info(msg, *args, **kwargs)
    logger.success = success
    logger.warning("Loguru not found. Falling back to standard Python logging.")

# --- TUI Imports (with fallback) ---
try:
    from textual.app import App, ComposeResult
    from textual.containers import Grid
    from textual.widgets import Header, Footer, Log, Static
    from textual.reactive import reactive
    import psutil
    TUI_AVAILABLE = True
except ImportError:
    TUI_AVAILABLE = False
# --- End TUI Imports ---

# --- Configuration ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_MAIN_DIR = os.path.join(ROOT_DIR, "systemCore", "engineMain")
BACKEND_SERVICE_DIR = os.path.join(ROOT_DIR, "systemCore", "UIEngine", "backend-service")
FRONTEND_DIR = os.path.join(ROOT_DIR, "systemCore", "UIEngine", "frontend-face-zephyrine")
LICENSE_DIR = os.path.join(ROOT_DIR, "licenses")
LICENSE_FLAG_FILE = os.path.join(ROOT_DIR, ".license_accepted_v1")
CUDA_TOOLKIT_INSTALLED_FLAG_FILE = os.path.join(ROOT_DIR, ".CUDA_Toolkit_Installed")
# Near the top with other path configurations
RELAUNCH_LOG_DIR = os.path.join(ROOT_DIR, "logs") # Or any preferred log directory
RELAUNCH_STDOUT_LOG = os.path.join(RELAUNCH_LOG_DIR, "relaunched_launcher_stdout.log")
RELAUNCH_STDERR_LOG = os.path.join(RELAUNCH_LOG_DIR, "relaunched_launcher_stderr.log")
RELAUNCH_LOG_DIR = os.path.join(ROOT_DIR, "logs")
ENGINE_PID_FILE = os.path.join(RELAUNCH_LOG_DIR, "engine.pid") # NEW: Path for the engine's PID file

SETUP_COMPLETE_FLAG_FILE = os.path.join(ROOT_DIR, ".setup_complete_v2")

# --- MeloTTS Configuration ---
MELO_TTS_SUBMODULE_DIR_NAME = "MeloAudioTTS_SubEngine"
MELO_TTS_PATH = os.path.join(ENGINE_MAIN_DIR, MELO_TTS_SUBMODULE_DIR_NAME)
MELO_TTS_LIB_INSTALLED_FLAG_FILE = os.path.join(ROOT_DIR, ".melo_tts_lib_installed_v1") # NEW: For the library itself
MELO_TTS_DATA_INSTALLED_FLAG_FILE = os.path.join(ROOT_DIR, ".melo_tts_data_installed_v1") # RENAMED: For data deps

# --- ChatterboxTTS Configuration (NEW) ---
CHATTERBOX_TTS_SUBMODULE_DIR_NAME = "ChatterboxTTS_subengine"
CHATTERBOX_TTS_PATH = os.path.join(ENGINE_MAIN_DIR, CHATTERBOX_TTS_SUBMODULE_DIR_NAME)
CHATTERBOX_TTS_INSTALLED_FLAG_FILE = os.path.join(ROOT_DIR, ".chatterbox_tts_installed_v1")
# --- END ChatterboxTTS Configuration ---

# --- LaTeX-OCR Local Sub-Engine Configuration ---
LATEX_OCR_SUBMODULE_DIR_NAME = "LaTeX_OCR-SubEngine"
LATEX_OCR_PATH = os.path.join(ENGINE_MAIN_DIR, LATEX_OCR_SUBMODULE_DIR_NAME)
LATEX_OCR_INSTALLED_FLAG_FILE = os.path.join(ROOT_DIR, ".latex_ocr_subengine_installed_v1")

# --- Playwright Flag Configuration

PLAYWRIGHT_BROWSERS_INSTALLED_FLAG_FILE = os.path.join(ROOT_DIR, ".playwright_browsers_installed_v1")

# --- Conda Configuration ---
# CONDA_ENV_NAME is no longer used for creation if prefix is used, but can be a descriptive base for the folder
RUNTIME_ENV_FOLDER_NAME = "zephyrineRuntimeVenv"
CONDA_EXECUTABLE = None
# TARGET_RUNTIME_ENV_PATH will now be ROOT_DIR + RUNTIME_ENV_FOLDER_NAME
TARGET_RUNTIME_ENV_PATH = os.path.join(ROOT_DIR, RUNTIME_ENV_FOLDER_NAME) # DIRECTLY DEFINE THE TARGET PATH
ACTIVE_ENV_PATH = None

if os.path.exists("/etc/debian_version") and "TERMUX_VERSION" not in os.environ:
    # We are likely in the proot (glibc) environment
    _cache_suffix = "_proot_glibc.txt"
else:
    # We are likely in the base Termux or a standard desktop Linux/macOS/Windows env
    _cache_suffix = "_main_env.txt"

IS_IN_PROOT_ENV = os.path.exists("/etc/debian_version") and "TERMUX_VERSION" not in os.environ

CONDA_PATH_CACHE_FILE = os.path.join(ROOT_DIR, f".conda_executable_path_cache{_cache_suffix}")

# --- Node.js Configuration (NEW) ---
REQUIRED_NODE_MAJOR = 20 # Or 18, if your project has a strict preference, but 20 is current LTS.
REQUIRED_NPM_MAJOR = 10  # npm 10 comes bundled with Node.js 20.
# --- END Node.js Configuration ---

# --- Python Dynamic Versioning for Conda ---
EPOCH_DATE = date(2025, 6, 1)
INITIAL_PYTHON_MAJOR = 3
INITIAL_PYTHON_MINOR = 12
FALLBACK_PYTHON_MAJOR = 3
FALLBACK_PYTHON_MINOR = 12

# --- GNAT Ada Compiler Configuration ---
GNAT_INSTALLED_FLAG_FILE = os.path.join(ROOT_DIR, ".gnat_compiler_installed_v1")
GNAT_TOOLCHAIN_INSTALLED_FLAG_FILE = os.path.join(ROOT_DIR, ".gnat_toolchain_installed_v1")
GNAT_BUILD_DIR = os.path.join(ROOT_DIR, "gnat_community_build")
GNAT_SOURCE_URL = "https://github.com/AdaCore/gnat-community-sources/archive/refs/tags/gnat-community-2021.tar.gz"
GNAT_TAR_FILENAME = "gnat-community-2021.tar.gz"
# --- END GNAT Configuration ---


# --- START of new retry logic ---
MAX_SETUP_ATTEMPTS = 4  # Initial run + 3 retries
final_setup_failures = []


running_processes = [] # For services started by the current script instance
process_lock = threading.Lock()
relaunched_conda_process_obj = None
tui_shutdown_event = threading.Event()

# --- Executable Paths (will be redefined after Conda activation) ---
IS_WINDOWS = os.name == 'nt'
PYTHON_EXECUTABLE = sys.executable
PIP_EXECUTABLE = ""
HYPERCORN_EXECUTABLE = ""
NPM_CMD = 'npm.cmd' if IS_WINDOWS else 'npm'
GIT_CMD = 'git.exe' if IS_WINDOWS else 'git'
CMAKE_CMD = 'cmake.exe' if IS_WINDOWS else 'cmake'

# --- ZephyMesh Configuration (MODIFIED FOR GO) ---
ZEPHYMESH_DIR = os.path.join(ROOT_DIR, "zephyMeshNetwork")
ZEPHYMESH_NODE_BINARY = os.path.join(ZEPHYMESH_DIR, "zephymesh_node_compiled.exe" if IS_WINDOWS else "zephymesh_node_compiled")
ZEPHYMESH_API_PORT = 22000 # The port you configured in the Go app's flags
zephymesh_api_url: Optional[str] = f"http://127.0.0.1:{ZEPHYMESH_API_PORT}"
zephymesh_process: Optional[subprocess.Popen] = None
ZEPHYMESH_PORT_INFO_FILE = os.path.join(ROOT_DIR, "zephymesh_ports.json")
# --- END ZephyMesh Configuration ---

# --- License Acceptance & Optional Imports (Initial state) ---
TIKTOKEN_AVAILABLE = False

# --- Static Model Pool Configuration ---
STATIC_MODEL_POOL_DIR_NAME = "staticmodelpool"
STATIC_MODEL_POOL_PATH = os.path.join(ENGINE_MAIN_DIR, STATIC_MODEL_POOL_DIR_NAME)
MODELS_TO_DOWNLOAD = [
    {
        "filename": "deepscaler.gguf",
        "url": "https://huggingface.co/bartowski/agentica-org_DeepScaleR-1.5B-Preview-GGUF/resolve/main/agentica-org_DeepScaleR-1.5B-Preview-f16.gguf?download=true",
        "description": "DeepScaleR Model (Agent/Router)"
    },
    {
        "filename": "qwen3EmbedCore.gguf",
        "url": "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/main/Qwen3-Embedding-0.6B-Q8_0.gguf?download=true",
        "description": "Qwen3 Embedding Vectorization Model"
    },
    {
        "filename": "NanoTranslator-immersive_translate-0.5B-GGUF-Q4_K_M.gguf",
        "url": "https://huggingface.co/mradermacher/NanoTranslator-immersive_translate-0.5B-GGUF/resolve/main/NanoTranslator-immersive_translate-0.5B.Q5_K_M.gguf?download=true",
        "description": "NanoTranslator Model"
    },
    {
        "filename": "Qwen3LowLatency.gguf",
        "url": "https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q5_K_M.gguf?download=true",
        "description": "Qwen 3 5-bit Hybrid Direct Mode (Fast Augmented General)"
    },
    {
        "filename": "Qwen3DeepseekDecomposer.gguf",
        "url": "https://huggingface.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF/resolve/main/DeepSeek-R1-0528-Qwen3-8B-Q4_K_S.gguf?download=true",
        "description": "Qwen 3 Deepseek R1 Decomposer 8B model, suboptimal for toolcall but a really good thinker and decompose solve complex things"
    },
    {
        "filename": "Qwen3ToolCall.gguf",
        "url": "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf?download=true",
        "description": "Qwen 3 Tool Calling, Not good thinker but good for controlling and tool calling and coding"
    },
    {
        "filename": "Qwen3-VL-ImageDescripter.gguf",
        "url": "https://huggingface.co/NexaAI/Qwen3-VL-4B-Instruct-GGUF/resolve/main/Qwen3-VL-4B-Instruct.Q4_K.gguf?download=true",
        "description": "Qwen3 Image Descriptor VL Model"
    },
    {
        "filename": "whisper-large-v3-q5_0.gguf",
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-q5_0.bin?download=true",
        "description": "Whisper Large v3 Model (ASR)"
    },
    {
        "filename": "whisper-lowlatency-direct.gguf",
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin?download=true",
        "description": "Whisper Direct Low Latency (ASR)"
    },
    {
        "filename": "flux1-schnell.gguf",
        "url": "https://huggingface.co/Green-Sky/flux.1-schnell-GGUF/resolve/main/flux1-schnell-q2_k.gguf?download=true",
        "description": "FLUX.1 Schnell GGUF (Main Q2_K)"
    },
    {
        "filename": "flux1-ae.gguf",
        "url": "https://huggingface.co/Green-Sky/flux.1-schnell-GGUF/resolve/main/ae-f16.gguf?download=true",
        "description": "FLUX.1 VAE GGUF (FP16)"
    },
    {
        "filename": "flux1-clip_l.gguf",
        "url": "https://huggingface.co/Green-Sky/flux.1-schnell-GGUF/resolve/main/clip_l-q8_0.gguf?download=true",
        "description": "FLUX.1 CLIP L GGUF (Q8_0)"
    },
    {
        "filename": "flux1-t5xxl.gguf",
        "url": "https://huggingface.co/Green-Sky/flux.1-schnell-GGUF/resolve/main/t5xxl_q2_k.gguf?download=true",
        "description": "FLUX.1 T5 XXL GGUF (Q2_K)"
    },
    #https://huggingface.co/second-state/stable-diffusion-v1-5-GGUF/resolve/main/stable-diffusion-v1-5-pruned-emaonly-Q5_0.gguf?download=true
    {
        "filename": "sd-refinement.gguf",
        "url": "https://huggingface.co/second-state/stable-diffusion-v1-5-GGUF/resolve/main/stable-diffusion-v1-5-pruned-emaonly-Q5_0.gguf?download=true",
        "description": "Stable Diffusion Refinement PostFlux"
    }
]

# Llama.cpp Fork Configuration
LLAMA_CPP_PYTHON_REPO_URL = "https://github.com/abetlen/llama-cpp-python.git"
LLAMA_CPP_PYTHON_CLONE_DIR_NAME = "llama-cpp-python_build"
LLAMA_CPP_PYTHON_CLONE_PATH = os.path.join(ROOT_DIR, LLAMA_CPP_PYTHON_CLONE_DIR_NAME)
CUSTOM_LLAMA_CPP_INSTALLED_FLAG_FILE = os.path.join(ROOT_DIR, ".custom_llama_cpp_installed_v1_conda")

# Stable Diffusion cpp python Configuration
STABLE_DIFFUSION_CPP_PYTHON_REPO_URL = "https://github.com/william-murray1204/stable-diffusion-cpp-python.git"
STABLE_DIFFUSION_CPP_PYTHON_CLONE_DIR_NAME = "stable-diffusion-cpp-python_build"
STABLE_DIFFUSION_CPP_PYTHON_CLONE_PATH = os.path.join(ROOT_DIR, STABLE_DIFFUSION_CPP_PYTHON_CLONE_DIR_NAME)
CUSTOM_SD_CPP_PYTHON_INSTALLED_FLAG_FILE = os.path.join(ROOT_DIR, ".custom_sd_cpp_python_installed_v1_conda")

# pywhispercpp python configuration
PYWHISPERCPP_REPO_URL = "https://github.com/absadiki/pywhispercpp"
PYWHISPERCPP_CLONE_DIR_NAME = "pywhispercpp_build" # Name for the local clone directory
PYWHISPERCPP_CLONE_PATH = os.path.join(ROOT_DIR, PYWHISPERCPP_CLONE_DIR_NAME)
PYWHISPERCPP_INSTALLED_FLAG_FILE = os.path.join(ROOT_DIR, ".pywhispercpp_installed_v1")

# pyaria2c python configuraiton
ARIA2P_INSTALLED_FLAG_FILE = os.path.join(ROOT_DIR, ".aria2p_installed_v1")


FLAG_FILES_TO_RESET_ON_ENV_RECREATE = [
    SETUP_COMPLETE_FLAG_FILE,
    LICENSE_FLAG_FILE,
    MELO_TTS_LIB_INSTALLED_FLAG_FILE,  
    MELO_TTS_DATA_INSTALLED_FLAG_FILE,  
    CUSTOM_LLAMA_CPP_INSTALLED_FLAG_FILE,
    CUSTOM_SD_CPP_PYTHON_INSTALLED_FLAG_FILE,
    CONDA_PATH_CACHE_FILE,
    PYWHISPERCPP_INSTALLED_FLAG_FILE,
    ARIA2P_INSTALLED_FLAG_FILE, # From previous step
    CHATTERBOX_TTS_INSTALLED_FLAG_FILE,
    LATEX_OCR_INSTALLED_FLAG_FILE,
    PLAYWRIGHT_BROWSERS_INSTALLED_FLAG_FILE
]


# Global list to keep track of running subprocesses
running_processes = []
process_lock = threading.Lock()


# --- Helper Functions ---
def print_colored(prefix, message, color=None):
    now = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    prefix_color = "\033[94m"
    if prefix == "ERROR":
        prefix_color = "\033[91m"
    elif prefix == "WARNING":
        prefix_color = "\033[93m"
    elif prefix == "SUCCESS":
        prefix_color = "\033[92m"
    reset_color = "\033[0m"
    if sys.stdout.isatty():
        print(f"[{now} | {prefix_color}{prefix.ljust(10)}{reset_color}] {message.strip()}")
    else:
        print(f"[{now} | {prefix.ljust(10)}] {message.strip()}")


def _compile_watchtowers() -> bool:
    """
    Compiles both the Go (Thread1) and Ada (Thread2) watchdogs.
    Returns True if both succeed, False otherwise.
    """
    print_system("--- Compiling ZephyWatchtower Components ---")

    # --- Compile Go Watchdog (Thread 1) ---
    go_watchdog_dir = os.path.join(ROOT_DIR, "ZephyWatchtower")
    go_watchdog_src = os.path.join(go_watchdog_dir, "watchdog_thread1.go")
    go_mod_path = os.path.join(go_watchdog_dir, "go.mod")
    go_exe_name = "watchdog_thread1.exe" if IS_WINDOWS else "watchdog_thread1"
    go_exe_path = os.path.join(go_watchdog_dir, go_exe_name)

    print_system("Compiling Go Watchdog (Thread 1)...")
    if not os.path.exists(go_watchdog_src):
        print_error(f"Go Watchdog source code not found: {go_watchdog_src}")
        return False

    if not os.path.exists(go_mod_path):
        print_warning(f"go.mod not found in {go_watchdog_dir}. Initializing module...")
        try:
            subprocess.run(["go", "mod", "init", "ZephyWatchtower"], cwd=go_watchdog_dir, check=True)
        except Exception as e:
            print_error(f"Failed to initialize Go module: {e}")
            return False

    print_system("Ensuring Go module dependencies are downloaded ('go mod tidy')...")
    if not run_command(["go", "mod", "tidy"], cwd=go_watchdog_dir, name="GO-MOD-TIDY-WATCHDOG"):
        print_error("Failed to download Go dependencies for the watchdog. Build will likely fail.")
        # You can decide to return False here or let the build attempt fail.
        # Letting it continue gives a more specific error from the build command itself.

    try:
        build_command_go = ["go", "build", "-o", go_exe_path, "."]
        process_go = subprocess.run(build_command_go, cwd=go_watchdog_dir, capture_output=True, text=True, check=False, timeout=180)
        if process_go.returncode != 0:
            print_error(f"Failed to build Go Watchdog. RC: {process_go.returncode}\nSTDERR: {process_go.stderr.strip()}")
            return False
        print_system("Go Watchdog (Thread 1) built successfully.")
    except Exception as e:
        print_error(f"An unexpected error occurred during Go watchdog build: {e}")
        return False

    # --- Compile Ada Watchdog (Thread 2) ---
    ada_watchdog_dir = os.path.join(ROOT_DIR, "ZephyWatchtower", "watchdog_thread2")
    
    print_system("Compiling Ada Watchdog (Thread 2)...")
    if not os.path.isdir(ada_watchdog_dir):
        print_error(f"Ada Watchdog project directory not found: {ada_watchdog_dir}")
        return False

    if not shutil.which("alr"):
        print_error("'alr' command not found. Cannot build Ada watchdog. Please ensure Alire is in your PATH.")
        return False
        
    try:
        ### MODIFICATION START ###
        
        # Base command is now 'alr exec -- gprbuild', which we proved is more reliable.
        build_command_ada = ["alr", "exec", "--", "gprbuild"]

        # Check if the OS is macOS and add the specific linker flag if it is.
        if platform.system() == 'Darwin':
            print_system("macOS detected, adding specific linker flags for Ada build...")
            try:
                # Get the SDK path from the system
                sdk_path_result = subprocess.run(['xcrun', '--show-sdk-path'], capture_output=True, text=True, check=True)
                sdk_path = sdk_path_result.stdout.strip()
                # Append the necessary linker arguments to the command
                build_command_ada.extend(['-largs', f'-L{sdk_path}/usr/lib'])
            except Exception as e:
                print_error(f"Could not get macOS SDK path for Ada build: {e}")
                return False
        
        ### MODIFICATION END ###

        process_ada = subprocess.run(build_command_ada, cwd=ada_watchdog_dir, capture_output=True, text=True, check=False, timeout=300)
        if process_ada.returncode == 0:
            print_system("Ada Watchdog (Thread 2) built successfully.")
        else:
            print_error(f"Failed to build Ada Watchdog. RC: {process_ada.returncode}\nSTDERR: {process_ada.stderr.strip()}")
            return False
    except Exception as e:
        print_error(f"An unexpected error occurred during Ada watchdog build: {e}")
        return False

    print_system("--- All ZephyWatchtower components compiled successfully. ---")
    return True

def _compile_and_run_watchdogs():
    """
    Compiles both watchdogs and then runs them. This version is corrected
    to not require any arguments, as the start_service_process function
    now handles logging to files automatically.
    """
    # First, call the existing compile function
    if not _compile_watchtowers():
        # _compile_watchtowers already prints detailed errors
        print_error("Halting watchdog startup due to compilation failure.")
        return

    print_system("--- Launching Watchdog Services ---")

    # Run Go Watchdog
    go_watchdog_dir = os.path.join(ROOT_DIR, "ZephyWatchtower")
    go_exe_name = "watchdog_thread1.exe" if IS_WINDOWS else "watchdog_thread1"
    go_exe_path = os.path.join(go_watchdog_dir, go_exe_name)
    if os.path.exists(go_exe_path):
        # Define the process names for all targets
        backend_exe_name = "zephyrine-backend.exe" if IS_WINDOWS else "zephyrine-backend"
        backend_process_name = backend_exe_name.replace(".exe", "")

        # The process name is derived from the binary file name, without the .exe
        mesh_process_name = "zephymesh_node_compiled"

        # Construct the command with the new target included
        watchdog_command = [
            go_exe_path,
            f"--targets={backend_process_name},node,{mesh_process_name}",  # ADDED mesh_process_name
            f"--pid-file={ENGINE_PID_FILE}"
        ]
        start_service_process(watchdog_command, go_watchdog_dir, "GO-WATCHDOG")
    else:
        print_error(f"Go watchdog executable not found after compile: {go_exe_path}")

    # Run Ada Watchdog
    ada_watchdog_dir = os.path.join(ROOT_DIR, "ZephyWatchtower", "watchdog_thread2")
    ada_exe_name = "watchdog_thread2.exe" if IS_WINDOWS else "watchdog_thread2"
    ada_exe_path = os.path.join(ada_watchdog_dir, "bin", ada_exe_name)
    if os.path.exists(ada_exe_path):
        # The Ada watchdog will now supervise the Go watchdog.
        # We must provide the full command for the Go watchdog as arguments
        # to the Ada watchdog.

        # Define the command for the Go watchdog (same as before)
        backend_exe_name = "zephyrine-backend.exe" if IS_WINDOWS else "zephyrine-backend"
        backend_process_name = backend_exe_name.replace(".exe", "")
        go_watchdog_command = [
            go_exe_path,
            f"--targets={backend_process_name},node",
            f"--pid-file={ENGINE_PID_FILE}"
        ]

        # Now, construct the command for the Ada watchdog.
        # It's the Ada executable, followed by the entire Go command.
        ada_watchdog_command = [ada_exe_path] + go_watchdog_command

        print_system(f"ADA Watchdog will supervise command: {' '.join(go_watchdog_command)}")

        # The working directory should be the project root so that relative paths
        # in the supervised command (like for the PID file) work correctly.
        start_service_process(ada_watchdog_command, ROOT_DIR, "ADA-WATCHDOG")
    else:
        print_error(f"Ada watchdog executable not found after compile: {ada_exe_path}")

def _compile_and_get_watchdog_path() -> Optional[str]:
    """
    Ensures the Go watchdog binary is compiled and returns its path.
    Includes logic to initialize the Go module if it's missing.
    Returns None on failure.
    """
    watchdog_dir = os.path.join(ROOT_DIR, "ZephyWatchtower")
    watchdog_src_path = os.path.join(watchdog_dir, "watchdog_thread1.go")
    go_mod_path = os.path.join(watchdog_dir, "go.mod")
    executable_name = "watchdog_thread1.exe" if IS_WINDOWS else "watchdog_thread1"
    watchdog_exe_path = os.path.join(watchdog_dir, executable_name)

    if not os.path.exists(watchdog_src_path):
        print_error(f"Watchdog source code not found at: {watchdog_src_path}")
        return None

    if not shutil.which("go"):
        print_error("Go compiler ('go') not found in PATH. Cannot build watchdog.")
        return None

    # --- NEW: Check for go.mod and initialize if missing ---
    if not os.path.exists(go_mod_path):
        print_warning(f"go.mod file not found in {watchdog_dir}. Initializing module...")
        try:
            # The module name should match the directory for convention
            mod_init_command = ["go", "mod", "init", "ZephyWatchtower"]
            init_process = subprocess.run(
                mod_init_command,
                cwd=watchdog_dir,
                capture_output=True, text=True, check=True, timeout=30
            )
            print_system(f"Go module 'ZephyWatchtower' initialized successfully.")
        except Exception as e:
            print_error(f"Failed to initialize Go module: {e}")
            if hasattr(e, 'stderr'):
                print_error(f"STDERR: {e.stderr}")
            return None
    # --- END NEW ---

    print_system(f"Attempting to build Go watchdog in: {watchdog_dir}")
    try:
        build_command = ["go", "build", "-o", watchdog_exe_path, "."]
        process = subprocess.run(
            build_command,
            cwd=watchdog_dir,
            capture_output=True, text=True, check=False, timeout=180
        )
        if process.returncode == 0:
            print_system(f"Go watchdog built successfully: {watchdog_exe_path}")
            return watchdog_exe_path
        else:
            print_error(f"Failed to build Go watchdog. RC: {process.returncode}")
            if process.stdout: print_error(f"STDOUT: {process.stdout.strip()}")
            if process.stderr: print_error(f"STDERR: {process.stderr.strip()}")
            return None
    except Exception as e:
        print_error(f"An unexpected error occurred during watchdog build: {e}")
        return None


def terminate_relaunched_process(process_obj, name="Relaunched Conda Process"):
    """
    Terminates a process and its entire descendant tree robustly using psutil.
    Falls back to OS-specific commands if psutil is not available.
    """
    # First, check if the process object is valid and running.
    if not process_obj or process_obj.poll() is not None:
        return  # Process doesn't exist or has already terminated.

    pid = process_obj.pid
    print_system(f"Attempting to terminate '{name}' (PID: {pid}) and its entire process tree...")

    # --- The psutil Method (Preferred) ---
    try:
        # We import psutil here, inside the function.
        # This makes it an optional dependency for cleanup. If it's not installed,
        # we can fall back gracefully to the less reliable method.
        import psutil
        print_system(f"Using psutil for robust process tree termination.")

        # Get the psutil Process object for our main target process.
        parent = psutil.Process(pid)

        # Get a list of all children, recursively.
        # This will include the 'conda run' shell, the relaunched python script,
        # hypercorn, node, etc.
        children = parent.children(recursive=True)

        # First, terminate all the descendant processes.
        # It's good practice to terminate children before the parent.
        for child in children:
            print_system(f"  -> Terminating child process '{child.name()}' (PID: {child.pid})")
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass  # Child already disappeared, that's fine.

        # Wait for the children to die.
        gone, alive = psutil.wait_procs(children, timeout=3)

        # If any children are still alive, kill them forcefully.
        for p in alive:
            print_warning(f"  -> Child process {p.pid} did not terminate gracefully. Killing.")
            try:
                p.kill()
            except psutil.NoSuchProcess:
                pass  # Already gone, fine.

        # Finally, terminate the main parent process itself.
        print_system(f"  -> Terminating main process '{parent.name()}' (PID: {parent.pid})")
        try:
            parent.terminate()
            parent.wait(timeout=5)
        except psutil.NoSuchProcess:
            print_system(f"Main process (PID: {pid}) already gone.")
        except psutil.TimeoutExpired:
            print_warning(f"Main process (PID: {pid}) did not terminate gracefully. Killing.")
            parent.kill()

        print_system(f"Termination of '{name}' (PID: {pid}) tree complete.")
        return  # Success

    except ImportError:
        print_warning("`psutil` library not found. Falling back to less reliable OS-level termination.")
    except psutil.NoSuchProcess:
        print_system(f"Process {pid} for '{name}' was already gone when cleanup started.")
        return  # Success, it's already dead
    except Exception as e:
        print_error(f"An unexpected error occurred during psutil termination: {e}")
        # Continue to fallback method.

    # --- Fallback Method (Your Original Code) ---
    # This block runs if psutil is not installed or if it failed unexpectedly.
    print_system(f"Executing fallback termination for '{name}' (PID: {pid})...")
    try:
        if IS_WINDOWS:
            # Use taskkill to terminate the process tree. /F for force, /T for tree.
            kill_cmd = ['taskkill', '/F', '/T', '/PID', str(pid)]
            subprocess.run(kill_cmd, check=False, capture_output=True, text=True)
        else:
            # On Unix, send SIGTERM to the entire process group.
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGTERM)

        process_obj.wait(timeout=5)
        print_system(f"Fallback termination for '{name}' completed.")
    except (ProcessLookupError, subprocess.TimeoutExpired):
        # If it times out, try a more forceful kill on the original process
        print_warning(f"Fallback termination for '{name}' timed out or failed. Attempting final kill.")
        process_obj.kill()
    except Exception as e:
        print_error(f"Error during fallback termination of '{name}' tree: {e}")

#=-=-=-=- this is Printing style for the log sysout syserr on stdio, wait, this is python
def print_system(message): print_colored("SYSTEM", message)

def print_aetherhand(message): print_colored("AETHERHAND", message)

def print_error(message): print_colored("ERROR", message)

def print_warning(message): print_colored("WARNING", message)


def stream_output(pipe, name=None, tui_log_widget: Optional['Log'] = None):
    """
    Streams output from a subprocess pipe. If a Textual Log widget is provided,
    writes to it in a thread-safe manner; otherwise, prints to stdout.
    """
    try:
        # Use iter(pipe.readline, '') to read line by line
        for line in iter(pipe.readline, ''):
            if line:
                # If a TUI widget was provided...
                if tui_log_widget:
                    # Use call_from_thread to safely update the UI from this background thread.
                    # This is the critical fix.
                    tui_log_widget.app.call_from_thread(tui_log_widget.write, f"[{name.upper()}] {line.strip()}")
                else:
                    # Fallback to simple terminal output if not in TUI mode
                    sys.stdout.write(f"[{name.upper()}] {line.strip()}\n")
                    sys.stdout.flush()
    except Exception as e:
        # Avoid printing errors for closed files, which is normal on shutdown
        if 'read of closed file' not in str(e).lower() and 'Stream is closed' not in str(e):
            # Log the error using the appropriate method
            log_line = f"[STREAM_ERROR] Error in stream '{name}': {e}"
            if tui_log_widget:
                tui_log_widget.app.call_from_thread(tui_log_widget.write, log_line)
            else:
                print_error(log_line)
    finally:
        if pipe:
            try:
                pipe.close()
            except Exception:
                pass # Ignore errors on closing a pipe that might already be closed


def _remove_flag_files(flags_to_remove: list):
    print_system("Attempting to remove existing state/completion flag files...")
    for flag_file in flags_to_remove:
        if os.path.exists(flag_file):
            try:
                os.remove(flag_file)
                print_system(f"  Removed flag file: {flag_file}")
            except OSError as e:
                print_warning(f"  Could not remove flag file {flag_file}: {e}")
        else:
            print_system(f"  Flag file not found (already removed or never created): {flag_file}")

def run_command(command, cwd, name, color=None, check=True, capture_output=False, env_override=None):
    log_name_prefix = name if name else os.path.basename(command[0])
    print_system(f"Running command in '{os.path.basename(cwd)}' for '{log_name_prefix}': {' '.join(command)}")
    # Manually construct the environment to ensure Conda's paths are prioritized.
    env = os.environ.copy()

    conda_bin_path = os.path.dirname(PYTHON_EXECUTABLE)
    conda_lib_path = os.path.join(TARGET_RUNTIME_ENV_PATH, "lib")

    # Force Conda's bin directory to be at the front of the PATH
    env['PATH'] = f"{conda_bin_path}{os.pathsep}{env.get('PATH', '')}"

    # Force Conda's lib directory to be at the front of the linker path
    if platform.system() == "Linux":
        env['LD_LIBRARY_PATH'] = f"{conda_lib_path}{os.pathsep}{env.get('LD_LIBRARY_PATH', '')}"
    elif platform.system() == "Darwin":  # macOS
        env['DYLD_LIBRARY_PATH'] = f"{conda_lib_path}{os.pathsep}{env.get('DYLD_LIBRARY_PATH', '')}"

    # Apply any specific overrides from the caller *after* setting our base env
    if env_override:
        str_env_override = {k: str(v) for k, v in env_override.items()}
        print_system(f"  with custom env for '{log_name_prefix}': {str_env_override}")
        env.update(str_env_override)

    command = [str(c) for c in command]
    use_shell = False
    if IS_WINDOWS:
        if command[0] == NPM_CMD or command[0].lower().endswith('.cmd') or command[0].lower().endswith('.bat'):
            use_shell = True

    try:
        process = subprocess.Popen(
            command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding='utf-8', errors='replace', bufsize=1,
            shell=use_shell, env=env
        )

        stdout_thread = threading.Thread(target=stream_output, args=(process.stdout, f"{log_name_prefix}-OUT"),
                                         daemon=True)
        stdout_thread.start()
        stderr_thread = threading.Thread(target=stream_output, args=(process.stderr, f"{log_name_prefix}-LogStream"),
                                         daemon=True)
        stderr_thread.start()

        process.wait()
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)

        if check and process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command)

        print_system(f"Command finished successfully for '{log_name_prefix}' in '{os.path.basename(cwd)}'.")
        return True
    except FileNotFoundError:
        print_error(f"Command not found for '{log_name_prefix}': {command[0]}. Is it installed and in PATH?")
        if command[0] == GIT_CMD:
            print_error("Ensure Git is installed: https://git-scm.com/downloads")
        elif command[0] == NPM_CMD:
            print_error("Ensure Node.js and npm are installed: https://nodejs.org/")
        elif command[0] == CMAKE_CMD:
            print_error("Ensure CMake is installed: https://cmake.org/download/")
        elif command[0] == HYPERCORN_EXECUTABLE:
            print_error(f"Ensure '{os.path.basename(HYPERCORN_EXECUTABLE)}' is installed in the Conda env.")
        return False
    except subprocess.CalledProcessError as e:
        print_error(
            f"Command failed for '{log_name_prefix}' in '{os.path.basename(cwd)}' with exit code {e.returncode}.")
        return False
    except Exception as e:
        print_error(
            f"An unexpected error occurred while running command for '{log_name_prefix}' in '{os.path.basename(cwd)}': {e}")
        return False


# --- Service Start Functions ---
def start_service_process(command, cwd, name, use_shell_windows=False):
    """
    Launches a service as a background process and redirects its
    stdout and stderr to a dedicated log file in the ./logs/ directory.
    """
    log_dir = os.path.join(ROOT_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a clean log file name, e.g., "ENGINE.log"
    log_file_path = os.path.join(log_dir, f"{name.replace(' ', '_')}.log")
    
    print_system(f"Launching {name}: {' '.join(command)} in {os.path.basename(cwd)}")
    print_system(f"  --> Log output will be streamed to: {log_file_path}")
    
    try:
        # Open the log file in write mode, which will truncate it on each start
        log_file_handle = open(log_file_path, 'w', encoding='utf-8')
        
        shell_val = use_shell_windows if IS_WINDOWS else False
        
        # Redirect both stdout and stderr to the same log file handle
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=log_file_handle,
            stderr=log_file_handle,
            text=True,
            encoding='utf-8',
            errors='replace',
            shell=shell_val
        )
        
        # Keep track of the process and its log file for the TUI to use
        with process_lock:
            running_processes.append((process, name, log_file_path, log_file_handle))

        return process
        
    except FileNotFoundError:
        print_error(f"Command failed for {name}: '{command[0]}' not found.")
        sys.exit(1)
    except Exception as e:
        print_error(f"Failed to start {name}: {e}")
        sys.exit(1)


def _terminate_service_robustly(process: subprocess.Popen, name: str):
    """Uses psutil to terminate a process and its entire descendant tree."""
    if process.poll() is not None:
        print_system(f"{name} already exited (return code: {process.poll()}).")
        return

    print_system(f"Terminating {name} (PID: {process.pid}) and its children...")
    try:
        import psutil
        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)

        for child in children:
            try:
                print_system(f"  -> Terminating child {child.name()} (PID: {child.pid}) of {name}")
                child.terminate()
            except psutil.NoSuchProcess:
                pass # Already gone

        # Wait for children to die
        gone, alive = psutil.wait_procs(children, timeout=3)
        for p in alive:
            print_warning(f"  -> Child {p.pid} of {name} did not terminate. Killing.")
            try:
                p.kill()
            except psutil.NoSuchProcess:
                pass

        # Now terminate the main process
        try:
            parent.terminate()
            parent.wait(timeout=5)
            print_system(f"{name} terminated gracefully.")
        except psutil.TimeoutExpired:
            print_warning(f"{name} did not terminate gracefully, killing (PID: {parent.pid})...")
            parent.kill()
            parent.wait(timeout=2)
            print_system(f"{name} killed.")

    except (ImportError, psutil.NoSuchProcess):
        # Fallback to the original method if psutil is not found or the process is already gone
        print_warning(f"psutil not found or process gone. Using standard terminate for {name}.")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
    except Exception as e:
        print_error(f"Error terminating {name} (PID: {process.pid}): {e}")
        # Final resort: kill the original process handle
        if process.poll() is None:
            process.kill()

def cleanup_processes():
    print_system("\nShutting down services...")

    global port_shield_stop_event
    if port_shield_stop_event and not port_shield_stop_event.is_set():
        print_system("Disarming port shield...")
        port_shield_stop_event.set()

    # Use global to ensure we are modifying the correct variables
    global zephymesh_process, relaunched_conda_process_obj

    # --- Shutdown ZephyMesh Node First ---
    if zephymesh_process and zephymesh_process.poll() is None:
        print_system(f"Terminating ZephyMesh Node (PID: {zephymesh_process.pid})...")
        zephymesh_process.terminate()
        try:
            zephymesh_process.wait(timeout=5)
            print_system("ZephyMesh Node terminated gracefully.")
        except subprocess.TimeoutExpired:
            print_warning("ZephyMesh Node did not terminate gracefully, killing...")
            zephymesh_process.kill()
    
    # --- Handle the main relaunched 'conda run' process ---
    if relaunched_conda_process_obj and relaunched_conda_process_obj.poll() is None:
        print_system("Terminating the 'conda run' process and its children...")
        terminate_relaunched_process(relaunched_conda_process_obj, "Relaunched 'conda run' process")
        relaunched_conda_process_obj = None

    # --- Handle locally managed services (Engine, Backend, etc.) ---
    with process_lock:
        # Make a copy to avoid issues while iterating and modifying
        procs_to_terminate = list(running_processes) 
        running_processes.clear()

    # Iterate over the copied list
    for proc, name, log_path, log_handle in reversed(procs_to_terminate):
        # First, ensure the log file handle is closed
        try:
            if log_handle and not log_handle.closed:
                log_handle.close()
        except Exception:
            pass  # Ignore errors on closing handle

        # Now, use our new robust terminator for the process
        _terminate_service_robustly(proc, name) #Use the helper function instead of coagulate into one position

atexit.register(cleanup_processes)


def signal_handler(sig, frame):
    print_system(f"\nSignal {sig} received. Initiating shutdown via atexit handlers...")
    # The atexit handler (cleanup_processes) will perform the necessary terminations.
    # We set a non-zero exit code, common for interruption.
    # 130 for SIGINT (Ctrl+C), 128 + signal number for others.
    sys.exit(130 if sig == signal.SIGINT else 128 + sig)

signal.signal(signal.SIGINT, signal_handler)
if not IS_WINDOWS: # SIGTERM is not really a thing for console apps on Windows this way for Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler) # Handle SIGTERM for graceful shutdown requests on Unix

def tail_log_to_widget(log_file_path: str, widget: 'Log'):
    """
    A function to be run in a background thread that tails a log file
    and writes new lines to a Textual Log widget.
    """
    logger.info(f"Tailing '{log_file_path}' to widget '#{widget.id}'...")
    try:
        with open(log_file_path, "r", encoding="utf-8", errors="replace") as f:
            f.seek(0, 2)
            # CHANGE THIS LINE:
            while not tui_shutdown_event.is_set():
                line = f.readline()
                if not line:
                    time.sleep(0.1)
                    continue
                # Check the event AGAIN before calling the app thread
                if tui_shutdown_event.is_set():
                    break
                widget.app.call_from_thread(widget.write, line)
    except FileNotFoundError:
        if not tui_shutdown_event.is_set():  # Only log if not shutting down
            widget.app.call_from_thread(widget.write, f"[ERROR] Log file not found: {log_file_path}")
    except Exception as e:
        if not tui_shutdown_event.is_set():  # Only log if not shutting down
            error_msg = f"[ERROR] Tailing failed for {os.path.basename(log_file_path)}: {e}"
            logger.error(error_msg)
            widget.app.call_from_thread(widget.write, error_msg)


def start_engine_main():
    name = "ENGINE"
    MAX_UPLOAD_SIZE_BYTES = 2 ** 63

    if os.path.exists(ENGINE_PID_FILE):
        os.remove(ENGINE_PID_FILE)

    command = [
        HYPERCORN_EXECUTABLE,
        "AdelaideAlbertCortex:system",
        "--bind", "127.0.0.1:11434",
        "--workers", "1",
        "--log-level", "info",
        "--pid", ENGINE_PID_FILE
    ]
    hypercorn_env = {
        "HYPERCORN_MAX_REQUEST_SIZE": str(MAX_UPLOAD_SIZE_BYTES)
    }
    # The function name start_service_process is a bit of a misnomer here,
    # as we aren't passing hypercorn_env. Assuming that's intended.
    # If not, start_service_process would need to be updated to accept an env dict.
    # Based on the original code, this seems to be the intent.
    start_service_process(command, ENGINE_MAIN_DIR, name)


def start_backend_service():
    """
    Launches the Go backend service.
    First runs 'go mod tidy', then starts the server with 'go run .'.
    """
    name = "BACKEND"
    # The directory is already defined globally as BACKEND_SERVICE_DIR
    print_system(f"Preparing to launch {name} service from: {BACKEND_SERVICE_DIR}")

    if not os.path.isdir(BACKEND_SERVICE_DIR):
        print_error(f"Backend service directory not found: {BACKEND_SERVICE_DIR}")
        sys.exit(1) # This is a critical failure
    backend_exe_name = "zephyrine-backend.exe" if IS_WINDOWS else "zephyrine-backend"
    backend_exe_path = os.path.join(BACKEND_SERVICE_DIR, backend_exe_name) #pointing to the compiled backend temp golang binary

    # --- Step 1: Run 'go mod tidy' ---
    # We use your 'run_command' helper because this is a one-off task we need to wait for.
    print_system(f"Building Go backend binary to: {backend_exe_path}")
    build_command = ["go", "build", "-o", backend_exe_path, "."]
    # We use run_command here because building is a one-time task we must wait for.
    if not run_command(build_command, cwd=BACKEND_SERVICE_DIR, name=f"{name}-BUILD", check=True):
        print_error("Failed to build Go backend. Backend service cannot start.")
        sys.exit(1)

    # --- Step 2: Launch 'go run .' ---
    # We use your 'start_service_process' helper as it correctly handles
    # long-running services, adds them to the cleanup list, and streams output.
    start_service_process([backend_exe_path], BACKEND_SERVICE_DIR, name)


def start_frontend():
    name = "FRONTEND"
    command = [NPM_CMD, "run", "dev"]
    start_service_process(command, FRONTEND_DIR, name, use_shell_windows=True)


def start_backend_service_fast():
    """Fast-path version: Assumes the backend is already compiled."""
    name = "BACKEND"
    backend_exe_name = "zephyrine-backend.exe" if IS_WINDOWS else "zephyrine-backend"
    backend_exe_path = os.path.join(BACKEND_SERVICE_DIR, backend_exe_name)

    if not os.path.exists(backend_exe_path):
        print_error(f"FATAL (Fast Path): Backend executable not found at {backend_exe_path}. Please run a full setup.")
        # In a fast path, this is a critical failure.
        return

    start_service_process([backend_exe_path], BACKEND_SERVICE_DIR, name)

def start_zephymesh_service_fast():
    """Fast-path version: Assumes ZephyMesh is compiled and launches it without waiting."""
    name = "ZEPHYMESH-NODE"
    mesh_exe_name = "zephymesh_node_compiled.exe" if IS_WINDOWS else "zephymesh_node_compiled"
    mesh_exe_path = os.path.join(ZEPHYMESH_DIR, mesh_exe_name)

    if not os.path.exists(mesh_exe_path):
        print_error(f"FATAL (Fast Path): ZephyMesh executable not found at {mesh_exe_path}. Please run a full setup.")
        return

    # Clean up old port file before starting
    if os.path.exists(ZEPHYMESH_PORT_INFO_FILE):
        os.remove(ZEPHYMESH_PORT_INFO_FILE)

    start_service_process([mesh_exe_path], ROOT_DIR, name)


def start_watchdogs_service_fast():
    """Fast-path version: Assumes watchdogs are compiled and launches them directly."""
    print_system("--- Launching Watchdog Services (Fast Path) ---")

    # Run Go Watchdog
    go_watchdog_dir = os.path.join(ROOT_DIR, "ZephyWatchtower")
    go_exe_name = "watchdog_thread1.exe" if IS_WINDOWS else "watchdog_thread1"
    go_exe_path = os.path.join(go_watchdog_dir, go_exe_name)
    if os.path.exists(go_exe_path):
        backend_exe_name = "zephyrine-backend.exe" if IS_WINDOWS else "zephyrine-backend"
        backend_process_name = backend_exe_name.replace(".exe", "")
        mesh_process_name = "zephymesh_node_compiled"
        watchdog_command = [
            go_exe_path,
            f"--targets={backend_process_name},node,{mesh_process_name}",
            f"--pid-file={ENGINE_PID_FILE}"
        ]
        start_service_process(watchdog_command, go_watchdog_dir, "GO-WATCHDOG")
    else:
        print_error(f"FATAL (Fast Path): Go watchdog executable not found at {go_exe_path}.")

    # Run Ada Watchdog
    ada_watchdog_dir = os.path.join(ROOT_DIR, "ZephyWatchtower", "watchdog_thread2")
    ada_exe_name = "watchdog_thread2.exe" if IS_WINDOWS else "watchdog_thread2"
    ada_exe_path = os.path.join(ada_watchdog_dir, "bin", ada_exe_name)
    if os.path.exists(ada_exe_path):
        # The logic for the Ada watchdog supervising the Go watchdog remains the same.
        backend_exe_name = "zephyrine-backend.exe" if IS_WINDOWS else "zephyrine-backend"
        backend_process_name = backend_exe_name.replace(".exe", "")
        go_watchdog_command = [
            go_exe_path,
            f"--targets={backend_process_name},node",
            f"--pid-file={ENGINE_PID_FILE}"
        ]
        ada_watchdog_command = [ada_exe_path] + go_watchdog_command
        start_service_process(ada_watchdog_command, ROOT_DIR, "ADA-WATCHDOG")
    else:
        print_error(f"FATAL (Fast Path): Ada watchdog executable not found at {ada_exe_path}.")

def monitor_services_fallback():
    """A simple loop to watch processes in non-TUI mode."""
    print_colored("SUCCESS", "All services launching. Press Ctrl+C to shut down.", "SUCCESS")
    try:
        while True:
            all_ok = True
            active_procs_found = False
            with process_lock:
                current_procs_snapshot = list(running_processes)

            if not current_procs_snapshot:
                # This could happen right at the start before any process is registered.
                time.sleep(1)
                continue

            for proc, name_s, _, _ in current_procs_snapshot:
                if proc.poll() is None:
                    active_procs_found = True
                else:
                    print_error(f"Service '{name_s}' exited unexpectedly (RC: {proc.poll()}).")
                    all_ok = False

            if not all_ok:
                print_error("One or more services terminated. Shutting down launcher.")
                break
            # If no processes are active anymore (e.g., they all finished), exit.
            if not active_procs_found:
                print_system("All services seem to have finished. Exiting launcher.")
                break

            time.sleep(5)
    except KeyboardInterrupt:
        print_system("\nKeyboardInterrupt received by main thread. Shutting down...")
    finally:
        print_system("Launcher main loop finished. Ensuring cleanup via atexit...")


def launch_all_services_in_parallel_and_monitor():
    """
    The "fast path" launch sequence. Launches dependencies first, then the engine.
    """
    print_system("--- IGNITION: Launching all services in parallel ---")

    # --- PHASE 1: IGNITE MAIN ENGINE & COMPLEMENTARY SERVICES ---
    print_system("Running main engine (Hypercorn)...")
    start_engine_main()

    # --- PHASE 2: LAUNCH CORE DEPENDENCIES ---
    core_dependency_threads = []
    tasks_to_launch_first = {
        "ZephyMesh": start_zephymesh_service_fast,
        "Backend": start_backend_service_fast,
    }
    for name, task_func in tasks_to_launch_first.items():
        thread = threading.Thread(target=task_func, name=f"LaunchThread-{name}", daemon=True)
        core_dependency_threads.append(thread)
        print_system(f"Dispatching core dependency launch for: {name}")
        thread.start()

    print_system("Waiting briefly for core services to initialize...")

    # Launch the Frontend immediately
    print_system("Dispatching complementary service launch for: Frontend")
    threading.Thread(target=start_frontend, name="LaunchThread-Frontend", daemon=True).start()

    # --- NEW: PHASE 3: SCHEDULE WATCHDOGS WITH A DELAY ---
    def delayed_watchdog_launch():
        """A simple wrapper function to be called by the timer."""
        print_system("--- [Delayed Task] --- Launching Watchdogs now...")
        start_watchdogs_service_fast()

    print_warning("Scheduling Watchdogs to launch in 5 seconds...")
    # Create a Timer thread that will call our wrapper function after 5 seconds.
    # This is non-blocking. The script will continue immediately.
    watchdog_timer = threading.Timer(1.0, delayed_watchdog_launch)
    watchdog_timer.daemon = True  # Ensure it doesn't prevent the app from exiting
    watchdog_timer.start()

    print_colored("SUCCESS", "All services Loaded! Zephy Loaded! Moving to monitoring.", "SUCCESS")

    # --- PHASE 4: MONITORING ---
    # Now, jump into the TUI or fallback monitoring loop.
    if TUI_LIBRARIES_AVAILABLE:
        print_system("TUI libraries found. Transitioning Zephyrine Terminal UI...")
        time.sleep(1) # Give services a moment to create their log files before TUI tails them
        try:
            app = ZephyrineTUI()
            app.run()
            print_system("TUI has exited. Launcher finished.")
        except Exception as e_tui:
            print_error(f"The Terminal UI failed to launch: {e_tui}")
            print_warning("Falling back to simple terminal output for monitoring.")
            monitor_services_fallback()
    else:
        print_warning("Textual or psutil not found. Falling back to simple terminal output for monitoring.")
        monitor_services_fallback()
        



# --- Conda Python Version Calculation ---
def get_conda_python_versions_to_try(current_dt=None):
    """
    Calculates a list of Python versions for Conda to try.
    - PRIORITIZES STABILITY: The initial, known-good version is always tried first.
    - SEMESTER-BASED UPGRADES: The dynamic version increases every 6 months
      past the EPOCH_DATE to provide future-proofing without being too aggressive.
    """
    if current_dt is None:
        current_dt = datetime.now()
    
    # Use the global EPOCH_DATE which should now be set to date(2025, 6, 1)
    epoch_date = EPOCH_DATE 
    current_date_obj = current_dt.date()

    # --- New Semester-Based Calculation ---
    # Calculate the total number of full months that have passed since the epoch.
    total_months_elapsed = (current_date_obj.year - epoch_date.year) * 12 + \
                           (current_date_obj.month - epoch_date.month)

    semesters_offset = 0
    if total_months_elapsed > 0:
        # Integer division by 6 gives us the number of completed semesters.
        semesters_offset = total_months_elapsed // 6

    # Calculate the new, dynamic version based on the semester offset.
    dynamic_target_minor = INITIAL_PYTHON_MINOR + semesters_offset

    # --- New "Stability First" Version Ordering ---
    versions_to_try = []
    
    # 1. PRIORITY 1: The stable, developer-tested initial version.
    versions_to_try.append(f"{INITIAL_PYTHON_MAJOR}.{INITIAL_PYTHON_MINOR}")

    # 2. PRIORITY 2: The newer, dynamically-calculated version (if different).
    #    This provides future-proofing as a fallback, not the primary choice.
    if dynamic_target_minor > INITIAL_PYTHON_MINOR:
        versions_to_try.append(f"{INITIAL_PYTHON_MAJOR}.{dynamic_target_minor}")

    # 3. PRIORITY 3: The ultimate fallbacks.
    versions_to_try.append(f"{FALLBACK_PYTHON_MAJOR}.{FALLBACK_PYTHON_MINOR}")
    # You can add more fallbacks here if needed, e.g.:
    # versions_to_try.append(f"{FALLBACK_PYTHON_MAJOR}.{FALLBACK_PYTHON_MINOR + 1}")

    # --- Final de-duplication to ensure a clean list ---
    final_versions_to_try = []
    for v_str in versions_to_try:
        if v_str not in final_versions_to_try:
            final_versions_to_try.append(v_str)

    print_system(f"Calculated preferred Python versions for Conda (Stability First): {final_versions_to_try}")
    return final_versions_to_try


# --- Conda Utility Functions ---
def _verify_conda_path(conda_path_to_verify):
    """Helper to verify if a given path is a functional Conda executable."""
    if not conda_path_to_verify or not os.path.isfile(conda_path_to_verify):
        return False

    is_executable = os.access(conda_path_to_verify, os.X_OK)
    if not is_executable and IS_WINDOWS and conda_path_to_verify.lower().endswith((".bat", ".exe")):
        is_executable = True  # .bat/.exe might not report X_OK but are runnable on Windows

    if not is_executable:
        return False

    cmd_list = []
    if IS_WINDOWS and conda_path_to_verify.lower().endswith(".bat"):
        cmd_list = ['cmd', '/c', conda_path_to_verify, 'info', '--base']
    else:
        cmd_list = [conda_path_to_verify, 'info', '--base']

    try:
        proc = subprocess.run(cmd_list, capture_output=True, text=True, timeout=10, check=False, errors='replace')
        if proc.returncode == 0 and proc.stdout and os.path.isdir(proc.stdout.strip()):
            return True  # Verified
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False  # Timed out or somehow still not found
    except Exception:  # Other errors during verification
        return False
    return False


def _find_shell_executable() -> Optional[str]:
    """
    Finds a suitable shell executable on non-Windows systems.
    It checks for 'bash', then 'sh', then 'ash' in the system's PATH.

    Returns:
        The absolute path to the first shell found, or None if none are found.
    """
    print_system("Searching for a suitable shell (bash, sh, ash)...")
    # A prioritized list of shells to search for.
    # Bash is often preferred if available, but sh is the most portable.
    shells_to_try = ["bash", "sh", "ash"]
    for shell in shells_to_try:
        # shutil.which is the correct, cross-platform way to find an executable in the PATH.
        found_path = shutil.which(shell)
        if found_path:
            print_system(f"Found shell: {found_path}")
            return found_path

    print_error("Could not find a suitable shell (bash, sh, or ash) in the system's PATH.")
    return None


def _handle_android_become_smartphone_setup() -> None:
    """
    Handles the fully automated setup of a proot-distro environment on Termux.
    It creates a resumable, one-time execution script in .bashrc that provides
    critical user warnings, sanitizes the build environment, installs a comprehensive
    set of dependencies, builds a custom graphics stack, and finally executes the launcher.
    This function does not return, as it replaces the current process.
    """
    PROOT_ALIAS = "zephyglibccompatlayer"
    MAX_PROOT_INSTALL_ATTEMPTS = 3

    print_aetherhand("Handling Android incompatibility, executing other subroutines")
    print_system("This entire process can consume over 5.69 GB of RAM and significant processing time.")
    print_system("Smartphones are consumer devices designed for media consumption and doomscrolling;")
    print_system("they are not designed to be locally and independently 'smart'. Remember that You own nothing and be happy.")
    print_system("The Android OS is EXTREMELY AGGRESSIVE in killing background processes (SIGKILL 9) to save resources.")
    print_warning("A device with 12GB of RAM or more is highly recommended for a chance of success.")
    print_warning("An unlocked bootloader (superuser/root) is recommended to properly disable these limitations.")
    print_colored("SYSTEM", "--- RECOMMENDED ACTION TO PREVENT RANDOM KILLS ---", "SYSTEM")
    print_system("To disable Android's 'Phantom Process Killer', run these commands via ADB from a PC or adb wifi or Shizuku:")
    print_colored("COMMAND", '    adb shell "/system/bin/device_config set_sync_disabled_for_tests persistent"',
                  "SUCCESS")
    print_colored("COMMAND",
                  '    adb shell "/system/bin/device_config put activity_manager max_phantom_processes 2147483647"',
                  "SUCCESS")
    print_colored("COMMAND", '    adb shell settings put global settings_enable_monitor_phantom_procs false', "SUCCESS")
    print_system("\nPausing for 10 seconds. Press Ctrl+C to abort if your device is not prepared...")


    # 1. Ensure proot-distro is installed via apt
    print_system("Checking for 'proot-distro' package for containerization...")
    proot_distro_path = shutil.which("proot-distro")
    if not proot_distro_path:
        print_warning("'proot-distro' not found. Attempting to install with 'pkg'...")
        try:
            subprocess.run(["apt", "update", "-y"], check=True, capture_output=True)
            subprocess.run(["apt", "install", "proot-distro", "-y"], check=True)
            proot_distro_path = shutil.which("proot-distro")
            if not proot_distro_path:
                raise RuntimeError("pkg install seemed to succeed, but 'proot-distro' is still not in PATH.")
            print_system("✅ 'proot-distro' installed successfully.")
        except Exception as e:
            print_error(f"Failed to install 'proot-distro': {e}")
            if isinstance(e, subprocess.CalledProcessError):
                print_error(f"Stderr: {e.stderr.decode(errors='ignore')}")
            print_error("Please install it manually ('pkg install proot-distro') and restart.")
            sys.exit(1)

    # 2. Reliably find the base path for proot-distro root filesystems
    prefix = os.environ.get("PREFIX", "/data/data/com.termux/files/usr")
    distro_base_path = os.path.join(prefix, "var/lib/proot-distro/installed-rootfs")
    distro_fs_path = os.path.join(distro_base_path, PROOT_ALIAS)

    # 3. Ensure the Ubuntu distribution is fully installed
    print_system(f"Checking for proot-distro environment at: {distro_fs_path}")
    if not os.path.isdir(distro_fs_path):
        install_successful = False
        for attempt in range(1, MAX_PROOT_INSTALL_ATTEMPTS + 1):
            print_warning(
                f"Distro '{PROOT_ALIAS}' not found or incomplete. Installing (Attempt {attempt}/{MAX_PROOT_INSTALL_ATTEMPTS})...")
            print_system(f"Performing pre-install reset: Removing '{PROOT_ALIAS}'...")
            subprocess.run([proot_distro_path, "remove", PROOT_ALIAS], check=False, capture_output=True)
            print_system("Reset complete.")
            print_warning("This will download the debian rootfs and may take a long time.")
            try:
                install_cmd = [proot_distro_path, "install", "debian", "--override-alias", PROOT_ALIAS]
                subprocess.run(install_cmd, check=True, capture_output=True)
                print_system(f"✅ Virtual Environment '{PROOT_ALIAS}' installed successfully.")
                install_successful = True
                break
            except subprocess.CalledProcessError as e:
                print_error(f"Installation attempt {attempt} failed: {e.stderr.decode(errors='ignore').strip()}")
                if attempt < MAX_PROOT_INSTALL_ATTEMPTS:
                    print_warning("Retrying in 5 seconds...")
                    time.sleep(5)
        if not install_successful:
            print_error(f"Failed to install proot-distro environment after {MAX_PROOT_INSTALL_ATTEMPTS} attempts.")
            sys.exit(1)
    else:
        print_system(f"Found existing and valid distro directory: '{PROOT_ALIAS}'.")

    # 4. OVERWRITE the .bashrc inside the proot environment with our build script
    print_system("Generating and writing one-time build script to proot environment...")
    try:
        bashrc_path = os.path.join(distro_fs_path, "root", ".bashrc")

        # This comprehensive shell script is the new .bashrc.
        auto_exec_script = f"""#!/bin/bash
# This is a one-time execution script generated by the Zephyrine Launcher.
set -e

# --- Step 0: CRITICAL Environment Sanitization and Configuration ---
echo "--> Sanitizing build environment to prevent contamination from Termux..."
unset C_INCLUDE_PATH CPLUS_INCLUDE_PATH CPATH CFLAGS CXXFLAGS LDFLAGS
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# Apply compiler flags to disable specific errors that halt the build.
echo "--> Applying compiler flags to work around build errors..."
export CFLAGS="-Wno-error=implicit-function-declaration -Wno-error=unused-result"
export CXXFLAGS="-Wno-error=implicit-function-declaration -Wno-error=unused-result"

# This is the flag file to check if the intensive build has already been done.
MESA_BUILD_FLAG="/root/.mesa_build_complete_v1"

apt-get update
apt-get upgrade -y
apt-get install -y --install-recommends \\
    build-essential ca-certificates cmake git gettext \\
    python3-pip python3-dev libc6-dev libc6 \\
    libvulkan-dev vulkan-tools libudev-dev llvm-dev \\
    libgmp-dev libssl-dev libffi-dev 

echo "--> Removing any system-level Node.js and npm to avoid conflicts..."
# Use || true to prevent the script from failing if the packages aren't installed.
apt-get remove --purge nodejs npm -y || true
apt-get autoremove -y || true
echo "--> System-level Node.js removed."

if [ ! -f "$MESA_BUILD_FLAG" ]; then
    echo "--- Zephyrine Proot Setup: Mesa build flag not found. Starting environment build ---"

    # --- Step 1: Update system and install a comprehensive set of build packages ---
    echo "--> Updating package lists and installing dependencies..."
    export DEBIAN_FRONTEND=noninteractive
    

    # --- Step 2: Update Linker Cache ---
    echo "--> Updating dynamic linker cache..."
    ldconfig

    # --- Step 3: Robustly enable source repositories and install Mesa build deps ---
    echo "--> Enabling source repositories for 'apt-get build-dep'..."
    SOURCES_FILE="/etc/apt/sources.list.d/zephyrine-mesa-build.list"
    grep -v '^#' /etc/apt/sources.list | sed -e 's/^deb /deb-src /' > "$SOURCES_FILE"
    apt-get update
    echo "--> Installing Mesa build dependencies..."
    apt-get build-dep mesa -y

    # --- Step 4: Clone or update the Mesa repository ---
    echo "--> Cloning or updating Mesa repository..."
    cd /root
    if [ -d "mesa-build" ]; then
        echo "--> 'mesa-build' directory found. Updating repository..."
        cd mesa-build
        git reset --hard HEAD
        git pull
    else
        echo "--> 'mesa-build' directory not found. Cloning repository..."
        #This is modified Mesa for Android Container Specifics and constraints, usually only used for Wine and playing games consumptions
        git clone --depth 1 https://github.com/lfdevs/mesa-for-android-container mesa-build
        cd mesa-build
    fi

    # --- Step 5: Detect CPU and LLVM RTTI status, then build GPU-specific drivers ---
    echo "--> Detecting CPU and LLVM RTTI status for custom graphics driver build..."
    if [ -d "build" ]; then
        echo "--> Found stale build directory. Removing for a clean configuration."
        rm -rf build
    fi

    BASE_MESON_ARGS="-D platforms=x11,wayland -D egl=enabled -D gles2=enabled -D glx=dri --prefix /usr --buildtype=release"
    echo "--> Checking system LLVM RTTI support..."
    LLVM_CXX_FLAGS=$(llvm-config --cxxflags)
    if echo "$LLVM_CXX_FLAGS" | grep -q -- "-fno-rtti"; then
        echo "--> System LLVM was built WITHOUT RTTI. Disabling RTTI for Mesa."
        BASE_MESON_ARGS="$BASE_MESON_ARGS -D cpp_rtti=false"
    else
        echo "--> System LLVM was built WITH RTTI. Enabling RTTI for Mesa (default)."
    fi

    #Unification of the GPU Driver compilation and installation
    # Source of Manual : https://github.com/lfdevs/mesa-for-android-container/blob/adreno-main/README.rst
    echo "--> Configuring a universal Mesa build for Adreno (Freedreno) and Mali (Panfrost) GPUs..."
    meson setup build $BASE_MESON_ARGS \\
    -D gallium-drivers=freedreno,panfrost,zink,virgl \\
    -D vulkan-drivers=freedreno,panfrost

    # --- Step 6: Compile and install the custom Mesa build ---
    echo "--> Compiling and installing custom Mesa build... (This will take a long time)"
    ninja -C build install
    # Add verification step

    # --- Step 6a: Verify GPU Driver Installation ---
    echo "--> Verifying that the custom GPU driver is active..."

    # <<< NEW SECTION START >>>
    # --- Step 7: Configure Runtime Environment for Custom Drivers ---
    echo "--> Configuring runtime environment to use custom Adreno/Mali drivers..."
    
    # This is the path where Mesa's ICD files are installed
    ICD_PATH="/usr/share/vulkan/icd.d"
    ADRENO_ICD="/usr/share/vulkan/icd.d/freedreno_icd.aarch64.json"
    PANFROST_ICD="/usr/share/vulkan/icd.d/panfrost_icd.aarch64.json"
    PROFILE_SCRIPT="/etc/profile.d/mesa-drivers.sh"
    
    # For Adreno GPUs (Turnip driver)
    if [ -f "$ICD_PATH/freedreno_icd.aarch64.json" ]; then
        echo "--> Found Adreno (Turnip) Vulkan driver. Setting environment variables."
        # Force the Turnip driver for Vulkan
        echo 'export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/freedreno_icd.aarch64.json' >> /etc/profile.d/mesa-drivers.sh
        # Force the Freedreno driver for OpenGL
        echo 'export GALLIUM_DRIVER=freedreno' >> /etc/profile.d/mesa-drivers.sh
        echo "--> Adreno environment configured."
        export VK_ICD_FILENAMES="$ADRENO_ICD"
        export GALLIUM_DRIVER=freedreno
        
    
    # For Mali GPUs (Panfrost driver)
    elif [ -f "$ICD_PATH/panfrost_icd.aarch64.json" ]; then
        echo "--> Found Mali (Panfrost) Vulkan driver. Setting environment variables."
        # Force the Panfrost driver for Vulkan
        echo 'export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/panfrost_icd.aarch64.json' >> /etc/profile.d/mesa-drivers.sh
        # Force the Panfrost driver for OpenGL
        echo 'export GALLIUM_DRIVER=panfrost' >> /etc/profile.d/mesa-drivers.sh
        echo "--> Mali environment configured."
        export VK_ICD_FILENAMES="$PANFROST_ICD"
        export GALLIUM_DRIVER=panfrost
    
    else
        echo "--> WARNING: Could not find hardware-specific Vulkan ICD files. GPU acceleration may not work."
    fi
    
    # Make the script executable so it's loaded on login
    chmod +x /etc/profile.d/mesa-drivers.sh
    
    echo "--> Runtime environment configuration complete. Changes will apply on next login."

    if vulkaninfo --summary | grep -q "llvmpipe"; then
        # This block executes if the hardware driver FAILED to load.
        echo "------------------------------------------------------------"
        echo "GPU DRIVER VERIFICATION FAILED!"
        echo "The system has fallen back to the 'llvmpipe' software renderer."
        echo "This means hardware acceleration is NOT working."
        echo "Please report this issue."
        echo "------------------------------------------------------------"
        touch /root/.gpu_acceleration_failed
    
    else
        # This block executes if "llvmpipe" was NOT found, meaning SUCCESS.
        echo "--> SUCCESS: A non-CPU Vulkan driver is active."
        # We can also clean up any old failure flag just in case
        rm -f /root/.gpu_acceleration_failed
    
    fi
    echo "--- Zephyrine Proot Setup: Environment build complete! ---"
    touch "$MESA_BUILD_FLAG"
    echo "--> Created Mesa build flag file. Build will be skipped on next launch."
else
    echo "--- Zephyrine Proot Setup: Mesa build flag found. Skipping intensive build process. ---"
fi

# --- Final Step: Restore a clean .bashrc and execute the launcher ---
echo "--> Restoring .bashrc and launching application..."
LAUNCH_DIR="{ROOT_DIR}"
echo 'export PS1="[\\\\u@\\\\h \\\\W]\\\\$ "' > ~/.bashrc
if [ -d "$LAUNCH_DIR" ]; then
    cd "$LAUNCH_DIR"
    python3 launcher.py
else
    echo "ERROR: Could not find launcher directory: $LAUNCH_DIR"
fi

echo "--- Zephyrine Launcher: Proot execution finished. Exiting shell. ---"
exit $?
"""
        with open(bashrc_path, "w", encoding="utf-8") as f:
            f.write(auto_exec_script)
        os.chmod(bashrc_path, 0o755)

        print_system("✅ Proot environment configured for build and execution.")

    except Exception as e:
        print_error(f"Failed to write build script to proot environment: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 5. Re-execute the script inside the proot environment
    print_colored("SYSTEM", "--- Relaunching inside glibc Compatibility Layer for Build ---", "SUCCESS")
    relaunch_env = os.environ.copy()
    proot_distro_path = shutil.which("proot-distro")
    relaunch_cmd = [proot_distro_path, "login", PROOT_ALIAS]
    try:
        print_aetherhand("Executing Compatibility layer with Android with debian bootstrap")
        os.execve(relaunch_cmd[0], relaunch_cmd, relaunch_env)
    except Exception as e:
        print_error(f"Failed to execve into proot-distro: {e}")
        print_error("Your environment is likely set up. Please try running manually:")
        print_colored("COMMAND", f"proot-distro login {PROOT_ALIAS}", "SUCCESS")
        setup_failures.append("Failed to configure Hugging Face cache directory. Reporting to the launcher")

def _install_miniforge_and_check_overall_environment() -> Optional[str]:
    """
    Downloads and installs Miniforge silently if Conda is not found.
    This version correctly handles different C libraries and environments by delegating
    or providing specific instructions.

    Returns:
        The path to the newly installed conda executable, or None on failure.
    """
    # === Termux (Android) Special Handling ===
    if "TERMUX_VERSION" in os.environ:
        _handle_android_become_smartphone_setup()
        # The above function either exits or replaces the process, so we should never get here.
        # But as a safeguard, we return None to indicate failure to proceed.
        return None

    # --- Standard Installation for non-Termux systems ---
    print_warning("--- Conda Not Found: Initiating Miniforge Auto-Installation ---")
    try:
        import platform
        import urllib.request
    except ImportError as e:
        print_error(f"Missing essential built-in modules for installation: {e}")
        return None

    os_name = platform.system()
    machine_arch = platform.machine()
    installer_filename = ""

    if IS_IN_PROOT_ENV:
        # Before we do anything else, find and destroy any contaminated Miniforge
        # installation that might exist in the shared project directory.
        contaminated_path = os.path.join(ROOT_DIR, "miniforge3_local")
        if os.path.exists(contaminated_path):
            print_warning(f"CRITICAL: Found contaminated Miniforge install at '{contaminated_path}'.")
            print_warning("This is from a previous run and will be AGGRESSIVELY REMOVED to ensure a clean proot environment.")
            try:
                shutil.rmtree(contaminated_path)
                print_system("Successfully removed contaminated Miniforge installation.")
            except Exception as e_clean:
                print_error(f"Failed to remove contaminated Miniforge: {e_clean}")
                print_error("Please remove this directory manually and restart.")
                return None

    if os_name == "Linux":
        py_arch = "x86_64" if "x86_64" in machine_arch else "aarch64"
        is_musl = False
        try:
            libc_name, _ = platform.libc_ver()
            if 'musl' in libc_name.lower(): is_musl = True
        except Exception:
            if os.path.exists(f"/lib/ld-musl-{py_arch}.so.1"): is_musl = True

        if is_musl:
            print_warning("Detected 'musl' C library (e.g., Alpine Linux). Checking for glibc compatibility layer...")
            if not os.path.exists("/lib/libgcompat.so.0"):
                print_colored("ERROR", "------------------ ACTION REQUIRED (Alpine/musl) ------------------", "ERROR")
                print_error("This 'musl'-based system requires a compatibility layer to run Miniforge.")
                print_system("Please run the following command with root privileges, then restart this launcher:")
                print_colored("COMMAND", "    sudo apk add gcompat", "SUCCESS")
                print_colored("ERROR", "-------------------------------------------------------------------", "ERROR")
                return None
            else:
                print_system("Found 'gcompat' compatibility layer. Proceeding with standard installer.")
        py_os = "Linux"
        installer_filename = f"Miniforge3-{py_os}-{py_arch}.sh"

    elif os_name == "Darwin":  # macOS
        py_os = "MacOSX"
        py_arch = "x86_64" if "x86_64" in machine_arch else "arm64"
        installer_filename = f"Miniforge3-{py_os}-{py_arch}.sh"
    elif os_name == "Windows":
        py_os = "Windows"
        py_arch = "x86_64"
        installer_filename = f"Miniforge3-{py_os}-{py_arch}.exe"
    else:
        print_error(f"Unsupported Operating System for auto-installation: {os_name}")
        return None

    download_url = f"https://github.com/conda-forge/miniforge/releases/latest/download/{installer_filename}"
    installer_local_path = os.path.join(ROOT_DIR, installer_filename)

    # Download logic
    print_system(f"Downloading Miniforge installer for {py_os}-{py_arch}...")
    print_system(f"  from: {download_url}")
    print_system(f"  to:   {installer_local_path}")
    try:
        with urllib.request.urlopen(download_url) as response, open(installer_local_path, 'wb') as out_file:
            total_size = int(response.info().get('Content-Length', 0))
            bytes_so_far = 0
            chunk_size = 8192

            while True:
                chunk = response.read(chunk_size)
                if not chunk: break
                out_file.write(chunk)
                bytes_so_far += len(chunk)
                if total_size > 0:
                    percent = float(bytes_so_far) / total_size * 100
                    sys.stdout.write(
                        f"\r  Progress: [{int(percent / 5) * '#'}{int(20 - percent / 5) * ' '}] {percent:.1f}%")
                    sys.stdout.flush()
        print("\nDownload complete.")
    except Exception as e:
        print_error(f"\nFailed to download Miniforge installer: {e}")
        if os.path.exists(installer_local_path): os.remove(installer_local_path)
        return None
    install_path = os.path.join(ROOT_DIR, "miniforge3_local")
    # Installation logic
    print_system(f"Installing Miniforge to local directory: {install_path}")
    if os.path.exists(install_path):
        print_warning(f"Removing existing local installation at: {install_path}")
        shutil.rmtree(install_path)
    try:
        if os_name in ["Linux", "Darwin"]:
            shell_executable = _find_shell_executable()
            if not shell_executable: raise RuntimeError("Installation failed: No suitable shell found.")
            os.chmod(installer_local_path, 0o755)
            install_cmd = [shell_executable, installer_local_path, "-b", "-p", install_path]
            subprocess.run(install_cmd, check=True, capture_output=True, text=True)
            is_in_proot_env = (os.path.exists("/etc/debian_version") and
                               "TERMUX_VERSION" not in os.environ and
                               os_name == "Linux")
            if is_in_proot_env:
                print_system("Proot environment detected. Adding Miniforge to shell PATH...")
                # The install_path is where we just installed miniforge, e.g., /root/miniforge3_local
                conda_bin_path = os.path.join(install_path, "bin")
                bashrc_path = os.path.expanduser("~/.bashrc")
                export_line = f'\n# Added by Zephyrine Launcher\nexport PATH="{conda_bin_path}:$PATH"\n'
                
                try:
                    with open(bashrc_path, "a", encoding="utf-8") as f:
                        f.write(export_line)
                    print_system(f"Successfully added '{conda_bin_path}' to '{bashrc_path}'.")
                    print_warning("Please log out and log back into the proot shell for PATH changes to take full effect in manual sessions.")
                except Exception as e_bashrc:
                    print_error(f"Failed to automatically add Conda to PATH: {e_bashrc}")
                    print_warning("You may need to add it manually to your ~/.bashrc file.")
        elif os_name == "Windows":
            install_cmd = [installer_local_path, "/S", "/InstallationType=JustMe", "/RegisterPython=0",
                           f"/D={install_path}"]
            subprocess.run(install_cmd, check=True, capture_output=True, text=True)
        print_system("✅ Miniforge installed successfully.")
    except (subprocess.CalledProcessError, RuntimeError) as e:
        print_error("Miniforge installation failed.")
        if isinstance(e, subprocess.CalledProcessError):
            print_error(f"Return Code: {e.returncode}")
            if e.stdout: print_error(f"Stdout: {e.stdout.strip()}")
            if e.stderr: print_error(f"Stderr: {e.stderr.strip()}")
        else:
            print_error(f"Reason: {e}")
        return None
    finally:
        if os.path.exists(installer_local_path):
            print_system(f"Cleaning up installer: {installer_local_path}")
            os.remove(installer_local_path)

    # Return path logic
    if os_name == "Windows":
        conda_exe_path = os.path.join(install_path, "Scripts", "conda.exe")
    else:
        conda_exe_path = os.path.join(install_path, "bin", "conda")
    if os.path.exists(conda_exe_path):
        return conda_exe_path
    else:
        print_error(f"Installation seemed to succeed, but conda executable not found at: {conda_exe_path}")
        return None



def find_conda_executable(attempt_number: int):
    global CONDA_EXECUTABLE

    # 1. Check Cache First
    if os.path.exists(CONDA_PATH_CACHE_FILE):
        try:
            with open(CONDA_PATH_CACHE_FILE, 'r', encoding='utf-8') as f_cache:
                cached_path = f_cache.read().strip()
            if cached_path:
                print_system(f"Found cached Conda path: '{cached_path}'. Verifying...")
                if attempt_number == 1 and os.path.isfile(cached_path):
                    CONDA_EXECUTABLE = cached_path
                    return CONDA_EXECUTABLE
                else:
                    # On retries (attempt > 1), or if the file doesn't exist, we must verify.
                    print_system(f"Found cached Conda path: '{cached_path}'. Verifying (Attempt #{attempt_number})...")
                    if _verify_conda_path(cached_path):
                        CONDA_EXECUTABLE = cached_path
                        print_system(f"Using verified cached Conda: {CONDA_EXECUTABLE}")
                        return CONDA_EXECUTABLE
                    else:
                        print_warning(f"Cached Conda path '{cached_path}' is invalid. Clearing cache and re-searching.")
                        try:
                            os.remove(CONDA_PATH_CACHE_FILE)
                        except OSError:
                            pass  # Ignore if removal fails
        except Exception as e_cache_read:
            print_warning(f"Error reading Conda path cache file: {e_cache_read}. Proceeding with search.")

    # 2. Try shutil.which (standard PATH search)
    conda_exe_name = "conda.exe" if IS_WINDOWS else "conda"
    conda_path_from_which = shutil.which(conda_exe_name)
    if conda_path_from_which and _verify_conda_path(conda_path_from_which):
        CONDA_EXECUTABLE = conda_path_from_which
        print_system(f"Found and verified Conda via PATH: {CONDA_EXECUTABLE}")
        try:  # Write to cache
            with open(CONDA_PATH_CACHE_FILE, 'w', encoding='utf-8') as f_cache:
                f_cache.write(CONDA_EXECUTABLE)
            print_system(f"Cached Conda path: {CONDA_EXECUTABLE}")
        except IOError as e_cache_write:
            print_warning(f"Could not write to Conda path cache file: {e_cache_write}")
        return CONDA_EXECUTABLE

    # <<< MODIFICATION: CONTEXT-AWARE RECURSIVE SEARCH >>>
    # Only perform the slow, intensive recursive search if we are NOT in the base Termux environment.
    # The search fails due to permissions in Termux anyway, so it's pointless and generates errors.
    IS_IN_BASE_TERMUX = "TERMUX_VERSION" in os.environ and not os.path.exists("/etc/debian_version")

    if IS_IN_BASE_TERMUX:
        print_system("Termux environment detected. Skipping slow recursive file search due to permissions.")
        CONDA_EXECUTABLE = None
        return None

    # 3. If not found via PATH, perform recursive search
    print_warning(
        f"'{conda_exe_name}' not found in PATH or cache was invalid. Attempting recursive search from system root(s)...")
    print_warning("This can take a significant amount of time, especially on large file systems.")

    search_roots = []
    if IS_WINDOWS:
        import string
        available_drives = [f"{d}:\\" for d in string.ascii_uppercase if os.path.exists(f"{d}:\\")]
        if not available_drives: available_drives = ["C:\\"]
        search_roots.extend(available_drives)
    else:
        search_roots.append("/")

    target_filenames_to_search = [conda_exe_name]
    if IS_WINDOWS: target_filenames_to_search.append("conda.bat")

    for root_dir_to_search in search_roots:
        print_system(f"Searching for Conda in '{root_dir_to_search}'...")
        for dirpath, dirnames, filenames_in_dir in os.walk(root_dir_to_search, topdown=True,
                                                           onerror=lambda e: print_warning(
                                                                   f"Permission or other error accessing {e.filename}, skipping.")):
            if not IS_WINDOWS and root_dir_to_search == "/":
                dirs_to_skip = ['proc', 'sys', 'dev', 'run', 'lost+found', '.gvfs', 'snap', 'mnt', 'media']
                dirnames[:] = [d for d in dirnames if not (d.lower() in dirs_to_skip or (
                            d.startswith('.') and len(d) > 1 and d not in ['.conda', '.config',
                                                                           '.local'] and os.path.join(dirpath,
                                                                                                      d) == os.path.join(
                        root_dir_to_search, d)))]

            for target_fname in target_filenames_to_search:
                if target_fname in filenames_in_dir:
                    potential_path = os.path.join(dirpath, target_fname)
                    if _verify_conda_path(potential_path):
                        CONDA_EXECUTABLE = potential_path
                        print_system(f"Found and verified Conda via recursive search: {CONDA_EXECUTABLE}")
                        try:  # Write to cache
                            with open(CONDA_PATH_CACHE_FILE, 'w', encoding='utf-8') as f_cache:
                                f_cache.write(CONDA_EXECUTABLE)
                            print_system(f"Cached Conda path: {CONDA_EXECUTABLE}")
                        except IOError as e_cache_write:
                            print_warning(f"Could not write to Conda path cache file: {e_cache_write}")
                        return CONDA_EXECUTABLE
            time.sleep(0.0001)

    print_error("Conda executable could not be found even after recursive search.")
    CONDA_EXECUTABLE = None
    return None


def get_conda_base_path():
    if not CONDA_EXECUTABLE: return None
    try:
        cmd_list = [CONDA_EXECUTABLE, "info", "--base"]
        if IS_WINDOWS and CONDA_EXECUTABLE.lower().endswith(".bat"):
            cmd_list = ['cmd', '/c', CONDA_EXECUTABLE, "info", "--base"]
        result = subprocess.run(cmd_list, capture_output=True, text=True, check=True, timeout=10, errors='replace')
        return result.stdout.strip()
    except Exception as e:
        print_warning(f"Could not get conda base path using '{' '.join(cmd_list)}': {e}")
        return None


def get_conda_env_path(env_name_or_path):  # This function's usage for the *target* env is reduced
    if not CONDA_EXECUTABLE: return None
    # If it's already a path and looks like a conda env, return it
    if os.path.isdir(env_name_or_path) and os.path.exists(os.path.join(env_name_or_path, 'conda-meta')):
        return os.path.abspath(env_name_or_path)

    # If it's not a path, try to resolve it as a name (though we are moving away from this for the target env)
    cmd_list = [CONDA_EXECUTABLE, "info", "--envs", "--json"]
    if IS_WINDOWS and CONDA_EXECUTABLE.lower().endswith(".bat"):
        cmd_list = ['cmd', '/c', CONDA_EXECUTABLE, "info", "--envs", "--json"]

    try:
        result = subprocess.run(cmd_list, capture_output=True, text=True, check=True, timeout=15, errors='replace')
        envs_info = json.loads(result.stdout)
        # This part looks for environments by name in the standard envs_dirs
        for env_path_str in envs_info["envs"]:
            if os.path.basename(env_path_str) == env_name_or_path:  # env_name_or_path here would be a name
                return env_path_str
    except Exception as e:
        print_warning(f"Could not list/parse conda envs using '{' '.join(cmd_list)}': {e}")

    # If env_name_or_path was a name and not found, and not a direct path, then it might not exist
    # or is not in the standard conda envs directories.
    # For prefix-based envs not in standard dirs, this lookup by name won't find them unless they happen to be registered.
    return None


def create_conda_env(env_prefix_path, python_versions_to_try):  # Takes env_prefix_path directly
    if not CONDA_EXECUTABLE:
        print_error("Conda executable not found. Cannot create environment.")
        return False

    os.makedirs(os.path.dirname(env_prefix_path), exist_ok=True)

    for py_version in python_versions_to_try:
        print_system(
            f"Attempting to create Conda environment at prefix '{env_prefix_path}' with Python {py_version}...")
        cmd_list_base = ["create", "--yes", "--prefix", env_prefix_path, f"python={py_version}", "-c", "conda-forge"]

        cmd_to_run = [CONDA_EXECUTABLE] + cmd_list_base
        if IS_WINDOWS and CONDA_EXECUTABLE.lower().endswith(".bat"):
            cmd_to_run = ['cmd', '/c', CONDA_EXECUTABLE] + cmd_list_base

        print_system(f"Executing: {' '.join(cmd_to_run)}")
        try:
            process = subprocess.Popen(cmd_to_run, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                       errors='replace')
            stdout_thread = threading.Thread(target=stream_output, args=(process.stdout, f"CONDA-CREATE-PREFIX-OUT"),
                                             daemon=True)
            stderr_thread = threading.Thread(target=stream_output, args=(process.stderr, f"CONDA-CREATE-PREFIX-ERR"),
                                             daemon=True)
            stdout_thread.start()
            stderr_thread.start()

            process.wait(timeout=600)
            stdout_thread.join(timeout=5)
            stderr_thread.join(timeout=5)

            if process.returncode == 0:
                print_system(
                    f"Conda environment created successfully at prefix '{env_prefix_path}' with Python {py_version}.")
                if not (os.path.isdir(env_prefix_path) and os.path.exists(os.path.join(env_prefix_path, 'conda-meta'))):
                    print_error(
                        f"Critical: Conda environment at prefix '{env_prefix_path}' seems invalid post-creation.")
                    return False
                return True
            else:
                print_warning(
                    f"Failed to create Conda environment at prefix '{env_prefix_path}' with Python {py_version} (retcode: {process.returncode}).")
                if os.path.exists(env_prefix_path):
                    print_warning(f"Attempting to clean up partially created environment at '{env_prefix_path}'...")
                    try:
                        shutil.rmtree(env_prefix_path)
                        print_system(f"Successfully removed partially created environment: {env_prefix_path}")
                    except Exception as e_rm:
                        print_error(f"Could not remove partially created environment '{env_prefix_path}': {e_rm}")

        except subprocess.TimeoutExpired:
            print_error(
                f"Timeout while creating Conda environment at prefix '{env_prefix_path}' with Python {py_version}.")
            if process and process.poll() is None: process.kill()
        except Exception as e:
            print_error(
                f"An unexpected error occurred while trying to create Conda environment at prefix '{env_prefix_path}' with Python {py_version}: {e}")
            print_error(
                f"Failed to create Conda environment at prefix '{env_prefix_path}' with any of the specified Python versions: {python_versions_to_try}.")
            _remove_flag_files(FLAG_FILES_TO_RESET_ON_ENV_RECREATE)

    print_error(
        f"Failed to create Conda environment at prefix '{env_prefix_path}' with any of the specified Python versions: {python_versions_to_try}.")
    return False


# --- License Acceptance Logic ---
LICENSES_TO_ACCEPT = [
    ("APACHE_2.0.txt", "Primary Framework Components (Apache 2.0)"),
    ("MIT_Deepscaler.txt", "Deepscaler Model/Code Components (MIT)"),
    ("MIT_Zephyrine.txt", "Project Zephyrine Core & UI (MIT)"),
    ("LICENSE_THIRD_PARTY_MODELS.txt", "Various licenses for bundled AI Models (see file for details)")
]


def load_licenses() -> tuple[dict[str, str], str]:
    licenses_content = {}
    combined_text = "Please review the following licenses before proceeding. You must accept all terms to use this software.\n\n"
    print_system("Loading licenses...")
    if not os.path.isdir(LICENSE_DIR):
        print_error(f"License directory not found: {LICENSE_DIR}")
        sys.exit(1)

    for filename, description in LICENSES_TO_ACCEPT:
        filepath = os.path.join(LICENSE_DIR, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            licenses_content[filename] = content
            combined_text += f"\n\n{'=' * 20} LICENSE: {filename} ({description}) {'=' * 20}\n\n{content}"
            print_system(f"  Loaded: {filename}")
        except FileNotFoundError:
            print_error(
                f"License file not found: {filepath}. Please ensure all license files are present in the '{LICENSE_DIR}' directory.")
            sys.exit(1)
        except Exception as e:
            print_error(f"Error reading license file {filepath}: {e}")
            sys.exit(1)
    return licenses_content, combined_text.strip()


def calculate_reading_time(text: str) -> float:
    if not TIKTOKEN_AVAILABLE:
        print_warning("tiktoken not found, cannot estimate reading time. Defaulting to 60s estimate.")
        return 60.0
    if not text: return 0.0

    WORDS_PER_MINUTE = 200
    AVG_WORDS_PER_TOKEN = 0.75

    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
    except Exception as e:
        print_warning(f"Failed to get tiktoken encoding 'cl100k_base': {e}. Trying 'p50k_base'.")
        try:
            enc = tiktoken.get_encoding("p50k_base")
        except Exception as e2:
            print_warning(f"Failed to get tiktoken encoding 'p50k_base': {e2}. Defaulting reading time to 60s.")
            return 60.0

    try:
        tokens = enc.encode(text)
        num_tokens = len(tokens)
        estimated_words = num_tokens * AVG_WORDS_PER_TOKEN
        estimated_minutes = estimated_words / WORDS_PER_MINUTE
        estimated_seconds = estimated_minutes * 60
        print_system(
            f"Estimated reading time for licenses: ~{estimated_minutes:.1f} minutes ({estimated_seconds:.0f} seconds, based on {num_tokens} tokens).")
        return estimated_seconds
    except Exception as e:
        print_warning(f"Failed to calculate reading time with tiktoken: {e}. Defaulting to 60s.")
        return 60.0


def _stream_log_file(log_path, process_to_monitor, stream_name_prefix="LOG"):
    """Continuously streams a log file to stdout."""
    print_system(f"[{stream_name_prefix}-STREAMER] Starting to monitor log file: {log_path}")
    initial_wait_tries = 10  # Try for 1 second (10 * 0.1s)
    file_ready = False

    for _ in range(initial_wait_tries):
        if os.path.exists(log_path) and os.path.getsize(log_path) > 0:
            file_ready = True
            break
        if process_to_monitor.poll() is not None: # Child already exited
            if os.path.exists(log_path): # Check one last time if file appeared
                file_ready = True
            break
        time.sleep(0.1)

    if not file_ready:
        if os.path.exists(log_path): # Exists but is empty
            print_warning(f"[{stream_name_prefix}-STREAMER] Log file {log_path} exists but is empty. Will monitor.")
        else: # Does not exist
            print_error(f"[{stream_name_prefix}-STREAMER] Log file {log_path} not found or empty after initial wait. Monitoring will likely fail if not created shortly.")
            # We still proceed to open, it might be created very soon.

    try:
        with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
            # Optional: f.seek(0, os.SEEK_END) # If you only want *new* content after streamer starts
            while process_to_monitor.poll() is None:  # Loop while the relaunched process is running
                line = f.readline()
                if line:
                    sys.stdout.write(f"[RL-{stream_name_prefix}] {line.strip()}\n")
                    sys.stdout.flush()
                else:
                    time.sleep(0.05)  # Wait for new content
            # After process ends, read any remaining lines
            time.sleep(0.1) # Give a brief moment for final flushes from child
            for line in f.readlines(): # Read all remaining lines
                if line:
                    sys.stdout.write(f"[RL-{stream_name_prefix}|FINAL] {line.strip()}\n")
                    sys.stdout.flush()
    except FileNotFoundError:
        # This might happen if the process exits very quickly before the file is ever created/checked again
        print_warning(f"[{stream_name_prefix}-STREAMER] Log file {log_path} was not found during active streaming attempt.")
    except Exception as e:
        print_error(f"[{stream_name_prefix}-STREAMER] Error streaming log file {log_path}: {e}")
    finally:
        print_system(f"[{stream_name_prefix}-STREAMER] Stopped monitoring log file: {log_path}")


def display_license_prompt(stdscr, licenses_text_lines: list, estimated_seconds: float) -> tuple[bool, float]:
    # (This function is extracted from the launcher.py you provided)
    import curses  # Ensure curses is imported, typically done at the top of launcher.py

    # Initialize curses settings
    curses.curs_set(0)  # Hide the cursor
    stdscr.nodelay(False)  # Make getch() blocking
    stdscr.keypad(True)  # Enable keypad mode for special keys
    curses.noecho()  # Turn off automatic echoing of keys to the screen
    curses.cbreak()  # React to keys instantly, without requiring Enter

    # Color pair definitions (check if curses.has_colors() first in a real app)
    if curses.has_colors():
        curses.start_color()
        # Define color pairs: (pair_number, foreground_color, background_color)
        curses.init_pair(1, curses.COLOR_YELLOW, curses.COLOR_BLACK)  # Instructions, time
        curses.init_pair(2, curses.COLOR_WHITE, curses.COLOR_BLACK)  # License text
        curses.init_pair(3, curses.COLOR_BLACK, curses.COLOR_GREEN)  # Accept button selected
        curses.init_pair(4, curses.COLOR_BLACK, curses.COLOR_RED)  # Reject button selected
        curses.init_pair(5, curses.COLOR_CYAN, curses.COLOR_BLACK)  # Header
        curses.init_pair(6, curses.COLOR_GREEN, curses.COLOR_BLACK)  # Accept button normal
        curses.init_pair(7, curses.COLOR_RED, curses.COLOR_BLACK)  # Reject button normal
    else:  # Fallback for terminals without color
        curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_WHITE, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_WHITE, curses.COLOR_BLACK)  # Selected would be A_REVERSE
        curses.init_pair(4, curses.COLOR_WHITE, curses.COLOR_BLACK)  # Selected would be A_REVERSE
        curses.init_pair(5, curses.COLOR_WHITE, curses.COLOR_BLACK)
        curses.init_pair(6, curses.COLOR_WHITE, curses.COLOR_BLACK)
        curses.init_pair(7, curses.COLOR_WHITE, curses.COLOR_BLACK)

    max_y, max_x = stdscr.getmaxyx()

    # Initial size check
    if max_y < 10 or max_x < 40:  # Minimum reasonable size
        # curses.wrapper handles endwin() if we raise an exception or return.
        # For a cleaner exit here if too small *before* loop, we might raise.
        # However, the loop below also checks, so this could be omitted if preferred.
        # For now, let the loop handle the "too small" message drawing.
        pass

    top_line = 0  # First line of the license text to display
    total_lines = len(licenses_text_lines)
    accepted = False
    start_time = time.monotonic()
    current_selection = 0  # 0 for Accept, 1 for Not Accept

    last_key_error_message = None  # To display errors from getch within curses window

    while True:
        try:
            stdscr.clear()
            max_y, max_x = stdscr.getmaxyx()  # Get current dimensions

            # Check size again in case of resize during loop
            if max_y < 10 or max_x < 40:
                last_key_error_message = "Terminal too small! Resize or press 'N' to exit."
                # Draw minimal error and exit instruction
                try:
                    stdscr.addstr(0, 0, last_key_error_message[:max_x - 1],
                                  curses.color_pair(7) | curses.A_BOLD)  # type: ignore
                    stdscr.addstr(2, 0, "Press 'N' or Enter (if on Not Accept) to exit."[:max_x - 1],
                                  curses.color_pair(1))  # type: ignore

                    reject_button_text_small = "[ Not Accept (N) to Exit ]"
                    stdscr.addstr(max_y - 2, max(0, (max_x - len(reject_button_text_small)) // 2),
                                  reject_button_text_small, curses.color_pair(4) | curses.A_REVERSE)  # type: ignore
                except curses.error:
                    pass  # If even this fails, not much to do
                stdscr.refresh()

                key = stdscr.getch()  # Wait for key to exit or resize
                if key == ord('n') or key == ord('N') or key == curses.KEY_ENTER or key == 10 or key == 13:
                    accepted = False
                    break
                elif key == curses.KEY_RESIZE:
                    continue  # Loop again to redraw with new size
                continue  # For any other key, just re-loop and show small screen message

            # Define layout based on current terminal size
            TEXT_AREA_HEIGHT = max(1, max_y - 7)  # Number of lines for license text
            TEXT_START_Y = 1
            SCROLL_INFO_Y = max_y - 5
            EST_TIME_Y = max_y - 4
            INSTR_Y = max_y - 3
            PROMPT_Y = max_y - 2

            # 1. Header
            header = "--- SOFTWARE LICENSE AGREEMENT ---"
            stdscr.addstr(0, max(0, (max_x - len(header)) // 2), header,
                          curses.color_pair(5) | curses.A_BOLD)  # type: ignore

            # 2. License Text Area
            for i in range(TEXT_AREA_HEIGHT):
                line_idx = top_line + i
                if line_idx < total_lines:
                    display_line = licenses_text_lines[line_idx][:max_x - 1].rstrip()  # Truncate line if too long
                    stdscr.addstr(TEXT_START_Y + i, 0, display_line, curses.color_pair(2))  # type: ignore
                else:
                    break  # No more lines to display in the current view

            # 3. Scroll Information
            if total_lines > TEXT_AREA_HEIGHT:
                progress = f"Line {top_line + 1} - {min(top_line + TEXT_AREA_HEIGHT, total_lines)} of {total_lines}"
                stdscr.addstr(SCROLL_INFO_Y, max(0, max_x - len(progress) - 1), progress,
                              curses.color_pair(1))  # type: ignore
            else:
                stdscr.addstr(SCROLL_INFO_Y, 0, "All text visible.", curses.color_pair(1))  # type: ignore

            # 4. Estimated Time
            est_time_str = f"Estimated minimum review time: {estimated_seconds / 60:.1f} min ({estimated_seconds:.0f} sec)"
            stdscr.addstr(EST_TIME_Y, 0, est_time_str[:max_x - 1], curses.color_pair(1))  # type: ignore

            # 5. Instructions
            instr = "Scroll: UP/DOWN/PGUP/PGDN/HOME/END | Select: LEFT/RIGHT | Confirm: ENTER / (a/n)"
            stdscr.addstr(INSTR_Y, 0, instr[:max_x - 1], curses.color_pair(1))  # type: ignore

            # 6. Accept/Reject Buttons
            accept_button_text = "[ Accept (A) ]"
            reject_button_text = "[ Not Accept (N) ]"
            button_spacing = 4
            buttons_total_width = len(accept_button_text) + len(reject_button_text) + button_spacing
            start_buttons_x = max(0, (max_x - buttons_total_width) // 2)

            accept_style = curses.color_pair(3) | curses.A_REVERSE if current_selection == 0 else curses.color_pair(
                6)  # type: ignore
            reject_style = curses.color_pair(4) | curses.A_REVERSE if current_selection == 1 else curses.color_pair(
                7)  # type: ignore

            stdscr.addstr(PROMPT_Y, start_buttons_x, accept_button_text, accept_style)
            stdscr.addstr(PROMPT_Y, start_buttons_x + len(accept_button_text) + button_spacing, reject_button_text,
                          reject_style)

            if last_key_error_message:  # Display error from getch within curses window if any
                stdscr.addstr(max_y - 1, 0, last_key_error_message[:max_x - 1],
                              curses.color_pair(7) | curses.A_BOLD)  # type: ignore
                last_key_error_message = None  # Clear after displaying

            stdscr.refresh()

        except curses.error as e_curses_draw:
            # This error means drawing failed, perhaps due to a very rapid resize or unusual terminal state.
            # Log it if possible (launcher.py doesn't use loguru by default, so print_warning if available)
            # For now, just sleep briefly and let the loop retry drawing with new dimensions.
            if 'print_warning' in globals():  # Check if print_warning is available
                print_warning(f"Curses draw error: {e_curses_draw}. Will attempt redraw.")
            time.sleep(0.05)
            continue  # Retry drawing

        try:
            key = stdscr.getch()  # Get user input
        except KeyboardInterrupt:  # Handle Ctrl+C gracefully
            accepted = False
            break
        except curses.error as e_getch:  # If getch itself errors (e.g. during resize after clear but before refresh)
            last_key_error_message = f"Input error: {e_getch}. Resizing or retrying..."
            time.sleep(0.05)  # Small delay before retrying the loop
            continue  # Retry the main loop, which will re-check dimensions and redraw
        except Exception as e_getch_other:  # Catch any other unexpected error during getch
            last_key_error_message = f"Unexpected input error: {e_getch_other}. Retrying..."
            time.sleep(0.05)
            continue

        # Process Key Input
        if key == ord('a') or key == ord('A'):
            accepted = True
            break
        elif key == ord('n') or key == ord('N'):
            accepted = False
            break
        elif key == curses.KEY_ENTER or key == 10 or key == 13:  # Enter key
            if current_selection == 0:
                accepted = True
            else:
                accepted = False
            break
        elif key == curses.KEY_LEFT:
            current_selection = 0
        elif key == curses.KEY_RIGHT:
            current_selection = 1
        elif key == curses.KEY_DOWN:
            top_line = min(top_line + 1, max(0, total_lines - TEXT_AREA_HEIGHT))
        elif key == curses.KEY_UP:
            top_line = max(0, top_line - 1)
        elif key == curses.KEY_NPAGE:  # Page Down
            top_line = min(max(0, total_lines - TEXT_AREA_HEIGHT), top_line + TEXT_AREA_HEIGHT)
        elif key == curses.KEY_PPAGE:  # Page Up
            top_line = max(0, top_line - TEXT_AREA_HEIGHT)
        elif key == curses.KEY_HOME:
            top_line = 0
        elif key == curses.KEY_END:
            top_line = max(0, total_lines - TEXT_AREA_HEIGHT)
        elif key == curses.KEY_RESIZE:  # Terminal resize event
            # The loop will restart, re-calculate max_y, max_x, clear and redraw.
            # This is the standard way to handle resize in curses.
            if 'print_system' in globals():  # Check if print_system is available
                print_system("Terminal resized, redrawing curses UI...")  # Optional: log to external file if needed
            pass  # Just let the loop continue to redraw

    end_time = time.monotonic()
    time_taken_to_decide = end_time - start_time

    # curses.wrapper will automatically call endwin(), nocbreak(), keypad(False), echo()
    return accepted, time_taken_to_decide


def _ensure_libiconv_copy() -> bool:
    """
    Ensures libiconv is installed. On Linux, it also ensures a physical copy of
    libiconv.so.2 exists as libiconv.so.1 if .so.1 is missing. This is a robust
    fix for environments with dependencies on both legacy and modern versions,
    avoiding symlink issues.
    """
    # --- CRITICAL FIX: This entire special logic ONLY applies to Linux ---
    if platform.system() != "Linux":
        print_system("libiconv copy/check logic is for Linux only. Skipping on other OS.")
        # On non-Linux systems, the standard package is sufficient and correct.
        return _ensure_conda_package("libiconv", conda_channel="conda-forge", is_critical=True)

    print_system("--- [Linux-Only] Checking and enforcing libiconv shared library files ---")

    # 1. Ensure the modern package is installed.
    if not _ensure_conda_package("libiconv", conda_channel="conda-forge", is_critical=True):
        print_error("Failed to install the base 'libiconv' package.")
        return False

    lib_dir = os.path.join(TARGET_RUNTIME_ENV_PATH, "lib")
    so_2_path = os.path.join(lib_dir, "libiconv.so.2")
    so_1_path = os.path.join(lib_dir, "libiconv.so.1")

    # 2. Verify that the source of our copy (.so.2) exists.
    if not os.path.exists(so_2_path):
        print_error(f"CRITICAL: 'libiconv.so.2' not found at '{so_2_path}' after installation.")
        return False

    # 3. If .so.1 is missing, create a physical copy.
    if not os.path.exists(so_1_path):
        print_warning(f"'libiconv.so.1' not found. Creating a physical copy from 'libiconv.so.2'.")
        try:
            shutil.copy2(so_2_path, so_1_path)
            # copy2 also copies permission bits, so it should be executable if the original is.
            print_colored("SUCCESS", f"Successfully created copy: {so_1_path}", "SUCCESS")
        except Exception as e:
            print_error(f"Failed to create copy of libiconv: {e}")
            return False
    else:
        print_system("'libiconv.so.1' already exists. No action needed.")

    # 4. Final verification of both files
    if os.path.exists(so_1_path) and os.path.exists(so_2_path):
        print_system("Verified: Both libiconv.so.1 and libiconv.so.2 are present.")
        return True
    else:
        print_error("Final verification failed. One or more required libiconv files are missing.")
        return False

def _ensure_legacy_libiconv_so() -> bool:
    """
    Ensures that both modern and legacy libiconv shared libraries are available.

    This function implements a "install modern, inject legacy" strategy:
    1. It ensures the latest `libiconv` from conda-forge is installed, which
       provides `libiconv.so.2`.
    2. It then checks for `libiconv.so.1`.
    3. If `.so.1` is missing, it downloads an older `libiconv` package from a
       legacy channel and surgically extracts *only* the `.so.1` files into the
       environment's `lib/` directory, leaving the modern version intact.
    """
    # This check is only relevant for non-Windows systems with .so files.
    if IS_WINDOWS:
        print_system("Legacy libiconv check skipped on Windows.")
        return _ensure_conda_package("libiconv", conda_channel="conda-forge", is_critical=True)

    lib_dir = os.path.join(TARGET_RUNTIME_ENV_PATH, "lib")
    required_libs = {
        "libiconv.so.2": False,
        "libiconv.so.1": False
    }

    # --- Initial check: See what's already there ---
    for lib_name in required_libs:
        if os.path.exists(os.path.join(lib_dir, lib_name)):
            required_libs[lib_name] = True

    if all(required_libs.values()):
        print_system("All required libiconv versions (.so.1 and .so.2) are already present.")
        return True

    print_system("Performing check/repair for required libiconv versions...")

    # --- Step 1: Install the modern version to guarantee libiconv.so.2 ---
    print_system("Ensuring modern 'libiconv' is installed to provide libiconv.so.2...")
    if not _ensure_conda_package("libiconv", conda_channel="conda-forge", is_critical=True):
        print_error("Failed to install the base 'libiconv' package. Cannot proceed.")
        return False

    # Re-check for .so.2
    if os.path.exists(os.path.join(lib_dir, "libiconv.so.2")):
        print_system("Verified: 'libiconv.so.2' is present.")
        required_libs["libiconv.so.2"] = True
    else:
        print_error("Critical error: 'libiconv.so.2' is NOT present after installing the package.")
        # This shouldn't happen, but we check to be safe.
        return False

    # --- Step 2: If .so.1 is missing, surgically add it ---
    if not required_libs["libiconv.so.1"]:
        print_warning("'libiconv.so.1' is missing. Attempting to inject it from a legacy package.")

        legacy_channels_to_try = [
            "conda-forge/label/cf202003::libiconv",
            "conda-forge/label/cf201901::libiconv",
            "conda-forge/label/gcc7::libiconv"
        ]

        legacy_lib_found = False
        for package_spec in legacy_channels_to_try:
            print_system(f"Attempting to download legacy package: {package_spec}")

            # Use --download-only to fetch the package without installing it
            download_cmd = [
                CONDA_EXECUTABLE, "install", "--yes", "--download-only",
                "--prefix", TARGET_RUNTIME_ENV_PATH,
                "-c", "conda-forge",
                package_spec
            ]
            if not run_command(download_cmd, ROOT_DIR, f"CONDA-DOWNLOAD-ONLY-LIBICONV", check=True):
                print_warning(f"Could not download package spec '{package_spec}'. Trying next.")
                continue

            # Find the downloaded .tar.bz2 file in the conda package cache
            try:
                info_json = subprocess.check_output([CONDA_EXECUTABLE, "info", "--json"], text=True)
                conda_info = json.loads(info_json)
                pkg_cache_dir = conda_info['pkgs_dirs'][0]

                # The package name is 'libiconv'. We find the downloaded file.
                # Example filename: libiconv-1.16-h516909a_2.tar.bz2
                import glob
                downloaded_files = glob.glob(os.path.join(pkg_cache_dir, "libiconv-*.tar.bz2"))
                if not downloaded_files:
                    raise FileNotFoundError("Could not find downloaded libiconv tarball in cache.")

                # Sort by modification time to get the most recently downloaded one
                package_tarball_path = max(downloaded_files, key=os.path.getmtime)
                print_system(f"Found downloaded package: {package_tarball_path}")

                # Surgically extract the .so.1 files
                import tarfile
                with tarfile.open(package_tarball_path, 'r:bz2') as tar:
                    members_to_extract = [
                        m for m in tar.getmembers()
                        if m.name.startswith("lib/libiconv.so.1") and m.isfile()
                    ]
                    if not members_to_extract:
                        print_warning(f"Package '{package_spec}' did not contain 'libiconv.so.1'. Trying next.")
                        continue

                    print_system(f"Extracting {len(members_to_extract)} legacy lib files into '{lib_dir}'...")
                    # We need to extract them to the root of the environment, tarfile will handle the `lib/` part.
                    tar.extractall(path=TARGET_RUNTIME_ENV_PATH, members=members_to_extract)

                    # Create the main symlink `libiconv.so.1` -> `libiconv.so.1.x.y` if it doesn't exist
                    # This is often handled by post-link scripts, but we do it manually to be safe.
                    real_so_file = None
                    for m in members_to_extract:
                        if ".so.1." in m.name:  # find the real file, e.g. libiconv.so.1.1.1
                            real_so_file = os.path.basename(m.name)
                            break

                    if real_so_file and not os.path.exists(os.path.join(lib_dir, "libiconv.so.1")):
                        print_system(f"Creating missing symlink: libiconv.so.1 -> {real_so_file}")
                        os.symlink(real_so_file, os.path.join(lib_dir, "libiconv.so.1"))

                    legacy_lib_found = True
                    break  # Success, exit the loop

            except Exception as e:
                print_error(f"An error occurred during legacy lib injection: {e}")
                traceback.print_exc()
                continue

        if legacy_lib_found and os.path.exists(os.path.join(lib_dir, "libiconv.so.1")):
            required_libs["libiconv.so.1"] = True
            print_system("Verified: 'libiconv.so.1' is now present.")
        else:
            print_error("Failed to inject 'libiconv.so.1' from any legacy package.")

    # --- Step 3: Final Verification ---
    if all(required_libs.values()):
        print_colored("SUCCESS", "All required libiconv versions (.so.1 and .so.2) are present.", "SUCCESS")
        return True
    else:
        missing = [lib for lib, found in required_libs.items() if not found]
        print_error(f"FATAL: Final libiconv check failed. Missing: {', '.join(missing)}")
        return False


def _ensure_conda_package_version(package_name: str, executable_name: str, required_major_version: int,
                                  version_arg_prefix: str = "-v") -> bool:
    """
    Ensures a specific major version of a Conda package is installed.

    This is an aggressive function that will:
    1. Check the version of the `executable_name`.
    2. If the major version is incorrect, it forcefully removes the package.
    3. It then installs the package with the correct major version string.

    Args:
        package_name: The name of the package in Conda (e.g., "nodejs").
        executable_name: The command to run to check the version (e.g., "node").
        required_major_version: The integer of the major version required (e.g., 20).
        version_arg_prefix: The argument to get the version (e.g., "-v" or "--version").

    Returns:
        True on success, False on failure.
    """
    print_system(f"--- Aggressively checking for {package_name} major version {required_major_version} ---")
    current_major = 0

    try:
        # Use run_command to ensure we are using the correct PATH from the Conda env
        # We need to capture the output here, so we modify the Popen call temporarily.
        check_command = [executable_name, version_arg_prefix]
        process = subprocess.run(check_command, capture_output=True, text=True, check=False, timeout=10,
                                 env=run_command.__globals__.get(
                                     'env'))  # Hacky way to get env from run_command's scope if it was set

        output = process.stdout.strip()
        if process.returncode == 0 and output:
            # Handle versions like "v20.12.2" or just "20.12.2"
            version_str = output.lstrip('v')
            current_major = int(version_str.split('.')[0])
            print_system(f"Found installed version of {executable_name}: {output} (Major: {current_major})")
        else:
            print_system(f"{executable_name} not found or version check failed. Will proceed with installation.")
    except (FileNotFoundError, ValueError, IndexError):
        print_system(f"Could not determine version for {executable_name}. Assuming installation is required.")
    except Exception as e:
        print_warning(
            f"Unexpected error during version check for {executable_name}: {e}. Assuming install is required.")

    if current_major == required_major_version:
        print_system(f"{package_name} is already at the correct major version {required_major_version}.")
        return True

    # --- If version is wrong or not found, be aggressive ---
    if current_major != 0:
        print_warning(f"Incorrect version of {package_name} found (Major: {current_major}). Forcing removal.")
        remove_cmd = [CONDA_EXECUTABLE, "remove", "--yes", "--force", "--prefix", TARGET_RUNTIME_ENV_PATH, package_name]
        if not run_command(remove_cmd, cwd=ROOT_DIR, name=f"FORCE-REMOVE-{package_name.upper()}", check=False):
            print_error(f"Failed to forcefully remove {package_name}. The update may fail.")

    # --- Install the correct version ---
    print_system(f"Installing {package_name} with specified major version: {required_major_version}")
    package_spec = f"{package_name}={required_major_version}"
    install_cmd = [
        CONDA_EXECUTABLE, "install", "--yes",
        "--prefix", TARGET_RUNTIME_ENV_PATH,
        "-c", "conda-forge",
        package_spec
    ]
    if not run_command(install_cmd, cwd=ROOT_DIR, name=f"INSTALL-{package_name.upper()}-V{required_major_version}",
                       check=True):
        print_error(f"Failed to install {package_spec}. Setup cannot continue.")
        return False

    if platform.system() == "Linux":
        print_system(f"--- Performing post-install patching on '{executable_name}' ---")
        try:
            # Find the absolute path to the newly installed executable
            executable_path = shutil.which(executable_name, path=os.path.join(TARGET_RUNTIME_ENV_PATH, "bin"))
            if not executable_path:
                raise FileNotFoundError(f"Could not find '{executable_name}' in Conda bin directory after install.")

            # Define the library path we need to bake into the executable
            lib_path = os.path.join(TARGET_RUNTIME_ENV_PATH, "lib")

            # The RPATH is a special section in an ELF executable.
            # '$ORIGIN' is a special token that means 'the directory where this executable is'.
            # So, '$ORIGIN/../lib' resolves to the correct `lib` directory relative to the `bin` directory.
            rpath_to_set = f"$ORIGIN/../lib"

            print_system(f"Patching '{executable_path}' to set RPATH to '{rpath_to_set}'...")
            patch_cmd = ["patchelf", "--set-rpath", rpath_to_set, executable_path]

            if not run_command(patch_cmd, cwd=ROOT_DIR, name=f"PATCHELF-{executable_name.upper()}", check=True):
                raise RuntimeError("patchelf command failed.")

            print_system(f"Successfully patched {executable_name}.")
        except Exception as e:
            print_error(f"FATAL: Failed to patch {executable_name} after installation: {e}")
            print_error("The application will likely fail to run. Check if 'patchelf' is installed correctly.")
            return False
        
    # --- Final verification ---
    try:
        post_install_process = subprocess.run([executable_name, version_arg_prefix], capture_output=True, text=True,
                                              check=True, timeout=10)
        post_install_version = post_install_process.stdout.strip().lstrip('v')
        post_install_major = int(post_install_version.split('.')[0])
        if post_install_major == required_major_version:
            print_colored("SUCCESS",
                          f"Successfully installed and verified {package_name} version {post_install_version}.",
                          "SUCCESS")
            return True
        else:
            print_error(f"FATAL: Installed {package_name} but version is still wrong! Found {post_install_version}.")
            return False
    except Exception as e:
        print_error(f"FATAL: Failed to verify {package_name} after installation: {e}")
        return False

def _ensure_conda_package(package_spec: str,
                          executable_to_check: Optional[str] = None,
                          conda_channel: str = "conda-forge",
                          is_critical: bool = True) -> bool:
    """
    Ensures a Conda package is installed in the target environment.
    This function is bimodal:
    - If 'executable_to_check' IS provided, it searches for that executable.
    - If 'executable_to_check' is NOT provided, it uses `conda list` to
      check if the package itself is installed (ideal for libraries).
    """
    log_package_name = package_spec.split('=')[0]
    is_already_installed = False

    # --- Verification Step ---
    if executable_to_check:
        # --- Mode 1: Check for an executable ---
        print_system(f"Checking for executable '{executable_to_check}' (from Conda package '{log_package_name}')...")
        found_path = shutil.which(executable_to_check)
        if found_path:
            try:
                norm_found_path = os.path.normcase(os.path.realpath(found_path))
                norm_target_env_path = os.path.normcase(os.path.realpath(TARGET_RUNTIME_ENV_PATH))
                if norm_found_path.startswith(norm_target_env_path):
                    is_already_installed = True
                    print_system(f"Executable '{executable_to_check}' found in target Conda environment: {found_path}")
                else:
                    print_warning(
                        f"Executable '{executable_to_check}' found at '{found_path}', but it's OUTSIDE the target env. Will reinstall into env.")
            except Exception as e_path:
                print_warning(f"Error verifying path for '{executable_to_check}': {e_path}. Assuming not in env.")
    else:
        # --- Mode 2: Check for the library/package via `conda list` ---
        print_system(f"Checking for Conda library/package '{log_package_name}'...")
        if not CONDA_EXECUTABLE:
            print_error("Cannot check for library: CONDA_EXECUTABLE path is not set.")
            return False  # Cannot proceed
        try:
            list_cmd = [CONDA_EXECUTABLE, "list", "-p", TARGET_RUNTIME_ENV_PATH, log_package_name]
            result = subprocess.run(list_cmd, capture_output=True, text=True, check=False)
            # A successful find will have stdout containing the package name. An empty result means not found.
            if result.returncode == 0 and log_package_name in result.stdout:
                print_system(f"Package '{log_package_name}' found in Conda environment.")
                is_already_installed = True
            else:
                print_system(f"Package '{log_package_name}' not found.")
        except Exception as e:
            print_warning(f"An exception occurred while checking for Conda package '{log_package_name}': {e}")
            is_already_installed = False  # Assume not installed on error

    # --- Installation Step ---
    if not is_already_installed:
        print_system(f"Attempting to install '{package_spec}' from channel '{conda_channel}' into Conda env...")
        conda_install_cmd = [
            CONDA_EXECUTABLE, "install", "--yes",
            "--prefix", TARGET_RUNTIME_ENV_PATH,
            "-c", conda_channel,
            package_spec
        ]
        if not run_command(conda_install_cmd, cwd=ROOT_DIR, name=f"CONDA-INSTALL-{log_package_name.upper()}",
                           check=True):
            print_error(f"Failed to install Conda package '{package_spec}'.")
            if is_critical:
                print_error(f"'{package_spec}' is a critical dependency. Exiting.")
                sys.exit(1)
            return False
        else:
            print_system(f"Successfully installed Conda package '{package_spec}'.")
            # If we just installed a library, there's no executable to re-verify, so we're done.
            if not executable_to_check:
                return True

            # Re-verify executable path after install
            found_path_after = shutil.which(executable_to_check)
            if found_path_after and os.path.normcase(os.path.realpath(found_path_after)).startswith(
                    os.path.normcase(os.path.realpath(TARGET_RUNTIME_ENV_PATH))):
                print_system(f"Executable '{executable_to_check}' now correctly located in Conda environment.")
            else:
                print_error(
                    f"Executable '{executable_to_check}' still not found in the correct Conda path after installation.")
                if is_critical: sys.exit(1)
                return False

    return True


def _detect_and_prepare_acceleration_env_vars() -> Dict[str, str]:
    """
    Attempts to detect the best available hardware acceleration and prepares
    environment variables to guide the builds in the relaunched Conda environment.
    Hierarchy: CUDA > Metal (on Apple Silicon) > Vulkan > CoreML (for Whisper on Apple) > OpenCL > CPU (with BLAS/OpenMP).
    Returns a dictionary of 'AUTODETECTED_...' environment variables.
    """
    print_system("--- Starting Hardware Acceleration Auto-Detection ---")
    detected_env_vars = {}
    primary_gpu_backend_detected = "cpu"  # Default

    # --- CUDA Detection (Highest Priority) ---
    try:
        nvidia_smi_path = None
        if IS_WINDOWS:
            # Common paths for nvidia-smi on Windows
            smi_paths = [
                os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"), "NVIDIA Corporation", "NVSMI",
                             "nvidia-smi.exe"),
                os.path.join(os.environ.get("SystemRoot", "C:\\Windows"), "System32", "nvidia-smi.exe")
            ]
            for p in smi_paths:
                if os.path.exists(p):
                    nvidia_smi_path = p
                    break
        else:  # Linux/macOS
            nvidia_smi_path = shutil.which("nvidia-smi")

        if nvidia_smi_path:
            # Try to run nvidia-smi to confirm a CUDA-capable GPU is present
            # This is a more robust check than just CUDA_PATH
            result = subprocess.run([nvidia_smi_path, "-L"], capture_output=True, text=True, timeout=5, check=False)
            if result.returncode == 0 and "GPU" in result.stdout:
                print_system("CUDA: Detected NVIDIA GPU via nvidia-smi.")
                primary_gpu_backend_detected = "cuda"
                detected_env_vars["AUTODETECTED_CUDA_AVAILABLE"] = "1"
        elif os.getenv("CUDA_HOME") or os.getenv("CUDA_PATH"):
            print_system("CUDA: Detected CUDA_HOME/CUDA_PATH. Assuming CUDA might be available (nvidia-smi not found).")
            primary_gpu_backend_detected = "cuda"  # Less certain, but a strong hint
            detected_env_vars["AUTODETECTED_CUDA_AVAILABLE"] = "1"
        else:
            print_system("CUDA: No clear indication of CUDA found (nvidia-smi or CUDA_HOME/PATH).")
    except Exception as e_cuda:
        print_warning(f"CUDA detection failed: {e_cuda}")

    # --- Metal Detection (Apple Silicon - Second Priority if CUDA not found) ---
    if primary_gpu_backend_detected == "cpu" and platform.system() == "Darwin" and platform.machine() == "arm64":
        # Check for Xcode command line tools as a proxy for Metal capabilities
        # A more direct check for Metal SDK isn't straightforward from Python without specific libraries
        if shutil.which("xcodebuild") or shutil.which("swift") or shutil.which("metal"):  # Heuristic
            print_system("Metal: Detected Apple Silicon and likely Xcode tools. Assuming Metal is available.")
            primary_gpu_backend_detected = "metal"
            detected_env_vars["AUTODETECTED_METAL_AVAILABLE"] = "1"  # For Apple platforms
        else:
            print_system("Metal: Apple Silicon detected, but Xcode tools (heuristic for Metal) not clearly found.")

    # --- Vulkan Detection (Third Priority if CUDA/Metal not found) ---
    # This is a basic check. Full Vulkan setup is simple but I doesn't understand yet.
    if primary_gpu_backend_detected == "cpu":
        vulkan_sdk_env = os.getenv("VULKAN_SDK")
        vulkaninfo_path = shutil.which("vulkaninfo" if not IS_WINDOWS else "vulkaninfo.exe")
        if vulkan_sdk_env and os.path.isdir(vulkan_sdk_env):
            print_system(f"Vulkan: Detected VULKAN_SDK environment variable: {vulkan_sdk_env}")
            primary_gpu_backend_detected = "vulkan"
            detected_env_vars["AUTODETECTED_VULKAN_AVAILABLE"] = "1"
        elif vulkaninfo_path:
            print_system(
                f"Vulkan: Found 'vulkaninfo' executable at {vulkaninfo_path}. Assuming Vulkan might be available.")
            # Running vulkaninfo can be slow or verbose, so just checking for its presence.
            primary_gpu_backend_detected = "vulkan"
            detected_env_vars["AUTODETECTED_VULKAN_AVAILABLE"] = "1"
        else:
            print_system("Vulkan: No clear indication of Vulkan SDK or vulkaninfo found.")

    # --- CoreML (Specific to Whisper on Apple platforms, can co-exist with Metal) ---
    if platform.system() == "Darwin":
        print_system("CoreML: Detected Apple platform. CoreML support can be enabled for Whisper.")
        print_system("CoreML: However CoreML is garbage in performance and quality so it is disabled, if you wish to override do")
        print_system("CoreML: export WHISPER_COREML=1 ")
        detected_env_vars["AUTODETECTED_COREML_POSSIBLE"] = "0"

    # --- OpenCL Detection (Lower priority) ---
    # if primary_gpu_backend_detected == "cpu": # Only if no other GPU backend found
    #     clinfo_path = shutil.which("clinfo" if not IS_WINDOWS else "clinfo.exe")
    #     if clinfo_path:
    #         print_system(f"OpenCL: Found 'clinfo' executable at {clinfo_path}. OpenCL might be available.")
    #         # OpenCL is simple but I doesn't understand yet to reliably use for GGUF, so we won't set it as primary_gpu_backend
    #         # unless specifically requested by user. Just note its potential.
    #         detected_env_vars["AUTODETECTED_OPENCL_POSSIBLE"] = "1"
    #     else:
    #         print_system("OpenCL: 'clinfo' not found.")

    # --- CPU Enhancements (OpenBLAS for Whisper, OpenMP for Llama) ---
    # These are not primary backends but enhancements.
    # GGML_BLAS=1 for whisper.cpp often relies on OpenBLAS being found by its CMake.
    # LLAMA_OPENMP=ON is enabled by default for CPU builds of llama-cpp-python by launcher.
    # For now, we won't try to auto-detect BLAS presence, as it's simple but I doesn't understand yet.
    # Users can still set GGML_BLAS=1 manually.
    print_system(
        f"CPU: OpenMP will be enabled by default for Llama.cpp CPU builds. User can set GGML_BLAS=1 for PyWhisperCpp CPU.")
    detected_env_vars[
        "AUTODETECTED_CPU_ENHANCEMENTS_INFO"] = "OpenMP for Llama.cpp; User can set GGML_BLAS=1 for PyWhisperCpp"

    detected_env_vars["AUTODETECTED_PRIMARY_GPU_BACKEND"] = primary_gpu_backend_detected
    print_system(
        f"--- Hardware Acceleration Auto-Detection Complete. Preferred GPU Backend: {primary_gpu_backend_detected} ---")
    if detected_env_vars.get("AUTODETECTED_COREML_POSSIBLE") == "1":
        print_system("    (CoreML also noted as possible for this platform)")

    return detected_env_vars

# --- Add Aria2p Import ---
disablebrokenaria2=True
ARIA2P_AVAILABLE = False
if not disablebrokenaria2:

    try:
        import aria2p
        ARIA2P_AVAILABLE = True
        # print_system("aria2p Python library imported successfully.") # Optional: for relaunched script
    except ImportError:
        # This warning will appear if launcher.py itself cannot import it.
        # The relaunched script (in Conda env) is where it matters most.
        print_warning("aria2p Python library not found. Aria2c download method will be unavailable.")
    # --- End Aria2p Import ---



# --- File Download Logic ---

# Global placeholder for requests_session, use string literal for type hint here too
requests_session: Optional['requests.Session'] = None

def download_file_with_progress(
    url: str,
    destination_path: str,
    file_description: str,
    requests_session: 'requests.Session',
    max_retries_non_connection_error: int = 3,
    aria2_rpc_host: str = "localhost",
    aria2_rpc_port: int = 6800,
    aria2_rpc_secret: Optional[str] = None
):
    """
    Downloads a file with a progress bar.
    1. First, attempts to find the file on the ZephyMesh network via the local API.
    2. If found, downloads directly from the peer.
    3. If not found, falls back to the provided internet URL.
    4. If downloaded from the internet, notifies the local mesh node to start seeding it.
    """
    print_system(f"Requesting asset: {file_description}...")
    # This is the relative path key used in the manifest (e.g., "staticmodelpool/model.gguf")
    asset_relative_path = os.path.join(
        os.path.basename(os.path.dirname(destination_path)),
        os.path.basename(destination_path)
    ).replace("\\", "/")

    peer_download_url = None
    # FIX 1: Removed 'global zephymesh_api_url' as it's redundant and causes a SyntaxError
    # with type-annotated variables. The variable is only read, not reassigned here.
    if zephymesh_process and zephymesh_api_url:
        try:
            # TODO: This is where you would query your local mesh node's API.
            # The API server needs to be implemented in mesh_node.py.
            # This is a placeholder for that future logic.
            print_system(f"Querying ZephyMesh for '{asset_relative_path}'... (API call not yet implemented)")
            # Example of what the actual request would look like:
            # import requests
            # response = requests.get(f"{zephymesh_api_url}/query", params={"filepath": asset_relative_path}, timeout=5)
            # if response.status_code == 200:
            #     data = response.json()
            #     if data.get("found"):
            #         peer_download_url = data["url"] # e.g., "http://<peer_ip>:<peer_port>/staticmodelpool/model.gguf"
            #         print_colored("SUCCESS", f"Found on peer: {peer_download_url}")
        except Exception as e:
            print_warning(f"Could not query ZephyMesh node: {e}. Falling back to internet.")

    url_to_use = peer_download_url if peer_download_url else url
    download_successful = False

    # Start of the robust download logic, now using url_to_use
    print_system(f"Preparing to download {file_description} from {url_to_use} to {destination_path}...")

    try:
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    except OSError as e:
        print_error(f"Failed to create directory for {destination_path}: {e}")
        return False

    # --- Attempt Aria2c Download First ---
    aria2c_executable_path = shutil.which("aria2c")

    if ARIA2P_AVAILABLE and aria2c_executable_path:
        print_system(f"Aria2c found at '{aria2c_executable_path}'. Attempting download via aria2p...")
        aria2_api_instance = None
        try:
            aria2_client_instance = aria2p.Client(host=aria2_rpc_host, port=aria2_rpc_port, secret=aria2_rpc_secret)
            aria2_api_instance = aria2p.API(aria2_client_instance)

            global_stats = aria2_api_instance.get_global_stat()
            print_system(
                f"Connected to aria2c RPC server. Speed: {global_stats.download_speed_string()}/{global_stats.upload_speed_string()}, Active: {global_stats.num_active}, Waiting: {global_stats.num_waiting}")

            aria2_options_dict = {
                "dir": os.path.dirname(destination_path),
                "out": os.path.basename(destination_path),
                "max-connection-per-server": "16", "split": "16", "min-split-size": "1M",
                "stream-piece-selector": "geom", "continue": "true", "allow-overwrite": "true"
            }

            print_system(f"Adding URI to aria2c: {url_to_use} with options: {aria2_options_dict}")
            download_task = aria2_api_instance.add_uris([url_to_use], options=aria2_options_dict)

            if not download_task:
                raise RuntimeError("aria2_api.add_uris did not return a download task object.")

            print_system(f"Aria2c download started (GID: {download_task.gid}). Monitoring progress...")

            from tqdm import tqdm
            with tqdm(total=100, unit='%', desc=file_description[:30], ascii=IS_WINDOWS, leave=False,
                      bar_format='{l_bar}{bar}| {n_fmt}% [{elapsed}<{remaining}, {rate_fmt}{postfix}]') as progress_bar:
                last_completed_bytes = 0
                while not download_task.is_complete and not download_task.has_error:
                    download_task.update()

                    if download_task.total_length > 0:
                        if progress_bar.total != download_task.total_length:
                            progress_bar.total = download_task.total_length
                            progress_bar.unit = 'iB'; progress_bar.unit_scale = True
                            progress_bar.n = download_task.completed_length
                            last_completed_bytes = download_task.completed_length
                        else:
                            progress_bar.update(download_task.completed_length - last_completed_bytes)
                            last_completed_bytes = download_task.completed_length

                        postfix_details = f"S:{download_task.download_speed_string()} C:{download_task.connections}"
                        if download_task.eta_string(human_readable=True):
                            postfix_details += f" ETA:{download_task.eta_string(human_readable=True)}"
                        progress_bar.set_postfix_str(postfix_details, refresh=False)
                        progress_bar.refresh()
                    else:
                        if download_task.progress > progress_bar.n:
                            progress_bar.update(int(download_task.progress) - int(progress_bar.n))
                        progress_bar.set_postfix_str(
                            f"S:{download_task.download_speed_string()} C:{download_task.connections}", refresh=True)

                    if download_task.error_code is not None:
                        raise RuntimeError(
                            f"Aria2c download error (Code {download_task.error_code}): {download_task.error_message}")
                    time.sleep(0.5)

            download_task.update()
            if download_task.is_complete:
                print_system(f"Aria2c download successful for {file_description}!")
                final_path = download_task.files[0].path
                if os.path.exists(final_path) and os.path.getsize(final_path) == download_task.total_length:
                    if final_path != destination_path:
                        print_warning(f"Aria2c saved to '{final_path}', moving to '{destination_path}'.")
                        shutil.move(final_path, destination_path)
                    download_successful = True
                else:
                    err_msg_verify = f"Aria2c reported complete, but file verification failed at '{final_path}'. Expected: {download_task.total_length}, Actual: {os.path.getsize(final_path) if os.path.exists(final_path) else 'Not Found'}"
                    print_error(err_msg_verify)
            elif download_task.has_error:
                print_error(
                    f"Aria2c download failed for {file_description}. Error (Code {download_task.error_code}): {download_task.error_message}")
                if download_task.files:
                    for file_entry in download_task.files:
                        if os.path.exists(file_entry.path):
                            try:
                                os.remove(file_entry.path)
                                print_warning(f"Removed incomplete aria2c download: {file_entry.path}")
                            except Exception as e_rm_aria:
                                print_error(f"Failed to remove incomplete aria2c file {file_entry.path}: {e_rm_aria}")
            
            if not download_successful:
                 print_warning("Aria2c download path did not complete successfully. Falling back to requests.")

        except aria2p.client.ClientException as e_aria_client:
            print_warning(f"Aria2p client/RPC connection error: {e_aria_client}. Is aria2c daemon running with --enable-rpc?")
            print_warning("Falling back to requests-based download.")
        except Exception as e_aria_general:
            print_warning(f"Unexpected error during Aria2c download attempt: {e_aria_general}")
            print_warning("Falling back to requests-based download.")
    else:
        if not ARIA2P_AVAILABLE: print_system("Aria2p Python library not available. Using requests.")
        elif not aria2c_executable_path: print_system("aria2c executable not found in PATH. Using requests.")

    # --- Requests-based Download (Fallback or if Aria2c not used/failed) ---
    if not download_successful:
        print_system("Using requests-based download method...")
        temp_destination_path_requests = destination_path + ".tmp_download_requests"
        connection_error_retries = 0
        other_errors_retries = 0
        server_file_size = 0
        head_request_successful = False

        try:
            import requests
            head_resp = requests_session.head(url_to_use, timeout=10, allow_redirects=True)
            head_resp.raise_for_status()
            server_file_size = int(head_resp.headers.get('content-length', 0))
            head_request_successful = True
            print_system(f"File size (requests HEAD): {server_file_size / (1024 * 1024):.2f} MB.")
        except requests.exceptions.RequestException as e_head_req:
            print_warning(f"Requests: Could not get file size via HEAD request: {e_head_req}.")

        while True:
            try:
                current_attempt_log = f"(ConnRetry: {connection_error_retries + 1}, OtherErrRetry: {other_errors_retries + 1})"
                print_system(f"Attempting download (requests): {file_description} {current_attempt_log}")

                response = requests_session.get(url_to_use, stream=True, timeout=(15, 300))
                response.raise_for_status()

                if not head_request_successful and server_file_size == 0:
                    server_file_size = int(response.headers.get('content-length', 0))
                    if server_file_size > 0: print_system(f"File size (requests GET): {server_file_size / (1024 * 1024):.2f} MB.")

                block_size = 8192
                from tqdm import tqdm
                with tqdm(total=server_file_size, unit='iB', unit_scale=True, desc=file_description[:30],
                          ascii=IS_WINDOWS, leave=False,
                          bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]') as progress_bar:
                    downloaded_bytes_count = 0
                    with open(temp_destination_path_requests, 'wb') as file_handle:
                        for data_chunk_item in response.iter_content(block_size):
                            progress_bar.update(len(data_chunk_item))
                            file_handle.write(data_chunk_item)
                            downloaded_bytes_count += len(data_chunk_item)

                if server_file_size != 0 and downloaded_bytes_count != server_file_size:
                    print_error(f"Requests Download ERROR: Size mismatch for {file_description}.")
                    print_error(f"  Expected {server_file_size} bytes, downloaded {downloaded_bytes_count} bytes.")
                    if os.path.exists(temp_destination_path_requests): os.remove(temp_destination_path_requests)
                    other_errors_retries += 1
                    if other_errors_retries >= max_retries_non_connection_error:
                        print_error(f"Max retries ({max_retries_non_connection_error}) for size mismatch. Download failed.")
                        download_successful = False
                        break # Exit while loop
                    print_warning(f"Retrying (requests) due to size mismatch (retry {other_errors_retries}/{max_retries_non_connection_error})...")
                    time.sleep(5)
                    continue

                shutil.move(temp_destination_path_requests, destination_path)
                print_system(f"Successfully downloaded (requests) {file_description}")
                download_successful = True
                break # Exit while loop

            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.ChunkedEncodingError) as e_connection:
                connection_error_retries += 1
                print_warning(f"Connection error (requests): {type(e_connection).__name__}. Retrying (attempt {connection_error_retries + 1})...")
                if os.path.exists(temp_destination_path_requests):
                    try: os.remove(temp_destination_path_requests)
                    except Exception: pass
                time.sleep(5)
            
            except requests.exceptions.HTTPError as e_http_error:
                other_errors_retries += 1
                print_error(f"HTTP error (requests): {e_http_error.response.status_code} for URL {url_to_use}")
                if os.path.exists(temp_destination_path_requests): os.remove(temp_destination_path_requests)
                if e_http_error.response.status_code in [401, 403, 404]:
                    print_error(f"Fatal HTTP error {e_http_error.response.status_code}. Not retrying.")
                    download_successful = False
                    break # Exit while loop
                if other_errors_retries >= max_retries_non_connection_error:
                    print_error(f"Max retries ({max_retries_non_connection_error}) for HTTP error. Download failed.")
                    download_successful = False
                    break # Exit while loop
                print_warning(f"Retrying (requests) due to HTTP error (retry {other_errors_retries}/{max_retries_non_connection_error})...")
                time.sleep(5)
            
            except Exception as e_general_requests:
                other_errors_retries += 1
                print_error(f"Unexpected error during requests download: {type(e_general_requests).__name__}")
                traceback.print_exc(file=sys.stderr)
                if os.path.exists(temp_destination_path_requests): os.remove(temp_destination_path_requests)
                if other_errors_retries >= max_retries_non_connection_error:
                    print_error(f"Max retries ({max_retries_non_connection_error}) for general error. Download failed.")
                    download_successful = False
                    break # Exit while loop
                print_warning(f"Retrying (requests) due to general error (retry {other_errors_retries}/{max_retries_non_connection_error})...")
                time.sleep(5)

    if download_successful and not peer_download_url:
        print_system(f"Notifying ZephyMesh node to seed new file: {asset_relative_path}")
        # TODO: Implement API call to local mesh node to trigger manifest update
        # try:
        #     import requests
        #     requests.post(f"{zephymesh_api_url}/manifest/update", timeout=2)
        # except Exception as e:
        #      print_warning(f"Could not notify ZephyMesh node to update manifest: {e}")

    return download_successful


def _ensure_gnat_compiler_from_source():
    """
    Checks for the GNAT compiler and if not found, downloads and builds GCC with Ada support from source.
    This is a simple but I doesn't understand yet, multi-stage process.
    """
    if os.path.exists(GNAT_INSTALLED_FLAG_FILE):
        print_system("GNAT compiler previously installed (flag file found).")
        return True

    if shutil.which("gnat"):
        print_system("GNAT compiler found in PATH. Creating flag file and skipping build.")
        with open(GNAT_INSTALLED_FLAG_FILE, 'w') as f:
            f.write(f"Found at {shutil.which('gnat')} on {datetime.now().isoformat()}")
        return True
        

    print_warning("GNAT Ada compiler not found. Preparing to build from GCC source...")
    print_warning("This is a simple but I doesn't understand yet, one-time setup and will take a VERY long time.")

    # --- Configuration for GCC Build (with user-specified versions) ---
    GCC_VERSION = "15.1.0"
    GCC_SOURCE_URL = f"https://ftp.gnu.org/gnu/gcc/gcc-{GCC_VERSION}/gcc-{GCC_VERSION}.tar.gz"
    GCC_TAR_FILENAME = f"gcc-{GCC_VERSION}.tar.gz"

    # Dependencies (with user-specified versions)
    GMP_VERSION = "6.3.0"
    MPC_VERSION = "1.3.1"
    MPFR_VERSION = "4.2.1"  # Using latest stable as 4.2.2 is not on the server
    GMP_URL = f"https://ftp.gnu.org/gnu/gmp/gmp-{GMP_VERSION}.tar.bz2"
    MPC_URL = f"https://ftp.gnu.org/gnu/mpc/mpc-{MPC_VERSION}.tar.gz"
    MPFR_URL = f"https://ftp.gnu.org/gnu/mpfr/mpfr-{MPFR_VERSION}.tar.gz"

    build_dir = os.path.join(ROOT_DIR, "gnat_from_gcc_build")
    source_dir = os.path.join(build_dir, "sources")
    os.makedirs(source_dir, exist_ok=True)

    # --- Step 1: Download all sources ---
    sources_to_download = {
        "GCC": (GCC_SOURCE_URL, os.path.join(source_dir, GCC_TAR_FILENAME)),
        "GMP": (GMP_URL, os.path.join(source_dir, f"gmp-{GMP_VERSION}.tar.bz2")),
        "MPC": (MPC_URL, os.path.join(source_dir, f"mpc-{MPC_VERSION}.tar.gz")),
        "MPFR": (MPFR_URL, os.path.join(source_dir, f"mpfr-{MPFR_VERSION}.tar.gz")),
    }

    try:
        import requests
        from tqdm import tqdm
        for name, (url, path) in sources_to_download.items():
            if not os.path.exists(path):
                print_system(f"Downloading {name} source from {url}...")
                with requests.get(url, stream=True) as r:
                    r.raise_for_status()
                    total_size = int(r.headers.get('content-length', 0))
                    with tqdm(total=total_size, unit='iB', unit_scale=True, desc=os.path.basename(path),
                              ascii=IS_WINDOWS) as progress_bar:
                        with open(path, 'wb') as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                progress_bar.update(len(chunk))
                                f.write(chunk)
    except Exception as e:
        print_error(f"Failed to download sources: {e}")
        return False

    # --- Step 2: Extract sources and create symlinks ---
    print_system("Extracting all source archives...")
    try:
        gcc_source_path = os.path.join(source_dir, f"gcc-{GCC_VERSION}")
        if not os.path.isdir(gcc_source_path):
            for name, (url, path) in sources_to_download.items():
                shutil.unpack_archive(path, source_dir)

        if not os.path.lexists(os.path.join(gcc_source_path, "gmp")):
            os.symlink(os.path.join(source_dir, f"gmp-{GMP_VERSION}"), os.path.join(gcc_source_path, "gmp"))
        if not os.path.lexists(os.path.join(gcc_source_path, "mpc")):
            os.symlink(os.path.join(source_dir, f"mpc-{MPC_VERSION}"), os.path.join(gcc_source_path, "mpc"))
        if not os.path.lexists(os.path.join(gcc_source_path, "mpfr")):
            os.symlink(os.path.join(source_dir, f"mpfr-{MPFR_VERSION}"), os.path.join(gcc_source_path, "mpfr"))

        print_system("Sources extracted and symlinked.")
    except Exception as e:
        print_error(f"Failed to extract or symlink sources: {e}")
        return False

    # --- Step 3: Perform the two-stage build ---
    install_prefix = TARGET_RUNTIME_ENV_PATH
    bootstrap_build_path = os.path.join(build_dir, "gcc-bootstrap")
    final_build_path = os.path.join(build_dir, "gcc-final")

    print_system("Cleaning previous build artifacts to prevent conflicts...")
    if os.path.exists(bootstrap_build_path):
        shutil.rmtree(bootstrap_build_path)
    if os.path.exists(final_build_path):
        shutil.rmtree(final_build_path)
    os.makedirs(bootstrap_build_path, exist_ok=True)
    os.makedirs(final_build_path, exist_ok=True)

    build_flags = []
    if sys.platform == "darwin":
        machine_arch = platform.machine()
        if machine_arch == 'arm64':
            machine_arch = 'aarch64'
        build_target_str = f"{machine_arch}-apple-darwin"
        print_system(f"Applying macOS build flags for target: {build_target_str}")
        build_flags = [
            f"--build={build_target_str}",
            f"--host={build_target_str}",
            f"--target={build_target_str}"
        ]

    # A. Bootstrap GCC with C only
    print_system("Stage 1: Configuring bootstrap GCC (C only)...")
    bootstrap_configure_cmd = [
                                  os.path.join(gcc_source_path, "configure"),
                                  f"--prefix={install_prefix}",
                                  "--enable-languages=c",
                                  "--disable-multilib",
                                  "--disable-shared",
                                  "--disable-nls"
                              ] + build_flags
    if not run_command(bootstrap_configure_cmd, cwd=bootstrap_build_path, name="GCC-BOOTSTRAP-CONFIGURE"):
        print_error("Bootstrap GCC configure step failed.")
        return False

    # <<< FIX: Use specific 'make' targets for bootstrap >>>
    print_system("Stage 1: Building bootstrap GCC ('make all-gcc')...")
    if not run_command(["make", "all-gcc"], cwd=bootstrap_build_path, name="GCC-BOOTSTRAP-MAKE"):
        print_error("Bootstrap GCC 'make all-gcc' step failed.")
        return False

    print_system("Stage 1: Installing bootstrap GCC ('make install-gcc')...")
    if not run_command(["make", "install-gcc"], cwd=bootstrap_build_path, name="GCC-BOOTSTRAP-INSTALL"):
        print_error("Bootstrap GCC 'make install-gcc' step failed.")
        return False

    print_system("✅ Bootstrap GCC installed successfully.")

    # B. Final GCC with Ada support
    print_system("Stage 2: Configuring final GCC (with Ada support)...")
    final_configure_cmd = [
                              os.path.join(gcc_source_path, "configure"),
                              f"--prefix={install_prefix}",
                              "--enable-languages=c,c++,ada",
                              "--disable-libada",
                              "--disable-nls",
                              "--disable-threads",
                              "--disable-multilib",
                              "--disable-shared"
                          ] + build_flags
    if not run_command(final_configure_cmd, cwd=final_build_path, name="GCC-FINAL-CONFIGURE"):
        print_error("Final GCC configure step failed.")
        return False

    # <<< FIX: Use specific 'make' targets for final build >>>
    print_system("Stage 2: Building final GCC tools. This will take a very long time...")
    if not run_command(["make", "all-gcc"], cwd=final_build_path, name="GCC-FINAL-MAKE-GCC"):
        print_error("Final GCC 'make all-gcc' step failed.")
        return False
    if not run_command(["make", "all-target-libgcc"], cwd=final_build_path, name="GCC-FINAL-MAKE-LIBGCC"):
        print_error("Final GCC 'make all-target-libgcc' step failed.")
        return False
    if not run_command(["make", "-C", "gcc", "cross-gnattools", "ada.all.cross"], cwd=final_build_path,
                       name="GCC-FINAL-MAKE-GNAT"):
        print_error("Final GCC 'make gnatools' step failed.")
        return False

    print_system("Stage 2: Installing final GNAT and GCC components...")
    if not run_command(["make", "install-strip-gcc", "install-target-libgcc"], cwd=final_build_path,
                       name="GCC-FINAL-INSTALL"):
        print_error("Final GCC 'make install' step failed.")
        return False

    # --- Finalization ---
    print_system("✅ GNAT compiler built and installed successfully into the Conda environment.")
    with open(GNAT_INSTALLED_FLAG_FILE, 'w') as f:
        f.write(f"Installed on {datetime.now().isoformat()}")

    print_system("Cleaning up GCC build files...")
    shutil.rmtree(build_dir)

    return True


def _ensure_alire_and_gnat_toolchain():
    """
    Ensures the Alire package manager and the GNAT toolchain are available.
    This version correctly detects the system architecture to download the
    proper Alire binary (e.g., aarch64 vs x86_64).
    """
    if os.path.exists(GNAT_TOOLCHAIN_INSTALLED_FLAG_FILE):
        print_system("GNAT toolchain previously installed via Alire (flag file found).")
        return True

    if shutil.which("gnat"):
        print_system("GNAT compiler found in PATH. Assuming Alire setup is complete.")
        with open(GNAT_TOOLCHAIN_INSTALLED_FLAG_FILE, 'w') as f:
            f.write(f"Found at {shutil.which('gnat')} on {datetime.now().isoformat()}")
        return True

    alire_install_dir = os.path.join(ROOT_DIR, "alire_installer")

    if not shutil.which("alr"):
        print_warning("Alire 'alr' command not found. Downloading pre-compiled binary...")

        # --- Determine Correct Alire Binary for OS/Arch ---
        os_name = platform.system().lower()
        machine_arch_raw = platform.machine().lower()
        machine_arch = "" # Reset machine_arch to ensure it's set correctly below

        if os_name == "darwin":
            os_name = "macos"
            machine_arch = "aarch64" if "arm64" in machine_arch_raw else "x86_64"
        elif os_name == "linux":
            # THIS IS THE FIX: Dynamically determine the architecture instead of hardcoding.
            machine_arch = "aarch64" if "aarch64" in machine_arch_raw else "x86_64"
        else:
            print_error(f"Automated Alire download is not supported for OS: {os_name}")
            return False

        alire_release_tag = "v2.1.0"
        alire_version_str = "2.1.0"
        alire_filename = f"alr-{alire_version_str}-bin-{machine_arch}-{os_name}.zip"
        alire_url = f"https://github.com/alire-project/alire/releases/download/{alire_release_tag}/{alire_filename}"

        if os.path.exists(alire_install_dir):
            shutil.rmtree(alire_install_dir)
        os.makedirs(alire_install_dir)

        zip_path = os.path.join(alire_install_dir, alire_filename)

        try:
            import requests
            from tqdm import tqdm
            print_system(f"Downloading Alire for {machine_arch} from: {alire_url}")
            with requests.get(alire_url, stream=True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                with tqdm(total=total_size, unit='iB', unit_scale=True, desc=alire_filename, ascii=IS_WINDOWS) as pbar:
                    with open(zip_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            pbar.update(len(chunk))
                            f.write(chunk)
            print_system("Alire downloaded successfully.")
        except Exception as e:
            print_error(f"Failed to download Alire binary: {e}")
            return False

        # --- Extract and Install Alire ---
        print_system(f"Extracting {alire_filename}...")
        try:
            shutil.unpack_archive(zip_path, alire_install_dir)

            # The path to the 'alr' executable inside the extracted archive
            alr_binary_path = os.path.join(alire_install_dir, "bin", "alr")
            # The destination path inside our Conda environment's bin directory
            conda_bin_path = os.path.join(TARGET_RUNTIME_ENV_PATH, "bin")

            if not os.path.exists(alr_binary_path):
                raise FileNotFoundError(f"'alr' binary not found at expected path: {alr_binary_path}")

            print_system(f"Installing 'alr' executable to {conda_bin_path}...")
            os.makedirs(conda_bin_path, exist_ok=True)
            shutil.copy(alr_binary_path, conda_bin_path)
            # Ensure the copied binary is executable
            os.chmod(os.path.join(conda_bin_path, "alr"), 0o755)

            # Clean up the temporary download directory
            shutil.rmtree(alire_install_dir)
            print_system("✅ Alire 'alr' executable installed successfully.")
        except Exception as e:
            print_error(f"Failed to install Alire: {e}")
            return False

    # 3. Use Alire to get the GNAT toolchain
    print_system("Using Alire to install the GNAT toolchain and gprbuild for Ada Compiler. This may take a moment...")
    # Now that 'alr' is in the Conda env's bin, it should be found in the PATH.
    alr_get_gnat_command = ["alr", "toolchain", "--select", "gnat_native", "gprbuild"]

    if not run_command(alr_get_gnat_command, cwd=ROOT_DIR, name="ALIRE-GET-GNAT"):
        print_error("Alire failed to install the GNAT toolchain.")
        return False

    print_system("✅ GNAT toolchain successfully installed via Alire.")
    with open(GNAT_TOOLCHAIN_INSTALLED_FLAG_FILE, 'w') as f:
        f.write(f"Installed via Alire on {datetime.now().isoformat()}")

    return True



def _check_and_repair_conda_env(env_path: str) -> bool:
    """
    Runs `conda doctor` on the specified environment and triggers a repair if needed.
    The "repair" consists of deleting the corrupt environment so it can be recreated.

    Returns:
        True if the environment is healthy, False if it was corrupt and deleted.
    """
    print_system(f"--- Running Conda Environment Health Check on {os.path.basename(env_path)} ---")

    if not os.path.isdir(env_path):
        print_system("Health check skipped: Conda environment does not exist yet.")
        return True # It's not "unhealthy", it just needs to be created.

    if not CONDA_EXECUTABLE:
        print_error("Cannot run `conda doctor`: Conda executable path is not set.")
        return False # Cannot verify, assume failure to be safe

    try:
        doctor_command = [CONDA_EXECUTABLE, "doctor", "-v", "-p", env_path]
        print_system(f"Executing: {' '.join(doctor_command)}")
        process = subprocess.run(
            doctor_command,
            capture_output=True,
            text=True,
            check=False, # We need to check output manually, not just return code
            timeout=3600
        )

        output = process.stdout + "\n" + process.stderr
        
        # Check for specific failure signatures from the conda doctor output.
        # "Missing Files" is a critical error that requires a rebuild.
        # "Altered Files" is often benign (e.g., new .pyc files), so we ignore it.
        if "❌ Missing Files:" in output:
            print_error(f"Conda environment at '{env_path}' is CORRUPT (Missing Files detected).")
            print_warning("Triggering autofix: The environment will be completely removed and recreated.")
            print_error(f"--- Conda Doctor Report ---\n{output.strip()}\n--------------------------")
            
            # --- The "Autofix" ---
            # 1. Remove associated flag files to ensure fresh installations.
            _remove_flag_files(FLAG_FILES_TO_RESET_ON_ENV_RECREATE)
            
            # 2. Delete the entire corrupt Conda environment directory.
            try:
                shutil.rmtree(env_path)
                print_system(f"Successfully removed corrupt environment: {env_path}")
            except Exception as e:
                print_error(f"CRITICAL: Failed to remove corrupt Conda environment at '{env_path}': {e}")
                print_error("Please remove this directory manually and restart the launcher.")
                sys.exit(1)
            
            return False # Signal that the environment was corrupt and has been removed.

        elif "❌" in output: # Catch other potential doctor errors
             print_warning(f"Conda doctor reported issues, but not 'Missing Files'. Report:\n{output.strip()}")
             print_system("Proceeding cautiously, but a manual environment recreation might be needed if errors persist.")
             return True # Treat as healthy for now, but warn the user.

        else:
            print_system("✅ Conda environment health check passed.")
            return True # Environment is healthy.

    except subprocess.TimeoutExpired:
        print_error("`conda doctor` command timed out. Assuming environment is unhealthy.")
        # You could trigger the autofix here as well if a timeout is considered critical.
        return False
    except Exception as e:
        print_error(f"An unexpected error occurred during `conda doctor` check: {e}")
        return False


def _ensure_and_launch_zephymesh():
    """
    Ensures the ZephyMesh node is compiled and launches it as a managed service.
    It then waits for the node to become ready by checking for its port file.
    This function is synchronous and should be called from the main thread.
    """
    global zephymesh_api_url  # We still need to set the global URL

    print_system("--- Preparing ZephyMesh P2P Node ---")

    # Define the binary name based on OS
    mesh_exe_name = "zephymesh_node_compiled.exe" if IS_WINDOWS else "zephymesh_node_compiled"
    mesh_exe_path = os.path.join(ZEPHYMESH_DIR, mesh_exe_name)

    try:
        # --- Step 1: Compile the binary if it doesn't exist ---
        if not os.path.exists(mesh_exe_path):
            print_warning(f"ZephyMesh node binary not found. Compiling from source...")

            go_source_dir = ZEPHYMESH_DIR
            go_mod_file = os.path.join(go_source_dir, "go.mod")

            # Ensure go.mod and dependencies are handled
            if not os.path.exists(go_mod_file):
                print_warning(f"go.mod not found in {go_source_dir}. Initializing module...")
                if not run_command(["go", "mod", "init", "zephymesh"], cwd=go_source_dir, name="GO-MOD-INIT-MESH"):
                    print_error("Failed to initialize Go module. ZephyMesh cannot start.")
                    return False  # Return failure

            print_system("Ensuring Go dependencies for mesh node ('go mod tidy')...")
            if not run_command(["go", "mod", "tidy"], cwd=go_source_dir, name="GO-MOD-TIDY-MESH"):
                print_error("Failed to download Go dependencies for mesh node. ZephyMesh cannot start.")
                return False  # Return failure

            print_system("Compiling the Go mesh node...")
            go_package_path = "./cmd/zephymesh_node"
            compile_command = ["go", "build", "-buildvcs=false", "-o", mesh_exe_path, go_package_path]

            if not run_command(compile_command, cwd=go_source_dir, name="GO-COMPILE-MESHNODE"):
                print_error("Failed to compile the Go ZephyMesh node. ZephyMesh cannot start.")
                return False  # Return failure
            print_system("✅ Go ZephyMesh node compiled successfully.")

        # --- Step 2: Clean up old state files and launch the process ---
        if os.path.exists(ZEPHYMESH_PORT_INFO_FILE):
            os.remove(ZEPHYMESH_PORT_INFO_FILE)

        # Use the standard service starter. This is the crucial change for cleanup.
        start_service_process([mesh_exe_path], ROOT_DIR, "ZEPHYMESH-NODE")

        # --- Step 3: Wait for the node to be ready ---
        print_system(f"Waiting for ZephyMesh port file: {ZEPHYMESH_PORT_INFO_FILE}")
        port_file_wait_start = time.monotonic()
        api_ready = False
        while time.monotonic() - port_file_wait_start < 60:
            # Check if the process died prematurely
            # We need to find our process in the global list now
            mesh_proc_obj = None
            with process_lock:
                for proc, name, _, _ in running_processes:
                    if name == "ZEPHYMESH-NODE":
                        mesh_proc_obj = proc
                        break

            if mesh_proc_obj and mesh_proc_obj.poll() is not None:
                raise RuntimeError("ZephyMesh node process exited prematurely while waiting for port file.")

            if os.path.exists(ZEPHYMESH_PORT_INFO_FILE):
                api_ready = True
                break
            time.sleep(0.5)

        if not api_ready:
            raise TimeoutError("ZephyMesh node did not create its port file in time.")

        # --- Step 4: Read the port and set the global API URL ---
        with open(ZEPHYMESH_PORT_INFO_FILE, 'r') as f:
            port_info = json.load(f)
            api_port = port_info['api_port']
            zephymesh_api_url = f"http://127.0.0.1:{api_port}"
            print_colored("SUCCESS", f"✅ ZephyMesh Node is ready. API URL discovered: {zephymesh_api_url}", "SUCCESS")

        return True  # Indicate success

    except Exception as e:
        print_error(f"Failed to start ZephyMesh Node: {e}. P2P distribution will be unavailable.")
        # No need to terminate here, the main atexit handler will get it.
        return False  # Indicate failure

def _miniforge_alike_checking_config(CONDA_EXEC):
    print_system("--- Configuring newly installed Conda")

    # This is the path to the conda executable we just installed


    command_zst = [CONDA_EXEC, "config", "--set", "experimental", "repodata_from_zst"]

    command_priority = [CONDA_EXEC, "config", "--set", "channel_priority", "flexible"]
    command_bypass_ssl_test = [CONDA_EXEC, "config", "--set", "ssl_verify", "no"]
    #conda config --set ssl_verify no
    max_config_attempts = 3
    config_success = False

    for attempt in range(max_config_attempts):
        print_system(f"Attempt {attempt + 1}/{max_config_attempts} to apply Conda configuration...")
        try:
            # We run these as direct subprocess calls because our 'run_command' helper
            # is designed for the main Conda environment, not this brand new one.
            # We use check=True to raise an error on failure.
            #subprocess.run(command_zst, check=True, capture_output=True, text=True, timeout=60)
            #print_system("  -> Successfully enabled 'repodata_from_zst'.")

            subprocess.run(command_bypass_ssl_test, check=True, capture_output=True, text=True, timeout=60)
            print_system("  -> DANGER TEST: Successfully enabled command_bypass_ssl_test.") #HTTP 000 Error bypass on third-party conda miniforge  https://stackoverflow.com/questions/50305725/condahttperror-http-000-connection-failed-for-url-https-repo-continuum-io-pk

            #subprocess.run(command_priority, check=True, capture_output=True, text=True, timeout=60)
            #print_system("  -> Successfully set 'channel_priority' to 'flexible'.")

            config_success = True
            print_colored("SUCCESS", "portable miniforge successfully configured for unsupported repo.", "SUCCESS")
            return True
            break  # Exit the loop on success

        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print_warning(f"Conda config command failed on attempt {attempt + 1}: {e}")
            if hasattr(e, 'stderr'):
                print_warning(f"STDERR: {e.stderr.strip()}")
            if attempt < max_config_attempts - 1:
                print_warning("Retrying in 5 seconds...")
                time.sleep(5)
        except Exception as e_unexpected:
            print_error(f"An unexpected error occurred during Conda config: {e_unexpected}")
            return False
            break  # Do not retry on unexpected errors

    if not config_success:
        print_error("Failed to apply Conda performance configurations after multiple attempts.")
        print_warning("The launcher will proceed, but you may encounter metadata download failures.")

def _check_playwright_linux_deps():
    """
    Checks for Playwright's browser dependencies on Linux BEFORE the main relaunch.
    If dependencies are missing, it prints a user-friendly message with the
    required `sudo` command and exits the script.
    """
    # This check is only necessary and relevant for Linux that is not proot.
    if platform.system() != "Linux" or IS_IN_PROOT_ENV:
        return True

    print_system("--- Performing pre-flight check for Playwright's Linux dependencies ---")

    # We need a temporary, minimal Conda environment to run the check command.
    # This avoids polluting the base system or requiring Playwright to be installed globally.
    check_env_path = os.path.join(ROOT_DIR, "playwright_dep_check_env")

    # Clean up any previous check environment to ensure a fresh state.
    if os.path.exists(check_env_path):
        try:
            shutil.rmtree(check_env_path)
        except OSError as e:
            print_warning(f"Could not remove old Playwright check env: {e}")

    try:
        # Create a small, temporary environment with just playwright in it.
        # This is much faster than checking inside our main, large environment.
        print_system("Creating temporary environment to check dependencies...")
        create_cmd = [
            CONDA_EXECUTABLE, "create", "--yes", "--prefix", check_env_path, "playwright", "-c", "conda-forge"
        ]
        subprocess.run(create_cmd, check=True, capture_output=True, text=True, timeout=300)

        # The command to check dependencies. It will return a non-zero exit code
        # and print the necessary `apt-get install ...` command if deps are missing.
        # Dynamically find the python executable in the temporary environment
        temp_env_python_exe = None
        bin_dir = os.path.join(check_env_path, "bin")

        # Prioritize python3, as it's more explicit
        if os.path.exists(os.path.join(bin_dir, "python3")):
            temp_env_python_exe = os.path.join(bin_dir, "python3")
        # Fallback to python if python3 doesn't exist
        elif os.path.exists(os.path.join(bin_dir, "python")):
            temp_env_python_exe = os.path.join(bin_dir, "python")
        else:
            # If neither is found, we can't proceed.
            print_error(f"CRITICAL: Could not find 'python' or 'python3' executable in the temporary check environment at '{bin_dir}'.")
            # The finally block will still handle cleanup.
            return False

        print_system(f"Found temporary python executable: {temp_env_python_exe}")

        check_deps_cmd = [
            temp_env_python_exe,
            "-m", "playwright", "install-deps"
        ]

        print_system("Running Playwright's dependency check...")
        # We expect this might fail, so we don't use check=True.
        result = subprocess.run(check_deps_cmd, capture_output=True, text=True, timeout=120)

        # If the command succeeded (exit code 0), all dependencies are present.
        if result.returncode == 0:
            print_system("✅ Playwright Linux dependencies are satisfied.")
            return True

        # If it failed, it means dependencies are missing. We must extract the command.
        print_colored("ERROR", "--- ACTION REQUIRED: Missing System Libraries for Playwright ---", "ERROR")
        print_error("The web browser automation engine (Playwright) needs system libraries that are not installed.")
        print_system("To fix this, please run the following command with sudo privileges:")

        # Search the command's output for the line starting with "sudo"
        sudo_command_to_run = ""
        output_lines = (result.stdout + result.stderr).splitlines()
        for line in output_lines:
            if line.strip().startswith("sudo"):
                sudo_command_to_run = line.strip()
                break

        if sudo_command_to_run:
            # Print the extracted command in a very visible way.
            print_colored("COMMAND", f"\n    {sudo_command_to_run}\n", "SUCCESS")
        else:
            # Fallback message if we couldn't parse the command.
            print_error("Could not automatically determine the exact command. Please check the logs below.")
            print_warning("You may need to run 'npx playwright install-deps' manually to see the required command.")
            print_warning(f"--- Full output from dependency check ---\n{result.stdout}\n{result.stderr}\n---")

        print_system("After running the command, please restart this launcher script.")
        print_colored("ERROR", "----------------------------------------------------------------", "ERROR")

        # This is a fatal error for this run. We exit so the user can take action.
        return False

    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print_error(f"Failed to run the Playwright dependency check: {e}")
        if hasattr(e, 'stdout'):
            print_error(f"STDOUT: {e.stdout}")
        if hasattr(e, 'stderr'):
            print_error(f"STDERR: {e.stderr}")
        return False
    finally:
        # Crucially, clean up the temporary environment afterwards.
        if os.path.exists(check_env_path):
            print_system("Cleaning up temporary dependency-check environment...")
            shutil.rmtree(check_env_path)

def _ensure_playwright_browsers():
    """Checks if Playwright browsers are installed and installs them if not, using a flag file."""
    # First, check if the flag file exists. If so, we can skip the whole process.
    if os.path.exists(PLAYWRIGHT_BROWSERS_INSTALLED_FLAG_FILE):
        print_system("Playwright browsers previously installed (flag file found).")
        return True

    # If the flag doesn't exist, proceed with the installation.
    # The message is now more accurate, reflecting a one-time setup action.
    print_system("--- Installing/Verifying Playwright browser binaries for the first time ---")
    try:
        playwright_install_command = [
            PYTHON_EXECUTABLE, "-m", "playwright", "install", "--with-deps"
        ]

        # Use run_command for its nice logging and error handling.
        if not run_command(playwright_install_command, ROOT_DIR, "PLAYWRIGHT-INSTALL-BROWSERS"):
            print_error("Failed to install Playwright browsers. Web scraping features may be disabled.")
            return False

        # If the command succeeds, create the flag file to prevent this from running again.
        print_system("✅ Playwright browsers are installed and up-to-date.")
        with open(PLAYWRIGHT_BROWSERS_INSTALLED_FLAG_FILE, 'w', encoding='utf-8') as f_flag:
            f_flag.write(f"Installed on: {datetime.now().isoformat()}\n")
        print_system("Playwright browsers installation flag created.")
        return True

    except Exception as e:
        print_error(f"An error occurred during Playwright browser setup: {e}")
        return False

def _perform_pre_attempt_cleanup(attempt_number: int):
    """Performs cleanup actions before a setup retry."""
    if attempt_number == 1:
        # No cleanup on the first attempt
        return

    print_warning(f"Setup attempt {attempt_number-1} failed. Performing pre-retry cleanup for attempt {attempt_number}.")

    # Attempt 2 (first failure): Remove all install flags except license.
    if attempt_number == 2:
        print_system("Cleanup Action: Removing installation flag files...")
        # Create a list of flags to remove, excluding the license file
        flags_to_clear = [f for f in FLAG_FILES_TO_RESET_ON_ENV_RECREATE if f != LICENSE_FLAG_FILE]
        _remove_flag_files(flags_to_clear)

    # Attempts 3 and 4 (second and third failures): Remove flags AND the Conda environment.
    elif attempt_number >= 3:
        print_system("Cleanup Action: Removing installation flag files AND the Conda environment.")
        # Remove all flags (including the ones for the env itself)
        _remove_flag_files(FLAG_FILES_TO_RESET_ON_ENV_RECREATE)

        if os.path.isdir(TARGET_RUNTIME_ENV_PATH):
            print_warning(f"Removing Conda environment directory: {TARGET_RUNTIME_ENV_PATH}")
            try:
                shutil.rmtree(TARGET_RUNTIME_ENV_PATH)
                print_system("Conda environment directory removed successfully.")
            except Exception as e:
                print_error(f"CRITICAL ERROR during cleanup: Could not remove Conda env '{TARGET_RUNTIME_ENV_PATH}': {e}")
                print_error("Manual intervention is required. Please delete the directory and restart.")
                sys.exit(1) # This is a cleanup failure, which is critical.
    time.sleep(2) # Give a moment for filesystem to catch up

def start_service_thread(target_func, name):
    print_system(f"Preparing to start service: {name}")
    thread = threading.Thread(target=target_func, name=name, daemon=True)
    thread.start()
    return thread



port_shield_stop_event: Optional[threading.Event] = None


def _port_shield_worker_daemon(port: int, target_process_name: str, stop_event: threading.Event):
    """
    The daemon worker. It intelligently monitors a port.
    - When the port is free, it checks with low frequency.
    - When ANY process binds to the port, it switches to a high-aggression loop.
    - If the binding process is the target, it is brutally terminated.
    """
    try:
        import psutil
    except ImportError:
        print_error("[PORT SHIELD] CRITICAL: psutil library not found. The shield cannot operate.")
        return

    print_colored("ERROR",
                  f"[PORT SHIELD] ACTIVATED. Intelligently monitoring port {port} for illegal binding by '{target_process_name}'.",
                  "ERROR")

    while not stop_event.is_set():
        process_on_port = None
        try:
            # Check for any process listening on the target port
            for conn in psutil.net_connections(kind='inet'):
                if conn.laddr.port == port and conn.status == psutil.CONN_LISTEN and conn.pid is not None:
                    process_on_port = conn
                    break  # Found a process, exit the loop to handle it.
        except psutil.Error as e:
            # This can happen if psutil has issues scanning. Pause briefly to avoid spam.
            print_warning(f"[PORT SHIELD] psutil scan failed: {e}. Pausing.")
            time.sleep(0.5)
            continue

        # --- This is the core logic change ---
        if process_on_port:
            # A process is on the port. ENTER HIGH-AGGRESSION MODE. No sleep.
            try:
                p = psutil.Process(process_on_port.pid)

                # Check if the process is our unwanted target.
                if target_process_name in p.name().lower():
                    # Threat confirmed. Annihilate.
                    print_colored("ERROR",
                                  f"[PORT SHIELD] Hostile target '{p.name()}' (PID: {p.pid}) has illegally bound to port {port}. ANNIHILATING.",
                                  "ERROR")

                    for child in p.children(recursive=True):
                        try:
                            child.kill()  # SIGKILL
                        except psutil.NoSuchProcess:
                            pass

                    p.kill()  # SIGKILL

                    # Minimal sleep ONLY after a kill to allow the OS to reap the process.
                    time.sleep(0.01)

                # If it's NOT the target (e.g., our own hypercorn), this block is skipped,
                # and the while loop immediately re-runs, maintaining high-frequency checks.

            except psutil.NoSuchProcess:
                # THIS IS THE FIX. The process on the port disappeared between checks.
                # This is normal. We do nothing and let the loop run again immediately.
                pass
            except (psutil.AccessDenied, psutil.ZombieProcess) as e:
                print_warning(f"[PORT SHIELD] Cannot access process on port {port} (PID: {process_on_port.pid}): {e}")
            except Exception as e_inner:
                # Catch any other unexpected error during the process handling.
                print_error(f"[PORT SHIELD] Inner loop error for PID {process_on_port.pid}: {e_inner}")

        else:
            # Port is free. ENTER LOW-IMPACT MODE.
            # This honors the "don't run the loop" request by relaxing frequency.
            # It dramatically reduces CPU usage when idle but remains vigilant.
            pass

def _start_port_shield_daemon(port: int, target_process_name: str):
    """
    Creates and launches the asynchronous, non-blocking Port Shield daemon thread.
    This function returns immediately.
    """
    global port_shield_stop_event

    if port_shield_stop_event is not None:
        print_warning("[PORT SHIELD] Shield daemon already started.")
        return

    print_system(f"--- Deploying Asynchronous Port Shield ---")

    stop_event = threading.Event()
    port_shield_stop_event = stop_event

    shield_thread = threading.Thread(
        target=_port_shield_worker_daemon,
        args=(port, target_process_name.lower(), stop_event),
        name=f"PortShieldDaemon-{port}",
        daemon=True  # CRITICAL: This ensures the thread dies if the main script crashes.
    )
    shield_thread.start()

# --- End Helper Logic ---
# --- Textual TUI Application (Self-Contained) ---

if TUI_AVAILABLE:
    class SystemStats(Static):
        """A widget to display real-time system stats."""
        cpu_usage = reactive(0.0)
        ram_usage = reactive(0.0)
        disk_free = reactive(0.0)

        def render(self) -> str:
            return (f"💻 CPU: {self.cpu_usage:>5.1f}% | "
                    f"🧠 RAM: {self.ram_usage:>5.1f}% | "
                    f"💾 Disk Free: {self.disk_free:.1f} GB")
        
    

    class ZephyrineTUI(App):
        """A Textual UI for monitoring the Zephyrine Engine and its daemons."""
        log_tail_threads = [] #initializing tail array Before we use it for parent child Processing tracking

        CSS = """
                Grid {
                    /* Keep 2 columns, but now define row heights explicitly */
                    grid-size: 2; 
                    grid-rows: 2fr 1fr 1; /* Top: 2 parts, Middle: 1 part, Bottom: 1 fixed line */
                    grid-gutter: 1;
                }
                #engine {
                    column-span: 2;
                    /* This widget now sits in the first row, which has a height of 2fr */
                }
                #watchdog1 {
                    /* This widget sits in the second row, which has a height of 1fr */
                }
                #watchdog2 {
                    /* This widget also sits in the second row, height 1fr */
                }
                #stats {
                    column-span: 2;
                    /* This widget sits in the third row, which has a fixed height of 1 line */
                    height: 1;
                }
                Log {
                    border-title-align: center;
                }
                """

        def compose(self) -> ComposeResult:
            """Create child widgets for the TUI."""
            yield Header(show_clock=True)

            # The Grid will now contain ALL the main panels
            with Grid():
                yield Log(id="engine", max_lines=386) # Increased max_lines for the bigger pane
                yield Log(id="watchdog1", max_lines=256)
                yield Log(id="watchdog2", max_lines=256)
                yield SystemStatsGraph(id="stats") # SystemStatsGraph is now the new name for the stats widget

            yield Footer()

        def on_mount(self) -> None:
            """Called when the app is mounted."""

            # Set the border titles correctly after the widgets are created
            self.query_one("#engine", Log).border_title = "🚀 Main Engine & Services"
            self.query_one("#watchdog1", Log).border_title = "🛡️ Watchdog (Go)"
            self.query_one("#watchdog2", Log).border_title = "🛡️ Watchdog (Ada)"


            # Start system stats updates
            self.update_stats()
            self.set_interval(2, self.update_stats)

            # Start all services in background threads
            self.start_all_services_in_background()
        

        def update_stats(self) -> None:
            """Update the system stats widget with percentages for the bars."""
            # Target the new widget class
            stats_widget = self.query_one(SystemStatsGraph)
            
            # Update CPU and RAM percentages directly
            stats_widget.cpu_percent = psutil.cpu_percent()
            stats_widget.ram_percent = psutil.virtual_memory().percent
            
            # Get disk usage info
            disk_usage = psutil.disk_usage('/')
            stats_widget.disk_percent = disk_usage.percent
            stats_widget.disk_free_gb = disk_usage.free / (1024 ** 3)

        def start_all_services_in_background(self):
            """
            Starts the main application services and tails the log files for ALL
            running processes (including those started before the TUI).
            """
            logger.info("TUI: Starting main application services and initiating log tailing...")

            # The watchdogs and mesh node are already running.
            # We only need to start the remaining services here.
            
            if not os.path.exists(SETUP_COMPLETE_FLAG_FILE):
                start_engine_main()
                start_backend_service()
                start_frontend()
            # This is where the slow mode starts the services because in one go (look at the main slow logic how it rely on this function to launch the services). but on fast mode it done using async service launch itself (check on the main threads of the async io services)

            # --- Wait a moment for the new log files to be created ---
            time.sleep(1.0)

            # This logic remains the same, but it will now also pick up
            # the log files from the pre-started watchdogs.
            log_dir = os.path.join(ROOT_DIR, "logs")
            widget_service_map = {
                "engine": ["ENGINE", "BACKEND", "FRONTEND"],
                "watchdog1": ["GO-WATCHDOG"],
                "watchdog2": ["ADA-WATCHDOG"],
            }

            tailed_files = set()

            for widget_id, service_names in widget_service_map.items():
                widget = self.query_one(f"#{widget_id}", Log)
                for service_name in service_names:
                    # NOTE: This part needs a small fix. The mesh node also creates a log.
                    # We should decide if we want to display it. Let's add it to the 'engine' log.
                    log_file = os.path.join(log_dir, f"{service_name.replace(' ', '_')}.log")

                    if log_file not in tailed_files:
                        logger.info(f"TUI: Starting tail for '{log_file}' -> '#{widget_id}'")

                        tail_thread = threading.Thread(
                            target=tail_log_to_widget,
                            args=(log_file, widget),
                            name=f"TuiTail-{service_name}",
                            daemon=True
                        )
                        tail_thread.start()

                        self.log_tail_threads.append(tail_thread)
                        tailed_files.add(log_file)

        from textual.binding import Binding

        BINDINGS = [
            Binding(key="ctrl+c", action="quit", description="Quit App", show=True),
            Binding(key="ctrl+q", action="quit", description="Quit App", show=True),
        ]

        def action_quit(self) -> None:
            """An action to quit the application gracefully."""
            # You can log this to the TUI itself for user feedback
            self.query_one("#engine", Log).write("\n[SYSTEM] Shutdown requested. Terminating services...")
            # self.exit() will trigger the on_unmount event and then quit

            tui_shutdown_event.set() #making sure that the quit TUI event is stated
            self.exit()

        def on_unmount(self) -> None:
            """Perform cleanup when the TUI is unmounted (quitting)."""
            # This is the new, primary cleanup location for TUI mode.
            # We call the same logic as the atexit handler.
            logger.info("TUI is unmounting. Initiating process cleanup...")
            cleanup_processes()
            logger.info("Waiting for log tailing threads to exit...")
            for thread in self.log_tail_threads:
                # We give each thread a moment to finish its loop and exit.
                thread.join(timeout=2.0)

    from textual.widgets import Static
    from textual.reactive import reactive
    from rich.table import Table
    from rich.progress_bar import ProgressBar

    class SystemStatsGraph(Static):
        """A widget to display real-time system stats with graphical bars."""
        
        # Define reactive variables to hold the data
        cpu_percent = reactive(0.0)
        ram_percent = reactive(0.0)
        disk_percent = reactive(0.0)
        disk_free_gb = reactive(0.0)

        def render(self) -> Table:
            """Render the stats into a Rich Table with progress bars."""
            # A Table is a perfect layout tool for this
            stats_table = Table.grid(expand=True, padding=(0, 1))
            stats_table.add_column("label", width=5)
            stats_table.add_column("bar") # The bar column will expand
            stats_table.add_column("value", width=8, justify="right")

            # Create ProgressBar instances for each stat
            # `width=None` allows the bar to fill the available space in its column
            cpu_bar = ProgressBar(total=100, completed=self.cpu_percent, width=None, style="bright_green", complete_style="green")
            ram_bar = ProgressBar(total=100, completed=self.ram_percent, width=None, style="bright_cyan", complete_style="cyan")
            disk_bar = ProgressBar(total=100, completed=self.disk_percent, width=None, style="bright_magenta", complete_style="magenta")

            # Add a row for each statistic
            stats_table.add_row("CPU", cpu_bar, f"{self.cpu_percent:>5.1f}%")
            stats_table.add_row("RAM", ram_bar, f"{self.ram_percent:>5.1f}%")
            stats_table.add_row("Disk", disk_bar, f"{self.disk_free_gb:>5.1f}G")

            return stats_table
    # --- End of new widget ---

# --- End of TUI Application Block ---

# --- Main Execution Logic ---
# Assuming all your helper functions (print_system, run_command, _ensure_conda_package, etc.)
# and global configurations (ROOT_DIR, REQUIRED_NODE_MAJOR, etc.) are defined correctly above this block.

if __name__ == "__main__":
    for attempt in range(1, MAX_SETUP_ATTEMPTS + 1):
        print_colored("SYSTEM", f"--- Starting Setup Attempt {attempt}/{MAX_SETUP_ATTEMPTS} ---", "SUCCESS")
        setup_failures = [] # Reset failures for this attempt
        print_system("--- Project Zephyrine Launcher ---")
        print_system(f"Root directory: {ROOT_DIR}")
        print_system(f"Target Conda environment path: {TARGET_RUNTIME_ENV_PATH}")
        pythonruntimeversion = f"{sys.version.split()[0]}"
        print_system(f"Initial Python Version check on this runtime execution: {pythonruntimeversion} on {platform.system()} ({platform.machine()})")

        current_conda_env_path_check = os.getenv("CONDA_PREFIX")
        is_already_in_correct_env = False
        if current_conda_env_path_check:
            try:
                norm_current_env_path = os.path.normcase(os.path.realpath(current_conda_env_path_check))
                norm_target_env_path = os.path.normcase(os.path.realpath(TARGET_RUNTIME_ENV_PATH))
                if os.path.isdir(norm_current_env_path) and \
                        os.path.isdir(norm_target_env_path) and \
                        norm_current_env_path == norm_target_env_path and \
                        os.getenv("ZEPHYRINE_RELAUNCHED_IN_CONDA") == "1":  # Check the relaunch flag
                    is_already_in_correct_env = True
                    ACTIVE_ENV_PATH = current_conda_env_path_check
            except Exception as e_path_check:
                print_warning(f"Error comparing Conda paths during initial check: {e_path_check}")

        IS_IN_PROOT_ENV = os.path.exists("/etc/debian_version") and "TERMUX_VERSION" not in os.environ
        if IS_IN_PROOT_ENV:
            print_aetherhand("✅ Detected execution within containerized bootstrap (glibc) environment.")

        if is_already_in_correct_env:
            # --- This is the RELAUNCHED script, running inside the correct Conda environment ---
            print_system(f"Running inside target Conda environment (Prefix: {ACTIVE_ENV_PATH})")
            #-=-=-=-=-=[flickerPhoton]
            # This is the crucial fix: Use the trusted 'conda' executable to reinstall
            # a compatible set of packaging tools for the Python 3.12 environment.



            if os.path.exists(SETUP_COMPLETE_FLAG_FILE):
                # --- FAST PATH ---
                print_colored("SUCCESS", "--- Fast Path Launch Detected: Skipping all checks ---", "SUCCESS")

                # Set up minimal required paths for launch, as these are not set on the fast path.
                _sys_exec_dir = os.path.dirname(sys.executable)
                PYTHON_EXECUTABLE = sys.executable
                PIP_EXECUTABLE = os.path.join(_sys_exec_dir, "pip.exe" if IS_WINDOWS else "pip")
                HYPERCORN_EXECUTABLE = os.path.join(_sys_exec_dir, "hypercorn.exe" if IS_WINDOWS else "hypercorn")

                TUI_LIBRARIES_AVAILABLE = False
                try:
                    import textual
                    import psutil

                    TUI_LIBRARIES_AVAILABLE = True
                except ImportError:
                    TUI_LIBRARIES_AVAILABLE = False

                # Now, call the parallel launcher
                launch_all_services_in_parallel_and_monitor()

                # Crucially, we must exit the script here to prevent the slow path from running.
                # We also break the loop in case of any unforseen structure.
                # The cleanup will be handled by the atexit handler.
                break
            else:
                #Slow path installation mode
                print_warning("--- First-Time Verification Run: Performing all checks... ---")
                print_system(f"Updated PIP_EXECUTABLE to: {PIP_EXECUTABLE}")
            print_system(f"Updated HYPERCORN_EXECUTABLE to: {HYPERCORN_EXECUTABLE}")

            # --- START: Set Hugging Face Cache Directory (NEW) ---
            print_system("--- Configuring Hugging Face Cache Directory ---")
            try:
                # Use a subdirectory within the static model pool to keep all downloads together
                hf_cache_dir = os.path.join(STATIC_MODEL_POOL_PATH, 'huggingface_cache')
                os.makedirs(hf_cache_dir, exist_ok=True)
                os.environ['HF_HOME'] = hf_cache_dir
                print_system(f"Hugging Face cache directory set to: {os.environ['HF_HOME']}")
            except Exception as e_hf_cache:
                print_error(f"CRITICAL: Failed to set custom Hugging Face cache directory: {e_hf_cache}")
                setup_failures.append("Failed to configure Hugging Face cache directory.")

            # Ensure CONDA_EXECUTABLE is set in the relaunched script context
            if globals().get('CONDA_EXECUTABLE') is None:
                print_system("Relaunched script: CONDA_EXECUTABLE is None. Attempting to load from cache or PATH...")
                loaded_from_cache = False
                if os.path.exists(CONDA_PATH_CACHE_FILE):
                    try:
                        with open(CONDA_PATH_CACHE_FILE, 'r', encoding='utf-8') as f_cache:
                            cached_path = f_cache.read().strip()
                        if cached_path and _verify_conda_path(cached_path):
                            globals()['CONDA_EXECUTABLE'] = cached_path
                            print_system(f"Using Conda executable from cache in relaunched script: {CONDA_EXECUTABLE}")
                            loaded_from_cache = True
                        else:
                            print_warning("Cached Conda path invalid/unverified in relaunched script.")
                    except Exception as e_cache_read:
                        print_warning(f"Error reading Conda cache in relaunched script: {e_cache_read}")
                if not loaded_from_cache:
                    conda_exe_from_which = shutil.which("conda.exe" if IS_WINDOWS else "conda")
                    if conda_exe_from_which and _verify_conda_path(conda_exe_from_which):
                        globals()['CONDA_EXECUTABLE'] = conda_exe_from_which
                        print_system(f"Found Conda via shutil.which in relaunched script: {CONDA_EXECUTABLE}")
                    else:
                        print_error("CRITICAL: Conda executable could not be determined. Conda package installs will fail.")

            if globals().get('CONDA_EXECUTABLE') is None:
                print_error("CRITICAL: CONDA_EXECUTABLE is still None. Subsequent Conda operations WILL FAIL.")
                setup_failures.append("Failed to install/ensure critical tool: conda is not found on operational runtime")

            # --- Read AUTODETECTED environment variables ---
            AUTO_PRIMARY_GPU_BACKEND = os.getenv("AUTODETECTED_PRIMARY_GPU_BACKEND", "cpu")
            AUTO_CUDA_AVAILABLE = os.getenv("AUTODETECTED_CUDA_AVAILABLE") == "1"
            AUTO_METAL_AVAILABLE = os.getenv("AUTODETECTED_METAL_AVAILABLE") == "1"
            AUTO_VULKAN_AVAILABLE = os.getenv("AUTODETECTED_VULKAN_AVAILABLE") == "1"
            AUTO_COREML_POSSIBLE = os.getenv("AUTODETECTED_COREML_POSSIBLE") == "1"
            print_system(
                f"Auto-detected preferences received: GPU_BACKEND='{AUTO_PRIMARY_GPU_BACKEND}', CUDA={AUTO_CUDA_AVAILABLE}, METAL={AUTO_METAL_AVAILABLE}, VULKAN={AUTO_VULKAN_AVAILABLE}, COREML_POSSIBLE={AUTO_COREML_POSSIBLE}")

            if "TERMUX_VERSION" in os.environ:
                gpu_failure_flag_path = "/root/.gpu_acceleration_failed"
                if os.path.exists(gpu_failure_flag_path):
                    print_colored("WARNING", "------------------------------------------------------------", "WARNING")
                    print_colored("WARNING", "             Entering CPU Fallback Mode", "WARNING")
                    print_warning("GPU driver verification failed during the initial setup.")
                    print_warning("All models Will all run on CPU")
                    print_warning("Performance will be significantly reduced.")
                    print_colored("WARNING", "------------------------------------------------------------", "WARNING")
                    
                    # This is the override logic. We force all GPU flags to false.
                    AUTO_PRIMARY_GPU_BACKEND = "cpu"
                    AUTO_CUDA_AVAILABLE = False
                    AUTO_METAL_AVAILABLE = False
                    AUTO_VULKAN_AVAILABLE = False


            # Set up core Python executable paths (already in Conda env)
            _sys_exec_dir = os.path.dirname(sys.executable)
            PYTHON_EXECUTABLE = sys.executable

            print_system(f"Checking python version again, it could betrayed us! changed in the mid exec {sys.version.split()[0]} ")

            PIP_EXECUTABLE = os.path.join(_sys_exec_dir, "pip.exe" if IS_WINDOWS else "pip")
            if not os.path.exists(PIP_EXECUTABLE): PIP_EXECUTABLE = shutil.which("pip") or "pip" # Fallback if direct path missing
            HYPERCORN_EXECUTABLE = os.path.join(_sys_exec_dir, "hypercorn.exe" if IS_WINDOWS else "hypercorn")
            if not os.path.exists(HYPERCORN_EXECUTABLE): HYPERCORN_EXECUTABLE = shutil.which("hypercorn") or "hypercorn" # Fallback



            print_system(f"Updated PIP_EXECUTABLE to: {PIP_EXECUTABLE}")
            print_system(f"Updated HYPERCORN_EXECUTABLE to: {HYPERCORN_EXECUTABLE}")

            # --- Engine Python Dependencies (requirements.txt) ---
            engine_req_path = os.path.join(ROOT_DIR, "requirements.txt")
            if not os.path.exists(engine_req_path): print_error(
                f"requirements.txt not found: {engine_req_path}"); setup_failures.append(
                f"Guru Meditation: requirements.txt not found")
            pip_install_success = False
            MAX_PIP_RETRIES = int(
                os.getenv("PIP_INSTALL_RETRIES", 3))  # Reduced from 99999 to 3 for more realistic retry
            PIP_RETRY_DELAY_SECONDS = 5


            # Install fundamental Python libraries (requests for downloads, tqdm for progress)
            if not run_command([PIP_EXECUTABLE, "install", "-U", "tqdm", "requests", 'dotenv', 'rich', 'textual', 'playwright', 'multiprocess', 'requests', 'setuptools'], ROOT_DIR, "PIP-UTILS"):
                print_error("tqdm/requests install failed. Exiting as these are crucial for further setup."); setup_failures.append("failed to install tqdm and requests for file downloads")

            # Initialize requests session for file downloads
            try:
                import requests; from tqdm import tqdm # Import here to ensure they are available after pip install
            except ImportError:
                print_error("Failed to import requests/tqdm after installation. Exiting."); setup_failures.append("Failed to install/ensure critical tool: failed to import requests and tqdm")
            import requests #enforce import
            requests_session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=20)
            requests_session.mount('http://', adapter)
            requests_session.mount('https://', adapter)

            _ensure_playwright_browsers()

            print_system("--- Ensuring critical environment utilities (patchelf, ncurses) ---")
            if platform.system() == "Linux":
                if not _ensure_conda_package("patchelf", is_critical=True):
                    setup_failures.append("Failed to install patchelf, which is critical for fixing library paths.")
            if not _ensure_conda_package("ncurses", is_critical=True):
                setup_failures.append("Failed to install ncurses, which is critical for terminal UI operations.")

            # --- START: Robust Node.js & npm setup (MOVED HERE) ---
            print_system("--- Ensuring Node.js & npm Dependencies ---")

            node_package_spec = f"nodejs={REQUIRED_NODE_MAJOR}"
            needs_node_install_or_reinstall = False

            try:
                # Attempt to detect the currently active Node.js/npm versions
                # Use subprocess.run to capture output and check return code.
                # Running them via PYTHON_EXECUTABLE -c "import subprocess..." ensures it uses the Conda Python's PATH.
                node_check_cmd = [PYTHON_EXECUTABLE, "-c", "import subprocess; print(subprocess.check_output(['node', '-v'], text=True, stderr=subprocess.DEVNULL).strip())"]
                npm_check_cmd = [PYTHON_EXECUTABLE, "-c", "import subprocess; print(subprocess.check_output(['npm', '-v'], text=True, stderr=subprocess.DEVNULL).strip())"]

                node_proc = subprocess.run(node_check_cmd, capture_output=True, text=True, check=False, timeout=10)
                npm_proc = subprocess.run(npm_check_cmd, capture_output=True, text=True, check=False, timeout=10)

                current_node_version_output = node_proc.stdout.strip() if node_proc.returncode == 0 else ""
                current_npm_version_output = npm_proc.stdout.strip() if npm_proc.returncode == 0 else ""

                current_node_major = 0
                if current_node_version_output:
                    try:
                        current_node_major = int(current_node_version_output[1:].split('.')[0]) # Remove 'v' prefix
                    except ValueError:
                        print_warning(f"Could not parse Node.js version: '{current_node_version_output}'. Assuming old or invalid.")

                current_npm_major = 0
                if current_npm_version_output:
                    try:
                        current_npm_major = int(current_npm_version_output.split('.')[0])
                    except ValueError:
                        print_warning(f"Could not parse npm version: '{current_npm_version_output}'. Assuming old or invalid.")

                print_system(f"Currently active Node.js version: {current_node_version_output or 'Not Found'} (Major: {current_node_major})")
                print_system(f"Currently active npm version: {current_npm_version_output or 'Not Found'} (Major: {current_npm_major})")

                # Decision logic: If node/npm not found, or found but too old
                if current_node_major < REQUIRED_NODE_MAJOR or current_npm_major < REQUIRED_NPM_MAJOR:
                    print_warning(f"Active Node.js/npm version is too old or not found. Required: Node {REQUIRED_NODE_MAJOR}+, npm {REQUIRED_NPM_MAJOR}+. Found: Node {current_node_major}, npm {current_npm_major}.")
                    needs_node_install_or_reinstall = True
                else:
                    print_system("Active Node.js/npm versions meet requirements.")
                    # Warn if Node.js/npm is significantly newer than specified (usually okay, but good to note)
                    if current_node_major > REQUIRED_NODE_MAJOR:
                        print_warning(f"Installed Node.js version ({current_node_major}) is newer than specified ({REQUIRED_NODE_MAJOR}). Proceeding.")
                    if current_npm_major > REQUIRED_NPM_MAJOR:
                        print_warning(f"Installed npm version ({current_npm_major}) is newer than specified ({REQUIRED_NPM_MAJOR}). Proceeding.")

            except Exception as e:
                # Catch all errors during initial version detection (e.g., command not found)
                print_warning(f"Could not reliably determine current Node.js/npm versions ({e}). Assuming fresh install/reinstall is needed.")
                needs_node_install_or_reinstall = True

            if needs_node_install_or_reinstall:
                print_system(f"Attempting to install/reinstall Node.js ({node_package_spec}) and npm...")

                # CRITICAL: Force removal of 'nodejs' from the Conda environment before reinstalling.
                # This ensures that even if a corrupted or old version exists *inside* the Conda env,
                # it's properly removed before the new install.
                print_system(f"Attempting to remove old 'nodejs' packages from '{os.path.basename(TARGET_RUNTIME_ENV_PATH)}' before reinstall...")
                # Use check=False because the package might not be there, or removal might fail partially,
                # but we still want to try the install.
                remove_cmd = [CONDA_EXECUTABLE, "remove", "--yes", "--prefix", TARGET_RUNTIME_ENV_PATH, "nodejs", "npm"]
                if not run_command(remove_cmd, cwd=ROOT_DIR, name="CONDA-REMOVE-NODEJS-NPM", check=False):
                    print_warning("Failed to fully remove existing nodejs/npm packages. Proceeding with install, but issues may persist.")
                else:
                    print_system("Old nodejs/npm packages removed (if present).")

                # Now, proceed with the install via _ensure_conda_package.
                # This call will actually run the `conda install` command.


                if not _ensure_conda_package_version("nodejs", "node", REQUIRED_NODE_MAJOR):
                    print_error(f"Failed to install Node.js {REQUIRED_NODE_MAJOR}.x in Conda environment. Exiting.")
                    setup_failures.append("Failed to install/ensure critical legacy tool: node.js failed to install or mismatch on the version")
                print_system(f"Node.js {REQUIRED_NODE_MAJOR}.x and bundled npm should now be installed.")

                # Re-verify immediately after installation to ensure success
                try:
                    post_install_node_version_check_cmd = [PYTHON_EXECUTABLE, "-c", "import subprocess; print(subprocess.check_output(['node', '-v'], text=True, stderr=subprocess.PIPE).strip())"]
                    post_install_npm_version_check_cmd = [PYTHON_EXECUTABLE, "-c", "import subprocess; print(subprocess.check_output(['npm', '-v'], text=True, stderr=subprocess.PIPE).strip())"]

                    post_install_node_version = subprocess.check_output(post_install_node_version_check_cmd, text=True, stderr=subprocess.PIPE).strip()
                    post_install_npm_version = subprocess.check_output(post_install_npm_version_check_cmd, text=True, stderr=subprocess.PIPE).strip()

                    post_install_node_major = int(post_install_node_version[1:].split('.')[0])
                    post_install_npm_major = int(post_install_npm_version.split('.')[0])

                    if post_install_node_major < REQUIRED_NODE_MAJOR or post_install_npm_major < REQUIRED_NPM_MAJOR:
                        print_error(f"ERROR: Node.js/npm still too old AFTER installation attempt. Node: {post_install_node_version}, npm: {post_install_npm_version}")
                        print_error("This indicates a deeper Conda environment or PATH issue. Manual intervention needed (e.g., `rm -rf zephyrineRuntimeVenv` then retry).")
                        setup_failures.append("Failed to install/ensure critical tool: Version mismatch or unsupported old version after installation")
                    else:
                        print_system(f"SUCCESS: Node.js {post_install_node_version} and npm {post_install_npm_version} are now active and meet requirements.")

                except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
                    print_error(f"CRITICAL: Failed to verify Node.js/npm versions immediately after installation attempt: {e}")
                    setup_failures.append(f"Failed to install/ensure critical tool: Version mismatch or unsupported old version after installation: {e}")
            # End if needs_node_install_or_reinstall

            # Final update of global NPM_CMD variable with the path to the now-correct npm
            npm_exe_name = "npm.cmd" if IS_WINDOWS else "npm"
            # Use shutil.which to get the *absolute path* from the now-activated Conda environment
            conda_bin_dir = os.path.dirname(PYTHON_EXECUTABLE)
            explicit_npm_path = os.path.join(conda_bin_dir, npm_exe_name)
            if os.path.exists(explicit_npm_path):
                globals()['NPM_CMD'] = explicit_npm_path
            else:
                # If for some reason it's not there, fall back to shutil.which as a last resort.
                print_warning(
                    f"Could not find npm at expected path '{explicit_npm_path}'. Falling back to PATH search.")
                globals()['NPM_CMD'] = shutil.which(npm_exe_name) or npm_exe_name

            # Also check for node itself, as npm depends on it
            if not globals()['NPM_CMD'] or not shutil.which("node"):
                print_error(f"CRITICAL: '{npm_exe_name}' or 'node' executable not found in Conda environment PATH after all installation attempts. Node.js setup failed. Exiting.")
                setup_failures.append(f"Failed to install/ensure critical tool: '{npm_exe_name}' or 'node' executable not found in Conda environment PATH after all installation attempts. Node.js setup failed. Exiting.")
            print_system(f"Using NPM_CMD: {globals()['NPM_CMD']}")
            print_system("--- Node.js & npm setup complete ---")
            # --- END: Robust Node.js & npm setup ---


            # --- Core Command-Line Tools (conda-installed) ---
            print_system("--- Ensuring core command-line tools (libiconv, git, cmake, go, ffmpeg, aria2c) ---")
            if not _ensure_libiconv_copy():
                # The function handles its own error printing. We just need to record the failure.
                setup_failures.append("Failed to install a compatible version of libiconv providing libiconv.so.1")
            print_system("--- Ensuring executable patching utility (patchelf) ---")
            if not _ensure_conda_package("patchelf", is_critical=True):
                setup_failures.append("Failed to install patchelf, which is critical for fixing library paths.")
            if not _ensure_conda_package("ncurses", is_critical=True):
                setup_failures.append("Failed to install ncurses, which is critical for terminal operations.")
            if AUTO_VULKAN_AVAILABLE:
                print_system("--- Ensuring Vulkan build tools (shaderc for glslc) ---")
                if not _ensure_conda_package("shaderc", executable_to_check="glslc", is_critical=True):
                    print_warning("Failed to install 'shaderc' via Conda. Vulkan-accelerated builds will likely fail.")
                    # We mark it non-critical; the build will fall back to CPU.


            if not _ensure_conda_package("git", executable_to_check="git", is_critical=True): setup_failures.append("Failed to install git, which is critical for terminal operations. requesting post install retry")
            if not _ensure_conda_package("cmake", executable_to_check="cmake", is_critical=True): setup_failures.append("Failed to install cmake, which is critical for terminal operations. requesting post install retry")
            # Ensure Go is installed for ZephyMesh and Backend
            if not _ensure_conda_package("go", executable_to_check="go", conda_channel="conda-forge", is_critical=True): setup_failures.append("Failed to install go, which is critical for terminal operations. requesting post install retry")
            # Ensure Alire/GNAT for Ada compilation
            if not _ensure_alire_and_gnat_toolchain(): setup_failures.append(f"Failed to install/ensure critical tool: Ada toolchain failed to install")

            # ffmpeg (non-critical, can fallback)
            _ensure_conda_package("ffmpeg", executable_to_check="ffmpeg", is_critical=False)

            # Aria2c (command-line tool)
            print_system("--- Ensuring Aria2c (for multi-connection downloads) via Conda ---")
            if not _ensure_conda_package(package_spec="conda-forge::aria2", executable_to_check="aria2c", is_critical=False):
                print_warning(
                    "aria2c (command-line tool) could not be installed via Conda. Multi-connection downloads will be unavailable, falling back to requests.")
            else:
                print_system("aria2c command-line tool checked/installed via Conda.")

            # Install Python wrapper for Aria2 (aria2p) via Pip (if not already done)
            if not os.path.exists(ARIA2P_INSTALLED_FLAG_FILE):
                print_system("--- Installing Python wrapper for Aria2 (aria2p) ---")
                if not run_command([PIP_EXECUTABLE, "install", "aria2p"], ROOT_DIR, "PIP-ARIA2P"):
                    print_warning(
                        "Failed to install 'aria2p' Python library. Multi-connection downloads via Aria2 will not be available if chosen as the download method.")
                else:
                    print_system("'aria2p' Python library installed successfully.")
                    try:
                        with open(ARIA2P_INSTALLED_FLAG_FILE, 'w', encoding='utf-8') as f_aria2p:
                            f_aria2p.write(f"Installed on: {datetime.now().isoformat()}\n")
                        print_system("aria2p installation flag created.")
                    except IOError as flag_err_aria2p:
                        print_error(f"Could not create aria2p installation flag file: {flag_err_aria2p}")
            else:
                print_system("Python wrapper for Aria2 (aria2p) previously installed (flag file found).")
            print_system("--- Core command-line tools check/install complete ---")

            # --- Frontend Node.js Dependencies ---
            print_system("--- Installing/Checking Frontend Node.js Dependencies ---")
            # This will now use the Node.js/npm ensured by the block above
            if not run_command([NPM_CMD, "install"], FRONTEND_DIR, "NPM-FRONTEND"):
                print_error("NPM Frontend install failed. Exiting."); setup_failures.append(f"Failed to install/ensure critical legacy tool: npm node failed to install")
            print_system("--- Frontend Node.js Dependencies installed ---")


            # Ensure `tiktoken` for license reading time estimation
            try:
                import tiktoken; TIKTOKEN_AVAILABLE = True
            except ImportError:
                TIKTOKEN_AVAILABLE = False; print_warning("tiktoken not available. License reading time estimation will be less accurate.")

            # Windows-specific curses support
            if IS_WINDOWS:
                if not run_command([PIP_EXECUTABLE, "install", "windows-curses"], ROOT_DIR, "PIP-WINCURSES"):
                    print_warning("windows-curses install failed. Curses-based license prompt may not work fully.")
            try:
                import curses
            except ImportError:
                print_error("curses import failed. License agreement cannot be displayed. Exiting."); setup_failures.append(f"Failed to ensure License acceptance. Jurisdiction comprimised")

            # License Acceptance
                    # License Acceptance
            if not os.path.exists(LICENSE_FLAG_FILE):
                print_system("License agreement required.")
                _, combined_license_text = load_licenses()
                estimated_reading_seconds = calculate_reading_time(combined_license_text)

                accepted = False
                time_taken = 0.0

                try:
                    # Attempt to display the interactive license prompt
                    accepted, time_taken = curses.wrapper(display_license_prompt, combined_license_text.splitlines(),
                                                        estimated_reading_seconds) # type: ignore
                except curses.error as e:
                    # Handle the specific error for unknown terminals
                    if "setupterm" in str(e) and "terminfo" in str(e):
                        print_error("Your terminal type is unknown or unsupported (e.g., TERM=unknown).")
                        print_warning("The interactive license prompt cannot be displayed.")
                        print_system("The program will proceed in 30 seconds, implicitly accepting the software licenses.")
                        print_system("To review licenses manually, check the 'licenses' directory.")
                        print_system("Press Ctrl+C now to cancel.")
                        try:
                            # Wait for 30 seconds, allowing the user to cancel
                            time.sleep(30)
                            accepted = True
                            time_taken = 30.0 # Record the time for logging
                            print_system("Continuing with implicit license acceptance...")
                        except KeyboardInterrupt:
                            print_error("\nOperation cancelled by user during implicit wait. Exiting.")
                            sys.exit(1)
                    else:
                        # For any other terminal-related error, fail gracefully
                        print_error(f"A fatal terminal error occurred: {e}")
                        print_error("Please run this script in a standard interactive terminal. Exiting.")
                        sys.exit(1)

                # Check if the license was accepted (either interactively or implicitly)
                if not accepted:
                    print_error("License not accepted. Exiting.")
                    sys.exit(1)

                # If accepted, create the flag file
                with open(LICENSE_FLAG_FILE, 'w', encoding='utf-8') as f_license:
                    f_license.write(f"Accepted: {datetime.now().isoformat()}\nTime: {time_taken:.2f}s\n")
                print_system(f"Licenses accepted in {time_taken:.2f}s.")
                if estimated_reading_seconds > 30 and time_taken < (estimated_reading_seconds * 0.1):
                    print_warning("Warning: Licenses accepted very quickly. Please ensure you have reviewed the terms.")
                    time.sleep(3)
            else:
                print_system("License previously accepted.")

            #_start_port_shield_daemon(port=11434, target_process_name="ollama") #brutal way to destroy ollama but ofc its fail since ollama always wins

            print_system(f"--- Checking Static Model Pool: {STATIC_MODEL_POOL_PATH} ---")
            os.makedirs(STATIC_MODEL_POOL_PATH, exist_ok=True)
            all_models_ok = True # Keep track if all critical models downloaded successfully
            for model_info in MODELS_TO_DOWNLOAD:
                dest_path = os.path.join(STATIC_MODEL_POOL_PATH, model_info["filename"])
                if not os.path.exists(dest_path):
                    print_warning(f"Model '{model_info['description']}' ({model_info['filename']}) not found. Downloading.")
                    if not download_file_with_progress(model_info["url"], dest_path, model_info["description"],
                                                    requests_session):
                        print_error(f"Failed download for {model_info['filename']}. This model may be required.")
                        all_models_ok = False
                else:
                    print_system(f"Model '{model_info['description']}' ({model_info['filename']}) already present.")
            print_system(f"Static model pool checked. All critical models {'OK' if all_models_ok else 'NOT OK'}.")
            if not all_models_ok:
                print_error("Some critical models failed to download. Functionality may be limited. Proceeding with caution.")
                # Decide if this is a hard exit or just a warning based on criticality of models.
                setup_failures.append(f"Models failed to download, check static model pool. functionality compromised")

            print_system("--- Checking TTS Systems (Chatterbox & MeloTTS) ---")

            # --- Stage 1: ChatterboxTTS Library Installation ---
            if not os.path.exists(CHATTERBOX_TTS_INSTALLED_FLAG_FILE):
                print_system(f"--- ChatterboxTTS First-Time Library Install ---")
                if not os.path.isdir(CHATTERBOX_TTS_PATH):
                    print_error(f"ChatterboxTTS directory not found at: {CHATTERBOX_TTS_PATH}. Skipping.")
                    setup_failures.append("ChatterboxTTS directory not found.")
                else:
                    if not run_command([PIP_EXECUTABLE, "install", "-e", "."], CHATTERBOX_TTS_PATH,
                                       "PIP-CHATTERBOX-EDITABLE"):
                        print_error("ChatterboxTTS library installation failed.")
                        setup_failures.append("ChatterboxTTS pip install failed.")
                    else:
                        print_system("ChatterboxTTS library installed successfully.")
                        with open(CHATTERBOX_TTS_INSTALLED_FLAG_FILE, 'w', encoding='utf-8') as f:
                            f.write(f"Installed on: {datetime.now().isoformat()}\n")
            else:
                print_system("ChatterboxTTS library previously installed (flag file found).")

            # --- Stage 2: MeloTTS Library Installation ---
            if not os.path.exists(MELO_TTS_LIB_INSTALLED_FLAG_FILE):
                print_system("--- MeloTTS First-Time Library Install ---")
                if not os.path.isdir(MELO_TTS_PATH):
                    print_error(f"MeloTTS directory not found at: {MELO_TTS_PATH}. Skipping.")
                    setup_failures.append("MeloTTS directory not found.")
                else:
                    if not run_command([PIP_EXECUTABLE, "install", "torchcodec"], MELO_TTS_PATH, "PIP-MELO-EDITABLE"):
                        print_error("MeloTTS library torchcodec installation failed.")
                        setup_failures.append("MeloTTS pip install torchcodec phase failed.")
                    if not run_command([PIP_EXECUTABLE, "install", "-e", "."], MELO_TTS_PATH, "PIP-MELO-EDITABLE"):
                        print_error("MeloTTS library installation failed.")
                        setup_failures.append("MeloTTS pip install failed.")
                    else:
                        print_system("MeloTTS library installed successfully.")
                        with open(MELO_TTS_LIB_INSTALLED_FLAG_FILE, 'w', encoding='utf-8') as f:
                            f.write(f"Installed on: {datetime.now().isoformat()}\n")
            else:
                print_system("MeloTTS library previously installed (flag file found).")

            # --- Stage 3: Shared TTS Data Dependency Download ---
            # This step runs AFTER both libraries are confirmed to be installed.
            # It is guarded by its own flag for the data download.
            if not os.path.exists(MELO_TTS_DATA_INSTALLED_FLAG_FILE):
                print_system("--- Downloading Shared TTS Data Dependencies (Unidic & Models) ---")

                # This command triggers downloads for both Unidic (for Melo) and the required HF models.
                # We use the audio_worker in test mode as it initializes the necessary components.
                tts_data_download_command = [
                    PYTHON_EXECUTABLE, "audio_worker.py",
                    "--task-type", "tts",
                    "--test-mode",
                    "--model-dir", "./staticmodelpool",  # Relative to ENGINE_MAIN_DIR
                    "--temp-dir", "./temp",  # Relative to ENGINE_MAIN_DIR
                    "--output-file", "tts_result.wav"  # Relative to ENGINE_MAIN_DIR
                ]

                if not run_command(tts_data_download_command, ENGINE_MAIN_DIR, "TTS-DEPS-DOWNLOAD"):
                    print_error(
                        "TTS dependency download worker failed. This means model/dictionary download likely failed.")
                    setup_failures.append("TTS dependency download via test worker failed.")
                else:
                    print_system("Shared TTS data dependencies downloaded successfully.")
                    # Create the data-specific flag ONLY after the data is successfully downloaded.
                    with open(MELO_TTS_DATA_INSTALLED_FLAG_FILE, 'w', encoding='utf-8') as f:
                        f.write(f"Data dependencies downloaded on: {datetime.now().isoformat()}")
                    print_system("TTS data dependency flag created.")
            else:
                print_system("Shared TTS data dependencies previously downloaded (flag file found).")

            # --- PyWhisperCpp Installation (with auto-detection fallback) ---
            if not os.path.exists(PYWHISPERCPP_INSTALLED_FLAG_FILE):
                print_system(f"--- PyWhisperCpp Installation (from local clone) ---")
                if not GIT_CMD or not shutil.which(GIT_CMD.split()[0]):
                    print_error("'git' not found. Skipping PyWhisperCpp source install.")
                else:
                    if os.path.exists(PYWHISPERCPP_CLONE_PATH): shutil.rmtree(PYWHISPERCPP_CLONE_PATH)
                    if not run_command([GIT_CMD, "clone", PYWHISPERCPP_REPO_URL, PYWHISPERCPP_CLONE_PATH], ROOT_DIR,
                                    "GIT-CLONE-PYWHISPERCPP"):
                        print_error("Clone pywhispercpp failed.")
                    elif not run_command([GIT_CMD, "submodule", "update", "--init", "--recursive"], PYWHISPERCPP_CLONE_PATH,
                                        "GIT-SUBMODULE-PYWHISPERCPP"):
                        print_error("Pywhispercpp submodule update failed.")
                    else:
                        pip_cmd_whisper = [PIP_EXECUTABLE, "install", "."]
                        env_whisper = {}
                        backend_whisper = "cpu (default)"
                        user_ggml_cuda = os.getenv("GGML_CUDA")
                        user_whisper_coreml = os.getenv("WHISPER_COREML")
                        user_ggml_vulkan = os.getenv("GGML_VULKAN")
                        user_ggml_blas = os.getenv("GGML_BLAS")
                        user_whisper_openvino = os.getenv("WHISPER_OPENVINO")

                        if user_ggml_cuda == '1':
                            env_whisper['GGML_CUDA'] = '1'; backend_whisper = "CUDA (User)"
                        elif user_whisper_coreml == '1' and AUTO_COREML_POSSIBLE:
                            env_whisper['WHISPER_COREML'] = '1'; backend_whisper = "CoreML (User)"
                        elif user_ggml_vulkan == '1':
                            env_whisper['GGML_VULKAN'] = '1'; backend_whisper = "Vulkan (User)"
                        elif user_ggml_blas == '1':
                            env_whisper['GGML_BLAS'] = '1'; backend_whisper = "OpenBLAS (User)"
                        elif user_whisper_openvino == '1':
                            env_whisper['WHISPER_OPENVINO'] = '1'; backend_whisper = "OpenVINO (User)"
                        elif AUTO_CUDA_AVAILABLE:
                            env_whisper['GGML_CUDA'] = '1'; backend_whisper = "CUDA (Auto)"
                        elif AUTO_COREML_POSSIBLE and (AUTO_PRIMARY_GPU_BACKEND == "metal" or platform.system() == "Darwin"):
                            env_whisper['WHISPER_COREML'] = '1'; backend_whisper = "CoreML (Auto for Apple)"
                        elif AUTO_METAL_AVAILABLE and AUTO_PRIMARY_GPU_BACKEND == "metal":
                            backend_whisper = "Metal (Auto Default for Apple)"
                        elif AUTO_VULKAN_AVAILABLE:
                            env_whisper['GGML_VULKAN'] = '1'; backend_whisper = "Vulkan (Auto)"
                        else:
                            backend_whisper = "CPU (Default - Attempting OpenBLAS)"; env_whisper['GGML_BLAS'] = '1'

                        print_system(
                            f"Attempting pywhispercpp install. Effective Backend: {backend_whisper}. Build Env: {env_whisper}")
                        if not run_command(pip_cmd_whisper, PYWHISPERCPP_CLONE_PATH, "PIP-PYWHISPERCPP",
                                        env_override=env_whisper if env_whisper else None):
                            print_error("pywhispercpp install failed.")
                        else:
                            print_system("pywhispercpp installed.")
                            print_warning("For non-WAV ASR, ensure FFmpeg is in PATH (via conda install ffmpeg).")
                            with open(PYWHISPERCPP_INSTALLED_FLAG_FILE, 'w',
                                    encoding='utf-8') as f_pwc_flag:
                                f_pwc_flag.write(backend_whisper)
                            print_system("PyWhisperCpp installation flag created.")
            else:
                print_system("PyWhisperCpp previously installed.")

            # --- Install Local LaTeX-OCR Sub-Engine ---
            if not os.path.exists(LATEX_OCR_INSTALLED_FLAG_FILE):
                print_system(f"--- First-Time Setup for local LaTeX_OCR-SubEngine ---")
                if not os.path.isdir(LATEX_OCR_PATH):
                    print_error(f"LaTeX-OCR Sub-Engine directory not found at: {LATEX_OCR_PATH}")
                    print_error("Please ensure the submodule/directory exists. Skipping LaTeX-OCR installation.")
                else:
                    print_system(f"Installing local LaTeX-OCR in editable mode from: {LATEX_OCR_PATH}")
                    pip_cmd_latex_ocr_local = [PIP_EXECUTABLE, "install", "-e", "."]
                    if not run_command(pip_cmd_latex_ocr_local, LATEX_OCR_PATH, "PIP-LATEX-OCR-LOCAL"):
                        print_error(
                            "Failed to install local LaTeX-OCR Sub-Engine. This functionality will be unavailable.")
                    else:
                        print_system("Local LaTeX-OCR Sub-Engine installed successfully.")
                        try:
                            with open(LATEX_OCR_INSTALLED_FLAG_FILE, 'w', encoding='utf-8') as f:
                                f.write(f"Installed from local Sub-Engine on: {datetime.now().isoformat()}\n")
                            print_system("Local LaTeX-OCR installation flag created.")
                        except IOError as flag_err:
                            print_error(f"Could not create local LaTeX-OCR installation flag file: {flag_err}")
            else:
                print_system("Local LaTeX-OCR Sub-Engine previously installed (flag file found).")

            # --- Custom llama-cpp-python Installation ---
            if os.getenv("PROVIDER", "llama_cpp").lower() == "llama_cpp":
                if not os.path.exists(CUSTOM_LLAMA_CPP_INSTALLED_FLAG_FILE):
                    print_system("--- Custom llama-cpp-python Installation ---")
                    if not GIT_CMD or not shutil.which(GIT_CMD.split()[0]):
                        print_error("'git' not found. Skipping source install.")
                    else:
                        if os.path.exists(LLAMA_CPP_PYTHON_CLONE_PATH): shutil.rmtree(LLAMA_CPP_PYTHON_CLONE_PATH)
                        if not run_command([GIT_CMD, "clone", LLAMA_CPP_PYTHON_REPO_URL, LLAMA_CPP_PYTHON_CLONE_PATH],
                                        ROOT_DIR, "GIT-CLONE-LLAMA") or \
                                not run_command([GIT_CMD, "submodule", "update", "--init", "--recursive"],
                                                LLAMA_CPP_PYTHON_CLONE_PATH, "GIT-SUBMODULE-LLAMA"):
                            print_error("llama-cpp-python clone or submodule update failed.")
                        else:
                            build_env_llama = {'FORCE_CMAKE': '1'}
                            cmake_args_list_llama = ["-DLLAMA_BUILD_EXAMPLES=OFF", "-DLLAMA_BUILD_TESTS=OFF"]
                            chosen_llama_backend = os.getenv("LLAMA_CPP_BACKEND") or AUTO_PRIMARY_GPU_BACKEND
                            backend_log_llama = "cpu"
                            if chosen_llama_backend == "cuda" and AUTO_CUDA_AVAILABLE:
                                cmake_args_list_llama.append("-DGGML_CUDA=ON"); backend_log_llama = "CUDA"
                            elif chosen_llama_backend == "metal" and AUTO_METAL_AVAILABLE:
                                cmake_args_list_llama.append("-DGGML_METAL=ON"); backend_log_llama = "Metal"
                            elif chosen_llama_backend == "vulkan" and AUTO_VULKAN_AVAILABLE:
                                cmake_args_list_llama.append("-DGGML_VULKAN=ON"); backend_log_llama = "Vulkan"
                            else:
                                cmake_args_list_llama.append("-DLLAMA_OPENMP=ON"); backend_log_llama = "CPU (OpenMP)"
                            print_system(f"Configuring llama-cpp-python build with: {backend_log_llama}")
                            build_env_llama['CMAKE_ARGS'] = " ".join(cmake_args_list_llama)
                            if not run_command([PIP_EXECUTABLE, "install", ".", "--verbose"],
                                            LLAMA_CPP_PYTHON_CLONE_PATH, "PIP-BUILD-LLAMA",
                                            env_override=build_env_llama):
                                print_error("Build llama-cpp-python failed.")
                            else:
                                print_system("llama-cpp-python installed.")
                                with open(CUSTOM_LLAMA_CPP_INSTALLED_FLAG_FILE, 'w',
                                        encoding='utf-8') as f_lcpp_flag:
                                    f_lcpp_flag.write(backend_log_llama)
                else:
                    print_system("Custom llama-cpp-python previously installed.")

            # --- Custom stable-diffusion-cpp-python Installation ---
            if not os.path.exists(CUSTOM_SD_CPP_PYTHON_INSTALLED_FLAG_FILE):
                print_system("--- Custom stable-diffusion-cpp-python Installation ---")
                if not GIT_CMD or not shutil.which(GIT_CMD.split()[0]) or not CMAKE_CMD or not shutil.which(
                    CMAKE_CMD.split()[0]):
                    print_error("'git' or 'cmake' not found. Skipping source install.")
                else:
                    if os.path.exists(STABLE_DIFFUSION_CPP_PYTHON_CLONE_PATH): shutil.rmtree(
                        STABLE_DIFFUSION_CPP_PYTHON_CLONE_PATH)
                    if not run_command([GIT_CMD, "clone", "--recursive", STABLE_DIFFUSION_CPP_PYTHON_REPO_URL,
                                        STABLE_DIFFUSION_CPP_PYTHON_CLONE_PATH], ROOT_DIR, "GIT-CLONE-SD"):
                        print_error("Clone SD failed.")
                    else:
                        sd_cpp_sub_path = os.path.join(STABLE_DIFFUSION_CPP_PYTHON_CLONE_PATH, "vendor",
                                                    "stable-diffusion.cpp")
                        sd_cpp_build_path = os.path.join(sd_cpp_sub_path, "build")
                        os.makedirs(sd_cpp_build_path, exist_ok=True)
                        cmake_args_sd_lib = []
                        chosen_sd_backend = os.getenv("SD_CPP_BACKEND") or AUTO_PRIMARY_GPU_BACKEND
                        backend_log_sd = "cpu"
                        if chosen_sd_backend == "cuda" and AUTO_CUDA_AVAILABLE:
                            cmake_args_sd_lib.append("-DSD_CUDA=ON"); backend_log_sd = "CUDA"
                        elif chosen_sd_backend == "metal" and AUTO_METAL_AVAILABLE:
                            cmake_args_sd_lib.append("-DSD_METAL=ON"); backend_log_sd = "Metal"
                        elif chosen_sd_backend == "vulkan" and AUTO_VULKAN_AVAILABLE:
                            cmake_args_sd_lib.append("-DSD_VULKAN=ON"); backend_log_sd = "Vulkan"
                        else:
                            backend_log_sd = "CPU (Default)"
                        print_system(f"Configuring stable-diffusion.cpp library build with: {backend_log_sd}")
                        if not run_command([CMAKE_CMD, ".."] + cmake_args_sd_lib, sd_cpp_build_path, "CMAKE-SD-LIB-CFG"):
                            print_error("CMake SD lib config failed.")
                        elif not run_command([CMAKE_CMD, "--build", "."] + (["--config", "Release"] if IS_WINDOWS else []),
                                            sd_cpp_build_path, "CMAKE-BUILD-SD-LIB"):
                            print_error("Build SD lib failed.")
                        else:
                            pip_build_env_sd = {'FORCE_CMAKE': '1'}
                            if cmake_args_sd_lib: pip_build_env_sd['CMAKE_ARGS'] = " ".join(cmake_args_sd_lib)
                            if not run_command([PIP_EXECUTABLE, "install", "."], STABLE_DIFFUSION_CPP_PYTHON_CLONE_PATH,
                                            "PIP-SD-BINDINGS", env_override=pip_build_env_sd):
                                print_error("Install SD bindings failed.")
                            else:
                                print_system("stable-diffusion-cpp-python installed.")
                                with open(CUSTOM_SD_CPP_PYTHON_INSTALLED_FLAG_FILE, 'w',
                                        encoding='utf-8') as f_sd_flag:
                                    f_sd_flag.write(backend_log_sd)
            else:
                print_system("Custom stable-diffusion-cpp-python previously installed.")


            print_system(f"--- Installing Python dependencies from {os.path.basename(engine_req_path)} ---")
            for attempt_pip in range(MAX_PIP_RETRIES):
                if run_command([PIP_EXECUTABLE, "install", "-r", engine_req_path], ENGINE_MAIN_DIR, "PIP-ENGINE-REQ"):
                    pip_install_success = True
                    break
                else:
                    print_warning(
                        f"pip install failed on attempt {attempt_pip + 1}/{MAX_PIP_RETRIES}. Retrying in {PIP_RETRY_DELAY_SECONDS} seconds...")
                    time.sleep(PIP_RETRY_DELAY_SECONDS)

            if not pip_install_success:
                print_error(f"Failed to install Python dependencies for Engine after {MAX_PIP_RETRIES} attempts. Exiting.")
                setup_failures.append(f"Pip failed to install Engine dependencies")
            print_system("--- Python dependencies for Engine installed ---")

            # Global command variable final sanity check (should point to conda env versions now)
            globals()['GIT_CMD'] = shutil.which("git") or ('git.exe' if IS_WINDOWS else 'git')
            globals()['CMAKE_CMD'] = shutil.which("cmake") or ('cmake.exe' if IS_WINDOWS else 'cmake')
            # NPM_CMD is already set by the Node.js block, but this re-confirms or falls back if needed.
            npm_exe_name_check = "npm.cmd" if IS_WINDOWS else "npm"
            globals()['NPM_CMD'] = shutil.which(npm_exe_name_check) or npm_exe_name_check # Ensure this is updated from the conda env

            print_system(f"Using GIT_CMD: {GIT_CMD}")
            print_system(f"Using CMAKE_CMD: {CMAKE_CMD}")
            print_system(f"Using NPM_CMD: {NPM_CMD}")
            # One final comprehensive check for all critical tools
            if not all([
                GIT_CMD and shutil.which(GIT_CMD.split()[0]),
                CMAKE_CMD and shutil.which(CMAKE_CMD.split()[0]),
                NPM_CMD and shutil.which(NPM_CMD.split('.')[0]),
                shutil.which("node"), # Directly check 'node' executable
                shutil.which("go"), # Directly check 'go' executable
                shutil.which("alr"), # Directly check 'alr' executable
            ]):
                print_error("One or more critical tools (git, cmake, node, npm, go, alr) not found after attempts. Exiting.")
                setup_failures.append(f"At final check, critical tools not found after all attempts, this maybe these or other tools are not installed in the conda environment(git, cmake, node, npm, go, alr)")

            # --- [NEW] TUI vs. Fallback Logic ---
            TUI_AVAILABLE = False
            try:
                # The TUI class should be defined in a separate file for clarity.
                # Assuming it's in `zephyrine_tui.py` in the project root.
                import textual
                import psutil
                TUI_AVAILABLE = True
            except ImportError:
                TUI_AVAILABLE = False

            TUI_LIBRARIES_AVAILABLE = False
            try:
                # Check if the required libraries can be imported
                import textual
                import psutil

                TUI_LIBRARIES_AVAILABLE = True
            except ImportError:
                TUI_LIBRARIES_AVAILABLE = False


            # This function will contain the fallback launch logic to avoid code duplication
            def launch_in_fallback_mode():
                print_warning("Falling back to simple terminal output for monitoring.")
                print_system("--- Starting Main Application Services (Fallback Mode) ---")

                # The watchdogs and mesh node are already running. We just start the rest.
                # First, check if there were any setup failures before proceeding.
                if setup_failures:
                    print_error("Halting service launch due to previous setup failures.")
                    return

                service_threads = []
                # Start Engine Main
                service_threads.append(start_service_thread(start_engine_main, "EngineMainThread"))
                time.sleep(3)
                engine_ready = any(
                    proc.poll() is None and name_s == "ENGINE" for proc, name_s, _, _ in running_processes)
                if not engine_ready:
                    print_error("Engine Main failed to start. Exiting.")
                    setup_failures.append("Engine Main failed to start.")
                    return

                # Start Backend Service and Frontend
                service_threads.append(start_service_thread(start_backend_service, "BackendServiceThread"))
                time.sleep(2)
                service_threads.append(start_service_thread(start_frontend, "FrontendThread"))

                print_colored("SUCCESS", "All services launching. Press Ctrl+C to shut down.", "SUCCESS")
                try:
                    while True:
                        all_ok = True
                        active_procs_found = False
                        with process_lock:
                            current_procs_snapshot = list(running_processes)

                        if not current_procs_snapshot and service_threads:
                            all_ok = False

                        for proc, name_s, _, _ in current_procs_snapshot:
                            if proc.poll() is None:
                                active_procs_found = True
                            else:
                                print_error(f"Service '{name_s}' exited unexpectedly (RC: {proc.poll()}).")
                                all_ok = False

                        if not all_ok:
                            print_error("One or more services terminated. Shutting down launcher.")
                            break
                        if not active_procs_found and service_threads:
                            print_system("All services seem to have finished. Exiting launcher.")
                            break

                        time.sleep(5)
                except KeyboardInterrupt:
                    print_system("\nKeyboardInterrupt received by main thread. Shutting down...")
                finally:
                    print_system("Launcher main loop finished. Ensuring cleanup via atexit...")


            # --- Main Launch Decision ---
            # Step 1: Launch the ZephyMesh Node and wait for it to be ready.
            if not _ensure_and_launch_zephymesh():
                # If it fails to start, we can still continue, but P2P will be disabled.
                print_warning(
                    "ZephyMesh node failed to start. P2P functionality will be disabled for this session.")

            # Step 2: Compile and run the watchdogs.
            print_system("--- Compiling and Launching Core Infrastructure (Watchdogs) ---")
            _compile_and_run_watchdogs()

            # --- [UI/MONITORING LAUNCH] ---
            # Now, decide which monitoring interface to launch.

            #TUI_LIBRARIES CHECK ARE NOW CHANGED BY SOMETHING

            if not setup_failures:
                print_system("--- First-Time Verification Complete ---")

                print_system("Compiling project Python source to optimized bytecode (-OO)...")
                try:
                    # 1. Specifically compile the main launcher script itself.
                    launcher_path = os.path.abspath(__file__)
                    print_system(f"  -> Compiling file: {launcher_path}")
                    compileall.compile_file(launcher_path, force=True, quiet=1, legacy=True, optimize=2)

                    # 2. Specifically compile your project's source code directories.
                    #    DO NOT include ROOT_DIR here as it contains the Conda environment.
                    project_source_dirs = [
                        ENGINE_MAIN_DIR,
                        # Add any other of YOUR source code directories here.
                        # Example: os.path.join(ROOT_DIR, "another_module")
                    ]

                    for path in set(project_source_dirs):
                        if os.path.isdir(path):
                            print_system(f"  -> Compiling directory: {path}")
                            compileall.compile_dir(path, force=True, quiet=1, legacy=True, optimize=2)
                        else:
                            print_warning(f"  -> Skipped non-existent directory for compilation: {path}")

                    print_colored("SUCCESS", "Project bytecode compilation complete.", "SUCCESS")
                except Exception as e_compile:
                    print_error(f"Failed during bytecode compilation: {e_compile}")
                    # Making this a warning instead of a failure. The app can still run without .pyc files.
                    print_warning(
                        "Bytecode compilation failed. The application can still run from .py source files, but startup will be slower.")
                    # We remove it from setup_failures so the master flag can still be created.
                    # setup_failures.append("Bytecode compilation failed.")

                print_system(f"Creating fast-launch flag at: {SETUP_COMPLETE_FLAG_FILE}")
                with open(SETUP_COMPLETE_FLAG_FILE, 'w') as f:
                    f.write(f"Setup completed on {datetime.now().isoformat()}")

                print_system("Setup complete. Proceeding to launch.")
                # We will call our new parallel launch function here later.
                # For now, put a placeholder to signify completion.
                launch_all_services_in_parallel_and_monitor()

            else:
                print_error("Halting launch due to setup/verification failures.")
                # The script will then naturally exit the 'if/else' and the loop will handle the failure.

            # --- End of [NEW] TUI vs. Fallback Logic ---
            del os.environ["ZEPHYRINE_RELAUNCHED_IN_CONDA"]
            #-=-=-=-=-=


        else:  # Initial launcher instance (is_already_in_correct_env is False)
            print_system(f"--- Initial Launcher: Conda Setup & Hardware Detection ---")

            autodetected_build_env_vars = _detect_and_prepare_acceleration_env_vars()


            if not find_conda_executable(attempt_number=attempt):
                newly_installed_conda_path = _install_miniforge_and_check_overall_environment()
                if newly_installed_conda_path:
                    print_system("Miniforge auto-installation successful. Using new executable.")
                    # The find_conda_executable() function already sets the global CONDA_EXECUTABLE
                    # and caches the path, but we can re-set it here for absolute certainty.
                    CONDA_EXECUTABLE = newly_installed_conda_path
                    try:  # Ensure the cache is written with the newly installed path
                        with open(CONDA_PATH_CACHE_FILE, 'w', encoding='utf-8') as f_cache:
                            f_cache.write(CONDA_EXECUTABLE)
                        print_system(f"Cached new Conda path: {CONDA_EXECUTABLE}")
                    except IOError as e_cache_write:
                        print_warning(f"Could not write to Conda path cache file: {e_cache_write}")
                else:
                    # If installation fails, we MUST mark this attempt as a failure and stop.
                    print_error("Automated Miniforge installation failed.")
                    print_error(
                        "Please try installing Miniconda or Miniforge manually and ensure 'conda' is in your PATH.")
                    setup_failures.append("Conda not found and auto-install failed.")
                    # Continue to the end of the loop to trigger the retry/failure logic
                    continue

                # At this point, CONDA_EXECUTABLE *must* be set. If not, something is fundamentally wrong.
            if not CONDA_EXECUTABLE:
                print_error("CRITICAL: Conda executable is not set after search and install attempts.")
                setup_failures.append("Failed to find or install a Conda executable.")
                continue  # Skip to the next attempt

            if not _miniforge_alike_checking_config(CONDA_EXECUTABLE):
                print_error("Might be fatal due to missing system dependencies _miniforge_alike_checking_config. Requesting retry afterwards!")
                setup_failures.append("issue on _miniforge_alike_checking_config. Requesting Retry!")


            if not _check_playwright_linux_deps():
                print_error("Might be fatal due to missing system dependencies _check_playwright_linux_deps. Requesting retry afterwards!")
                setup_failures.append("issue on _check_playwright_linux_deps. Requesting Retry!")

            print_system(f"Using Conda executable: {CONDA_EXECUTABLE}")

            # Run the health check on the existing environment before proceeding.
            # If it returns False, it means the environment was corrupt and has been deleted.
            if attempt >= 2:
                print_system(f"--- This is a retry (Attempt {attempt}). Running Conda health check as a diagnostic. ---")
                # If it returns False, it means the environment was corrupt and has been deleted.
                if not _check_and_repair_conda_env(TARGET_RUNTIME_ENV_PATH):
                    print_warning("Environment was repaired by deletion. The launcher will now create a fresh one.")

            if not (os.path.isdir(TARGET_RUNTIME_ENV_PATH) and os.path.exists(
                    os.path.join(TARGET_RUNTIME_ENV_PATH, 'conda-meta'))):
                print_system(f"Target Conda env '{TARGET_RUNTIME_ENV_PATH}' not found/invalid. Creating...")
                _remove_flag_files(FLAG_FILES_TO_RESET_ON_ENV_RECREATE)
                target_python_versions = get_conda_python_versions_to_try()
                if not create_conda_env(TARGET_RUNTIME_ENV_PATH, target_python_versions):
                    print_error(f"Failed to create Conda env at '{TARGET_RUNTIME_ENV_PATH}'. Exiting.")
                    setup_failures.append(f"Weird, how did we get here? Conda env creation failed")
            else:
                print_system(f"Target Conda env '{TARGET_RUNTIME_ENV_PATH}' exists.")

            # --- Conda install NVIDIA CUDA Toolkit if detected ---
            if autodetected_build_env_vars.get("AUTODETECTED_CUDA_AVAILABLE") == "1":
                print_system("CUDA acceleration detected by initial launcher.")
                if not os.path.exists(CUDA_TOOLKIT_INSTALLED_FLAG_FILE):
                    print_warning("NVIDIA CUDA Toolkit not yet marked as installed in the Conda environment. Attempting installation...")
                    print_warning("This may take several minutes and download a significant amount of data (>2 GB).")

                    cuda_install_cmd = [
                        CONDA_EXECUTABLE, 'install', '--yes',
                        '--prefix', TARGET_RUNTIME_ENV_PATH,
                        '-c', 'nvidia',
                        'cuda-toolkit'
                    ]

                    if IS_WINDOWS and CONDA_EXECUTABLE and CONDA_EXECUTABLE.lower().endswith(".bat"):
                        cuda_install_cmd = ['cmd', '/c'] + cuda_install_cmd

                    print_system(f"Executing: {' '.join(cuda_install_cmd)}")
                    try:
                        process = subprocess.Popen(cuda_install_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors='replace')
                        stdout_thread = threading.Thread(target=stream_output, args=(process.stdout, "CUDA-INSTALL-OUT"), daemon=True)
                        stderr_thread = threading.Thread(target=stream_output, args=(process.stderr, "CUDA-INSTALL-ERR"), daemon=True)
                        stdout_thread.start()
                        stderr_thread.start()

                        process.wait()
                        stdout_thread.join(timeout=5)
                        stderr_thread.join(timeout=5)

                        if process.returncode == 0:
                            print_system("NVIDIA CUDA Toolkit successfully installed into Conda environment.")
                            with open(CUDA_TOOLKIT_INSTALLED_FLAG_FILE, 'w', encoding='utf-8') as f_flag:
                                f_flag.write(f"Installed on: {datetime.now().isoformat()}\n")
                        else:
                            print_error(f"Failed to install NVIDIA CUDA Toolkit (return code: {process.returncode}).")
                            print_error("The launcher will continue, but builds requiring 'nvcc' will likely fail.")
                            print_error("Please check the 'CUDA-INSTALL-ERR' logs above for details.")
                    except Exception as e_cuda_install:
                        print_error(f"An exception occurred during CUDA toolkit installation: {e_cuda_install}")
                        print_error("The launcher will continue, but CUDA support will likely be unavailable.")
                else:
                    print_system("NVIDIA CUDA Toolkit is already marked as installed in Conda environment (flag file found).")
            # --- END Conda install NVIDIA CUDA Toolkit ---

            # Prepare relaunch command
            script_to_run_abs_path = os.path.abspath(__file__)
            conda_run_cmd_list_base = [CONDA_EXECUTABLE, 'run', '--prefix', TARGET_RUNTIME_ENV_PATH]

            conda_supports_no_capture = False
            try:
                # Check if 'conda run' supports '--no-capture-output' for direct streaming
                test_no_capture_cmd = [CONDA_EXECUTABLE, 'run', '--help']
                help_output = subprocess.check_output(test_no_capture_cmd, text=True, stderr=subprocess.STDOUT, timeout=5)
                if '--no-capture-output' in help_output:
                    conda_supports_no_capture = True
                    conda_run_cmd_list_base.insert(2, '--no-capture-output') # Insert after 'run'
            except Exception:
                pass # Ignore errors if `conda run --help` fails

            conda_run_cmd_list = conda_run_cmd_list_base + ['python', script_to_run_abs_path] + sys.argv[1:]
            if IS_WINDOWS and CONDA_EXECUTABLE and CONDA_EXECUTABLE.lower().endswith(".bat"):
                conda_run_cmd_list = ['cmd', '/c'] + conda_run_cmd_list

            display_cmd_list = [str(c) if c is not None else "<CRITICAL_NONE_ERROR>" for c in conda_run_cmd_list]
            if "<CRITICAL_NONE_ERROR>" in display_cmd_list:
                print_error(f"FATAL: None value in conda_run_cmd_list: {display_cmd_list}"); setup_failures.append(f"Critical Error: None value in conda_run_cmd_list")
            print_system(f"Relaunching script using 'conda run': {' '.join(display_cmd_list)}")

            os.makedirs(RELAUNCH_LOG_DIR, exist_ok=True)
            popen_env = os.environ.copy()
            popen_env.update(autodetected_build_env_vars)
            popen_env["ZEPHYRINE_RELAUNCHED_IN_CONDA"] = "1" # Set flag for relaunched script

            log_stream_threads = []
            common_popen_kwargs_relaunch = {"text": True, "errors": 'replace', "bufsize": 1, "env": popen_env}
            if IS_WINDOWS:
                common_popen_kwargs_relaunch["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                common_popen_kwargs_relaunch["preexec_fn"] = os.setsid

            process_conda_run = None
            try:
                if conda_supports_no_capture:
                    print_system("Parent: Relaunched script output should stream directly.")
                    process_conda_run = subprocess.Popen(conda_run_cmd_list, **common_popen_kwargs_relaunch)
                else:
                    print_system(
                        f"Parent: Relaunched script output via logs: STDOUT='{RELAUNCH_STDOUT_LOG}', STDERR='{RELAUNCH_STDERR_LOG}'")
                    with open(RELAUNCH_STDOUT_LOG, 'w', encoding='utf-8') as f_stdout, \
                            open(RELAUNCH_STDERR_LOG, 'w', encoding='utf-8') as f_stderr:
                        file_capture_kwargs = common_popen_kwargs_relaunch.copy()
                        file_capture_kwargs["stdout"] = f_stdout
                        file_capture_kwargs["stderr"] = f_stderr
                        process_conda_run = subprocess.Popen(conda_run_cmd_list, **file_capture_kwargs)

                if not process_conda_run: raise RuntimeError("Popen for conda run failed.")
                relaunched_conda_process_obj = process_conda_run # Store for atexit cleanup

                if not conda_supports_no_capture:
                    # Start threads to stream captured logs from files
                    stdout_thread = threading.Thread(target=_stream_log_file,
                                                    args=(RELAUNCH_STDOUT_LOG, relaunched_conda_process_obj, "STDOUT"),
                                                    daemon=True)
                    stderr_thread = threading.Thread(target=_stream_log_file,
                                                    args=(RELAUNCH_STDERR_LOG, relaunched_conda_process_obj, "STDERR"),
                                                    daemon=True)
                    log_stream_threads.extend([stdout_thread, stderr_thread])
                    stdout_thread.start()
                    stderr_thread.start()

                exit_code_from_conda_run = -1
                if relaunched_conda_process_obj:
                    try:
                        relaunched_conda_process_obj.wait() # Wait for the relaunched script to complete
                        exit_code_from_conda_run = relaunched_conda_process_obj.returncode
                    except KeyboardInterrupt:
                        print_system("Parent script's wait for 'conda run' interrupted. Shutting down...")
                        # If SIGINT, exit with 130 (common for Ctrl+C)
                        if not getattr(sys, 'exitfunc_called', False): sys.exit(130)

                    if log_stream_threads:
                        print_system("Parent: Relaunched script finished. Waiting for log streamers (max 2s)...")
                        for t in log_stream_threads:
                            if t.is_alive(): t.join(timeout=1.0)
                        print_system("Parent: Log streamers finished.")

                    print_system(f"'conda run' process finished with code: {exit_code_from_conda_run}.")
                    relaunched_conda_process_obj = None # Mark as handled for atexit
                    sys.exit(exit_code_from_conda_run) # Exit parent with child's return code
                else:
                    print_error("Failed to start 'conda run' process. This should not happen after Popen check."); setup_failures.append(f"conda run process failed to start on the last phase Popen check issue?")
            except Exception as e_outer:
                print_error(f"Error during 'conda run' or wait: {e_outer}"); traceback.print_exc(); setup_failures.append(f"another error or timed out during wait on conda")

        if not setup_failures:
            print_colored("SUCCESS", f"Setup completed successfully on attempt {attempt}!", "SUCCESS")
            break  # Exit the retry loop
        else:
            print_colored("ERROR", f"Setup attempt {attempt} failed with the following errors:", "ERROR")
            for failure in setup_failures:
                print_error(f"  - {failure}")
            final_setup_failures = setup_failures # Save the last set of failures for the final report

    # --- This code runs AFTER the for loop is finished (either by 'break' or exhaustion) ---
    if final_setup_failures:
        print_colored("FATAL", "="*60, "ERROR")
        print_colored("FATAL", "           PROJECT ZEPHYRINE LAUNCHER FAILED OR GONE INOP", "ERROR")
        print_colored("FATAL", "="*60, "ERROR")
        print_error(f"All {MAX_SETUP_ATTEMPTS} setup attempts have failed. Giving up.")
        print_error("The final set of errors encountered was:")
        for i, failure in enumerate(final_setup_failures, 1):
            print_error(f"  {i}. {failure}")

        print_warning("\n--- Recommended Actions Memo ---")
        print_warning("0. Make sure you are not running Windows or NT based system/environemtn, because this script is not tested on Windows.")
        print_warning("1. Review the logs above for specific error messages (e.g., from pip, cmake, git).")
        print_warning("2. Ensure you have a stable internet connection.")
        print_warning("3. Check for sufficient disk space.")
        print_warning(f"4. As a last resort, manually delete the '{RUNTIME_ENV_FOLDER_NAME}' directory and all '.flag' files in the root folder, then run this script again.")
        sys.exit(1)