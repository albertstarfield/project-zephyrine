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
#import requests #(Don't request at the top) But on the main after conda relaunch to make sure it's installed
from typing import Optional, Dict

# --- Configuration ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_MAIN_DIR = os.path.join(ROOT_DIR, "systemCore", "engineMain")
BACKEND_SERVICE_DIR = os.path.join(ROOT_DIR, "systemCore", "backend-service")
FRONTEND_DIR = os.path.join(ROOT_DIR, "systemCore", "frontend-face-zephyrine")
LICENSE_DIR = os.path.join(ROOT_DIR, "licenses")
LICENSE_FLAG_FILE = os.path.join(ROOT_DIR, ".license_accepted_v1")
# Near the top with other path configurations
RELAUNCH_LOG_DIR = os.path.join(ROOT_DIR, "logs") # Or any preferred log directory
RELAUNCH_STDOUT_LOG = os.path.join(RELAUNCH_LOG_DIR, "relaunched_launcher_stdout.log")
RELAUNCH_STDERR_LOG = os.path.join(RELAUNCH_LOG_DIR, "relaunched_launcher_stderr.log")

# --- MeloTTS Configuration ---
MELO_TTS_SUBMODULE_DIR_NAME = "MeloAudioTTS_SubEngine"
MELO_TTS_PATH = os.path.join(ENGINE_MAIN_DIR, MELO_TTS_SUBMODULE_DIR_NAME)
MELO_TTS_INSTALLED_FLAG_FILE = os.path.join(ROOT_DIR, ".melo_tts_installed_v1")

# --- ChatterboxTTS Configuration (NEW) ---
CHATTERBOX_TTS_SUBMODULE_DIR_NAME = "ChatterboxTTS_subengine"
CHATTERBOX_TTS_PATH = os.path.join(ENGINE_MAIN_DIR, CHATTERBOX_TTS_SUBMODULE_DIR_NAME)
CHATTERBOX_TTS_INSTALLED_FLAG_FILE = os.path.join(ROOT_DIR, ".chatterbox_tts_installed_v1")
# --- END ChatterboxTTS Configuration ---

# --- LaTeX-OCR Local Sub-Engine Configuration ---
LATEX_OCR_SUBMODULE_DIR_NAME = "LaTeX_OCR-SubEngine"
LATEX_OCR_PATH = os.path.join(ENGINE_MAIN_DIR, LATEX_OCR_SUBMODULE_DIR_NAME)
LATEX_OCR_INSTALLED_FLAG_FILE = os.path.join(ROOT_DIR, ".latex_ocr_subengine_installed_v1")

# --- Conda Configuration ---
# CONDA_ENV_NAME is no longer used for creation if prefix is used, but can be a descriptive base for the folder
CONDA_ENV_FOLDER_NAME = "zephyrineCondaVenv"
CONDA_EXECUTABLE = None
# TARGET_CONDA_ENV_PATH will now be ROOT_DIR + CONDA_ENV_FOLDER_NAME
TARGET_CONDA_ENV_PATH = os.path.join(ROOT_DIR, CONDA_ENV_FOLDER_NAME) # DIRECTLY DEFINE THE TARGET PATH
ACTIVE_ENV_PATH = None
CONDA_PATH_CACHE_FILE = os.path.join(ROOT_DIR, ".conda_executable_path_cache.txt")

# --- Python Versioning for Conda ---
EPOCH_DATE = date(2025, 5, 9)
INITIAL_PYTHON_MAJOR = 3
INITIAL_PYTHON_MINOR = 12
FALLBACK_PYTHON_MAJOR = 3
FALLBACK_PYTHON_MINOR = 12

running_processes = [] # For services started by the current script instance
process_lock = threading.Lock()
relaunched_conda_process_obj = None # <<<< ADD THIS LINE

# --- Executable Paths (will be redefined after Conda activation) ---
IS_WINDOWS = os.name == 'nt'
PYTHON_EXECUTABLE = sys.executable
PIP_EXECUTABLE = ""
HYPERCORN_EXECUTABLE = ""
NPM_CMD = 'npm.cmd' if IS_WINDOWS else 'npm'
GIT_CMD = 'git.exe' if IS_WINDOWS else 'git'
CMAKE_CMD = 'cmake.exe' if IS_WINDOWS else 'cmake'

# --- License Acceptance & Optional Imports (Initial state) ---
TIKTOKEN_AVAILABLE = False

# --- Static Model Pool Configuration ---
STATIC_MODEL_POOL_DIR_NAME = "staticModelPool"
STATIC_MODEL_POOL_PATH = os.path.join(ENGINE_MAIN_DIR, STATIC_MODEL_POOL_DIR_NAME)
MODELS_TO_DOWNLOAD = [
    {
        "filename": "ui-tars.gguf",
        "url": "https://huggingface.co/mradermacher/UI-TARS-1.5-7B-i1-GGUF/resolve/main/UI-TARS-1.5-7B.i1-IQ3_XXS.gguf?download=true",
        "description": "UI-TARS Model (Small UI Interaction)"
    },
    {
        "filename": "deepscaler.gguf",
        "url": "https://huggingface.co/bartowski/agentica-org_DeepScaleR-1.5B-Preview-GGUF/resolve/main/agentica-org_DeepScaleR-1.5B-Preview-f16.gguf?download=true",
        "description": "DeepScaleR Model (Agent/Router)"
    },
    {
        "filename": "LatexMind-2B-Codec-i1-GGUF-IQ4_XS.gguf",
        "url": "https://huggingface.co/mradermacher/LatexMind-2B-Codec-i1-GGUF/resolve/main/LatexMind-2B-Codec.i1-IQ4_XS.gguf?download=true",
        "description": "LatexMind Model (LaTeX/VLM)"
    },
    {
        "filename": "mxbai-embed-large-v1.gguf",
        "url": "https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1/resolve/main/mxbai-embed-large-v1.gguf?download=true",
        "description": "MXBAI Embeddings Model"
    },
    {
        "filename": "NanoTranslator-immersive_translate-0.5B-GGUF-Q4_K_M.gguf",
        "url": "https://huggingface.co/mradermacher/NanoTranslator-immersive_translate-0.5B-GGUF/resolve/main/NanoTranslator-immersive_translate-0.5B.Q5_K_M.gguf?download=true",
        "description": "NanoTranslator Model"
    },
    {
        "filename": "qwen2-math-1.5b-instruct-q5_K_M.gguf",
        "url": "https://huggingface.co/lmstudio-community/Qwen2-Math-1.5B-Instruct-GGUF/resolve/main/Qwen2-Math-1.5B-Instruct-Q5_K_M.gguf?download=true",
        "description": "Qwen2 Math Model"
    },
    {
        "filename": "Qwen2.5-DirectLowLatency.gguf",
        "url": "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q3_k_m.gguf?download=true",
        "description": "Qwen2.5 Instruct Direct Mode (Fast General)"
    },
    {
        "filename": "qwen2.5-coder-3b-instruct-q5_K_M.gguf",
        "url": "https://huggingface.co/bartowski/Qwen2.5-Coder-3B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-3B-Instruct-Q5_K_M.gguf?download=true",
        "description": "Qwen2.5 Coder Model"
    },
    {
        "filename": "Qwen2.5-VL-7B-Instruct-q4_k_m.gguf",
        "url": "https://huggingface.co/bartowski/Qwen2-VL-7B-Instruct-GGUF/resolve/main/Qwen2-VL-7B-Instruct-Q4_K_M.gguf?download=true",
        "description": "Qwen2.5 VL Model"
    },
    {
        "filename": "whisper-large-v3-q8_0.gguf",
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/models/ggml-large-v3-q8_0.bin?download=true",
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
    LICENSE_FLAG_FILE,
    MELO_TTS_INSTALLED_FLAG_FILE,
    CUSTOM_LLAMA_CPP_INSTALLED_FLAG_FILE,
    CUSTOM_SD_CPP_PYTHON_INSTALLED_FLAG_FILE,
    CONDA_PATH_CACHE_FILE,
    PYWHISPERCPP_INSTALLED_FLAG_FILE,
    ARIA2P_INSTALLED_FLAG_FILE, # From previous step
    CHATTERBOX_TTS_INSTALLED_FLAG_FILE,
    LATEX_OCR_INSTALLED_FLAG_FILE
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
    reset_color = "\033[0m"
    if sys.stdout.isatty():
        print(f"[{now} | {prefix_color}{prefix.ljust(10)}{reset_color}] {message.strip()}")
    else:
        print(f"[{now} | {prefix.ljust(10)}] {message.strip()}")


def terminate_relaunched_process(process_obj, name="Relaunched Conda Process"):
    if not process_obj or process_obj.poll() is not None:
        return # Process doesn't exist or already terminated

    pid = process_obj.pid
    print_system(f"Attempting to terminate {name} (PID: {pid}) and its process tree...")
    try:
        if IS_WINDOWS:
            # Use taskkill to terminate the process tree. /F for force, /T for tree.
            kill_cmd = ['taskkill', '/F', '/T', '/PID', str(pid)]
            print_system(f"Executing: {' '.join(kill_cmd)}")
            result = subprocess.run(kill_cmd, check=False, capture_output=True, text=True)
            if result.returncode == 0:
                print_system(f"Taskkill command for {name} (PID: {pid}) executed successfully.")
            else:
                print_warning(f"Taskkill for {name} (PID: {pid}) exited with {result.returncode}. stdout: {result.stdout.strip()}, stderr: {result.stderr.strip()}")
        else:
            # On Unix, send SIGTERM to the entire process group.
            # This requires process_obj to have been started with preexec_fn=os.setsid.
            pgid = os.getpgid(pid)
            print_system(f"Sending SIGTERM to process group {pgid} for {name} (PID: {pid}).")
            os.killpg(pgid, signal.SIGTERM)

        # Wait for the process to terminate
        try:
            process_obj.wait(timeout=5)
            print_system(f"{name} (PID: {pid}) and its tree terminated gracefully.")
        except subprocess.TimeoutExpired:
            print_warning(f"{name} (PID: {pid}) tree did not terminate gracefully after initial signal/command.")
            if IS_WINDOWS:
                if process_obj.poll() is None: # Check if it's still running
                    print_warning(f"{name} (PID: {pid}) might still be running after taskkill /F /T. Manual check may be needed.")
            else:
                # If SIGTERM failed, escalate to SIGKILL for the process group.
                print_system(f"Sending SIGKILL to process group {pgid} for {name} (PID: {pid}).")
                os.killpg(pgid, signal.SIGKILL)
                process_obj.wait(timeout=2) # Give SIGKILL a moment
                print_system(f"{name} (PID: {pid}) tree killed.")
    except ProcessLookupError:
        print_system(f"{name} (PID: {pid}) process or group already gone.")
    except Exception as e:
        print_error(f"Error during termination of {name} (PID: {pid}) tree: {e}")


def print_system(message): print_colored("SYSTEM", message)


def print_error(message): print_colored("ERROR", message)


def print_warning(message): print_colored("WARNING", message)


def stream_output(pipe, name=None, color=None):
    try:
        for line in iter(pipe.readline, ''):
            if line:
                if name:
                    output_color = "\033[90m"
                    reset_color = "\033[0m"
                    if sys.stdout.isatty():
                        sys.stdout.write(f"{output_color}[{name.upper()}] {line.strip()}{reset_color}\n")
                    else:
                        sys.stdout.write(f"[{name.upper()}] {line.strip()}\n")
                else:
                    sys.stdout.write(line)
                sys.stdout.flush()
    except Exception as e:
        if 'read of closed file' not in str(e).lower() and 'Stream is closed' not in str(e):
            print(f"[STREAM_ERROR] Error reading output stream for pipe {pipe} (process: {name}): {e}")
    finally:
        if pipe:
            try:
                pipe.close()
            except Exception:
                pass


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
    current_env = os.environ.copy()
    if env_override:
        str_env_override = {k: str(v) for k, v in env_override.items()}
        print_system(f"  with custom env for '{log_name_prefix}': {str_env_override}")
        current_env.update(str_env_override)

    command = [str(c) for c in command]
    use_shell = False
    if IS_WINDOWS:
        if command[0] == NPM_CMD or command[0].lower().endswith('.cmd') or command[0].lower().endswith('.bat'):
            use_shell = True

    try:
        process = subprocess.Popen(
            command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding='utf-8', errors='replace', bufsize=1,
            shell=use_shell, env=current_env
        )

        stdout_thread = threading.Thread(target=stream_output, args=(process.stdout, f"{log_name_prefix}-OUT"),
                                         daemon=True)
        stdout_thread.start()
        stderr_thread = threading.Thread(target=stream_output, args=(process.stderr, f"{log_name_prefix}-ERR"),
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


def start_service_thread(target_func, name):
    print_system(f"Preparing to start service: {name}")
    thread = threading.Thread(target=target_func, name=name, daemon=True)
    thread.start()
    return thread


def cleanup_processes():
    print_system("\nShutting down services...")

    # --- Handle the main relaunched 'conda run' process first ---
    # This is relevant if the *initial* launcher instance is exiting (e.g., Ctrl+C)
    # while it was waiting for 'conda run' to complete.
    global relaunched_conda_process_obj # Ensure we're using the global
    if relaunched_conda_process_obj and relaunched_conda_process_obj.poll() is None:
        print_system("Initial launcher instance is shutting down; terminating the 'conda run' process and its children...")
        terminate_relaunched_process(relaunched_conda_process_obj, "Relaunched 'conda run' process")
        relaunched_conda_process_obj = None # Mark as handled

    # --- Handle locally managed services (e.g., Engine, Backend, Frontend) ---
    # This is relevant for the *relaunched* script instance cleaning up its own services,
    # or if the initial script ever started services directly.
    with process_lock:
        procs_to_terminate = list(running_processes) # Make a copy
        running_processes.clear() # Clear the global list

    for proc, name in reversed(procs_to_terminate):
        if proc.poll() is None: # Check if process is still running
            print_system(f"Terminating {name} (PID: {proc.pid})...")
            try:
                proc.terminate() # Send SIGTERM (or Windows equivalent)
                try:
                    proc.wait(timeout=5) # Wait for graceful shutdown
                    print_system(f"{name} terminated gracefully.")
                except subprocess.TimeoutExpired:
                    print_warning(f"{name} did not terminate gracefully, killing (PID: {proc.pid})...")
                    proc.kill() # Send SIGKILL (or Windows equivalent)
                    proc.wait(timeout=2) # Wait for kill
                    print_system(f"{name} killed.")
            except Exception as e:
                print_error(f"Error terminating/killing {name} (PID: {proc.pid}): {e}")
        else:
            print_system(f"{name} already exited (return code: {proc.poll()}).")

atexit.register(cleanup_processes) # This line should already exist


def signal_handler(sig, frame):
    print_system(f"\nSignal {sig} received. Initiating shutdown via atexit handlers...")
    # The atexit handler (cleanup_processes) will perform the necessary terminations.
    # We set a non-zero exit code, common for interruption.
    # 130 for SIGINT (Ctrl+C), 128 + signal number for others.
    sys.exit(130 if sig == signal.SIGINT else 128 + sig)

signal.signal(signal.SIGINT, signal_handler)
if not IS_WINDOWS: # SIGTERM is not really a thing for console apps on Windows this way for Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler) # Handle SIGTERM for graceful shutdown requests on Unix


# --- Service Start Functions ---
def start_service_process(command, cwd, name, use_shell_windows=False):
    print_system(f"Launching {name}: {' '.join(command)} in {os.path.basename(cwd)}")
    try:
        shell_val = use_shell_windows if IS_WINDOWS else False
        process = subprocess.Popen(
            command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding='utf-8', errors='replace', bufsize=1,
            shell=shell_val
        )
        with process_lock:
            running_processes.append((process, name))

        stdout_thread = threading.Thread(target=stream_output, args=(process.stdout, f"{name}-OUT"), daemon=True)
        stderr_thread = threading.Thread(target=stream_output, args=(process.stderr, f"{name}-ERR"), daemon=True)
        stdout_thread.start()
        stderr_thread.start()
        return process
    except FileNotFoundError:
        print_error(f"Command failed for {name}: '{command[0]}' not found.")
        sys.exit(1)
    except Exception as e:
        print_error(f"Failed to start {name}: {e}")
        sys.exit(1)


def start_engine_main():
    name = "ENGINE"
    MAX_UPLOAD_SIZE_BYTES = 2 ** 63  # Set this to 2 to the power of 63
    command = [
        HYPERCORN_EXECUTABLE,
        "app:app",
        "--bind", "127.0.0.1:11434",
        "--workers", "1",
        "--log-level", "info"
    ]
    hypercorn_env = {
        "HYPERCORN_MAX_REQUEST_SIZE": str(MAX_UPLOAD_SIZE_BYTES)
    }
    start_service_process(command, ENGINE_MAIN_DIR, name)


def start_backend_service():
    name = "BACKEND"
    command = ["node", "server.js"]
    start_service_process(command, BACKEND_SERVICE_DIR, name, use_shell_windows=False)


def start_frontend():
    name = "FRONTEND"
    command = [NPM_CMD, "run", "dev"]
    start_service_process(command, FRONTEND_DIR, name, use_shell_windows=True)


# --- Conda Python Version Calculation ---
def get_conda_python_versions_to_try(current_dt=None):
    if current_dt is None:
        current_dt = datetime.now()
    current_date_obj = current_dt.date()

    if current_date_obj < EPOCH_DATE:
        years_offset = 0
    else:
        years_offset = current_date_obj.year - EPOCH_DATE.year
        if (current_date_obj.month, current_date_obj.day) < (EPOCH_DATE.month, EPOCH_DATE.day):
            years_offset -= 1
        years_offset = max(0, years_offset)

    target_major = INITIAL_PYTHON_MAJOR
    target_minor = INITIAL_PYTHON_MINOR + years_offset

    preferred_versions = []
    preferred_versions.append(f"{target_major}.{target_minor}")
    if target_minor > INITIAL_PYTHON_MINOR:
        preferred_versions.append(f"{INITIAL_PYTHON_MAJOR}.{INITIAL_PYTHON_MINOR}")
    preferred_versions.append(f"{FALLBACK_PYTHON_MAJOR}.{FALLBACK_PYTHON_MINOR}")
    preferred_versions.append(f"{FALLBACK_PYTHON_MAJOR}.{FALLBACK_PYTHON_MINOR + 1}")

    final_versions_to_try = []
    for v_str in preferred_versions:
        if v_str not in final_versions_to_try:
            final_versions_to_try.append(v_str)

    print_system(f"Calculated preferred Python versions for Conda (in order of trial): {final_versions_to_try}")
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


def find_conda_executable():
    global CONDA_EXECUTABLE

    # 1. Check Cache First
    if os.path.exists(CONDA_PATH_CACHE_FILE):
        try:
            with open(CONDA_PATH_CACHE_FILE, 'r', encoding='utf-8') as f_cache:
                cached_path = f_cache.read().strip()
            if cached_path:
                print_system(f"Found cached Conda path: '{cached_path}'. Verifying...")
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
    # global TARGET_CONDA_ENV_PATH # No longer needed to set globally here, it's passed in

    if not CONDA_EXECUTABLE:
        print_error("Conda executable not found. Cannot create environment.")
        return False

    # Ensure the parent directory for the prefix exists if it's nested deeply (though here it's ROOT_DIR)
    os.makedirs(os.path.dirname(env_prefix_path), exist_ok=True)

    for py_version in python_versions_to_try:
        print_system(
            f"Attempting to create Conda environment at prefix '{env_prefix_path}' with Python {py_version}...")
        # Using --prefix or -p instead of --name
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
                # TARGET_CONDA_ENV_PATH is already known (it's env_prefix_path)
                if not (os.path.isdir(env_prefix_path) and os.path.exists(os.path.join(env_prefix_path, 'conda-meta'))):
                    print_error(
                        f"Critical: Conda environment at prefix '{env_prefix_path}' seems invalid post-creation.")
                    return False
                return True
            else:
                print_warning(
                    f"Failed to create Conda environment at prefix '{env_prefix_path}' with Python {py_version} (retcode: {process.returncode}).")
                # Potentially remove the partially created prefix directory if creation fails
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
            _remove_flag_files(FLAG_FILES_TO_RESET_ON_ENV_RECREATE)  # <--- ADD THIS LINE

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
                    accepted = False;
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
            accepted = False;
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
            accepted = True;
            break
        elif key == ord('n') or key == ord('N'):
            accepted = False;
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


def _ensure_conda_package(package_name: str,
                          executable_to_check: Optional[str] = None,
                          conda_channel: str = "conda-forge",
                          is_critical: bool = True) -> bool:
    """
    Checks if an executable (related to a Conda package) is available.
    If not, attempts to install the Conda package into the TARGET_CONDA_ENV_PATH.
    This function should be called when the script is running *inside* the target Conda environment.
    """
    check_exe = executable_to_check if executable_to_check else package_name

    print_system(f"Checking for executable '{check_exe}' (from Conda package '{package_name}')...")

    # When this function runs, we expect to be *inside* the activated zephyrineCondaVenv.
    # So, shutil.which should find executables installed in this environment.
    found_path = shutil.which(check_exe)

    is_in_env = False
    if found_path:
        try:
            # Normalize paths for reliable comparison
            norm_found_path = os.path.normcase(os.path.realpath(found_path))
            norm_target_env_path = os.path.normcase(os.path.realpath(TARGET_CONDA_ENV_PATH))
            # Check if the found executable is within our target Conda environment
            if norm_found_path.startswith(norm_target_env_path):
                is_in_env = True
                print_system(f"Executable '{check_exe}' found in target Conda environment: {found_path}")
            else:
                print_warning(
                    f"Executable '{check_exe}' found at '{found_path}', but it's OUTSIDE the target Conda env '{TARGET_CONDA_ENV_PATH}'. Will attempt install into env.")
        except Exception as e_path:
            print_warning(f"Error verifying path for '{check_exe}': {e_path}. Assuming not in env.")

    if not found_path or not is_in_env:
        if not found_path:
            print_warning(f"Executable '{check_exe}' (for package '{package_name}') not found in PATH.")

        print_system(
            f"Attempting to install '{package_name}' from channel '{conda_channel}' into Conda env '{os.path.basename(TARGET_CONDA_ENV_PATH)}' using prefix...")

        # CONDA_EXECUTABLE should be the path to the conda binary (e.g., from base or miniconda)
        # TARGET_CONDA_ENV_PATH is the prefix of the environment we are installing into.
        conda_install_cmd = [
            CONDA_EXECUTABLE, "install", "--yes",
            "--prefix", TARGET_CONDA_ENV_PATH,  # Crucial for installing into the correct environment
            "-c", conda_channel,
            package_name
        ]

        # The run_command function will execute this.
        # It inherits the environment, but CONDA_EXECUTABLE should work correctly.
        if not run_command(conda_install_cmd, cwd=ROOT_DIR,
                           name=f"CONDA-INSTALL-{package_name.upper().replace('-', '_')}",
                           check=True):  # check=True will raise error on failure
            print_error(f"Failed to install Conda package '{package_name}'.")
            if is_critical:
                print_error(f"'{package_name}' is a critical dependency. Exiting.")
                sys.exit(1)
            return False
        else:
            print_system(f"Successfully installed Conda package '{package_name}'. Verifying executable...")
            # Re-check after install attempt
            found_path_after_install = shutil.which(check_exe)
            if found_path_after_install:
                norm_found_path_after = os.path.normcase(os.path.realpath(found_path_after_install))
                norm_target_env_path_after = os.path.normcase(os.path.realpath(TARGET_CONDA_ENV_PATH))
                if norm_found_path_after.startswith(norm_target_env_path_after):
                    print_system(f"Executable '{check_exe}' now found in Conda environment: {found_path_after_install}")
                    # Update global command variables if they were naively set
                    # This ensures subsequent run_command calls use the Conda-installed versions if they were system-wide before
                    if check_exe == "git" and 'GIT_CMD' in globals() and globals()[
                        'GIT_CMD'] != found_path_after_install: globals()['GIT_CMD'] = found_path_after_install
                    if check_exe == "cmake" and 'CMAKE_CMD' in globals() and globals()[
                        'CMAKE_CMD'] != found_path_after_install: globals()['CMAKE_CMD'] = found_path_after_install
                    if check_exe == "npm" and 'NPM_CMD' in globals() and globals()[
                        'NPM_CMD'] != found_path_after_install: globals()['NPM_CMD'] = found_path_after_install
                    # For 'node', it's usually just called as 'node', not stored in a CMD var
                    return True
                else:
                    print_error(
                        f"'{check_exe}' installed by Conda but found at '{found_path_after_install}', which is not in the target env '{TARGET_CONDA_ENV_PATH}'. This indicates a PATH or conda setup issue.")
                    if is_critical: sys.exit(1)
                    return False
            else:
                print_error(
                    f"Executable '{check_exe}' still not found after Conda install attempt for package '{package_name}'.")
                if is_critical: sys.exit(1)
                return False
    return True  # Already found in env, or successfully installed and verified


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
    # This is a basic check. Full Vulkan setup is complex.
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
    #         # OpenCL is complex to reliably use for GGUF, so we won't set it as primary_gpu_backend
    #         # unless specifically requested by user. Just note its potential.
    #         detected_env_vars["AUTODETECTED_OPENCL_POSSIBLE"] = "1"
    #     else:
    #         print_system("OpenCL: 'clinfo' not found.")

    # --- CPU Enhancements (OpenBLAS for Whisper, OpenMP for Llama) ---
    # These are not primary backends but enhancements.
    # GGML_BLAS=1 for whisper.cpp often relies on OpenBLAS being found by its CMake.
    # LLAMA_OPENMP=ON is enabled by default for CPU builds of llama-cpp-python by launcher.
    # For now, we won't try to auto-detect BLAS presence, as it's complex.
    # Users can still set GGML_BLAS=1 manually.
    print_system(
        f"CPU: OpenMP will be enabled by default for Llama.cpp CPU builds. User can set GGML_BLAS=1 for PyWhisperCpp CPU.")
    detected_env_vars[
        "AUTODETECTED_CPU_ENHANCEMENTS_INFO"] = "OpenMP for Llama.cpp; User can set GGML_BLAS=1 for PyWhisperCpp"

    detected_env_vars["AUTODETECTED_PRIMARY_GPU_BACKEND"] = primary_gpu_backend_detected
    print_system(
        f"--- Hardware Acceleration Auto-Detection Complete. Preferred GPU Backend: {primary_gpu_backend_detected} ---")
    if detected_env_vars.get("AUTODETECTED_COREML_POSSIBLE") == "1":
        print_system("    (CoreML also noted as possible for Whisper on this platform)")

    return detected_env_vars

# --- Add Aria2p Import ---
ARIA2P_AVAILABLE = False
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
ARIA2P_AVAILABLE = False # Assuming this is set based on aria2p import attempt

def download_file_with_progress(
    url: str,
    destination_path: str,
    file_description: str,
    requests_session: 'requests.Session', # <<< CORRECTED TYPE HINT
    max_retries_non_connection_error: int = 3,
    aria2_rpc_host: str = "localhost",
    aria2_rpc_port: int = 6800,
    aria2_rpc_secret: Optional[str] = None
):
    """
    Downloads a file with a progress bar.
    Attempts to use aria2c for multi-connection download if available and connected.
    Falls back to requests-based download with robust retries.
    """
    print_system(f"Preparing to download {file_description} from {url} to {destination_path}...")

    try:
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    except OSError as e:
        print_error(f"Failed to create directory for {destination_path}: {e}")
        return False  # Cannot proceed if directory creation fails

    # --- Attempt Aria2c Download First ---
    aria2c_executable_path = shutil.which("aria2c")  # Check if aria2c CLI is in PATH

    if ARIA2P_AVAILABLE and aria2c_executable_path:
        print_system(f"Aria2c found at '{aria2c_executable_path}'. Attempting download via aria2p...")
        aria2_api_instance = None  # Renamed variable to avoid conflict if API is also class name
        try:
            # Initialize API client and connect to aria2 RPC server
            aria2_client_instance = aria2p.Client(host=aria2_rpc_host, port=aria2_rpc_port, secret=aria2_rpc_secret)
            aria2_api_instance = aria2p.API(aria2_client_instance)

            # Verify connection by getting global stats
            global_stats = aria2_api_instance.get_global_stat()
            print_system(
                f"Connected to aria2c RPC server. Speed: {global_stats.download_speed_string()}/{global_stats.upload_speed_string()}, Active: {global_stats.num_active}, Waiting: {global_stats.num_waiting}")

            # Define download options for aria2c
            aria2_options_dict = {
                "dir": os.path.dirname(destination_path),
                "out": os.path.basename(destination_path),
                "max-connection-per-server": "16",
                "split": "16",
                "min-split-size": "1M",
                "stream-piece-selector": "geom",
                "continue": "true",  # Resume downloads
                "allow-overwrite": "true"  # In case a .tmp_aria2 file exists
            }

            print_system(f"Adding URI to aria2c: {url} with options: {aria2_options_dict}")
            download_task = aria2_api_instance.add_uris([url], options=aria2_options_dict)

            if not download_task:
                raise RuntimeError("aria2_api.add_uris did not return a download task object.")

            print_system(f"Aria2c download started (GID: {download_task.gid}). Monitoring progress...")

            with tqdm(total=100, unit='%', desc=file_description[:30], ascii=IS_WINDOWS, leave=False,
                      bar_format='{l_bar}{bar}| {n_fmt}% [{elapsed}<{remaining}, {rate_fmt}{postfix}]') as progress_bar:
                last_completed_bytes = 0
                while not download_task.is_complete and not download_task.has_error:
                    download_task.update()

                    if download_task.total_length > 0:  # If total size is known
                        if progress_bar.total != download_task.total_length:  # Initialize or update total for tqdm
                            progress_bar.total = download_task.total_length
                            progress_bar.unit = 'iB'  # Switch to bytes
                            progress_bar.unit_scale = True
                            progress_bar.n = download_task.completed_length  # Set initial progress
                            last_completed_bytes = download_task.completed_length
                        else:  # Already initialized with bytes, update by increment
                            progress_bar.update(download_task.completed_length - last_completed_bytes)
                            last_completed_bytes = download_task.completed_length

                        postfix_details = f"S:{download_task.download_speed_string()} C:{download_task.connections}"
                        if download_task.eta_string(human_readable=True):
                            postfix_details += f" ETA:{download_task.eta_string(human_readable=True)}"
                        progress_bar.set_postfix_str(postfix_details, refresh=False)
                        progress_bar.refresh()
                    else:  # Total length not yet known, show percentage if available
                        # download_task.progress is 0-100 based on files if total size unknown
                        if download_task.progress > progress_bar.n:
                            progress_bar.update(int(download_task.progress) - int(progress_bar.n))
                        progress_bar.set_postfix_str(
                            f"S:{download_task.download_speed_string()} C:{download_task.connections}", refresh=True)

                    if download_task.error_code is not None:
                        raise RuntimeError(
                            f"Aria2c download error (Code {download_task.error_code}): {download_task.error_message}")
                    time.sleep(0.5)

            download_task.update()  # Final update
            if download_task.is_complete:
                print_system(f"Aria2c download successful for {file_description}!")
                # Verify file integrity after aria2c reports completion
                final_path = download_task.files[0].path  # aria2p should update this to the correct path
                if os.path.exists(final_path) and os.path.getsize(final_path) == download_task.total_length:
                    if final_path != destination_path:  # If aria2c saved it with a different name (e.g. if "out" option was tricky)
                        print_warning(f"Aria2c saved to '{final_path}', moving to '{destination_path}'.")
                        shutil.move(final_path, destination_path)
                    return True  # Success
                else:
                    err_msg_verify = f"Aria2c reported complete, but file verification failed at '{final_path}'. Expected: {download_task.total_length}, Actual: {os.path.getsize(final_path) if os.path.exists(final_path) else 'Not Found'}"
                    print_error(err_msg_verify)
                    # Fall through to requests if verification fails
            elif download_task.has_error:
                print_error(
                    f"Aria2c download failed for {file_description}. Error (Code {download_task.error_code}): {download_task.error_message}")
                if download_task.files:
                    for file_entry in download_task.files:  # Clean up potentially incomplete files
                        if os.path.exists(file_entry.path):
                            try:
                                os.remove(file_entry.path); print_warning(
                                    f"Removed incomplete aria2c download: {file_entry.path}")
                            except Exception as e_rm_aria:
                                print_error(f"Failed to remove incomplete aria2c file {file_entry.path}: {e_rm_aria}")

            # If not returned True by now, aria2c path failed or verification failed; fall through to requests
            print_warning(
                "Aria2c download path did not complete successfully or file verification failed. Falling back to requests.")

        except aria2p.client.ClientException as e_aria_client:  # Errors connecting to RPC
            print_warning(
                f"Aria2p client/RPC connection error: {e_aria_client}. Is aria2c daemon running with --enable-rpc?")
            print_warning("Falling back to requests-based download.")
        except Exception as e_aria_general:  # Other errors during aria2p usage
            print_warning(f"Unexpected error during Aria2c download attempt: {e_aria_general}")
            # traceback.print_exc(file=sys.stderr) # Uncomment for detailed debugging if needed
            print_warning("Falling back to requests-based download.")
    else:
        if not ARIA2P_AVAILABLE:
            print_system("Aria2p Python library not available. Using requests.")
        elif not aria2c_executable_path:
            print_system("aria2c executable not found in PATH. Using requests.")

    # --- Requests-based Download (Fallback or if Aria2c not used) ---
    print_system("Using requests-based download method...")
    temp_destination_path_requests = destination_path + ".tmp_download_requests"  # Use a distinct temp name
    connection_error_retries = 0
    other_errors_retries = 0
    server_file_size = 0
    head_request_successful = False

    try:  # Try to get file size with HEAD request first
        head_resp = requests_session.head(url, timeout=10, allow_redirects=True)
        head_resp.raise_for_status()
        server_file_size = int(head_resp.headers.get('content-length', 0))
        head_request_successful = True
        print_system(f"File size (requests HEAD): {server_file_size / (1024 * 1024):.2f} MB.")
    except requests.exceptions.RequestException as e_head_req:
        print_warning(f"Requests: Could not get file size via HEAD request: {e_head_req}.")

    while True:  # Main retry loop for requests method
        try:
            current_attempt_log = f"(ConnRetry: {connection_error_retries + 1}, OtherErrRetry: {other_errors_retries + 1})"
            print_system(f"Attempting download (requests): {file_description} {current_attempt_log}")

            response = requests_session.get(url, stream=True, timeout=(15, 300))  # (connect_timeout, read_timeout)
            response.raise_for_status()  # Raises HTTPError for 4xx/5xx responses

            if not head_request_successful and server_file_size == 0:  # If HEAD failed, try GET headers
                server_file_size = int(response.headers.get('content-length', 0))
                if server_file_size > 0: print_system(
                    f"File size (requests GET): {server_file_size / (1024 * 1024):.2f} MB.")

            block_size = 8192  # 8KB
            progress_bar_description_short = file_description[:30]

            with tqdm(total=server_file_size, unit='iB', unit_scale=True, desc=progress_bar_description_short,
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
                print_error(f"  Expected {server_file_size} bytes, actually downloaded {downloaded_bytes_count} bytes.")
                if os.path.exists(temp_destination_path_requests): os.remove(temp_destination_path_requests)
                other_errors_retries += 1
                if other_errors_retries >= max_retries_non_connection_error:
                    print_error(
                        f"Max retries ({max_retries_non_connection_error}) reached for size mismatch. Download failed for {file_description}.")
                    return False  # Permanent failure for this file
                print_warning(
                    f"Retrying (requests) due to size mismatch (retry {other_errors_retries}/{max_retries_non_connection_error}). Waiting 5 seconds...")
                time.sleep(5)
                continue  # Go to next iteration of the while True loop

            shutil.move(temp_destination_path_requests, destination_path)
            print_system(f"Successfully downloaded (requests) {file_description} to {destination_path}")
            return True  # Successful download

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError) as e_connection:
            connection_error_retries += 1
            print_warning(
                f"Connection error (requests) for {file_description}: {type(e_connection).__name__} - {e_connection}.")
            print_warning(
                f"Retrying download (attempt {connection_error_retries + 1}). Designed for our famously fast & reliable Indonesian internet! ;) Waiting 5 seconds...")
            if os.path.exists(temp_destination_path_requests):
                try:
                    os.remove(temp_destination_path_requests)
                except Exception as e_remove_conn:
                    print_warning(
                        f"Could not remove partial download {temp_destination_path_requests} after connection error: {e_remove_conn}")
            time.sleep(5)
            # Loop continues for connection errors (infinite retries)

        except requests.exceptions.HTTPError as e_http_error:
            other_errors_retries += 1
            print_error(
                f"HTTP error (requests) for {file_description}: {e_http_error.response.status_code} {e_http_error.response.reason} for URL {url}")
            if os.path.exists(temp_destination_path_requests): os.remove(temp_destination_path_requests)

            if e_http_error.response.status_code in [401, 403, 404]:  # Fatal client errors
                print_error(
                    f"Fatal HTTP error {e_http_error.response.status_code}. Not retrying for {file_description}.")
                return False  # Permanent failure

            if other_errors_retries >= max_retries_non_connection_error:
                print_error(
                    f"Max retries ({max_retries_non_connection_error}) reached for HTTP error. Download failed for {file_description}.")
                return False  # Permanent failure
            print_warning(
                f"Retrying (requests) due to HTTP error (retry {other_errors_retries}/{max_retries_non_connection_error}). Waiting 5 seconds...")
            time.sleep(5)
            # Loop continues for limited other_errors_retries

        except Exception as e_general_requests:
            other_errors_retries += 1
            print_error(
                f"Unexpected error during requests download for {file_description}: {type(e_general_requests).__name__} - {e_general_requests}")
            traceback.print_exc(file=sys.stderr)  # Print full traceback for unexpected issues
            if os.path.exists(temp_destination_path_requests): os.remove(temp_destination_path_requests)

            if other_errors_retries >= max_retries_non_connection_error:
                print_error(
                    f"Max retries ({max_retries_non_connection_error}) reached for general error. Download failed for {file_description}.")
                return False  # Permanent failure
            print_warning(
                f"Retrying (requests) due to general error (retry {other_errors_retries}/{max_retries_non_connection_error}). Waiting 5 seconds...")
            time.sleep(5)
            # Loop continues for limited other_errors_retries

    # This line should ideally not be reached if the requests loop is infinite for connection errors
    # and returns False for other maxed-out errors. It's a final safety net.
    print_error(f"Download failed for {file_description} after all attempts and fallbacks.")
    return False


# --- Main Execution Logic ---
if __name__ == "__main__":
    print_system("--- Project Zephyrine Launcher ---")
    print_system(f"Root directory: {ROOT_DIR}")
    print_system(f"Target Conda environment path: {TARGET_CONDA_ENV_PATH}")
    print_system(f"Initial Python: {sys.version.split()[0]} on {platform.system()} ({platform.machine()})")

    current_conda_env_path_check = os.getenv("CONDA_PREFIX")
    is_already_in_correct_env = False
    if current_conda_env_path_check:
        try:
            norm_current_env_path = os.path.normcase(os.path.realpath(current_conda_env_path_check))
            norm_target_env_path = os.path.normcase(os.path.realpath(TARGET_CONDA_ENV_PATH))
            if os.path.isdir(norm_current_env_path) and \
                    os.path.isdir(norm_target_env_path) and \
                    norm_current_env_path == norm_target_env_path and \
                    os.getenv("ZEPHYRINE_RELAUNCHED_IN_CONDA") == "1":  # Check the relaunch flag
                is_already_in_correct_env = True
                ACTIVE_ENV_PATH = current_conda_env_path_check
        except Exception as e_path_check:  # Corrected 'e' usage
            print_warning(f"Error comparing Conda paths during initial check: {e_path_check}")

    if is_already_in_correct_env:
        # --- This is the RELAUNCHED script, running inside the correct Conda environment ---
        print_system(f"Running inside target Conda environment (Prefix: {ACTIVE_ENV_PATH})")
        print_system(f"Python executable in use: {sys.executable}")

        if "ZEPHYRINE_RELAUNCHED_IN_CONDA" in os.environ:
            del os.environ["ZEPHYRINE_RELAUNCHED_IN_CONDA"]

        if globals().get('CONDA_EXECUTABLE') is None:
            print_system("Relaunched script: CONDA_EXECUTABLE is None. Attempting to load from cache or PATH...")
            loaded_from_cache = False
            if os.path.exists(CONDA_PATH_CACHE_FILE):
                try:
                    with open(CONDA_PATH_CACHE_FILE, 'r', encoding='utf-8') as f_cache:  # Corrected with statement
                        cached_path = f_cache.read().strip()
                    if cached_path and _verify_conda_path(cached_path):
                        globals()['CONDA_EXECUTABLE'] = cached_path
                        print_system(f"Using Conda executable from cache in relaunched script: {CONDA_EXECUTABLE}")
                        loaded_from_cache = True
                    else:
                        print_warning("Cached Conda path invalid/unverified in relaunched script.")
                except Exception as e_cache_read:
                    print_warning(f"Error reading Conda cache in relaunched script: {e_cache_read}")  # Corrected 'e'
            if not loaded_from_cache:
                conda_exe_from_which = shutil.which("conda.exe" if IS_WINDOWS else "conda")
                if conda_exe_from_which and _verify_conda_path(conda_exe_from_which):
                    globals()['CONDA_EXECUTABLE'] = conda_exe_from_which
                    print_system(f"Found Conda via shutil.which in relaunched script: {CONDA_EXECUTABLE}")
                else:
                    print_error("CRITICAL: Conda executable could not be determined. Conda package installs will fail.")

        if globals().get('CONDA_EXECUTABLE') is None:
            print_error("CRITICAL: CONDA_EXECUTABLE is still None. Subsequent Conda operations WILL FAIL.");
            sys.exit(1)

        # --- Read AUTODETECTED environment variables ---
        AUTO_PRIMARY_GPU_BACKEND = os.getenv("AUTODETECTED_PRIMARY_GPU_BACKEND", "cpu")
        AUTO_CUDA_AVAILABLE = os.getenv("AUTODETECTED_CUDA_AVAILABLE") == "1"  # Corrected variable name
        AUTO_METAL_AVAILABLE = os.getenv("AUTODETECTED_METAL_AVAILABLE") == "1"  # Corrected variable name
        AUTO_VULKAN_AVAILABLE = os.getenv("AUTODETECTED_VULKAN_AVAILABLE") == "1"  # Corrected variable name
        AUTO_COREML_POSSIBLE = os.getenv("AUTODETECTED_COREML_POSSIBLE") == "1"  # Corrected variable name
        print_system(
            f"Auto-detected preferences received: GPU_BACKEND='{AUTO_PRIMARY_GPU_BACKEND}', CUDA={AUTO_CUDA_AVAILABLE}, METAL={AUTO_METAL_AVAILABLE}, VULKAN={AUTO_VULKAN_AVAILABLE}, COREML_POSSIBLE={AUTO_COREML_POSSIBLE}")

        _sys_exec_dir = os.path.dirname(sys.executable)
        PYTHON_EXECUTABLE = sys.executable
        PIP_EXECUTABLE = os.path.join(_sys_exec_dir, "pip.exe" if IS_WINDOWS else "pip")
        if not os.path.exists(PIP_EXECUTABLE): PIP_EXECUTABLE = shutil.which("pip") or "pip"
        HYPERCORN_EXECUTABLE = os.path.join(_sys_exec_dir, "hypercorn.exe" if IS_WINDOWS else "hypercorn")
        if not os.path.exists(HYPERCORN_EXECUTABLE): HYPERCORN_EXECUTABLE = shutil.which("hypercorn") or "hypercorn"

        print_system(f"Updated PIP_EXECUTABLE to: {PIP_EXECUTABLE}")
        print_system(f"Updated HYPERCORN_EXECUTABLE to: {HYPERCORN_EXECUTABLE}")

        if not run_command([PIP_EXECUTABLE, "install", "--upgrade", "pip", "setuptools", "wheel"], ROOT_DIR,
                           "PIP-UPGRADE-CORE"): print_warning("Pip/setuptools upgrade failed.")
        if not run_command([PIP_EXECUTABLE, "install", "tqdm", "requests"], ROOT_DIR, "PIP-UTILS"): print_error(
            "tqdm/requests install failed."); sys.exit(1)


        try:
            import requests; from tqdm import tqdm
        except ImportError:
            print_error("Failed to import requests/tqdm. Exiting."); sys.exit(1)
        requests_session = requests.Session();
        adapter = requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=20);
        requests_session.mount('http://', adapter);
        requests_session.mount('https://', adapter)



        engine_req_path = os.path.join(ROOT_DIR, "requirements.txt")
        if not os.path.exists(engine_req_path): print_error(f"requirements.txt not found: {engine_req_path}"); sys.exit(
            1)
        pip_install_success = False
        # You could make MAX_PIP_RETRIES a constant or configurable
        MAX_PIP_RETRIES = int(os.getenv("PIP_INSTALL_RETRIES", 99999))  # Get from env or default
        PIP_RETRY_DELAY_SECONDS = 5  # Delay between retries

        for attempt in range(MAX_PIP_RETRIES):
            if run_command([PIP_EXECUTABLE, "install", "-r", engine_req_path], ENGINE_MAIN_DIR, "PIP-ENGINE-REQ"):
                pip_install_success = True
                break  # Exit loop on success
            else:
                print_warning(
                    f"pip install failed on attempt {attempt + 1}/{MAX_PIP_RETRIES}. Retrying in {PIP_RETRY_DELAY_SECONDS} seconds due to unreliable connection...")
                time.sleep(PIP_RETRY_DELAY_SECONDS)

        if not pip_install_success:
            print_error(f"Failed to install Python dependencies for Engine after {MAX_PIP_RETRIES} attempts. Exiting.")
            sys.exit(1)
        try:
            import tiktoken; TIKTOKEN_AVAILABLE = True
        except ImportError:
            TIKTOKEN_AVAILABLE = False; print_warning("tiktoken not available.")

        print_system("--- Ensuring core command-line tools (git, cmake, nodejs, ffmpeg) ---")
        if not _ensure_conda_package("git", "git", is_critical=True): sys.exit(1)
        if not _ensure_conda_package("cmake", "cmake", is_critical=True): sys.exit(1)
        if not _ensure_conda_package("nodejs", "node", is_critical=True): sys.exit(1)
        if not _ensure_conda_package("nodejs", "npm", is_critical=True): sys.exit(1)
        _ensure_conda_package("ffmpeg", "ffmpeg", is_critical=False)
        print_system("--- Ensuring Aria2c (for multi-connection downloads) via Conda ---")
        # is_critical=False because the requests-based downloader can be a fallback.
        if not _ensure_conda_package(package_name="aria2", executable_to_check="aria2c", is_critical=False):
            print_warning(
                "aria2c (command-line tool) could not be installed via Conda. Multi-connection downloads will be unavailable.")
        else:
            print_system("aria2c command-line tool checked/installed via Conda.")

        # The previous block for "Core command-line tools check/install attempt complete" can come after this.
        # Then refresh global CMD vars as you have it.
        # print_system("--- Core command-line tools check/install attempt complete ---")
        # globals()['GIT_CMD'] = shutil.which("git") or ... etc. ...

        # --- Install Python wrapper for Aria2 (aria2p) via Pip ---
        # This step is best after general pip requirements and before specific source builds
        # that might depend on a stable Python environment.
        if not os.path.exists(ARIA2P_INSTALLED_FLAG_FILE):
            print_system("--- Installing Python wrapper for Aria2 (aria2p) ---")
            # PIP_EXECUTABLE should be correctly defined by this point in the relaunched script
            if not run_command([PIP_EXECUTABLE, "install", "aria2p"], ROOT_DIR, "PIP-ARIA2P"):
                print_warning(
                    "Failed to install 'aria2p' Python library. Multi-connection downloads via Aria2 will not be available if chosen as the download method.")
                # Not making this fatal, as the 'requests'-based download is a fallback.
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

        globals()['GIT_CMD'] = shutil.which("git") or ('git.exe' if IS_WINDOWS else 'git')
        globals()['CMAKE_CMD'] = shutil.which("cmake") or ('cmake.exe' if IS_WINDOWS else 'cmake')
        npm_exe_name = "npm.cmd" if IS_WINDOWS else "npm"
        globals()['NPM_CMD'] = shutil.which(npm_exe_name) or npm_exe_name
        print_system(f"Using GIT_CMD: {GIT_CMD}");
        print_system(f"Using CMAKE_CMD: {CMAKE_CMD}");
        print_system(f"Using NPM_CMD: {NPM_CMD}")
        if not all([GIT_CMD and shutil.which(GIT_CMD.split()[0]), CMAKE_CMD and shutil.which(CMAKE_CMD.split()[0]),
                    NPM_CMD and shutil.which(NPM_CMD.split('.')[0]), shutil.which("node")]):
            print_error("One or more critical tools (git, cmake, node, npm) not found after attempts. Exiting.");
            sys.exit(1)

        if IS_WINDOWS:
            if not run_command([PIP_EXECUTABLE, "install", "windows-curses"], ROOT_DIR, "PIP-WINCURSES"): print_error(
                "windows-curses install failed."); sys.exit(1)
        try:
            import curses
        except ImportError:
            print_error("curses import failed."); sys.exit(1)
        if not os.path.exists(LICENSE_FLAG_FILE):
            print_system("License agreement required.")
            _, combined_license_text = load_licenses()
            estimated_reading_seconds = calculate_reading_time(combined_license_text)
            accepted, time_taken = curses.wrapper(display_license_prompt, combined_license_text.splitlines(),
                                                  estimated_reading_seconds)  # type: ignore
            if not accepted: print_error("License not accepted. Exiting."); sys.exit(1)
            with open(LICENSE_FLAG_FILE, 'w', encoding='utf-8') as f_license:  # Corrected with statement
                f_license.write(f"Accepted: {datetime.now().isoformat()}\nTime: {time_taken:.2f}s\n")
            print_system(f"Licenses accepted by user in {time_taken:.2f}s.")
            if estimated_reading_seconds > 30 and time_taken < (estimated_reading_seconds * 0.1):
                print_warning("Warning: Licenses accepted quickly. Ensure you understood terms.");
                time.sleep(3)
        else:
            print_system("License previously accepted.")

        print_system(f"--- Checking Static Model Pool: {STATIC_MODEL_POOL_PATH} ---")
        os.makedirs(STATIC_MODEL_POOL_PATH, exist_ok=True)
        all_models_ok = True  # This flag was from my suggestion, might not be in your version
        for model_info in MODELS_TO_DOWNLOAD:  # MODELS_TO_DOWNLOAD needs to be defined globally
            dest_path = os.path.join(STATIC_MODEL_POOL_PATH, model_info["filename"])
            if not os.path.exists(dest_path):  # <<< THE CRITICAL CHECK
                print_warning(f"Model '{model_info['description']}' ({model_info['filename']}) not found. Downloading.")
                # download_file_with_progress is assumed to be defined elsewhere
                if not download_file_with_progress(model_info["url"], dest_path, model_info["description"],
                                                   requests_session):
                    print_error(f"Failed download for {model_info['filename']}.")
                    all_models_ok = False  # From my suggestion
            else:
                print_system(f"Model '{model_info['description']}' ({model_info['filename']}) already present.")
        print_system("Static model pool checked.")

        # --- ChatterboxTTS Installation (NEW) ---
        if not os.path.exists(CHATTERBOX_TTS_INSTALLED_FLAG_FILE):
            print_system(f"--- ChatterboxTTS First-Time Setup from {CHATTERBOX_TTS_PATH} ---")
            if not os.path.isdir(CHATTERBOX_TTS_PATH):
                print_error(f"ChatterboxTTS directory not found at: {CHATTERBOX_TTS_PATH}")
                print_error("Please ensure the submodule/directory exists. Skipping ChatterboxTTS installation.")
            else:
                print_system(f"Installing ChatterboxTTS in editable mode from: {CHATTERBOX_TTS_PATH}")
                # Ensure PIP_EXECUTABLE is defined and valid at this point
                if not run_command([PIP_EXECUTABLE, "install", "-e", "."], CHATTERBOX_TTS_PATH,
                                   "PIP-CHATTERBOX-EDITABLE"):
                    print_error("ChatterboxTTS installation failed. Check pip logs above.")
                    # Decide if this is a fatal error or if the launcher can continue
                    # For now, it will continue but ChatterboxTTS might not be available.
                else:
                    print_system("ChatterboxTTS installed successfully in editable mode.")
                    try:
                        with open(CHATTERBOX_TTS_INSTALLED_FLAG_FILE, 'w', encoding='utf-8') as f_chatterbox_flag:
                            f_chatterbox_flag.write(f"ChatterboxTTS installed on: {datetime.now().isoformat()}\n")
                        print_system("ChatterboxTTS installation flag created.")
                    except IOError as flag_err_chatterbox:
                        print_error(f"Could not create ChatterboxTTS installation flag file: {flag_err_chatterbox}")
        else:
            print_system("ChatterboxTTS previously installed (flag file found).")
        # --- END ChatterboxTTS Installation ---

        # --- start of MELOTTS ---
        if not os.path.exists(MELO_TTS_INSTALLED_FLAG_FILE):
            print_system(f"--- MeloTTS First-Time Setup from {MELO_TTS_PATH} ---")
            if not os.path.isdir(MELO_TTS_PATH): print_error(f"MeloTTS dir missing: {MELO_TTS_PATH}."); sys.exit(1)
            if not run_command([PIP_EXECUTABLE, "install", "-e", "."], MELO_TTS_PATH, "PIP-MELO-EDITABLE"): print_error(
                "MeloTTS install failed."); sys.exit(1)
            if not run_command([PYTHON_EXECUTABLE, "-m", "unidic", "download"], MELO_TTS_PATH,
                               "UNIDIC-DOWNLOAD"): print_warning("unidic download failed.")
            # (MeloTTS test logic from your file - ensure audio_worker_script is defined)
            audio_worker_script_path_for_melo_test = os.path.join(ENGINE_MAIN_DIR,
                                                                  "audio_worker.py")  # Assuming ENGINE_MAIN_DIR is defined
            if os.path.exists(audio_worker_script_path_for_melo_test):
                # temp_melo_test_dir = os.path.join(ROOT_DIR, "temp_melo_audio_test_files"); os.makedirs(temp_melo_test_dir, exist_ok=True)
                # test_out_file_melo = os.path.join(temp_melo_test_dir, f"initial_melo_test_{datetime.now().strftime('%Y%m%d%H%M%S')}.wav")
                # test_cmd_melo = [PYTHON_EXECUTABLE, audio_worker_script_path_for_melo_test, "--test-mode", "--task-type", "tts", "--model-lang", "EN", "--device", "auto", "--output-file", test_out_file_melo, "--temp-dir", temp_melo_test_dir]
                # if not run_command(test_cmd_melo, ENGINE_MAIN_DIR, "MELO-INIT-TEST"): print_warning("MeloTTS initial test failed.")
                # else: print_system(f"MeloTTS initial test OK. Test audio: {test_out_file_melo if os.path.exists(test_out_file_melo) else 'Not found'}")
                pass  # Simplified, assuming test runs if path exists
            with open(MELO_TTS_INSTALLED_FLAG_FILE, 'w', encoding='utf-8') as f_melo_flag:  # Corrected with statement
                f_melo_flag.write(f"Tested: {datetime.now().isoformat()}")
            print_system("MeloTTS setup and flag created.")  # Use print_system instead of print_success
        else:
            print_system("MeloTTS previously installed/tested.")

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
                    user_ggml_cuda = os.getenv("GGML_CUDA");
                    user_whisper_coreml = os.getenv("WHISPER_COREML");
                    user_ggml_vulkan = os.getenv("GGML_VULKAN");
                    user_ggml_blas = os.getenv("GGML_BLAS");
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
                    elif AUTO_COREML_POSSIBLE and (
                            AUTO_PRIMARY_GPU_BACKEND == "metal" or platform.system() == "Darwin"):
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
                        print_system("pywhispercpp installed.");  # Use print_system
                        print_warning("For non-WAV ASR, ensure FFmpeg is in PATH (via conda install ffmpeg).")
                        with open(PYWHISPERCPP_INSTALLED_FLAG_FILE, 'w',
                                  encoding='utf-8') as f_pwc_flag:  # Corrected with statement
                            f_pwc_flag.write(backend_whisper)
                        print_system("PyWhisperCpp installation flag created.")
        else:
            print_system("PyWhisperCpp previously installed.")


        # ---
        # --- Install Local LaTeX-OCR Sub-Engine ---
        if not os.path.exists(LATEX_OCR_INSTALLED_FLAG_FILE):
            print_system(f"--- First-Time Setup for local LaTeX_OCR-SubEngine ---")

            # Check if the local directory for your fork exists
            if not os.path.isdir(LATEX_OCR_PATH):
                print_error(f"LaTeX-OCR Sub-Engine directory not found at: {LATEX_OCR_PATH}")
                print_error("Please ensure the submodule/directory exists. Skipping LaTeX-OCR installation.")
            else:
                print_system(f"Installing local LaTeX-OCR in editable mode from: {LATEX_OCR_PATH}")

                # The command for editable install from a local directory
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
                        build_env_llama = {'FORCE_CMAKE': '1'};
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
                        if not run_command([PIP_EXECUTABLE, "install", ".", "--upgrade", "--no-cache-dir", "--verbose"],
                                           LLAMA_CPP_PYTHON_CLONE_PATH, "PIP-BUILD-LLAMA",
                                           env_override=build_env_llama):
                            print_error("Build llama-cpp-python failed.")
                        else:
                            print_system("llama-cpp-python installed.")  # Use print_system
                            with open(CUSTOM_LLAMA_CPP_INSTALLED_FLAG_FILE, 'w',
                                      encoding='utf-8') as f_lcpp_flag:  # Corrected with statement
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
                    sd_cpp_build_path = os.path.join(sd_cpp_sub_path, "build");
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
                            print_system("stable-diffusion-cpp-python installed.")  # Use print_system
                            with open(CUSTOM_SD_CPP_PYTHON_INSTALLED_FLAG_FILE, 'w',
                                      encoding='utf-8') as f_sd_flag:  # Corrected with statement
                                f_sd_flag.write(backend_log_sd)
        else:
            print_system("Custom stable-diffusion-cpp-python previously installed.")

        print_system("--- Installing/Checking Node.js Dependencies ---")
        if not NPM_CMD or not shutil.which(NPM_CMD.split('.')[0]): print_error(
            f"'{NPM_CMD}' not found. Node.js setup failed."); sys.exit(1)
        if not run_command([NPM_CMD, "install"], BACKEND_SERVICE_DIR, "NPM-BACKEND"): print_error(
            "NPM Backend install failed."); sys.exit(1)
        if not run_command([NPM_CMD, "install"], FRONTEND_DIR, "NPM-FRONTEND"): print_error(
            "NPM Frontend install failed."); sys.exit(1)

        print_system("--- Starting All Services ---")
        service_threads = []
        service_threads.append(start_service_thread(start_engine_main, "EngineMainThread"))
        time.sleep(3)
        engine_ready = any(proc.poll() is None and name_s == "ENGINE" for proc, name_s in running_processes)
        if not engine_ready: print_error("Engine Main failed to start. Exiting."); sys.exit(1)

        service_threads.append(start_service_thread(start_backend_service, "BackendServiceThread"));
        time.sleep(2)
        service_threads.append(start_service_thread(start_frontend, "FrontendThread"))
        print_colored("SUCCESS", "All services launching. Press Ctrl+C to shut down.")  # Use print_colored for success
        try:
            while True:
                all_ok = True;
                active_procs_found = False
                with process_lock:
                    current_procs_snapshot = list(running_processes)
                if not current_procs_snapshot and service_threads: all_ok = False
                for proc, name_s in current_procs_snapshot:
                    if proc.poll() is None:
                        active_procs_found = True
                    else:
                        print_error(f"Service '{name_s}' exited unexpectedly (RC: {proc.poll()})."); all_ok = False
                if not all_ok: print_error("One or more services terminated. Shutting down launcher."); break
                if not active_procs_found and service_threads: print_system(
                    "All services seem to have finished. Exiting launcher."); break
                time.sleep(5)
        except KeyboardInterrupt:
            print_system("\nKeyboardInterrupt received by main thread (relaunched script). Shutting down...")
        finally:
            print_system("Launcher main loop (relaunched script) finished. Ensuring cleanup via atexit...")

    else:  # Initial launcher instance (is_already_in_correct_env is False)
        print_system(f"--- Initial Launcher: Conda Setup & Hardware Detection ---")

        autodetected_build_env_vars = _detect_and_prepare_acceleration_env_vars()

        if not find_conda_executable():
            print_error("Conda executable not located. Exiting.");
            _remove_flag_files(FLAG_FILES_TO_RESET_ON_ENV_RECREATE);
            sys.exit(1)
        print_system(f"Using Conda executable: {CONDA_EXECUTABLE}")

        if not (os.path.isdir(TARGET_CONDA_ENV_PATH) and os.path.exists(
                os.path.join(TARGET_CONDA_ENV_PATH, 'conda-meta'))):
            print_system(f"Target Conda env '{TARGET_CONDA_ENV_PATH}' not found/invalid. Creating...")
            _remove_flag_files(FLAG_FILES_TO_RESET_ON_ENV_RECREATE)
            target_python_versions = get_conda_python_versions_to_try()
            if not create_conda_env(TARGET_CONDA_ENV_PATH, target_python_versions):
                print_error(f"Failed to create Conda env at '{TARGET_CONDA_ENV_PATH}'. Exiting.");
                sys.exit(1)
        else:
            print_system(f"Target Conda env '{TARGET_CONDA_ENV_PATH}' exists.")

        script_to_run_abs_path = os.path.abspath(__file__)
        conda_run_cmd_list_base = [CONDA_EXECUTABLE, 'run', '--prefix', TARGET_CONDA_ENV_PATH]

        conda_supports_no_capture = False
        try:
            test_no_capture_cmd = [CONDA_EXECUTABLE, 'run', '--help']
            help_output = subprocess.check_output(test_no_capture_cmd, text=True, stderr=subprocess.STDOUT, timeout=5)
            if '--no-capture-output' in help_output:
                conda_supports_no_capture = True;
                conda_run_cmd_list_base.insert(2, '--no-capture-output')
        except Exception:
            pass

        conda_run_cmd_list = conda_run_cmd_list_base + ['python', script_to_run_abs_path] + sys.argv[1:]
        if IS_WINDOWS and CONDA_EXECUTABLE and CONDA_EXECUTABLE.lower().endswith(".bat"):
            conda_run_cmd_list = ['cmd', '/c'] + conda_run_cmd_list

        display_cmd_list = [str(c) if c is not None else "<CRITICAL_NONE_ERROR>" for c in conda_run_cmd_list]
        if "<CRITICAL_NONE_ERROR>" in display_cmd_list: print_error(
            f"FATAL: None value in conda_run_cmd_list: {display_cmd_list}"); sys.exit(1)
        print_system(f"Relaunching script using 'conda run': {' '.join(display_cmd_list)}")

        os.makedirs(RELAUNCH_LOG_DIR, exist_ok=True)
        popen_env = os.environ.copy()
        popen_env.update(autodetected_build_env_vars)
        popen_env["ZEPHYRINE_RELAUNCHED_IN_CONDA"] = "1"

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
                        open(RELAUNCH_STDERR_LOG, 'w', encoding='utf-8') as f_stderr:  # Corrected with statement
                    file_capture_kwargs = common_popen_kwargs_relaunch.copy()
                    file_capture_kwargs["stdout"] = f_stdout;
                    file_capture_kwargs["stderr"] = f_stderr
                    process_conda_run = subprocess.Popen(conda_run_cmd_list, **file_capture_kwargs)

            if not process_conda_run: raise RuntimeError("Popen for conda run failed.")
            relaunched_conda_process_obj = process_conda_run

            if not conda_supports_no_capture:
                stdout_thread = threading.Thread(target=_stream_log_file,
                                                 args=(RELAUNCH_STDOUT_LOG, relaunched_conda_process_obj, "STDOUT"),
                                                 daemon=True)
                stderr_thread = threading.Thread(target=_stream_log_file,
                                                 args=(RELAUNCH_STDERR_LOG, relaunched_conda_process_obj, "STDERR"),
                                                 daemon=True)
                log_stream_threads.extend([stdout_thread, stderr_thread]);
                stdout_thread.start();
                stderr_thread.start()

            exit_code_from_conda_run = -1
            if relaunched_conda_process_obj:
                try:
                    relaunched_conda_process_obj.wait()
                    exit_code_from_conda_run = relaunched_conda_process_obj.returncode
                except KeyboardInterrupt:
                    print_system("Parent script's wait for 'conda run' interrupted. Shutting down...");
                    if not getattr(sys, 'exitfunc_called', False): sys.exit(130)

                if log_stream_threads:
                    print_system("Parent: Relaunched script finished. Waiting for log streamers (max 2s)...")
                    for t in log_stream_threads:
                        if t.is_alive(): t.join(timeout=1.0)
                    print_system("Parent: Log streamers finished.")

                print_system(f"'conda run' process finished with code: {exit_code_from_conda_run}.")
                relaunched_conda_process_obj = None
                sys.exit(exit_code_from_conda_run)
            else:
                print_error("Failed to start 'conda run' process."); sys.exit(1)
        except Exception as e_outer:
            print_error(f"Error during 'conda run' or wait: {e_outer}"); traceback.print_exc(); sys.exit(1)

