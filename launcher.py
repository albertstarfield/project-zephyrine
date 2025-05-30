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
from typing import Optional

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



FLAG_FILES_TO_RESET_ON_ENV_RECREATE = [
    LICENSE_FLAG_FILE,
    MELO_TTS_INSTALLED_FLAG_FILE,
    CUSTOM_LLAMA_CPP_INSTALLED_FLAG_FILE,
    CUSTOM_SD_CPP_PYTHON_INSTALLED_FLAG_FILE,
    CONDA_PATH_CACHE_FILE,
    PYWHISPERCPP_INSTALLED_FLAG_FILE # <-- Add this line
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
    command = [HYPERCORN_EXECUTABLE, "app:app", "--bind", "127.0.0.1:11434", "--workers", "1", "--log-level", "info"]
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
    import curses
    curses.curs_set(0)
    stdscr.nodelay(False)
    stdscr.keypad(True)
    curses.noecho()
    curses.cbreak()

    if curses.has_colors():
        curses.start_color()
        curses.init_pair(1, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_WHITE, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_BLACK, curses.COLOR_GREEN)
        curses.init_pair(4, curses.COLOR_BLACK, curses.COLOR_RED)
        curses.init_pair(5, curses.COLOR_CYAN, curses.COLOR_BLACK)
        curses.init_pair(6, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(7, curses.COLOR_RED, curses.COLOR_BLACK)
    else:
        curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_WHITE, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_WHITE, curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_WHITE, curses.COLOR_BLACK)
        curses.init_pair(5, curses.COLOR_WHITE, curses.COLOR_BLACK)
        curses.init_pair(6, curses.COLOR_WHITE, curses.COLOR_BLACK)
        curses.init_pair(7, curses.COLOR_WHITE, curses.COLOR_BLACK)

    max_y, max_x = stdscr.getmaxyx()
    if max_y < 10 or max_x < 40:
        curses.nocbreak()
        stdscr.keypad(False)
        curses.echo()
        curses.endwin()
        print_error("Terminal window is too small to display licenses. Please resize and try again.")
        sys.exit(1)

    TEXT_AREA_HEIGHT = max_y - 7
    TEXT_START_Y = 1
    SCROLL_INFO_Y = max_y - 5
    EST_TIME_Y = max_y - 4
    INSTR_Y = max_y - 3
    PROMPT_Y = max_y - 2

    top_line = 0
    total_lines = len(licenses_text_lines)
    accepted = False
    start_time = time.monotonic()
    current_selection = 0

    while True:
        try:
            stdscr.clear()
            max_y, max_x = stdscr.getmaxyx()

            TEXT_AREA_HEIGHT = max(1, max_y - 7)
            TEXT_START_Y = 1
            SCROLL_INFO_Y = max_y - 5
            EST_TIME_Y = max_y - 4
            INSTR_Y = max_y - 3
            PROMPT_Y = max_y - 2

            header = "--- SOFTWARE LICENSE AGREEMENT ---"
            stdscr.addstr(0, max(0, (max_x - len(header)) // 2), header, curses.color_pair(5) | curses.A_BOLD)

            for i in range(TEXT_AREA_HEIGHT):
                line_idx = top_line + i
                if line_idx < total_lines:
                    display_line = licenses_text_lines[line_idx][:max_x - 1].rstrip()
                    stdscr.addstr(TEXT_START_Y + i, 0, display_line, curses.color_pair(2))
                else:
                    break

            if total_lines > TEXT_AREA_HEIGHT:
                progress = f"Line {top_line + 1} - {min(top_line + TEXT_AREA_HEIGHT, total_lines)} of {total_lines}"
                stdscr.addstr(SCROLL_INFO_Y, max(0, max_x - len(progress) - 1), progress, curses.color_pair(1))
            else:
                stdscr.addstr(SCROLL_INFO_Y, 0, "All text visible.", curses.color_pair(1))

            est_time_str = f"Estimated minimum review time: {estimated_seconds / 60:.1f} min ({estimated_seconds:.0f} sec)"
            stdscr.addstr(EST_TIME_Y, 0, est_time_str[:max_x - 1], curses.color_pair(1))

            instr = "Scroll: UP/DOWN/PGUP/PGDN | Select: LEFT/RIGHT | Confirm: ENTER / (a/n)"
            stdscr.addstr(INSTR_Y, 0, instr[:max_x - 1], curses.color_pair(1))

            accept_button_text = "[ Accept (A) ]"
            reject_button_text = "[ Not Accept (N) ]"

            button_spacing = 4
            buttons_total_width = len(accept_button_text) + len(reject_button_text) + button_spacing
            start_buttons_x = max(0, (max_x - buttons_total_width) // 2)

            accept_style = curses.color_pair(3) | curses.A_REVERSE if current_selection == 0 else curses.color_pair(6)
            reject_style = curses.color_pair(4) | curses.A_REVERSE if current_selection == 1 else curses.color_pair(7)

            stdscr.addstr(PROMPT_Y, start_buttons_x, accept_button_text, accept_style)
            stdscr.addstr(PROMPT_Y, start_buttons_x + len(accept_button_text) + button_spacing, reject_button_text,
                          reject_style)

            stdscr.refresh()

        except curses.error as e:
            print_warning(f"Curses display error: {e}. If persists, resize terminal.")
            time.sleep(0.1)
            max_y, max_x = stdscr.getmaxyx()
            if max_y < 10 or max_x < 40:
                curses.nocbreak()
                stdscr.keypad(False)
                curses.echo()
                curses.endwin()
                print_error("Terminal window became too small. Please resize and try again.")
                sys.exit(1)
            continue

        try:
            key = stdscr.getch()
        except KeyboardInterrupt:
            accepted = False
            break
        except Exception as e:
            print_warning(f"Error getting input: {e}")
            key = -1

        if key == ord('a') or key == ord('A'):
            accepted = True
            break
        elif key == ord('n') or key == ord('N'):
            accepted = False
            break
        elif key == curses.KEY_ENTER or key == 10 or key == 13:
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
        elif key == curses.KEY_NPAGE:
            top_line = min(max(0, total_lines - TEXT_AREA_HEIGHT), top_line + TEXT_AREA_HEIGHT)
        elif key == curses.KEY_PPAGE:
            top_line = max(0, top_line - TEXT_AREA_HEIGHT)
        elif key == curses.KEY_HOME:
            top_line = 0
        elif key == curses.KEY_END:
            top_line = max(0, total_lines - TEXT_AREA_HEIGHT)
        elif key == curses.KEY_RESIZE:
            pass

    end_time = time.monotonic()
    time_taken = end_time - start_time

    curses.nocbreak()
    stdscr.keypad(False)
    curses.echo()
    curses.endwin()
    return accepted, time_taken


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
            print_warning(f"Error verifying path for '{check_exe}': {e}. Assuming not in env.")

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

# --- File Download Logic ---
def download_file_with_progress(url, destination_path, file_description, requests_session):
    from tqdm import tqdm
    import requests

    print_system(f"Downloading {file_description} from {url} to {destination_path}...")
    try:
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)

        response = requests_session.get(url, stream=True, timeout=(10, 300))
        response.raise_for_status()

        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 8192

        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, desc=file_description,
                            ascii=IS_WINDOWS, leave=False)
        temp_destination_path = destination_path + ".tmp_download"

        with open(temp_destination_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()

        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print_error(
                f"Download ERROR for {file_description}: Size mismatch (expected {total_size_in_bytes}, got {progress_bar.n}).")
            if os.path.exists(temp_destination_path): os.remove(temp_destination_path)
            return False

        shutil.move(temp_destination_path, destination_path)
        print_system(f"Successfully downloaded and verified {file_description}.")
        return True
    except requests.exceptions.RequestException as e:
        print_error(f"Failed to download {file_description} due to network/request error: {e}")
        if os.path.exists(destination_path + ".tmp_download"):
            try:
                os.remove(destination_path + ".tmp_download")
            except Exception as rm_err:
                print_warning(f"Could not remove partial download {destination_path + '.tmp_download'}: {rm_err}")
        return False
    except Exception as e:
        print_error(f"An unexpected error occurred during download of {file_description}: {e}")
        if os.path.exists(destination_path + ".tmp_download"):
            try:
                os.remove(destination_path + ".tmp_download")
            except Exception as rm_err:
                print_warning(f"Could not remove partial download {destination_path + '.tmp_download'}: {rm_err}")
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
                    norm_current_env_path == norm_target_env_path:
                is_already_in_correct_env = True
                ACTIVE_ENV_PATH = current_conda_env_path_check
        except FileNotFoundError:
            pass
        except Exception as e:
            print_warning(f"Error comparing Conda paths: {e}")

    if is_already_in_correct_env:
        print_system(f"Running inside target Conda environment (Prefix: {ACTIVE_ENV_PATH})")
        print_system(f"Python executable in use: {sys.executable}")

        # --- ADD THIS BLOCK TO ENSURE CONDA_EXECUTABLE IS SET IN RELAUNCHED SCRIPT ---
        if globals().get('CONDA_EXECUTABLE') is None:  # Check if it wasn't set (it wouldn't be on relaunch)
            print_system("Relaunched script: CONDA_EXECUTABLE is None. Attempting to load from cache or PATH...")
            loaded_from_cache = False
            if os.path.exists(CONDA_PATH_CACHE_FILE):
                try:
                    with open(CONDA_PATH_CACHE_FILE, 'r', encoding='utf-8') as f_cache:
                        cached_path = f_cache.read().strip()
                    if cached_path and _verify_conda_path(cached_path):  # _verify_conda_path must be globally defined
                        globals()['CONDA_EXECUTABLE'] = cached_path
                        print_system(f"Using Conda executable from cache in relaunched script: {CONDA_EXECUTABLE}")
                        loaded_from_cache = True
                    else:
                        print_warning("Cached Conda path was invalid or verification failed in relaunched script.")
                except Exception as e_cache_read:
                    print_warning(f"Error reading Conda cache in relaunched script: {e_cache_read}")

            if not loaded_from_cache:  # If cache didn't work or didn't exist
                print_system("Attempting to find Conda via shutil.which('conda') in relaunched script...")
                conda_exe_from_which = shutil.which("conda.exe" if IS_WINDOWS else "conda")
                if conda_exe_from_which and _verify_conda_path(conda_exe_from_which):
                    globals()['CONDA_EXECUTABLE'] = conda_exe_from_which
                    print_system(f"Found Conda via shutil.which in relaunched script: {CONDA_EXECUTABLE}")
                    # Optionally, re-cache it here if you want to update a potentially stale cache
                    # try:
                    #     with open(CONDA_PATH_CACHE_FILE, 'w', encoding='utf-8') as f_cache_update:
                    #         f_cache_update.write(CONDA_EXECUTABLE)
                    # except IOError: pass
                else:
                    print_error(
                        "CRITICAL: Conda executable could not be determined in relaunched script (cache & shutil.which failed). Conda package installations will fail.")
                    # You might want to sys.exit(1) here if CONDA_EXECUTABLE is essential for subsequent steps
                    # that absolutely require it, even in the relaunched script. For _ensure_conda_package, it is.
                    # For now, it will proceed and _ensure_conda_package will fail if CONDA_EXECUTABLE is still None.

        if globals().get('CONDA_EXECUTABLE') is None:
            print_error(
                "CRITICAL: CONDA_EXECUTABLE is still None after attempting to load in relaunched script. Subsequent Conda operations WILL FAIL.")
            # sys.exit("Failed to determine CONDA_EXECUTABLE in relaunched context.") # Consider exiting
        # --- END ADDED BLOCK ---

        print_system("<<<<< RELAUNCHED SCRIPT IS ALIVE AND RUNNING THIS LINE >>>>>")
        sys.stdout.flush()
        time.sleep(1)
        print_system("<<<<< RELAUNCHED SCRIPT CONTINUING AFTER PAUSE >>>>>")
        sys.stdout.flush()

        if "ZEPHYRINE_RELAUNCHED_IN_CONDA" in os.environ:
            del os.environ["ZEPHYRINE_RELAUNCHED_IN_CONDA"]

        # --- Application setup logic (runs inside the Conda environment) ---
        if not ACTIVE_ENV_PATH:
            ACTIVE_ENV_PATH = os.getenv("CONDA_PREFIX")
            if not ACTIVE_ENV_PATH:
                print_error("CRITICAL: CONDA_PREFIX is not set even after relaunch. Exiting.")
                sys.exit(1)
            try:
                if not (os.path.isdir(ACTIVE_ENV_PATH) and os.path.isdir(TARGET_CONDA_ENV_PATH) and \
                        os.path.normcase(os.path.realpath(ACTIVE_ENV_PATH)) == os.path.normcase(
                            os.path.realpath(TARGET_CONDA_ENV_PATH))):
                    print_error(
                        f"CRITICAL: Re-fetched CONDA_PREFIX '{ACTIVE_ENV_PATH}' does not match target '{TARGET_CONDA_ENV_PATH}'. Exiting.")
                    sys.exit(1)
            except Exception as path_e:
                print_error(f"CRITICAL: Error comparing re-fetched CONDA_PREFIX: {path_e}. Exiting.")
                sys.exit(1)
            print_system(
                f"Confirmed active Conda environment from CONDA_PREFIX (re-fetched): {os.path.basename(ACTIVE_ENV_PATH)}")



        _sys_exec_dir = os.path.dirname(sys.executable)
        PYTHON_EXECUTABLE = sys.executable
        PIP_EXECUTABLE = os.path.join(_sys_exec_dir, "pip.exe" if IS_WINDOWS else "pip")
        HYPERCORN_EXECUTABLE = os.path.join(_sys_exec_dir, "hypercorn.exe" if IS_WINDOWS else "hypercorn")

        if not os.path.exists(PIP_EXECUTABLE):
            print_warning(f"Pip executable not found at derived path {PIP_EXECUTABLE}. Trying shutil.which('pip')...")
            found_pip = shutil.which("pip")
            if found_pip and os.path.normcase(os.path.realpath(os.path.dirname(found_pip))).startswith(
                    os.path.normcase(os.path.realpath(ACTIVE_ENV_PATH))):
                PIP_EXECUTABLE = found_pip
                print_system(f"Using pip found at: {PIP_EXECUTABLE}")
            else:
                print_error(f"Could not locate pip in the active Conda environment ({ACTIVE_ENV_PATH}). Exiting.")
                sys.exit(1)

        if IS_WINDOWS:
            print_system("Checking/Installing windows-curses for the license prompt...")
            if not run_command([PIP_EXECUTABLE, "install", "windows-curses"], ROOT_DIR, "PIP-WINCURSES"):
                print_error("Failed to install windows-curses. License prompt may not work. Exiting.")
                sys.exit(1)
        try:
            import curses
        except ImportError:
            print_error("Failed to import curses. Exiting."); sys.exit(1)

        print_system("--- Installing Core Helper Utilities (tqdm, requests) ---")
        if not run_command([PIP_EXECUTABLE, "install", "tqdm", "requests"], ROOT_DIR, "PIP-UTILS"):
            print_error("Failed to install tqdm/requests. Exiting.");
            sys.exit(1)
        try:
            import requests
        except ImportError:
            print_error("Failed to import 'requests'. Exiting."); sys.exit(1)

        requests_session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=20)
        requests_session.mount('http://', adapter);
        requests_session.mount('https://', adapter)

        print_system("--- Installing/Checking Python Dependencies (Engine) from requirements.txt ---")
        engine_req_path = os.path.join(ROOT_DIR, "requirements.txt")
        if not os.path.exists(engine_req_path): print_error(
            f"Engine requirements.txt not found: {engine_req_path}"); sys.exit(1)
        if not run_command([PIP_EXECUTABLE, "install", "-r", engine_req_path], ENGINE_MAIN_DIR, "PIP-ENGINE-REQ"):
            print_error("Failed to install Python dependencies for Engine. Exiting.");
            sys.exit(1)

        # --- Ensure Core Command-Line Tools via Conda ---
        print_system("--- Ensuring core command-line tools are present in Conda environment ---")
        if not _ensure_conda_package(package_name="git", executable_to_check="git", is_critical=True):
            print_error("Git could not be ensured. Many setup steps will fail.");
            sys.exit(1)

        if not _ensure_conda_package(package_name="cmake", executable_to_check="cmake", is_critical=True):
            print_error("CMake could not be ensured. Builds will fail.");
            sys.exit(1)

        # Node.js provides both 'node' and 'npm'
        if not _ensure_conda_package(package_name="nodejs", executable_to_check="node", is_critical=True):
            print_error("Node.js (node) could not be ensured. Backend/Frontend setup will fail.");
            sys.exit(1)
        if not _ensure_conda_package(package_name="nodejs", executable_to_check="npm", is_critical=True):
            # Technically, nodejs should provide npm, but an explicit check can be good.
            # If node was installed, npm should be there. If not, this might re-trigger nodejs install.
            print_error("Node.js (npm) could not be ensured. Backend/Frontend setup will fail.");
            sys.exit(1)

        # For ffmpeg, it's not strictly critical for the launcher to exit if missing, but ASR/TTS for some formats will fail.
        _ensure_conda_package(package_name="ffmpeg", executable_to_check="ffmpeg", is_critical=False)
        # The existing warning about ffmpeg for pywhispercpp can remain as an extra reminder.
        print_system("--- Core command-line tools check/install attempt complete ---")

        # --- Re-evaluate global CMD variables AFTER potential Conda installs ---
        # This ensures we use the Conda-installed versions if they were just put there.
        # shutil.which() will search the PATH, which should now prioritize the Conda env's bin.
        globals()['GIT_CMD'] = shutil.which("git") or ('git.exe' if IS_WINDOWS else 'git')
        globals()['CMAKE_CMD'] = shutil.which("cmake") or ('cmake.exe' if IS_WINDOWS else 'cmake')
        npm_exe_check = "npm.cmd" if IS_WINDOWS else "npm"
        globals()['NPM_CMD'] = shutil.which(npm_exe_check) or npm_exe_check

        print_system(f"Updated GIT_CMD to: {GIT_CMD}")
        print_system(f"Updated CMAKE_CMD to: {CMAKE_CMD}")
        print_system(f"Updated NPM_CMD to: {NPM_CMD}")
        if not all([shutil.which("git"), shutil.which("cmake"), shutil.which("npm"), shutil.which("node")]):
            print_warning(
                "One or more critical tools (git, cmake, node, npm) still not found after conda install attempts. Subsequent steps might fail.")

        try:
            import tiktoken; TIKTOKEN_AVAILABLE = True; print_system("tiktoken is available.")
        except ImportError:
            TIKTOKEN_AVAILABLE = False; print_warning("tiktoken NOT available. License time estimate default.")

        if not os.path.exists(LICENSE_FLAG_FILE):
            print_system("License agreement required.")
            try:
                _, combined_license_text = load_licenses()
                estimated_reading_seconds = calculate_reading_time(combined_license_text)
                accepted, time_taken = curses.wrapper(display_license_prompt, combined_license_text.splitlines(),
                                                      estimated_reading_seconds)
                if not accepted: print_error("License terms not accepted. Exiting."); sys.exit(1)
                with open(LICENSE_FLAG_FILE, 'w', encoding='utf-8') as f:
                    f.write(f"Accepted: {datetime.now().isoformat()}\nTime: {time_taken:.2f}s\n")
                print_system(f"Licenses accepted by user in {time_taken:.2f}s.")
                if estimated_reading_seconds > 30 and time_taken < (estimated_reading_seconds * 0.1):
                    print_warning("Warning: Licenses accepted quickly. Ensure you understood terms.");
                    time.sleep(3)
            except curses.error as e:
                print_error(f"Curses error during license: {e}"); sys.exit(1)
            except Exception as e:
                print_error(f"Error during license: {e}"); curses.endwin(); sys.exit(1)
        else:
            print_system(f"License previously accepted (flag: {LICENSE_FLAG_FILE}).")

        print_system(f"--- Checking Static Model Pool: {STATIC_MODEL_POOL_PATH} ---")
        os.makedirs(STATIC_MODEL_POOL_PATH, exist_ok=True)
        all_models_ok = True
        for model_info in MODELS_TO_DOWNLOAD:
            dest_path = os.path.join(STATIC_MODEL_POOL_PATH, model_info["filename"])
            if not os.path.exists(dest_path):
                print_warning(f"Model '{model_info['description']}' not found. Downloading.")
                if not download_file_with_progress(model_info["url"], dest_path, model_info["description"],
                                                   requests_session):
                    print_error(f"Failed download '{model_info['filename']}'.");
                    all_models_ok = False
            else:
                print_system(f"Model '{model_info['description']}' present.")
        if not all_models_ok:
            print_error("One or more models failed to download.")
        else:
            print_system("Static model pool checked/populated.")

        if not os.path.exists(MELO_TTS_INSTALLED_FLAG_FILE):
            print_system(f"--- MeloTTS First-Time Setup from {MELO_TTS_PATH} ---")
            if not os.path.isdir(MELO_TTS_PATH): print_error(
                f"MeloTTS dir missing: {MELO_TTS_PATH}. Ensure submodule init."); sys.exit(1)
            if not run_command([PIP_EXECUTABLE, "install", "-e", "."], MELO_TTS_PATH, "PIP-MELO-EDITABLE"): print_error(
                "MeloTTS install failed. Exiting."); sys.exit(1)
            if not run_command([PYTHON_EXECUTABLE, "-m", "unidic", "download"], MELO_TTS_PATH,
                               "UNIDIC-DOWNLOAD"): print_warning("'unidic' download failed.")

            audio_worker_script = os.path.join(ENGINE_MAIN_DIR, "audio_worker.py")
            if not os.path.exists(audio_worker_script): print_error(
                f"audio_worker.py not found: '{audio_worker_script}'. Exiting."); sys.exit(1)
            temp_melo_test_dir = os.path.join(ROOT_DIR, "temp_melo_audio_test_files");
            os.makedirs(temp_melo_test_dir, exist_ok=True)
            test_out_file = os.path.join(temp_melo_test_dir,
                                         f"initial_melo_test_{datetime.now().strftime('%Y%m%d%H%M%S')}.wav")
            test_cmd = [PYTHON_EXECUTABLE, audio_worker_script, "--test-mode", "--model-lang", "EN", "--device", "auto",
                        "--output-file", test_out_file, "--temp-dir", temp_melo_test_dir]
            if not run_command(test_cmd, ENGINE_MAIN_DIR, "MELO-INIT-TEST"):
                print_warning("MeloTTS initial test failed.")
            else:
                print_system(
                    f"MeloTTS initial test OK. Test audio: {test_out_file if os.path.exists(test_out_file) else 'Not found'}")
            with open(MELO_TTS_INSTALLED_FLAG_FILE, 'w', encoding='utf-8') as f:
                f.write(f"Tested: {datetime.now().isoformat()}\n")
        else:
            print_system("MeloTTS previously installed/tested.")

        if not os.path.exists(PYWHISPERCPP_INSTALLED_FLAG_FILE):
            print_system(f"--- PyWhisperCpp Installation (from local clone) ---")

            if not shutil.which(GIT_CMD):
                print_error(f"'{GIT_CMD}' not found. Git is required to install pywhispercpp from source.")
                print_warning("Skipping PyWhisperCpp installation. ASR functionality will be unavailable.")
            else:
                # 1. Clean and Clone the repository
                if os.path.exists(PYWHISPERCPP_CLONE_PATH):
                    print_system(f"Removing existing pywhispercpp build directory: {PYWHISPERCPP_CLONE_PATH}")
                    try:
                        shutil.rmtree(PYWHISPERCPP_CLONE_PATH)
                    except Exception as e:
                        print_error(
                            f"Failed to remove existing pywhispercpp build directory: {e}. Please remove it manually and retry.")
                        # Consider this a blocking error for this install attempt

                if not os.path.exists(PYWHISPERCPP_CLONE_PATH):  # Proceed only if old one is gone or never existed
                    print_system(
                        f"Cloning pywhispercpp from '{PYWHISPERCPP_REPO_URL}' to '{PYWHISPERCPP_CLONE_PATH}'...")
                    if not run_command([GIT_CMD, "clone", PYWHISPERCPP_REPO_URL, PYWHISPERCPP_CLONE_PATH],
                                       cwd=ROOT_DIR, name="GIT-CLONE-PYWHISPERCPP"):
                        print_error("Failed to clone pywhispercpp repository. Installation aborted for this run.")
                    else:
                        print_system(f"Successfully cloned pywhispercpp to {PYWHISPERCPP_CLONE_PATH}.")

                        # --- ADDED: Initialize and update submodules ---
                        print_system(
                            "Initializing and updating pywhispercpp submodules (e.g., whisper.cpp, pybind11)...")
                        if not run_command([GIT_CMD, "submodule", "update", "--init", "--recursive"],
                                           cwd=PYWHISPERCPP_CLONE_PATH,  # Run this inside the cloned repo
                                           name="GIT-SUBMODULE-PYWHISPERCPP"):
                            print_error(
                                "Failed to update pywhispercpp submodules. Installation may fail or be incomplete.")
                            # This is often critical, so you might consider exiting or marking install as failed.
                        else:
                            print_system("Pywhispercpp submodules updated successfully.")
                        # --- END ADDED STEP ---

                        # 2. Prepare environment and pip install from local clone
                        pip_local_install_cmd = [PIP_EXECUTABLE, "install", "."]

                        build_env_override = {}
                        backend_detected = "default (CPU)"

                        if os.getenv("GGML_CUDA") == '1':
                            build_env_override['GGML_CUDA'] = '1';
                            backend_detected = "CUDA"
                            print_system("CUDA backend for PyWhisperCpp build detected via GGML_CUDA=1 env var.")
                        elif os.getenv("WHISPER_COREML") == '1':
                            build_env_override['WHISPER_COREML'] = '1';
                            backend_detected = "CoreML"
                            print_system(
                                "CoreML backend for PyWhisperCpp build detected via WHISPER_COREML=1 env var.")
                        # ... (add other elif for GGML_VULKAN, GGML_BLAS, WHISPER_OPENVINO as before) ...

                        print_system(
                            f"Attempting to build and install pywhispercpp from local source '{PYWHISPERCPP_CLONE_PATH}' (Target Backend: {backend_detected})...")

                        if not run_command(pip_local_install_cmd,
                                           cwd=PYWHISPERCPP_CLONE_PATH,
                                           name="PIP-PYWHISPERCPP-LOCAL",
                                           env_override=build_env_override if build_env_override else None):
                            print_error("Failed to build/install pywhispercpp from local source.")
                            print_warning("ASR functionality using Whisper.cpp might be impaired.")
                        else:
                            print_system("pywhispercpp built and installed successfully from local source.")
                            print_warning(
                                "For pywhispercpp to transcribe audio formats other than WAV, ensure FFmpeg is installed and accessible in your system's PATH.")
                            try:
                                with open(PYWHISPERCPP_INSTALLED_FLAG_FILE, 'w', encoding='utf-8') as f:
                                    f.write(
                                        f"Installed from local source, backend hint '{backend_detected}' on: {datetime.now().isoformat()}\n")
                                print_system("pywhispercpp installation flag created.")
                            except IOError as flag_err:
                                print_error(f"Could not create pywhispercpp installation flag file: {flag_err}")
        else:
            print_system("PyWhisperCpp previously installed (flag file found).")

        provider_env_check = os.getenv("PROVIDER", "llama_cpp").lower()
        if provider_env_check == "llama_cpp":
            if not os.path.exists(CUSTOM_LLAMA_CPP_INSTALLED_FLAG_FILE):
                print_system(f"--- Custom llama-cpp-python Installation ---")
                if not shutil.which(GIT_CMD): print_error(f"'{GIT_CMD}' not found. Exiting."); sys.exit(1)
                run_command([PIP_EXECUTABLE, "uninstall", "llama-cpp-python", "-y"], ROOT_DIR, "PIP-UNINSTALL-LLAMA",
                            check=False)
                if os.path.exists(LLAMA_CPP_PYTHON_CLONE_PATH): shutil.rmtree(LLAMA_CPP_PYTHON_CLONE_PATH)
                if not run_command([GIT_CMD, "clone", LLAMA_CPP_PYTHON_REPO_URL, LLAMA_CPP_PYTHON_CLONE_PATH], ROOT_DIR,
                                   "GIT-CLONE-LLAMA"): print_error("Clone llama-cpp-python failed."); sys.exit(1)
                if not run_command([GIT_CMD, "submodule", "update", "--init", "--recursive"],
                                   LLAMA_CPP_PYTHON_CLONE_PATH, "GIT-SUBMODULE-LLAMA"): print_error(
                    "Submodule update failed."); sys.exit(1)

                build_env = {'FORCE_CMAKE': '1'};
                cmake_args_list = ["-DLLAMA_BUILD_EXAMPLES=OFF", "-DLLAMA_BUILD_TESTS=OFF"]
                default_backend = 'cpu';
                llama_backend = os.getenv("LLAMA_CPP_BACKEND", default_backend).lower()
                if llama_backend == "cuda":
                    cmake_args_list.append("-DGGML_CUDA=ON")
                elif llama_backend == "metal":
                    cmake_args_list.append("-DGGML_METAL=ON")
                elif llama_backend == "cpu":
                    cmake_args_list.append("-DLLAMA_OPENMP=ON")
                else:
                    print_warning(
                        f"Unknown LLAMA_CPP_BACKEND '{llama_backend}'. Defaulting CPU."); cmake_args_list.append(
                        "-DLLAMA_OPENMP=ON")

                effective_cmake_args = " ".join(filter(None, cmake_args_list))
                if effective_cmake_args: build_env['CMAKE_ARGS'] = effective_cmake_args

                pip_cmd_llama = [PIP_EXECUTABLE, "install", ".", "--upgrade", "--no-cache-dir", "--verbose"]
                if not run_command(pip_cmd_llama, LLAMA_CPP_PYTHON_CLONE_PATH, "PIP-BUILD-LLAMA",
                                   env_override=build_env): print_error("Build llama-cpp-python failed."); sys.exit(1)
                with open(CUSTOM_LLAMA_CPP_INSTALLED_FLAG_FILE, 'w', encoding='utf-8') as f:
                    f.write(f"Installed {llama_backend} on {datetime.now().isoformat()}\n")
            else:
                print_system("Custom llama-cpp-python previously installed.")
        else:
            print_system(f"PROVIDER is '{provider_env_check}'. Skipping custom llama-cpp-python.")

        if not os.path.exists(CUSTOM_SD_CPP_PYTHON_INSTALLED_FLAG_FILE):
            print_system(f"--- Custom stable-diffusion-cpp-python Installation ---")
            if not shutil.which(GIT_CMD) or not shutil.which(CMAKE_CMD): print_error(
                "Git or CMake not found. Exiting."); sys.exit(1)
            run_command([PIP_EXECUTABLE, "uninstall", "stable-diffusion-cpp-python", "-y"], ROOT_DIR,
                        "PIP-UNINSTALL-SD", check=False)
            if os.path.exists(STABLE_DIFFUSION_CPP_PYTHON_CLONE_PATH): shutil.rmtree(
                STABLE_DIFFUSION_CPP_PYTHON_CLONE_PATH)
            if not run_command([GIT_CMD, "clone", "--recursive", STABLE_DIFFUSION_CPP_PYTHON_REPO_URL,
                                STABLE_DIFFUSION_CPP_PYTHON_CLONE_PATH], ROOT_DIR, "GIT-CLONE-SD"): print_error(
                "Clone SD failed."); sys.exit(1)

            sd_cpp_sub_path = os.path.join(STABLE_DIFFUSION_CPP_PYTHON_CLONE_PATH, "vendor", "stable-diffusion.cpp")
            sd_cpp_build_path = os.path.join(sd_cpp_sub_path, "build");
            os.makedirs(sd_cpp_build_path, exist_ok=True)
            cmake_args_sd = ["-DCMAKE_POLICY_VERSION_MINIMUM=3.13"];
            sd_backend_env = os.getenv("SD_CPP_BACKEND", "").lower()
            if sd_backend_env == "cuda": cmake_args_sd.append("-DSD_CUDA=ON")  # Add other backends as needed
            if not run_command([CMAKE_CMD, ".."] + cmake_args_sd, sd_cpp_build_path, "CMAKE-SD-LIB-CFG"): print_error(
                "CMake SD lib failed."); sys.exit(1)
            if not run_command([CMAKE_CMD, "--build", "."] + (["--config", "Release"] if IS_WINDOWS else []),
                               sd_cpp_build_path, "CMAKE-BUILD-SD-LIB"): print_error("Build SD lib failed."); sys.exit(
                1)

            pip_build_env_sd = os.environ.copy();
            pip_build_env_sd['FORCE_CMAKE'] = '1'
            if " ".join(cmake_args_sd): pip_build_env_sd['CMAKE_ARGS'] = " ".join(cmake_args_sd)
            pip_cmd_sd = [PIP_EXECUTABLE, "install", ".", "--upgrade", "--no-cache-dir", "--verbose"]
            if not run_command(pip_cmd_sd, STABLE_DIFFUSION_CPP_PYTHON_CLONE_PATH, "PIP-SD-BINDINGS",
                               env_override=pip_build_env_sd): print_error("Install SD bindings failed."); sys.exit(1)
            with open(CUSTOM_SD_CPP_PYTHON_INSTALLED_FLAG_FILE, 'w', encoding='utf-8') as f:
                f.write(f"Installed {sd_backend_env or 'cpu'} on {datetime.now().isoformat()}\n")
        else:
            print_system("Custom stable-diffusion-cpp-python previously installed.")

        print_system("--- Installing/Checking Node.js Backend Dependencies ---")
        if not shutil.which(NPM_CMD.split('.')[0]): print_error(
            f"'{NPM_CMD}' not found. Install Node.js. Exiting."); sys.exit(1)
        if not run_command([NPM_CMD, "install"], BACKEND_SERVICE_DIR, "NPM-BACKEND"): print_error(
            "NPM Backend install failed."); sys.exit(1)
        print_system("--- Installing/Checking Node.js Frontend Dependencies ---")
        if not run_command([NPM_CMD, "install"], FRONTEND_DIR, "NPM-FRONTEND"): print_error(
            "NPM Frontend install failed."); sys.exit(1)

        print_system("--- Starting All Services ---")
        service_threads = []
        service_threads.append(start_service_thread(start_engine_main, "EngineMainThread"))
        time.sleep(2);
        engine_ready = False
        with process_lock:
            for proc, name in running_processes:
                if name == "ENGINE" and proc.poll() is None: engine_ready = True; break
        if not engine_ready: print_error("Engine Main failed to start. Exiting."); sys.exit(1)

        service_threads.append(start_service_thread(start_backend_service, "BackendServiceThread"));
        time.sleep(2)
        service_threads.append(start_service_thread(start_frontend, "FrontendThread"))
        print_system("All services launching. Press Ctrl+C to shut down.")

        try:
            while True:
                active_procs = False;
                all_ok = True
                with process_lock:
                    current_procs = list(running_processes)
                if not current_procs and service_threads: all_ok = False  # No procs but threads trying to start them
                for proc, name in current_procs:
                    if proc.poll() is None:
                        active_procs = True
                    else:
                        print_error(f"Service '{name}' exited (RC: {proc.poll()})."); all_ok = False
                if not all_ok: print_error(
                    "One or more services terminated. Initiating shutdown of remaining services."); break
                if not active_procs and service_threads: print_system(
                    "All managed services have finished or exited. Launcher will now shut down."); break
                if not service_threads: print_system(
                    "No services were configured to start. Setup complete. Exiting launcher."); break  # Should not happen if services started
                time.sleep(5)
        except KeyboardInterrupt:
            print_system("\nKeyboardInterrupt received by main thread (relaunched script). Shutting down...")
        finally:
            print_system("Launcher main loop (relaunched script) finished. Ensuring cleanup...")
        # atexit handler 'cleanup_processes' will run automatically on script exit.

    else:
        # --- This is the INITIAL script instance, needs to set up Conda and relaunch ---
        print_system(f"--- Conda Environment Setup (Target Prefix: {TARGET_CONDA_ENV_PATH}) ---")
        if not find_conda_executable():
            print_error("Conda executable could not be located. Please install Anaconda/Miniconda. Exiting.")
            _remove_flag_files(FLAG_FILES_TO_RESET_ON_ENV_RECREATE)
            sys.exit(1)
        print_system(f"Using Conda executable: {CONDA_EXECUTABLE}")

        if current_conda_env_path_check:
            print_warning(
                f"Currently in Conda env '{current_conda_env_path_check}', but target is '{TARGET_CONDA_ENV_PATH}'. Will attempt to switch/relaunch.")
        else:
            print_system(
                "Not currently in an active Conda environment (CONDA_PREFIX not set or empty). Will attempt to use/create target.")

        if not (os.path.isdir(TARGET_CONDA_ENV_PATH) and os.path.exists(
                os.path.join(TARGET_CONDA_ENV_PATH, 'conda-meta'))):
            print_system(f"Target Conda environment prefix '{TARGET_CONDA_ENV_PATH}' not found or invalid. Creating...")
            _remove_flag_files(FLAG_FILES_TO_RESET_ON_ENV_RECREATE)
            target_python_versions = get_conda_python_versions_to_try()
            if not create_conda_env(TARGET_CONDA_ENV_PATH, target_python_versions):
                print_error(f"Failed to create the Conda environment at prefix '{TARGET_CONDA_ENV_PATH}'. Exiting.")
                # create_conda_env calls _remove_flag_files on its internal failure
                sys.exit(1)
        else:
            print_system(f"Target Conda environment prefix '{TARGET_CONDA_ENV_PATH}' exists.")

        script_to_run_abs_path = os.path.abspath(__file__)

        # --- DEBUG POINT A from previous response ---
        print_system(f"[DEBUG_POINT_A] CONDA_EXECUTABLE is: '{CONDA_EXECUTABLE}' (Type: {type(CONDA_EXECUTABLE)})")
        if CONDA_EXECUTABLE is None:  # This check is critical
            print_error(
                "CRITICAL DEBUG: CONDA_EXECUTABLE is None immediately before 'conda_run_base_cmd' definition. Exiting.")
            sys.exit("FATAL: CONDA_EXECUTABLE became None unexpectedly after find_conda_executable check.")
        # --- END DEBUG POINT A ---

        conda_run_base_cmd = [CONDA_EXECUTABLE, 'run', '--prefix', TARGET_CONDA_ENV_PATH, 'python',
                              script_to_run_abs_path]

        conda_supports_no_capture_flag = False
        try:
            test_no_capture_cmd = [CONDA_EXECUTABLE, 'run', '--help']
            help_output = subprocess.check_output(test_no_capture_cmd, text=True, stderr=subprocess.STDOUT, timeout=5)
            if '--no-capture-output' in help_output:
                conda_supports_no_capture_flag = True
                print_system("Conda supports '--no-capture-output', will use for direct streaming.")
                conda_run_base_cmd.insert(2, '--no-capture-output')
            else:
                print_system(
                    "Conda does not appear to support '--no-capture-output'. Relaunch output will be captured to files.")
        except Exception as e_conda_help:
            print_warning(
                f"Could not determine Conda's '--no-capture-output' support: {e_conda_help}. Assuming not supported.")

        conda_run_cmd_list = conda_run_base_cmd + sys.argv[1:]

        if IS_WINDOWS and CONDA_EXECUTABLE and CONDA_EXECUTABLE.lower().endswith(".bat"):
            conda_run_cmd_list = ['cmd', '/c'] + conda_run_cmd_list

        display_cmd_list_for_log = []
        none_found_in_cmd_list_for_display = False
        for i_item, item_val in enumerate(conda_run_cmd_list):
            if item_val is None:
                display_cmd_list_for_log.append(f"<NONE_VALUE_AT_INDEX_{i_item}>")
                none_found_in_cmd_list_for_display = True
            else:
                display_cmd_list_for_log.append(str(item_val))

        if none_found_in_cmd_list_for_display:
            print_error(
                f"DEBUG: One or more None values found in conda_run_cmd_list before join: {display_cmd_list_for_log}")
            sys.exit("FATAL: None value found in command list for Popen, which is invalid.")

        log_message_for_conda_run_display = ' '.join(display_cmd_list_for_log)
        print_system(f"Relaunching script using 'conda run': {log_message_for_conda_run_display}")

        os.makedirs(RELAUNCH_LOG_DIR, exist_ok=True)

        log_stream_threads = []

        common_popen_kwargs = {"text": True, "errors": 'replace', "bufsize": 1}
        if IS_WINDOWS:
            common_popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            common_popen_kwargs["preexec_fn"] = os.setsid

        process_conda_run = None

        try:
            if conda_supports_no_capture_flag:  # If --no-capture-output was added
                print_system("Parent launcher: Relaunched script output should stream directly to this terminal.")
                process_conda_run = subprocess.Popen(conda_run_cmd_list, **common_popen_kwargs)
            else:
                print_system(
                    f"Parent launcher: Relaunched script output will be captured to files (and tailed by parent):")
                print_system(f"  Stdout log: {RELAUNCH_STDOUT_LOG}")
                print_system(f"  Stderr log: {RELAUNCH_STDERR_LOG}")

                # This with block needs to be outside the Popen call if f_stdout/f_stderr are passed to it.
                # The Popen call should happen inside this block.
                # Corrected structure:
                with open(RELAUNCH_STDOUT_LOG, 'w', encoding='utf-8') as f_stdout, \
                        open(RELAUNCH_STDERR_LOG, 'w', encoding='utf-8') as f_stderr:

                    file_capture_popen_kwargs = common_popen_kwargs.copy()
                    file_capture_popen_kwargs["stdout"] = f_stdout
                    file_capture_popen_kwargs["stderr"] = f_stderr
                    process_conda_run = subprocess.Popen(conda_run_cmd_list, **file_capture_popen_kwargs)

            if not process_conda_run:  # Should be caught by exceptions below if Popen fails
                raise RuntimeError("subprocess.Popen failed to create process_conda_run object.")

            relaunched_conda_process_obj = process_conda_run

            if not conda_supports_no_capture_flag:  # Only start file tailers if not streaming directly
                print_system("Parent launcher: Starting file log streamers for relaunched script (from files)...")
                stdout_stream_thread = threading.Thread(
                    target=_stream_log_file,
                    args=(RELAUNCH_STDOUT_LOG, relaunched_conda_process_obj, "STDOUT"), daemon=True)
                stderr_stream_thread = threading.Thread(
                    target=_stream_log_file,
                    args=(RELAUNCH_STDERR_LOG, relaunched_conda_process_obj, "STDERR"), daemon=True)
                log_stream_threads.extend([stdout_stream_thread, stderr_stream_thread])
                stdout_stream_thread.start();
                stderr_stream_thread.start()

            exit_code_from_conda_run = -1
            if relaunched_conda_process_obj:
                try:
                    relaunched_conda_process_obj.wait()
                    exit_code_from_conda_run = relaunched_conda_process_obj.returncode
                except KeyboardInterrupt:
                    print_system("Parent script's wait for 'conda run' was interrupted by KeyboardInterrupt.")
                    if not getattr(sys, 'exitfunc_called', False): sys.exit(130)

                if log_stream_threads:
                    print_system(
                        "Parent launcher: Relaunched script finished. Waiting for file log streamers (max 2s)...")
                    for t in log_stream_threads:
                        if t.is_alive(): t.join(timeout=1.0)
                    print_system("Parent launcher: File log streamers finished.")

                print_system(f"'conda run' process finished with code: {exit_code_from_conda_run}.")
                relaunched_conda_process_obj = None

                if exit_code_from_conda_run != 0:
                    error_msg = f"'conda run' process exited with non-zero code: {exit_code_from_conda_run}."
                    if not conda_supports_no_capture_flag: error_msg += f" Check logs: STDOUT='{RELAUNCH_STDOUT_LOG}', STDERR='{RELAUNCH_STDERR_LOG}'"
                    print_error(error_msg)
                else:
                    success_msg = "'conda run' process completed successfully."
                    if not conda_supports_no_capture_flag: success_msg += f" Check logs: STDOUT='{RELAUNCH_STDOUT_LOG}', STDERR='{RELAUNCH_STDERR_LOG}'"
                    print_system(success_msg)
                sys.exit(exit_code_from_conda_run)
            else:
                print_error("Failed to start the 'conda run' process. Cannot proceed.")
                sys.exit(1)

        except FileNotFoundError as e_fnf:
            print_error(
                f"Failed to execute 'conda run' (cmd or component not found: {(conda_run_cmd_list[0] if conda_run_cmd_list else '<UNKNOWN>')}): {e_fnf}")
            sys.exit(1)
        except Exception as e_outer:
            print_error(f"An unexpected error occurred while trying to execute 'conda run' or wait for it: {e_outer}")
            traceback.print_exc()
            sys.exit(1)