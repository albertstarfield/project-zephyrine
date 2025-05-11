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
    with process_lock:
        procs_to_terminate = list(running_processes)
        running_processes.clear()

    for proc, name in reversed(procs_to_terminate):
        if proc.poll() is None:
            print_system(f"Terminating {name} (PID: {proc.pid})...")
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                    print_system(f"{name} terminated gracefully.")
                except subprocess.TimeoutExpired:
                    print_warning(f"{name} did not terminate gracefully, killing (PID: {proc.pid})...")
                    proc.kill()
                    proc.wait(timeout=2)
                    print_system(f"{name} killed.")
            except Exception as e:
                print_error(f"Error terminating/killing {name} (PID: {proc.pid}): {e}")
        else:
            print_system(f"{name} already exited (return code: {proc.poll()}).")


atexit.register(cleanup_processes)


def signal_handler(sig, frame):
    print_system("\nCtrl+C received. Initiating shutdown...")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


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
            if os.path.isdir(current_conda_env_path_check) and \
                    os.path.isdir(TARGET_CONDA_ENV_PATH) and \
                    os.path.normcase(os.path.realpath(current_conda_env_path_check)) == os.path.normcase(
                os.path.realpath(TARGET_CONDA_ENV_PATH)):
                is_already_in_correct_env = True
                ACTIVE_ENV_PATH = current_conda_env_path_check
        except FileNotFoundError:
            # This can happen if one of the paths is invalid (e.g., symlink broken)
            pass

    if is_already_in_correct_env:
        print_system(f"Running inside target Conda environment (Prefix: {ACTIVE_ENV_PATH})")
        print_system(f"Python executable in use: {sys.executable}")
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # +++ ADD DIAGNOSTIC PRINT AND SLEEP HERE FOR THE RELAUNCHED SCRIPT +++
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        print_system("<<<<< RELAUNCHED SCRIPT IS ALIVE AND RUNNING THIS LINE >>>>>")
        sys.stdout.flush()  # Ensure this diagnostic message is flushed
        time.sleep(3)  # Pause to make it very obvious if this part is reached
        print_system("<<<<< RELAUNCHED SCRIPT CONTINUING AFTER PAUSE >>>>>")
        sys.stdout.flush()
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # The ZEPHYRINE_RELAUNCHED_IN_CONDA flag is less critical with 'conda run'
        # as 'conda run' itself manages the environment activation robustly.
        # However, cleaning it up doesn't hurt if it was set by a previous os.execv attempt.
        if "ZEPHYRINE_RELAUNCHED_IN_CONDA" in os.environ:
            del os.environ["ZEPHYRINE_RELAUNCHED_IN_CONDA"]
    else:
        # This block executes if not already in the correct environment.
        # It will find conda, create env if needed, then relaunch using 'conda run'.
        print_system(f"--- Conda Environment Setup (Target Prefix: {TARGET_CONDA_ENV_PATH}) ---")
        if not find_conda_executable():
            print_error("Conda executable could not be located. Please install Anaconda/Miniconda. Exiting.")
            sys.exit(1)
        print_system(f"Using Conda executable: {CONDA_EXECUTABLE}")

        if current_conda_env_path_check:
            print_warning(
                f"Currently in Conda env '{current_conda_env_path_check}', but target is '{TARGET_CONDA_ENV_PATH}'. Will attempt to switch/relaunch into target.")
        else:
            print_system(
                "Not currently in an active Conda environment (CONDA_PREFIX not set or empty). Will attempt to use/create target.")

        if not (os.path.isdir(TARGET_CONDA_ENV_PATH) and os.path.exists(
                os.path.join(TARGET_CONDA_ENV_PATH, 'conda-meta'))):
            print_system(f"Target Conda environment prefix '{TARGET_CONDA_ENV_PATH}' not found or invalid. Creating...")
            target_python_versions = get_conda_python_versions_to_try()
            if not create_conda_env(TARGET_CONDA_ENV_PATH, target_python_versions):
                print_error(f"Failed to create the Conda environment at prefix '{TARGET_CONDA_ENV_PATH}'. Exiting.")
                sys.exit(1)
        else:
            print_system(f"Target Conda environment prefix '{TARGET_CONDA_ENV_PATH}' exists.")

        script_to_run_abs_path = os.path.abspath(__file__)

        # ---- Choose ONE of these conda_run_cmd_list constructions ----
        # Option 1: With --no-capture-output (preferred if your conda supports it)
        conda_run_cmd_list = [
                                 CONDA_EXECUTABLE,
                                 'run',
                                 '--no-capture-output',
                                 '--prefix', TARGET_CONDA_ENV_PATH,
                                 'python',
                                 script_to_run_abs_path
                             ] + sys.argv[1:]

        # Option 2: Without --no-capture-output (if flag is not supported)
        # conda_run_cmd_list = [
        #     CONDA_EXECUTABLE,
        #     'run',
        #     '--prefix', TARGET_CONDA_ENV_PATH,
        #     'python',
        #     script_to_run_abs_path
        # ] + sys.argv[1:]
        # ---- End of choice ----

        if IS_WINDOWS and CONDA_EXECUTABLE.lower().endswith(".bat"):
            # Reconstruct if it's a .bat for Windows
            if '--no-capture-output' in conda_run_cmd_list:
                base_conda_cmd = [CONDA_EXECUTABLE, 'run', '--no-capture-output', '--prefix', TARGET_CONDA_ENV_PATH,
                                  'python', script_to_run_abs_path] + sys.argv[1:]
            else:
                base_conda_cmd = [CONDA_EXECUTABLE, 'run', '--prefix', TARGET_CONDA_ENV_PATH, 'python',
                                  script_to_run_abs_path] + sys.argv[1:]
            conda_run_cmd_list = ['cmd', '/c'] + base_conda_cmd

        print_system(f"Relaunching script using 'conda run': {' '.join(conda_run_cmd_list)}")
        print_system(f"Stdout of relaunched script will be logged to: {RELAUNCH_STDOUT_LOG}")
        print_system(f"Stderr of relaunched script will be logged to: {RELAUNCH_STDERR_LOG}")

        # Ensure log directory exists
        os.makedirs(RELAUNCH_LOG_DIR, exist_ok=True)

        try:
            # Open log files in append mode ('a') or write mode ('w')
            # Using 'w' will clear the log on each relaunch attempt, good for fresh diagnostics
            # Using 'a' will append, good for seeing history across multiple attempts if needed
            with open(RELAUNCH_STDOUT_LOG, 'w', encoding='utf-8') as f_stdout, \
                    open(RELAUNCH_STDERR_LOG, 'w', encoding='utf-8') as f_stderr:

                process_conda_run = subprocess.Popen(
                    conda_run_cmd_list,
                    stdout=f_stdout,  # Redirect stdout to file
                    stderr=f_stderr,  # Redirect stderr to file
                    text=True,  # Still good practice for process interaction
                    errors='replace',
                    bufsize=1
                )

                # We are no longer streaming to parent's TTY for the relaunch itself.
                # Parent script will wait for conda run to finish.
                process_conda_run.wait()

                return_code = process_conda_run.returncode
                if return_code != 0:
                    print_error(
                        f"'conda run' process (executing the relaunched script) exited with code {return_code}.")
                    print_error(f"Check logs: STDOUT='{RELAUNCH_STDOUT_LOG}', STDERR='{RELAUNCH_STDERR_LOG}'")
                else:
                    print_system(f"'conda run' process completed. Check logs for output from relaunched script.")

                sys.exit(return_code)

        except Exception as e:
            print_error(f"Failed to execute 'conda run' and log its output: {e}")
            sys.exit(1)

    # --- From this point onwards, we are confirmed to be in the correct Conda environment ---
    # This ACTIVE_ENV_PATH should have been set if is_already_in_correct_env was true at the start.
    # If not, it implies an issue with the script's logic flow or Conda environment detection.
    if not ACTIVE_ENV_PATH:
        ACTIVE_ENV_PATH = os.getenv("CONDA_PREFIX")  # Try to get it again if somehow missed
        if not ACTIVE_ENV_PATH:
            print_error(
                "CRITICAL: CONDA_PREFIX is still not set after all checks. This should not happen if 'conda run' was successful or if already in env. Exiting.")
            sys.exit(1)
        # Final verification if ACTIVE_ENV_PATH was just re-fetched
        try:
            if not (os.path.isdir(ACTIVE_ENV_PATH) and \
                    os.path.isdir(TARGET_CONDA_ENV_PATH) and \
                    os.path.normcase(os.path.realpath(ACTIVE_ENV_PATH)) == os.path.normcase(
                        os.path.realpath(TARGET_CONDA_ENV_PATH))):
                print_error(
                    f"CRITICAL: Final check failed. CONDA_PREFIX '{ACTIVE_ENV_PATH}' (re-fetched) does not match target prefix '{TARGET_CONDA_ENV_PATH}'. Exiting.")
                sys.exit(1)
        except FileNotFoundError:
            print_error(
                f"CRITICAL: Final path comparison error for re-fetched CONDA_PREFIX '{ACTIVE_ENV_PATH}' and TARGET_CONDA_ENV_PATH '{TARGET_CONDA_ENV_PATH}'. Exiting.")
            sys.exit(1)
        print_system(
            f"Confirmed active Conda environment from CONDA_PREFIX (re-fetched): {os.path.basename(ACTIVE_ENV_PATH)} (Path: {ACTIVE_ENV_PATH})")

    # Define executable paths based on the active Conda environment (sys.executable should be from this env)
    _sys_exec_dir = os.path.dirname(sys.executable)
    PYTHON_EXECUTABLE = sys.executable
    PIP_EXECUTABLE = os.path.join(_sys_exec_dir, "pip.exe" if IS_WINDOWS else "pip")
    HYPERCORN_EXECUTABLE = os.path.join(_sys_exec_dir, "hypercorn.exe" if IS_WINDOWS else "hypercorn")

    if not os.path.exists(PIP_EXECUTABLE):
        print_warning(f"Pip executable not found at derived path {PIP_EXECUTABLE}. Trying shutil.which('pip')...")
        found_pip = shutil.which("pip")
        # Ensure the found pip is from the *current* active conda environment
        if found_pip and os.path.normcase(os.path.realpath(os.path.dirname(found_pip))).startswith(
                os.path.normcase(os.path.realpath(ACTIVE_ENV_PATH))):
            PIP_EXECUTABLE = found_pip
            print_system(f"Using pip found at: {PIP_EXECUTABLE}")
        else:
            print_error(
                f"Could not locate pip in the active Conda environment ({ACTIVE_ENV_PATH}). Searched {found_pip if found_pip else 'nowhere specific outside derived path'}. Exiting.")
            sys.exit(1)

    # --- Install windows-curses if on Windows (for license prompt) ---
    if IS_WINDOWS:
        print_system("Checking/Installing windows-curses for the license prompt...")
        if not run_command([PIP_EXECUTABLE, "install", "windows-curses"], ROOT_DIR, "PIP-WINCURSES"):
            print_error("Failed to install windows-curses. License prompt may not work. Exiting.")
            sys.exit(1)
        try:
            import curses
        except ImportError:
            print_error("Failed to import curses even after attempting install of windows-curses. Exiting.")
            sys.exit(1)
    else:  # For non-Windows, curses should be part of standard library or provided by python install
        try:
            import curses
        except ImportError:
            print_error(
                "Failed to import curses on non-Windows system. Python build might be incomplete or incompatible. Exiting.")
            sys.exit(1)

    # --- Install Core Helper Utilities (tqdm, requests) ---
    print_system("--- Installing Core Helper Utilities (tqdm, requests) ---")
    if not run_command([PIP_EXECUTABLE, "install", "tqdm", "requests"], ROOT_DIR, "PIP-UTILS"):
        print_error("Failed to install tqdm/requests. These are essential. Exiting.")
        sys.exit(1)
    print_system("Core helper utilities (tqdm, requests) installed/checked.")
    try:
        import requests
    except ImportError:
        print_error("Failed to import 'requests' after attempting installation. Exiting.")
        sys.exit(1)

    requests_session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=20)
    requests_session.mount('http://', adapter)
    requests_session.mount('https://', adapter)

    # --- Install Engine Dependencies (requirements.txt) ---
    print_system("--- Installing/Checking Python Dependencies (Engine) from requirements.txt ---")
    engine_req_path = os.path.join(ENGINE_MAIN_DIR, "requirements.txt")
    if not os.path.exists(engine_req_path):
        print_error(f"Engine requirements.txt not found at {engine_req_path}")
        sys.exit(1)
    if not run_command([PIP_EXECUTABLE, "install", "-r", engine_req_path], ENGINE_MAIN_DIR, "PIP-ENGINE-REQ"):
        print_error("Failed to install Python dependencies for Engine. Exiting.")
        sys.exit(1)
    print_system("Standard Python dependencies for Engine checked/installed.")

    # --- Re-evaluate TIKTOKEN_AVAILABLE ---
    try:
        import tiktoken

        TIKTOKEN_AVAILABLE = True
        print_system("tiktoken is available.")
    except ImportError:
        TIKTOKEN_AVAILABLE = False
        print_warning(
            "tiktoken is NOT available after dependency install. License reading time estimation will use default.")

    # --- License Acceptance Step ---
    if not os.path.exists(LICENSE_FLAG_FILE):
        print_system("License agreement required for first run or if flag file is missing.")
        try:
            _, combined_license_text = load_licenses()
            estimated_reading_seconds = calculate_reading_time(combined_license_text)
            license_lines = combined_license_text.splitlines()

            accepted, time_taken = curses.wrapper(display_license_prompt, license_lines, estimated_reading_seconds)

            if not accepted:
                print_error("License terms not accepted. Exiting application.")
                sys.exit(1)
            else:
                try:
                    with open(LICENSE_FLAG_FILE, 'w', encoding='utf-8') as f:
                        f.write(f"Accepted on: {datetime.now().isoformat()}\n")
                        f.write(f"Time taken to review: {time_taken:.2f} seconds\n")
                    print_system("License acceptance recorded.")
                except IOError as flag_err:
                    print_error(
                        f"Critical: Could not create license acceptance flag file '{LICENSE_FLAG_FILE}': {flag_err}")
                    sys.exit(1)

                print_system(f"Licenses accepted by user in {time_taken:.2f} seconds.")
                MIN_REASONABLE_TIME_FACTOR = 0.1
                if estimated_reading_seconds > 30 and time_taken < (
                        estimated_reading_seconds * MIN_REASONABLE_TIME_FACTOR):
                    print_warning(
                        f"Warning: Licenses were accepted very quickly ({time_taken:.2f}s vs estimated {estimated_reading_seconds:.2f}s).")
                    print_warning("Please ensure you have understood the terms.")
                    time.sleep(3)

        except curses.error as e:
            print_error(f"A Curses error occurred during license display: {e}")
            sys.exit(1)
        except Exception as e:
            print_error(f"An unexpected error occurred during the license acceptance process: {e}")
            try:
                curses.endwin()
            except:
                pass
            sys.exit(1)
    else:
        print_system(f"License previously accepted (flag file found: {LICENSE_FLAG_FILE}).")

    # --- Static Model Pool Setup ---
    print_system(f"--- Checking/Populating Static Model Pool at {STATIC_MODEL_POOL_PATH} ---")
    if not os.path.isdir(STATIC_MODEL_POOL_PATH):
        print_system(f"Static model pool directory not found. Creating: {STATIC_MODEL_POOL_PATH}")
        try:
            os.makedirs(STATIC_MODEL_POOL_PATH, exist_ok=True)
        except OSError as e:
            print_error(f"Failed to create static model pool directory '{STATIC_MODEL_POOL_PATH}': {e}")
            sys.exit(1)

    all_models_present_and_correct = True
    for model_info in MODELS_TO_DOWNLOAD:
        model_dest_path = os.path.join(STATIC_MODEL_POOL_PATH, model_info["filename"])
        if not os.path.exists(model_dest_path):
            print_warning(f"Model '{model_info['description']}' ({model_info['filename']}) not found. Downloading.")
            if not download_file_with_progress(model_info["url"], model_dest_path, model_info["description"],
                                               requests_session):
                print_error(f"Failed to download '{model_info['filename']}'.")
                all_models_present_and_correct = False
            else:
                print_system(f"Model '{model_info['filename']}' downloaded.")
        else:
            print_system(f"Model '{model_info['description']}' ({model_info['filename']}) already present.")

    if not all_models_present_and_correct:
        print_error("One or more models could not be downloaded. Functionality might be impaired.")
    else:
        print_system("All required models for static pool are present or downloaded.")

    # --- MeloTTS Installation & Initial Test ---
    if not os.path.exists(MELO_TTS_INSTALLED_FLAG_FILE):
        print_system(f"--- MeloTTS First-Time Setup from {MELO_TTS_PATH} ---")
        if not os.path.isdir(MELO_TTS_PATH):
            print_error(f"MeloTTS submodule directory not found at: {MELO_TTS_PATH}")
            print_error(
                "Please ensure MeloTTS is cloned or placed there (e.g., 'git submodule update --init --recursive').")
            sys.exit(1)

        print_system("Installing MeloTTS in editable mode...")
        if not run_command([PIP_EXECUTABLE, "install", "-e", "."], MELO_TTS_PATH, "PIP-MELO-EDITABLE"):
            print_error("Failed to install MeloTTS (editable). Check pip logs. Exiting.")
            sys.exit(1)

        print_system("Downloading 'unidic' dictionary for MeloTTS (for Japanese)...")
        if not run_command([PYTHON_EXECUTABLE, "-m", "unidic", "download"], MELO_TTS_PATH, "UNIDIC-DOWNLOAD"):
            print_warning("Failed to download 'unidic' dictionary. Japanese TTS might not work.")
        else:
            print_system("'unidic' dictionary downloaded/present.")

        print_system("--- Running MeloTTS Initial Test (triggers internal model downloads) ---")
        audio_worker_script = os.path.join(ENGINE_MAIN_DIR, "audio_worker.py")
        if not os.path.exists(audio_worker_script):
            print_error(f"audio_worker.py not found at '{audio_worker_script}'. Exiting.")
            sys.exit(1)

        test_mode_temp_dir = os.path.join(ROOT_DIR, "temp_melo_audio_test_files")
        os.makedirs(test_mode_temp_dir, exist_ok=True)
        test_output_filename = f"initial_melo_test_{datetime.now().strftime('%Y%m%d%H%M%S')}.wav"
        abs_test_output_file = os.path.join(test_mode_temp_dir, test_output_filename)

        test_command = [
            PYTHON_EXECUTABLE, audio_worker_script, "--test-mode",
            "--model-lang", "EN", "--device", "auto",
            "--output-file", abs_test_output_file, "--temp-dir", test_mode_temp_dir
        ]
        if not run_command(test_command, ENGINE_MAIN_DIR, "MELO-INIT-TEST"):
            print_warning("MeloTTS initial test failed. TTS functionality could be impaired.")
        else:
            print_system("MeloTTS initial test completed.")
            if os.path.exists(abs_test_output_file):
                print_system(f"Test audio file at: {abs_test_output_file}")
            else:
                print_warning(f"Test audio file '{abs_test_output_file}' not found despite test success.")
        try:
            with open(MELO_TTS_INSTALLED_FLAG_FILE, 'w', encoding='utf-8') as f:
                f.write(f"MeloTTS installed and tested on: {datetime.now().isoformat()}\n")
            print_system("MeloTTS installation flag created.")
        except IOError as flag_err:
            print_error(f"Could not create MeloTTS flag file '{MELO_TTS_INSTALLED_FLAG_FILE}': {flag_err}")
    else:
        print_system("MeloTTS previously installed/tested (flag file found).")

    # --- Custom llama-cpp-python Installation ---
    provider_env = os.getenv("PROVIDER", "llama_cpp").lower()
    if provider_env == "llama_cpp":
        if not os.path.exists(CUSTOM_LLAMA_CPP_INSTALLED_FLAG_FILE):
            print_system(f"--- Custom llama-cpp-python Installation (PROVIDER=llama_cpp) ---")

            if not shutil.which(GIT_CMD):
                print_error(
                    f"'{GIT_CMD}' not found. Git is required. Install from https://git-scm.com/downloads. Exiting.")
                sys.exit(1)

            print_system("Attempting to uninstall any existing standard 'llama-cpp-python'...")
            run_command([PIP_EXECUTABLE, "uninstall", "llama-cpp-python", "-y"], ROOT_DIR, "PIP-UNINSTALL-LLAMA",
                        check=False)

            if os.path.exists(LLAMA_CPP_PYTHON_CLONE_PATH):
                print_system(f"Cleaning previous llama-cpp-python build directory: {LLAMA_CPP_PYTHON_CLONE_PATH}")
                try:
                    shutil.rmtree(LLAMA_CPP_PYTHON_CLONE_PATH)
                except Exception as e:
                    print_error(f"Failed to remove '{LLAMA_CPP_PYTHON_CLONE_PATH}': {e}. Remove manually. Exiting.");
                    sys.exit(1)

            print_system(f"Cloning '{LLAMA_CPP_PYTHON_REPO_URL}' into '{LLAMA_CPP_PYTHON_CLONE_PATH}'...")
            if not run_command([GIT_CMD, "clone", LLAMA_CPP_PYTHON_REPO_URL, LLAMA_CPP_PYTHON_CLONE_PATH], ROOT_DIR,
                               "GIT-CLONE-LLAMA"):
                print_error("Failed to clone llama-cpp-python. Exiting.");
                sys.exit(1)

            print_system("Initializing/updating llama.cpp submodule...")
            if not run_command([GIT_CMD, "submodule", "update", "--init", "--recursive"], LLAMA_CPP_PYTHON_CLONE_PATH,
                               "GIT-SUBMODULE-LLAMA"):
                print_error("Failed to init/update llama-cpp-python submodules. Exiting.");
                sys.exit(1)

            build_env = {'FORCE_CMAKE': '1'}
            cmake_args_list = ["-DLLAMA_BUILD_EXAMPLES=OFF", "-DLLAMA_BUILD_TESTS=OFF"]

            default_backend = 'cpu'
            if sys.platform == "darwin": default_backend = 'metal'
            llama_backend = os.getenv("LLAMA_CPP_BACKEND", default_backend).lower()
            print_system(f"Configuring llama.cpp build for backend: {llama_backend}")

            if llama_backend == "cuda":
                cmake_args_list.append("-DGGML_CUDA=ON")
            elif llama_backend == "metal":
                if sys.platform != "darwin": print_warning("Metal backend selected, but system is not macOS.")
                cmake_args_list.append("-DGGML_METAL=ON")
                if platform.machine() == "arm64":
                    cmake_args_list.extend(["-DCMAKE_SYSTEM_PROCESSOR=arm64", "-DCMAKE_OSX_ARCHITECTURES=arm64"])
            elif llama_backend == "rocm":
                cmake_args_list.append("-DGGML_HIPBLAS=ON")
            elif llama_backend == "openblas":
                cmake_args_list.extend(["-DGGML_BLAS=ON", "-DGGML_BLAS_VENDOR=OpenBLAS"])
            elif llama_backend == "vulkan":
                cmake_args_list.append("-DGGML_VULKAN=ON")
            elif llama_backend == "sycl":
                cmake_args_list.append("-DGGML_SYCL=ON")
            elif llama_backend == "rpc":
                cmake_args_list.append("-DGGML_RPC=ON")
            elif llama_backend == "cpu":
                print_system("Building llama.cpp for CPU with OpenMP (if available).")
                cmake_args_list.append("-DLLAMA_OPENMP=ON")
                if sys.platform.startswith("linux"):
                    print_warning(
                        "For CPU builds on Linux with Conda, ensure Conda's OpenMP runtime is available (e.g., 'conda install libgomp' if using conda's gcc, or system 'libgomp1').")
            else:
                print_warning(f"Unknown LLAMA_CPP_BACKEND '{llama_backend}'. Defaulting to CPU-only build.")
                cmake_args_list.append("-DLLAMA_OPENMP=ON")

            effective_cmake_args = " ".join(filter(None, cmake_args_list))
            if effective_cmake_args: build_env['CMAKE_ARGS'] = effective_cmake_args

            print_system(
                f"Running pip install for custom llama-cpp-python from '{LLAMA_CPP_PYTHON_CLONE_PATH}' with CMAKE_ARGS: {build_env.get('CMAKE_ARGS', 'None')}")
            pip_install_command = [PIP_EXECUTABLE, "install", ".", "--upgrade", "--no-cache-dir", "--verbose"]

            if not run_command(pip_install_command, LLAMA_CPP_PYTHON_CLONE_PATH, "PIP-BUILD-LLAMA",
                               env_override=build_env):
                print_error("Failed to build/install custom llama-cpp-python. Check logs. Exiting.")
                sys.exit(1)

            print_system("Custom llama-cpp-python built and installed successfully.")
            try:
                with open(CUSTOM_LLAMA_CPP_INSTALLED_FLAG_FILE, 'w', encoding='utf-8') as f:
                    f.write(
                        f"Custom Llama CPP Python (backend: {llama_backend}) installed on: {datetime.now().isoformat()}\n")
                print_system("Custom llama-cpp-python installation flag created.")
            except IOError as flag_err:
                print_error(f"Could not create custom llama-cpp-python flag file: {flag_err}")
        else:
            print_system("Custom llama-cpp-python previously installed (flag file found). Skipping build.")
    else:
        print_system(
            f"PROVIDER is '{provider_env}'. Skipping custom llama-cpp-python. Standard version from reqs will be used if listed.")

    # --- Custom stable-diffusion-cpp-python Installation ---
    if not os.path.exists(CUSTOM_SD_CPP_PYTHON_INSTALLED_FLAG_FILE):
        print_system(f"--- Custom stable-diffusion-cpp-python Installation ---")
        if not shutil.which(GIT_CMD):
            print_error(f"'{GIT_CMD}' not found. Git is required. Exiting.");
            sys.exit(1)
        if not shutil.which(CMAKE_CMD):
            print_error(
                f"'{CMAKE_CMD}' not found. CMake is required. Install from https://cmake.org/download/ or via Conda ('conda install cmake'). Exiting.")
            sys.exit(1)

        try:
            cmake_ver_proc = subprocess.run([CMAKE_CMD, "--version"], capture_output=True, text=True, check=True,
                                            timeout=5)
            cmake_version = cmake_ver_proc.stdout.splitlines()[0].split()[-1]
            print_system(f"Found CMake version: {cmake_version}")
            major, minor = map(int, cmake_version.split('.')[:2])
            if major < 3 or (major == 3 and minor < 13):
                print_warning(f"CMake version {cmake_version} might be too old (>=3.13 recommended).")
        except Exception as cmake_err:
            print_warning(f"Could not verify CMake version: {cmake_err}.")

        print_system("Uninstalling existing 'stable-diffusion-cpp-python'...")
        run_command([PIP_EXECUTABLE, "uninstall", "stable-diffusion-cpp-python", "-y"], ROOT_DIR, "PIP-UNINSTALL-SD",
                    check=False)

        if os.path.exists(STABLE_DIFFUSION_CPP_PYTHON_CLONE_PATH):
            print_system(f"Cleaning existing SD build directory: {STABLE_DIFFUSION_CPP_PYTHON_CLONE_PATH}")
            try:
                shutil.rmtree(STABLE_DIFFUSION_CPP_PYTHON_CLONE_PATH)
            except Exception as e:
                print_error(f"Failed to remove SD dir: {e}. Exiting."); sys.exit(1)

        print_system(
            f"Cloning '{STABLE_DIFFUSION_CPP_PYTHON_REPO_URL}' into '{STABLE_DIFFUSION_CPP_PYTHON_CLONE_PATH}'...")
        if not run_command([GIT_CMD, "clone", "--recursive", STABLE_DIFFUSION_CPP_PYTHON_REPO_URL,
                            STABLE_DIFFUSION_CPP_PYTHON_CLONE_PATH], ROOT_DIR, "GIT-CLONE-SD"):
            print_error("Failed to clone SD repo. Exiting.");
            sys.exit(1)

        sd_cpp_submodule_path = os.path.join(STABLE_DIFFUSION_CPP_PYTHON_CLONE_PATH, "vendor", "stable-diffusion.cpp")
        if not os.path.isdir(sd_cpp_submodule_path):
            print_error(f"SD submodule not found at: {sd_cpp_submodule_path}. Exiting.");
            sys.exit(1)

        sd_cpp_build_path = os.path.join(sd_cpp_submodule_path, "build")
        os.makedirs(sd_cpp_build_path, exist_ok=True)

        print_system("--- Configuring stable-diffusion.cpp C++ library build ---")
        cmake_args_sd_cpp = ["-DCMAKE_POLICY_VERSION_MINIMUM=3.13"]
        sd_backend = os.getenv("SD_CPP_BACKEND", "").lower()
        print_system(f"Selected SD_CPP_BACKEND: '{sd_backend or 'auto/default'}'")

        if sd_backend == "cuda" or (not sd_backend and sys.platform in ["linux", "win32"] and shutil.which('nvcc')):
            cmake_args_sd_cpp.append("-DSD_CUDA=ON")
        elif sd_backend == "hipblas" or (not sd_backend and sys.platform == "linux" and os.path.isdir("/opt/rocm")):
            cmake_args_sd_cpp.append("-DSD_HIPBLAS=ON")
        elif sd_backend == "metal" or (not sd_backend and sys.platform == "darwin"):
            cmake_args_sd_cpp.append("-DSD_METAL=ON")
        elif sd_backend == "openblas":
            cmake_args_sd_cpp.append("-DGGML_OPENBLAS=ON")
        elif sd_backend == "vulkan":
            cmake_args_sd_cpp.append("-DSD_VULKAN=ON")
        elif sd_backend == "sycl":
            cmake_args_sd_cpp.append("-DSD_SYCL=ON")
        elif sd_backend == "musa":
            cmake_args_sd_cpp.append("-DSD_MUSA=ON")
        else:
            print_system("Configuring SD for CPU backend.")

        if os.getenv("SD_FLASH_ATTENTION", "OFF").upper() == "ON":
            cmake_args_sd_cpp.append("-DSD_FLASH_ATTN=ON")

        cmake_configure_command = [CMAKE_CMD, ".."] + cmake_args_sd_cpp
        if not run_command(cmake_configure_command, sd_cpp_build_path, "CMAKE-SD-LIB-CFG"):
            print_error("CMake config failed for SD library. Exiting.");
            sys.exit(1)

        print_system("--- Building stable-diffusion.cpp C++ library ---")
        cmake_build_command = [CMAKE_CMD, "--build", "."]
        if IS_WINDOWS: cmake_build_command.extend(["--config", "Release"])
        if not run_command(cmake_build_command, sd_cpp_build_path, "CMAKE-BUILD-SD-LIB"):
            print_error("Build failed for SD library. Exiting.");
            sys.exit(1)
        print_system("Stable-diffusion.cpp C++ library built successfully.")

        print_system(f"--- Installing stable-diffusion-cpp-python bindings ---")
        pip_build_env = os.environ.copy()
        pip_build_env['FORCE_CMAKE'] = '1'
        if sd_backend == "hipblas" and sys.platform == "linux":
            print_system("Setting CC=clang, CXX=clang++ for ROCm pip build environment.")
            pip_build_env['CC'] = 'clang'
            pip_build_env['CXX'] = 'clang++'
        elif sd_backend == "sycl":
            print_system("Setting CC=icx, CXX=icpx for SYCL pip build environment.")
            pip_build_env['CC'] = 'icx'
            pip_build_env['CXX'] = 'icpx'
        elif sd_backend == "musa":
            print_system("Setting CC/CXX to MUSA compilers for MUSA pip build environment.")
            musa_clang = "/usr/local/musa/bin/clang"
            musa_clang_pp = "/usr/local/musa/bin/clang++"
            if not os.path.exists(musa_clang) or not os.path.exists(musa_clang_pp):
                print_error(f"MUSA compilers not found at: {musa_clang}, {musa_clang_pp}")
                sys.exit(1)
            pip_build_env['CC'] = musa_clang
            pip_build_env['CXX'] = musa_clang_pp

        pip_cmake_args_str = " ".join(cmake_args_sd_cpp)
        if pip_cmake_args_str:
            pip_build_env['CMAKE_ARGS'] = pip_cmake_args_str
            print_system(f"Setting CMAKE_ARGS for pip install: {pip_cmake_args_str}")

        pip_install_sd_command = [PIP_EXECUTABLE, "install", ".", "--upgrade", "--no-cache-dir", "--verbose"]
        if not run_command(pip_install_sd_command, STABLE_DIFFUSION_CPP_PYTHON_CLONE_PATH, "PIP-SD-BINDINGS",
                           env_override=pip_build_env):
            print_error("Failed to install stable-diffusion-cpp-python bindings.")
            print_error("Check pip build logs above. Ensure C++ library built correctly and setup.py can find it.")
            sys.exit(1)
        print_system("Custom stable-diffusion-cpp-python bindings installed successfully.")

        try:
            with open(CUSTOM_SD_CPP_PYTHON_INSTALLED_FLAG_FILE, 'w', encoding='utf-8') as f:
                f.write(
                    f"Custom Stable Diffusion CPP Python (backend: {sd_backend or 'cpu'}) installed on: {datetime.now().isoformat()}\n")
            print_system("Custom stable-diffusion-cpp-python installation flag created.")
        except IOError as flag_err:
            print_error(
                f"Could not create custom stable-diffusion-cpp-python installation flag file '{CUSTOM_SD_CPP_PYTHON_INSTALLED_FLAG_FILE}': {flag_err}")
    else:
        print_system("Custom stable-diffusion-cpp-python previously installed (flag file found). Skipping build.")

    # --- Node.js Dependencies ---
    print_system("--- Installing/Checking Node.js Backend Dependencies ---")
    if not shutil.which(NPM_CMD.split('.')[0]):
        print_error(
            f"'{NPM_CMD}' not found. Please install Node.js and npm, or ensure it's in PATH (e.g. 'conda install nodejs'). Exiting.")
        sys.exit(1)

    backend_pkg_path = os.path.join(BACKEND_SERVICE_DIR, "package.json")
    if not os.path.exists(backend_pkg_path): print_error(
        f"Backend package.json not found at {backend_pkg_path}. Cannot install dependencies. Exiting."); sys.exit(1)
    if not run_command([NPM_CMD, "install"], BACKEND_SERVICE_DIR, "NPM-BACKEND"):
        print_error("Failed to install Node.js backend dependencies. Check npm logs. Exiting.");
        sys.exit(1)
    print_system("Node.js backend dependencies checked/installed.")

    print_system("--- Installing/Checking Node.js Frontend Dependencies ---")
    frontend_pkg_path = os.path.join(FRONTEND_DIR, "package.json")
    if not os.path.exists(frontend_pkg_path): print_error(
        f"Frontend package.json not found at {frontend_pkg_path}. Cannot install dependencies. Exiting."); sys.exit(1)
    if not run_command([NPM_CMD, "install"], FRONTEND_DIR, "NPM-FRONTEND"):
        print_error("Failed to install Node.js frontend dependencies. Check npm logs. Exiting.");
        sys.exit(1)
    print_system("Node.js frontend dependencies checked/installed.")

    # --- Start All Services ---
    print_system("--- Starting All Services ---")
    service_threads = []
    service_threads.append(start_service_thread(start_engine_main, "EngineMainThread"))

    print_system("Waiting for Engine Main (Hypercorn) to initialize (up to 10 seconds)...")
    time.sleep(2)
    engine_ready = False
    engine_exited = False
    for _ in range(8):
        with process_lock:
            for proc, name in running_processes:
                if name == "ENGINE":
                    if proc.poll() is None:
                        engine_ready = True; break
                    else:
                        print_error(
                            f"Engine Main (Hypercorn) seems to have exited prematurely with code {proc.poll()}."); engine_exited = True; break
        if engine_ready or engine_exited: break
        time.sleep(1)

    if not engine_ready and not engine_exited:
        with process_lock:
            for proc, name in running_processes:
                if name == "ENGINE" and proc.poll() is None: engine_ready = True

    if not engine_ready:
        print_error("Engine Main (Hypercorn) failed to start or stay running. Check logs above. Exiting.")
        sys.exit(1)
    print_system("Engine Main appears to be running.")

    service_threads.append(start_service_thread(start_backend_service, "BackendServiceThread"))
    time.sleep(2)
    service_threads.append(start_service_thread(start_frontend, "FrontendThread"))

    print_system("All services are being started. Launcher will monitor them. Press Ctrl+C to shut down.")

    # --- Monitoring Loop ---
    try:
        while True:
            active_managed_process_found = False
            all_processes_ok = True

            with process_lock:
                current_procs_snapshot = list(running_processes)
                procs_to_remove = []

            if not current_procs_snapshot and service_threads:
                print_warning(
                    "No managed processes are running, but service threads exist. This might indicate startup failures not caught earlier.")
                all_processes_ok = False

            for proc, name in current_procs_snapshot:
                if proc.poll() is None:
                    active_managed_process_found = True
                else:
                    print_error(f"Service '{name}' has exited unexpectedly with return code {proc.poll()}.")
                    procs_to_remove.append((proc, name))
                    all_processes_ok = False

            if procs_to_remove:
                with process_lock:
                    for item in procs_to_remove:
                        if item in running_processes:
                            running_processes.remove(item)

            if not all_processes_ok:
                print_error("One or more services terminated. Initiating shutdown of remaining services.")
                break

            if not active_managed_process_found and service_threads:
                print_system("All managed services have finished or exited. Launcher will now shut down.")
                break

            if not service_threads:
                print_system("No services were configured to start. Setup complete. Exiting launcher.")
                break

            time.sleep(5)

    except KeyboardInterrupt:
        print_system("\nKeyboardInterrupt received by main thread. Shutting down...")
    finally:
        print_system("Launcher main loop finished. Ensuring cleanup...")