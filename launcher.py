#!/usr/bin/env python3
import os
import sys
import subprocess
import threading
import time
import signal
import atexit
import shutil
import platform # Added for architecture check
from datetime import datetime

try:
    import colorama
    colorama.init()
    COLOR_RESET = colorama.Style.RESET_ALL
    COLOR_ENGINE = colorama.Fore.CYAN + colorama.Style.BRIGHT
    COLOR_BACKEND = colorama.Fore.MAGENTA + colorama.Style.BRIGHT
    COLOR_FRONTEND = colorama.Fore.GREEN + colorama.Style.BRIGHT
    COLOR_SYSTEM = colorama.Fore.YELLOW + colorama.Style.BRIGHT
    COLOR_ERROR = colorama.Fore.RED + colorama.Style.BRIGHT
except ImportError:
    print("Warning: colorama not found. Logs will not be colored.")
    print("Install it using: pip install colorama")
    COLOR_RESET = ""
    COLOR_ENGINE = ""
    COLOR_BACKEND = ""
    COLOR_FRONTEND = ""
    COLOR_SYSTEM = ""
    COLOR_ERROR = ""

# --- Configuration ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(ROOT_DIR, "venv")
ENGINE_MAIN_DIR = os.path.join(ROOT_DIR, "systemCore", "engineMain")
BACKEND_SERVICE_DIR = os.path.join(ROOT_DIR, "systemCore", "backend-service")
FRONTEND_DIR = os.path.join(ROOT_DIR, "systemCore", "frontend-face-zephyrine")

# Define paths to executables within the virtual environment
IS_WINDOWS = os.name == 'nt'
VENV_BIN_DIR = os.path.join(VENV_DIR, "Scripts" if IS_WINDOWS else "bin")
PYTHON_EXECUTABLE = os.path.join(VENV_BIN_DIR, "python.exe" if IS_WINDOWS else "python")
PIP_EXECUTABLE = os.path.join(VENV_BIN_DIR, "pip.exe" if IS_WINDOWS else "pip")
HYPERCORN_EXECUTABLE = os.path.join(VENV_BIN_DIR, "hypercorn.exe" if IS_WINDOWS else "hypercorn")
NPM_CMD = 'npm.cmd' if IS_WINDOWS else 'npm'
GIT_CMD = 'git.exe' if IS_WINDOWS else 'git'

# Llama.cpp Fork Configuration (URLs/Branches are now informational only unless uncommented)
LLAMA_CPP_PYTHON_REPO_URL = "https://github.com/abetlen/llama-cpp-python.git"
LLAMA_CPP_PYTHON_CLONE_DIR_NAME = "llama-cpp-python_build" # Directory name within ROOT_DIR for the bindings repo
LLAMA_CPP_PYTHON_CLONE_PATH = os.path.join(ROOT_DIR, LLAMA_CPP_PYTHON_CLONE_DIR_NAME)
# LLAMA_CPP_FORK_URL = "https://github.com/HimariO/llama.cpp.qwen2vl.git" # Your C++ fork URL (INFO ONLY)
# LLAMA_CPP_FORK_BRANCH = "qwen25-vl-20250404" # Your specific C++ fork branch (INFO ONLY)
LLAMA_CPP_SUBMODULE_PATH = "vendor/llama.cpp" # Relative path within llama-cpp-python repo

# Global list to keep track of running subprocesses
running_processes = []
process_lock = threading.Lock()

# --- Helper Functions (print_colored, print_system, print_error, run_command, stream_output, etc.) ---
def print_colored(prefix, message, color):
    """Prints a message with a timestamp, colored prefix, and resets color."""
    now = datetime.now().strftime('%H:%M:%S.%f')[:-3] # HH:MM:SS.ms
    print(f"{color}[{now} | {prefix.ljust(10)}] {COLOR_RESET}{message.strip()}")

def print_system(message):
    print_colored("SYSTEM", message, COLOR_SYSTEM)

def print_error(message):
    print_colored("ERROR", message, COLOR_ERROR)

def print_warning(message):
    print_colored("WARNING", message, COLOR_SYSTEM) # Use yellow for warnings too

def run_command(command, cwd, name, color, check=True, capture_output=False, env_override=None):
    """
    Runs a command, optionally checks for errors, and can stream output.
    Allows overriding environment variables.
    """
    print_system(f"Running command in '{os.path.basename(cwd)}': {' '.join(command)}")
    current_env = os.environ.copy()
    if env_override:
        # Make sure env var values are strings
        str_env_override = {k: str(v) for k, v in env_override.items()}
        print_system(f"  with custom env: {str_env_override}")
        current_env.update(str_env_override)

    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1,
            shell=(IS_WINDOWS and command[0] in [NPM_CMD, 'node', GIT_CMD]),
            env=current_env
        )

        stdout_thread = threading.Thread(target=stream_output, args=(process.stdout, name, color), daemon=True)
        stdout_thread.start()
        stderr_thread = threading.Thread(target=stream_output, args=(process.stderr, f"{name}-ERR", COLOR_ERROR), daemon=True)
        stderr_thread.start()

        process.wait() # Wait for the process to complete
        stdout_thread.join() # Wait for the stream readers to finish
        stderr_thread.join()

        if check and process.returncode != 0:
            # Error message logged by stderr thread, raise exception to signal failure
            raise subprocess.CalledProcessError(process.returncode, command)

        print_system(f"Command finished successfully in '{os.path.basename(cwd)}': {' '.join(command)}")
        return True

    except FileNotFoundError:
        print_error(f"Command not found: {command[0]}. Is it installed and in PATH?")
        if command[0] == GIT_CMD: print_error("Ensure Git is installed: https://git-scm.com/downloads")
        elif command[0] in [NPM_CMD, 'node']: print_error("Ensure Node.js and npm are installed: https://nodejs.org/")
        elif command[0] == HYPERCORN_EXECUTABLE: print_error(f"Ensure '{os.path.basename(HYPERCORN_EXECUTABLE)}' is installed in the venv (check requirements.txt).")
        return False
    except subprocess.CalledProcessError as e:
        # Error message was already printed by the stderr thread
        print_error(f"Command failed in '{os.path.basename(cwd)}' with exit code {e.returncode}.")
        return False
    except Exception as e:
        print_error(f"An unexpected error occurred while running command in '{os.path.basename(cwd)}': {e}")
        return False

def stream_output(pipe, name, color):
    """Reads lines from a subprocess pipe and prints them with prefix and color."""
    try:
        for line in iter(pipe.readline, ''):
            if line:
                print_colored(name, line, color)
    except Exception as e:
        # Avoid printing error if pipe is closed expectedly during cleanup
        if 'read of closed file' not in str(e).lower():
            print_error(f"Error reading output stream for {name}: {e}")
    finally:
        if pipe:
            try:
                pipe.close()
            except Exception:
                pass # Ignore errors during pipe closing

def start_service_thread(target_func, name):
    """Starts a service in a daemon thread."""
    print_system(f"Preparing to start service: {name}")
    thread = threading.Thread(target=target_func, name=name, daemon=True)
    thread.start()
    return thread

def cleanup_processes():
    """Attempts to terminate all tracked subprocesses."""
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
                    proc.wait(timeout=3)
                    print_system(f"{name} terminated gracefully.")
                except subprocess.TimeoutExpired:
                    print_system(f"{name} did not terminate gracefully, killing...")
                    proc.kill()
                    proc.wait()
                    print_system(f"{name} killed.")
            except Exception as e:
                print_error(f"Error terminating {name}: {e}")
        else:
            print_system(f"{name} already exited (return code: {proc.poll()}).")

atexit.register(cleanup_processes)

def signal_handler(sig, frame):
    print_system("\nCtrl+C received. Initiating shutdown...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# --- Service Start Functions (start_engine_main, start_backend_service, start_frontend) ---
def start_engine_main():
    """Starts the Python Engine Main service using Hypercorn."""
    name = "ENGINE"
    color = COLOR_ENGINE
    command = [
        HYPERCORN_EXECUTABLE,
        "app:app",
        "--bind", "127.0.0.1:11434",
        "--workers", "1",
        "--log-level", "info"
    ]
    print_system(f"Launching Engine Main via Hypercorn: {' '.join(command)} in {ENGINE_MAIN_DIR}")
    try:
        process = subprocess.Popen(
            command,
            cwd=ENGINE_MAIN_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1
        )
        with process_lock:
            running_processes.append((process, name))

        stdout_thread = threading.Thread(target=stream_output, args=(process.stdout, name, color), daemon=True)
        stderr_thread = threading.Thread(target=stream_output, args=(process.stderr, f"{name}-ERR", COLOR_ERROR), daemon=True)
        stdout_thread.start()
        stderr_thread.start()

    except FileNotFoundError:
         print_error(f"Command failed: '{os.path.basename(HYPERCORN_EXECUTABLE)}' not found in venv.")
         print_error(f"Searched in: {HYPERCORN_EXECUTABLE}")
         print_error("Ensure 'hypercorn' is listed in systemCore/engineMain/requirements.txt.")
         sys.exit(1)
    except Exception as e:
        print_error(f"Failed to start {name}: {e}")
        sys.exit(1)

def start_backend_service():
    """Starts the Node.js Backend Service."""
    name = "BACKEND"
    color = COLOR_BACKEND
    command = ["node", "server.js"]
    print_system(f"Launching Backend Service: {' '.join(command)} in {BACKEND_SERVICE_DIR}")
    try:
        process = subprocess.Popen(
            command,
            cwd=BACKEND_SERVICE_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1,
            shell=IS_WINDOWS
        )
        with process_lock:
            running_processes.append((process, name))
        stdout_thread = threading.Thread(target=stream_output, args=(process.stdout, name, color), daemon=True)
        stderr_thread = threading.Thread(target=stream_output, args=(process.stderr, f"{name}-ERR", COLOR_ERROR), daemon=True)
        stdout_thread.start()
        stderr_thread.start()
    except FileNotFoundError:
        print_error("Command failed: 'node' not found. Is Node.js installed and in PATH?")
        sys.exit(1)
    except Exception as e:
        print_error(f"Failed to start {name}: {e}")

def start_frontend():
    """Starts the Vite Frontend Development Server."""
    name = "FRONTEND"
    color = COLOR_FRONTEND
    command = [NPM_CMD, "run", "dev"]
    print_system(f"Launching Frontend (Vite): {' '.join(command)} in {FRONTEND_DIR}")
    try:
        process = subprocess.Popen(
            command,
            cwd=FRONTEND_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1,
            shell=IS_WINDOWS
        )
        with process_lock:
            running_processes.append((process, name))
        stdout_thread = threading.Thread(target=stream_output, args=(process.stdout, name, color), daemon=True)
        stderr_thread = threading.Thread(target=stream_output, args=(process.stderr, f"{name}-ERR", COLOR_ERROR), daemon=True)
        stdout_thread.start()
        stderr_thread.start()
    except FileNotFoundError:
        print_error(f"Command failed: '{NPM_CMD}' not found. Is Node.js/npm installed and in PATH?")
        sys.exit(1)
    except Exception as e:
        print_error(f"Failed to start {name}: {e}")

# --- Main Execution Logic ---
if __name__ == "__main__":
    print_system("--- Project Zephyrine Launcher ---")

    # 1. Check/Create/Relaunch in Virtual Environment
    is_in_target_venv = (os.getenv('VIRTUAL_ENV') == VENV_DIR) or \
                         (hasattr(sys, 'prefix') and sys.prefix == VENV_DIR) or \
                         (hasattr(sys, 'real_prefix') and sys.real_prefix == VENV_DIR)

    if not is_in_target_venv:
        print_system(f"Not running in the target virtual environment: {VENV_DIR}")
        if not os.path.exists(VENV_DIR):
            print_system(f"Virtual environment not found. Creating it at: {VENV_DIR}")
            try:
                python_cmd_for_venv = sys.executable
                print_system(f"Using Python interpreter for venv creation: {python_cmd_for_venv}")
                subprocess.run([python_cmd_for_venv, "-m", "venv", VENV_DIR], check=True)
                print_system("Virtual environment created successfully.")
            except subprocess.CalledProcessError as e: print_error(f"Failed to create venv: {e}"); sys.exit(1)
            except Exception as e: print_error(f"Unexpected error during venv creation: {e}"); sys.exit(1)

        print_system(f"Relaunching script using Python from virtual environment: {PYTHON_EXECUTABLE}")
        if not os.path.exists(PYTHON_EXECUTABLE): print_error(f"Python executable not found in venv: {PYTHON_EXECUTABLE}"); sys.exit(1)
        try: os.execv(PYTHON_EXECUTABLE, [PYTHON_EXECUTABLE] + sys.argv)
        except Exception as e: print_error(f"Failed to relaunch script in venv: {e}"); sys.exit(1)

    # --- Running inside the correct venv ---
    print_system(f"Running inside the correct virtual environment: {VENV_DIR}")

    # 2. Install/Check Dependencies (Python)
    print_system("--- Installing/Checking Python Dependencies (Engine) ---")
    req_path = os.path.join(ENGINE_MAIN_DIR, "requirements.txt")
    if not os.path.exists(req_path): print_error(f"requirements.txt not found at {req_path}"); sys.exit(1)

    # Install standard requirements first
    if not run_command([PIP_EXECUTABLE, "install", "-r", req_path], ENGINE_MAIN_DIR, "PIP", COLOR_SYSTEM):
         print_error("Failed to install Python dependencies from requirements.txt. Exiting.")
         sys.exit(1)
    print_system("Standard Python dependencies checked/installed.")

    # --- Custom llama-cpp-python Installation ---
    provider_env = os.getenv("PROVIDER", "llama_cpp").lower()
    print_system(f"Checking PROVIDER environment variable: {provider_env}")

    if provider_env == "llama_cpp":
        print_system(f"--- Attempting Custom llama-cpp-python Installation (Using Default Submodule) ---")

        # Check for Git
        if not shutil.which(GIT_CMD): print_error(f"'{GIT_CMD}' command not found. Git is required. Exiting."); sys.exit(1)

        # Uninstall standard llama-cpp-python if it exists
        print_system("Attempting to uninstall standard llama-cpp-python (if present)...")
        run_command([PIP_EXECUTABLE, "uninstall", "llama-cpp-python", "-y"], ROOT_DIR, "PIP", COLOR_SYSTEM, check=False)

        # Clone llama-cpp-python repo if it doesn't exist
        if not os.path.exists(LLAMA_CPP_PYTHON_CLONE_PATH):
            print_system(f"Cloning '{LLAMA_CPP_PYTHON_REPO_URL}' into '{LLAMA_CPP_PYTHON_CLONE_PATH}'...")
            if not run_command([GIT_CMD, "clone", LLAMA_CPP_PYTHON_REPO_URL, LLAMA_CPP_PYTHON_CLONE_PATH], ROOT_DIR, "GIT", COLOR_SYSTEM):
                print_error("Failed to clone llama-cpp-python repository. Exiting."); sys.exit(1)
        else:
            print_system(f"llama-cpp-python directory already exists at '{LLAMA_CPP_PYTHON_CLONE_PATH}'. Skipping clone.")

        # Initialize/Update submodules within llama-cpp-python repo
        # This command will pull the default llama.cpp commit specified by llama-cpp-python
        print_system("Initializing/updating submodules within llama-cpp-python (using default llama.cpp commit)...")
        if not run_command([GIT_CMD, "submodule", "update", "--init", "--recursive"], LLAMA_CPP_PYTHON_CLONE_PATH, "GIT", COLOR_SYSTEM):
            print_error("Failed to update submodules. Exiting."); sys.exit(1)

        # --- REMOVED/COMMENTED OUT: Fork Manipulation ---
        # Navigate into the submodule and point it to the custom fork/branch
        submodule_full_path = os.path.join(LLAMA_CPP_PYTHON_CLONE_PATH, LLAMA_CPP_SUBMODULE_PATH)
        if not os.path.isdir(submodule_full_path):
             print_error(f"llama.cpp submodule directory not found at: {submodule_full_path}"); sys.exit(1)
        #
        # print_system(f"Skipping update of submodule to point to specific fork/branch.")
        # # Check remote command removed
        # # Add remote command removed
        # # Fetch command removed
        # # Checkout command removed
        # --- END REMOVED/COMMENTED OUT ---


        # --- Prepare environment for build with Dynamic CMAKE_ARGS ---
        print_system("Preparing environment for llama.cpp build...")
        build_env = {'FORCE_CMAKE': '1'}
        cmake_args_list = []

        # Read desired backend from environment variable
        default_backend = 'cpu'
        if sys.platform == "darwin": default_backend = 'metal'
        elif sys.platform in ["linux", "win32"]: default_backend = 'cuda'

        llama_backend = os.getenv("LLAMA_CPP_BACKEND", default_backend).lower()
        print_system(f"Selected llama.cpp backend: {llama_backend} (Control with LLAMA_CPP_BACKEND env var: cpu, cuda, metal, rocm, openblas, vulkan, sycl, rpc)")

        # Construct CMAKE_ARGS based on selected backend
        if llama_backend == "cuda":
            cmake_args_list.append("-DGGML_CUDA=on")
        elif llama_backend == "metal":
            if sys.platform != "darwin": print_warning("Metal backend selected but platform is not macOS. Build might fail.")
            cmake_args_list.append("-DGGML_METAL=on")
            if platform.machine() == "arm64":
                print_system("Adding Metal arm64 CMake flags for Apple Silicon.")
                cmake_args_list.extend(["-DCMAKE_OSX_ARCHITECTURES=arm64", "-DCMAKE_APPLE_SILICON_PROCESSOR=arm64"])
        elif llama_backend == "rocm":
            cmake_args_list.append("-DGGML_HIPBLAS=on")
            # AMDGPU_TARGETS might be needed, instruct user if build fails
            if not os.getenv("AMDGPU_TARGETS"):
                print_warning("ROCm backend selected. You might need to set the AMDGPU_TARGETS environment variable for your specific GPU (e.g., export AMDGPU_TARGETS=gfx1100).")
        elif llama_backend == "openblas":
            cmake_args_list.append("-DGGML_BLAS=ON")
            cmake_args_list.append("-DGGML_BLAS_VENDOR=OpenBLAS")
            if IS_WINDOWS: print_warning("Windows OpenBLAS build might require manual CMAKE_ARGS for compiler paths if auto-detection fails.")
        elif llama_backend == "vulkan":
            cmake_args_list.append("-DGGML_VULKAN=on")
            if not os.getenv("VULKAN_SDK"):
                 print_warning("Vulkan backend selected. Ensure Vulkan SDK is installed and VULKAN_SDK environment variable is set.")
        elif llama_backend == "sycl":
             cmake_args_list.append("-DGGML_SYCL=on")
             print_warning("SYCL backend selected. Ensure oneAPI is installed and you have sourced 'setvars.sh' (Linux) or run 'setvars.bat' (Windows) *before* launching this script.")
             print_warning("Also ensure icx/icpx compilers are in PATH or set CC/CXX environment variables.")
             # Example CMake args if CC/CXX not set - remove if user manages env:
             # cmake_args_list.extend(["-DCMAKE_C_COMPILER=icx", "-DCMAKE_CXX_COMPILER=icpx"])
        elif llama_backend == "rpc":
             cmake_args_list.append("-DGGML_RPC=on")
        elif llama_backend == "cpu":
             print_system("Building for CPU only (no BLAS/GPU acceleration flags).")
             cmake_args_list = [] # Ensure empty
        else:
            print_warning(f"Unknown LLAMA_CPP_BACKEND '{llama_backend}'. Building for CPU only.")
            cmake_args_list = []

        # Combine args into string and add to build environment
        if cmake_args_list:
            build_env['CMAKE_ARGS'] = " ".join(cmake_args_list)

        # Install llama-cpp-python from its local directory (now containing default submodule)
        print_system(f"Running pip install from '{LLAMA_CPP_PYTHON_CLONE_PATH}' with build flags...")
        pip_install_command = [
            PIP_EXECUTABLE, "install", ".",
            "--upgrade" # Ensure rebuild
        ]
        if not run_command(pip_install_command, LLAMA_CPP_PYTHON_CLONE_PATH, "PIP-LLAMA", COLOR_SYSTEM, env_override=build_env):
            print_error("Failed to install custom llama-cpp-python build. Exiting.")
            sys.exit(1)

        print_system("Custom llama-cpp-python build (using default submodule) installed successfully.")

    # --- Continue with other dependencies (Node.js Backend/Frontend) ---
    # 3. Install/Check Dependencies (Node.js Backend Service)
    print_system("--- Installing/Checking Node Backend Dependencies ---")
    pkg_path = os.path.join(BACKEND_SERVICE_DIR, "package.json")
    if not os.path.exists(pkg_path): print_error(f"package.json not found at {pkg_path}"); sys.exit(1)
    if not run_command([NPM_CMD, "install"], BACKEND_SERVICE_DIR, "NPM-BE", COLOR_SYSTEM):
         print_error("Failed to install Node backend dependencies. Exiting."); sys.exit(1)
    print_system("Node backend dependencies checked/installed.")

    # 4. Install/Check Dependencies (Node.js Frontend)
    print_system("--- Installing/Checking Node Frontend Dependencies ---")
    pkg_path_fe = os.path.join(FRONTEND_DIR, "package.json")
    if not os.path.exists(pkg_path_fe): print_error(f"package.json not found at {pkg_path_fe}"); sys.exit(1)
    if not run_command([NPM_CMD, "install"], FRONTEND_DIR, "NPM-FE", COLOR_SYSTEM):
         print_error("Failed to install Node frontend dependencies. Exiting."); sys.exit(1)
    print_system("Node frontend dependencies checked/installed.")


    # 5. Start all services concurrently
    print_system("--- Starting Services ---")
    threads = []
    threads.append(start_service_thread(start_engine_main, "EngineMainThread"))
    print_system("Waiting a few seconds for Engine to initialize...")
    time.sleep(5) # Allow time for Hypercorn to bind and app.py to load

    engine_proc_ok = False
    with process_lock:
        for proc, name in running_processes:
            if name == "ENGINE" and proc.poll() is None: engine_proc_ok = True; break
    if not engine_proc_ok: print_error("Engine Main (Hypercorn) failed to stay running. Check ENGINE logs."); sys.exit(1)
    else: print_system("Engine Main appears to be running. Starting other services...")

    threads.append(start_service_thread(start_backend_service, "BackendServiceThread"))
    time.sleep(2)
    threads.append(start_service_thread(start_frontend, "FrontendThread"))

    print_system("All services are starting up. Press Ctrl+C to shut down.")

    # 6. Keep the main thread alive and monitor processes
    try:
        while True:
            active_process_found = False
            with process_lock:
                current_running = list(running_processes)
                processes_to_remove = []

            all_processes_ok = True
            for i in range(len(current_running) - 1, -1, -1):
                 proc, name = current_running[i]
                 if proc.poll() is None: active_process_found = True
                 else: print_error(f"Service '{name}' exited unexpectedly with code {proc.poll()}."); processes_to_remove.append((proc, name)); all_processes_ok = False

            if processes_to_remove:
                 with process_lock: running_processes[:] = [p for p in running_processes if p not in processes_to_remove]
            if not all_processes_ok: print_error("A service exited unexpectedly. Shutting down all services."); break
            if not active_process_found and threads: print_system("All managed service processes seem to have exited."); break
            time.sleep(5)

    except KeyboardInterrupt: pass
    finally: print_system("Launcher shutting down or finished.")