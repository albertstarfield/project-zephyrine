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
from datetime import datetime
import shlex       # <<< --- ADD THIS LINE --- >>>


# --- License Acceptance Imports ---
import curses # For terminal UI
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
# --- End License Acceptance Imports ---



# --- Configuration ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(ROOT_DIR, "venv")
ENGINE_MAIN_DIR = os.path.join(ROOT_DIR, "systemCore", "engineMain")
BACKEND_SERVICE_DIR = os.path.join(ROOT_DIR, "systemCore", "backend-service")
FRONTEND_DIR = os.path.join(ROOT_DIR, "systemCore", "frontend-face-zephyrine")
LICENSE_DIR = os.path.join(ROOT_DIR, "licenses") # <<< NEW

# Define paths to executables within the virtual environment
IS_WINDOWS = os.name == 'nt'
# <<< Add windows-curses handling >>>
if IS_WINDOWS:
    try:
        import curses # Try importing the windows-curses version
    except ImportError:
        print(f"{COLOR_ERROR}ERROR: `windows-curses` not found. Please install it:{COLOR_RESET}")
        print(f"  {os.path.join(VENV_DIR, 'Scripts', 'pip.exe')} install windows-curses")
        sys.exit(1)
# <<< End windows-curses handling >>>

VENV_BIN_DIR = os.path.join(VENV_DIR, "Scripts" if IS_WINDOWS else "bin")
PYTHON_EXECUTABLE = os.path.join(VENV_BIN_DIR, "python.exe" if IS_WINDOWS else "python")
PIP_EXECUTABLE = os.path.join(VENV_BIN_DIR, "pip.exe" if IS_WINDOWS else "pip")
HYPERCORN_EXECUTABLE = os.path.join(VENV_BIN_DIR, "hypercorn.exe" if IS_WINDOWS else "hypercorn")
NPM_CMD = 'npm.cmd' if IS_WINDOWS else 'npm'
GIT_CMD = 'git.exe' if IS_WINDOWS else 'git'

# Llama.cpp Fork Configuration
LLAMA_CPP_PYTHON_REPO_URL = "https://github.com/abetlen/llama-cpp-python.git"
LLAMA_CPP_PYTHON_CLONE_DIR_NAME = "llama-cpp-python_build"
LLAMA_CPP_PYTHON_CLONE_PATH = os.path.join(ROOT_DIR, LLAMA_CPP_PYTHON_CLONE_DIR_NAME)
LLAMA_CPP_SUBMODULE_PATH = "vendor/llama.cpp"

# Global list to keep track of running subprocesses
running_processes = []
process_lock = threading.Lock()

# --- Helper Functions (print_colored, print_system, print_error, run_command, stream_output, etc.) ---
# (Keep existing helper functions: print_colored, print_system, print_error, print_warning, run_command, stream_output)
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
    """Runs a command, optionally checks for errors, and can stream output."""
    # (Keep existing run_command implementation)
    print_system(f"Running command in '{os.path.basename(cwd)}': {' '.join(command)}")
    current_env = os.environ.copy()
    if env_override:
        str_env_override = {k: str(v) for k, v in env_override.items()}
        print_system(f"  with custom env: {str_env_override}")
        current_env.update(str_env_override)

    try:
        process = subprocess.Popen(
            command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding='utf-8', errors='replace', bufsize=1,
            shell=(IS_WINDOWS and command[0] in [NPM_CMD, 'node', GIT_CMD]),
            env=current_env
        )
        # Add process to global list *before* starting threads
        # This doesn't apply here as run_command waits, unlike service starters
        stdout_thread = threading.Thread(target=stream_output, args=(process.stdout, name, color), daemon=True)
        stdout_thread.start()
        stderr_thread = threading.Thread(target=stream_output, args=(process.stderr, f"{name}-ERR", COLOR_ERROR), daemon=True)
        stderr_thread.start()
        process.wait()
        stdout_thread.join()
        stderr_thread.join()
        if check and process.returncode != 0: raise subprocess.CalledProcessError(process.returncode, command)
        print_system(f"Command finished successfully in '{os.path.basename(cwd)}': {' '.join(command)}")
        return True
    except FileNotFoundError: print_error(f"Command not found: {command[0]}. In PATH?"); return False
    except subprocess.CalledProcessError as e: print_error(f"Command failed in '{os.path.basename(cwd)}' RC={e.returncode}."); return False
    except Exception as e: print_error(f"Unexpected error running command in '{os.path.basename(cwd)}': {e}"); return False

def stream_output(pipe, name, color):
    """Reads lines from a subprocess pipe and prints them with prefix and color."""
    # (Keep existing stream_output implementation)
    try:
        for line in iter(pipe.readline, ''):
            if line: print_colored(name, line, color)
    except Exception as e:
        if 'read of closed file' not in str(e).lower(): print_error(f"Error reading output stream for {name}: {e}")
    finally:
        if pipe:
            try: pipe.close()
            except Exception: pass

def start_service_thread(target_func, name):
    """Starts a service in a daemon thread."""
    # (Keep existing start_service_thread implementation)
    print_system(f"Preparing to start service: {name}")
    thread = threading.Thread(target=target_func, name=name, daemon=True)
    thread.start()
    return thread

def cleanup_processes():
    """Attempts to terminate all tracked subprocesses."""
    # (Keep existing cleanup_processes implementation)
    print_system("\nShutting down services...")
    with process_lock:
        procs_to_terminate = list(running_processes)
        running_processes.clear()
    for proc, name in reversed(procs_to_terminate):
        if proc.poll() is None:
            print_system(f"Terminating {name} (PID: {proc.pid})...")
            try:
                proc.terminate()
                try: proc.wait(timeout=3); print_system(f"{name} terminated gracefully.")
                except subprocess.TimeoutExpired: print_system(f"{name} kill..."); proc.kill(); proc.wait(); print_system(f"{name} killed.")
            except Exception as e: print_error(f"Error terminating {name}: {e}")
        else: print_system(f"{name} already exited (RC: {proc.poll()}).")

atexit.register(cleanup_processes)

def signal_handler(sig, frame):
    print_system("\nCtrl+C received. Initiating shutdown...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# --- Service Start Functions (start_engine_main, start_backend_service, start_frontend) ---
# (Keep existing service start functions)
def start_engine_main():
    """Starts the Python Engine Main service using Hypercorn."""
    name = "ENGINE"; color = COLOR_ENGINE
    command = [ HYPERCORN_EXECUTABLE, "app:app", "--bind", "127.0.0.1:11434", "--workers", "1", "--log-level", "info" ]
    print_system(f"Launching Engine Main via Hypercorn: {' '.join(command)} in {ENGINE_MAIN_DIR}")
    try:
        process = subprocess.Popen( command, cwd=ENGINE_MAIN_DIR, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace', bufsize=1 )
        with process_lock: running_processes.append((process, name))
        stdout_thread = threading.Thread(target=stream_output, args=(process.stdout, name, color), daemon=True); stderr_thread = threading.Thread(target=stream_output, args=(process.stderr, f"{name}-ERR", COLOR_ERROR), daemon=True)
        stdout_thread.start(); stderr_thread.start()
    except FileNotFoundError: print_error(f"Command failed: '{os.path.basename(HYPERCORN_EXECUTABLE)}' not found. In venv?"); sys.exit(1)
    except Exception as e: print_error(f"Failed to start {name}: {e}"); sys.exit(1)

def start_backend_service():
    """Starts the Node.js Backend Service."""
    name = "BACKEND"; color = COLOR_BACKEND
    command = ["node", "server.js"]
    print_system(f"Launching Backend Service: {' '.join(command)} in {BACKEND_SERVICE_DIR}")
    try:
        process = subprocess.Popen( command, cwd=BACKEND_SERVICE_DIR, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace', bufsize=1, shell=IS_WINDOWS )
        with process_lock: running_processes.append((process, name))
        stdout_thread = threading.Thread(target=stream_output, args=(process.stdout, name, color), daemon=True); stderr_thread = threading.Thread(target=stream_output, args=(process.stderr, f"{name}-ERR", COLOR_ERROR), daemon=True)
        stdout_thread.start(); stderr_thread.start()
    except FileNotFoundError: print_error("Command failed: 'node' not found. Is Node.js installed?"); sys.exit(1)
    except Exception as e: print_error(f"Failed to start {name}: {e}")

def start_frontend():
    """Starts the Vite Frontend Development Server."""
    name = "FRONTEND"; color = COLOR_FRONTEND
    command = [NPM_CMD, "run", "dev"]
    print_system(f"Launching Frontend (Vite): {' '.join(command)} in {FRONTEND_DIR}")
    try:
        process = subprocess.Popen( command, cwd=FRONTEND_DIR, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace', bufsize=1, shell=IS_WINDOWS )
        with process_lock: running_processes.append((process, name))
        stdout_thread = threading.Thread(target=stream_output, args=(process.stdout, name, color), daemon=True); stderr_thread = threading.Thread(target=stream_output, args=(process.stderr, f"{name}-ERR", COLOR_ERROR), daemon=True)
        stdout_thread.start(); stderr_thread.start()
    except FileNotFoundError: print_error(f"Command failed: '{NPM_CMD}' not found. Is Node.js/npm installed?"); sys.exit(1)
    except Exception as e: print_error(f"Failed to start {name}: {e}")


# --- License Acceptance Logic ---

# List of licenses: (filename, associated component/model group)
LICENSES_TO_ACCEPT = [
    ("APACHE_2.0.txt", "Flux1-Schnell, Qwen 2.5 VL Arch, Qwen2, mxbai-embed-large, Stable Diffusion GGUF"),
    ("MIT_Deepscaler.txt", "Deepscaler Model"),
    ("MIT_Zephyrine.txt", "Project Zephyrine (Launcher, Engine, Frontend, Backend)")
]

def load_licenses() -> tuple[dict[str, str], str]:
    """Loads license texts from files and returns a dict and combined string."""
    licenses_content = {}
    combined_text = ""
    print_system("Loading licenses...")
    for filename, description in LICENSES_TO_ACCEPT:
        filepath = os.path.join(LICENSE_DIR, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                licenses_content[filename] = content
                combined_text += f"\n\n--- LICENSE: {filename} ({description}) ---\n\n" + content
                print_system(f"  Loaded: {filename}")
        except FileNotFoundError:
            print_error(f"License file not found: {filepath}. Please create it.")
            sys.exit(1)
        except Exception as e:
            print_error(f"Error reading license file {filepath}: {e}")
            sys.exit(1)
    return licenses_content, combined_text.strip()

def calculate_reading_time(text: str) -> float:
    """Estimates reading time in seconds using tiktoken."""
    if not TIKTOKEN_AVAILABLE:
        print_warning("tiktoken not found, cannot estimate reading time. Skipping timing check.")
        return 0.0 # Return 0 if tiktoken is unavailable

    if not text:
        return 0.0

    WORDS_PER_MINUTE = 238 # Average reading speed
    AVG_WORDS_PER_TOKEN = 0.75 # Rough estimate for cl100k_base

    try:
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(text)
        num_tokens = len(tokens)
        estimated_words = num_tokens * AVG_WORDS_PER_TOKEN
        estimated_minutes = estimated_words / WORDS_PER_MINUTE
        estimated_seconds = estimated_minutes * 60
        print_system(f"Estimated reading time: {estimated_seconds:.2f} seconds ({num_tokens} tokens)")
        return estimated_seconds
    except Exception as e:
        print_warning(f"Failed to calculate reading time with tiktoken: {e}")
        return 0.0 # Return 0 on error

def display_license_prompt(stdscr, licenses_text_lines: list, estimated_seconds: float) -> tuple[bool, float]:
    """
    Uses curses to display licenses and get acceptance.
    Returns (accepted: bool, time_taken: float).
    """
    curses.curs_set(0) # Hide cursor
    stdscr.nodelay(False) # Wait for input
    stdscr.keypad(True) # Enable special keys
    curses.noecho()
    curses.cbreak()

    # Colors (optional, add more pairs if needed)
    curses.start_color()
    curses.init_pair(1, curses.COLOR_YELLOW, curses.COLOR_BLACK) # Instructions
    curses.init_pair(2, curses.COLOR_WHITE, curses.COLOR_BLACK) # Normal text
    curses.init_pair(3, curses.COLOR_BLACK, curses.COLOR_GREEN) # Accept button
    curses.init_pair(4, curses.COLOR_BLACK, curses.COLOR_RED) # Reject button
    curses.init_pair(5, curses.COLOR_CYAN, curses.COLOR_BLACK) # Header

    max_y, max_x = stdscr.getmaxyx()
    if max_y < 10 or max_x < 40: # Basic size check
        return False, 0.0 # Indicate failure due to small terminal

    # Display layout constants
    TEXT_AREA_HEIGHT = max_y - 6 # Leave lines for header, instructions, prompt
    TEXT_START_Y = 2
    INSTR_Y = max_y - 3
    PROMPT_Y = max_y - 2
    ACCEPT_X = max_x // 2 - 15
    REJECT_X = max_x // 2 + 5

    top_line = 0
    total_lines = len(licenses_text_lines)
    accepted = False
    start_time = time.monotonic() # Start timing when display begins

    while True:
        stdscr.clear()

        # Header
        header = "--- SOFTWARE LICENSES ---"
        stdscr.addstr(0, (max_x - len(header)) // 2, header, curses.color_pair(5) | curses.A_BOLD)
        assoc = "Models: Flux1, Qwen2.5VL, Qwen2, Embeddings, SD | Code: Deepscaler, Zephyrine"
        stdscr.addstr(1, 0, assoc[:max_x-1], curses.color_pair(5)) # Truncate association if needed

        # Display license text
        for i in range(TEXT_AREA_HEIGHT):
            line_idx = top_line + i
            if line_idx < total_lines:
                # Truncate lines that are too long for the screen width
                line_text = licenses_text_lines[line_idx][:max_x - 1]
                try:
                    stdscr.addstr(TEXT_START_Y + i, 0, line_text, curses.color_pair(2))
                except curses.error:
                    pass # Ignore error if writing fails (e.g., at bottom right)
            else:
                break # Stop drawing if we run out of lines

        # Instructions
        instr = "Scroll: UP/DOWN Arrows, PgUp/PgDn | Press 'a' to Accept, 'n' to Not Accept"
        stdscr.addstr(INSTR_Y, 0, instr[:max_x-1], curses.color_pair(1))

        # Prompt buttons
        stdscr.addstr(PROMPT_Y, ACCEPT_X, "[ Accept (a) ]", curses.color_pair(3) | curses.A_REVERSE)
        stdscr.addstr(PROMPT_Y, REJECT_X, "[Not Accept (n)]", curses.color_pair(4) | curses.A_REVERSE)

        stdscr.refresh()

        # Get user input
        key = stdscr.getch()

        if key == ord('a') or key == ord('A'):
            accepted = True
            break
        elif key == ord('n') or key == ord('N'):
            accepted = False
            break
        elif key == curses.KEY_DOWN:
            if top_line < total_lines - TEXT_AREA_HEIGHT:
                top_line += 1
        elif key == curses.KEY_UP:
            if top_line > 0:
                top_line -= 1
        elif key == curses.KEY_NPAGE: # Page Down
             top_line = min(total_lines - TEXT_AREA_HEIGHT, top_line + TEXT_AREA_HEIGHT)
             if top_line < 0: top_line = 0 # Ensure non-negative
        elif key == curses.KEY_PPAGE: # Page Up
             top_line = max(0, top_line - TEXT_AREA_HEIGHT)

    end_time = time.monotonic()
    time_taken = end_time - start_time
    return accepted, time_taken

# --- Main Execution Logic ---
if __name__ == "__main__":
    print_system("--- Project Zephyrine Launcher ---")

    # --- License Acceptance Step ---
    LICENSE_FLAG_FILE = os.path.join(ROOT_DIR, ".license_accepted_v1") # Added version suffix

    # <<< --- START MODIFICATION --- >>>
    license_already_accepted = os.path.exists(LICENSE_FLAG_FILE)

    if not license_already_accepted:
        print_system("License agreement required for first run.")
        try:
            licenses_data, combined_license_text = load_licenses()
            estimated_reading_seconds = calculate_reading_time(combined_license_text)
            license_lines = combined_license_text.splitlines()

            # Use curses to display prompt
            accepted, time_taken = curses.wrapper(display_license_prompt, license_lines, estimated_reading_seconds)

            if not accepted:
                print_error("License not accepted. Exiting.")
                sys.exit(1)
            else:
                # Acceptance granted, create the flag file
                try:
                    with open(LICENSE_FLAG_FILE, 'w') as f:
                        f.write(f"Accepted on: {datetime.now().isoformat()}\n")
                    print_system(f"Acceptance recorded.")
                except Exception as flag_err:
                    print_error(f"Could not create acceptance flag file: {flag_err}")
                    # Proceed anyway, but warn the user they might see the prompt again
                    print_warning("License accepted for this session, but recording failed.")
                    time.sleep(2)

                print_system(f"Licenses accepted in {time_taken:.2f} seconds.")
                # Warn if accepted too quickly
                if estimated_reading_seconds > 10 and time_taken < (estimated_reading_seconds * 0.5):
                    print_warning("Warning: Licenses accepted very quickly.")
                    print_warning(f"         Estimated reading time was ~{estimated_reading_seconds:.0f}s, you took {time_taken:.1f}s.")
                    print_warning("         Please ensure you have reviewed the terms.")
                    time.sleep(3) # Pause to let user see warning

        except Exception as e:
            print_error(f"An error occurred during the license acceptance process: {e}")
            try: curses.endwin() # Attempt cleanup
            except: pass
            sys.exit(1)
    else:
        print_system("License previously accepted (found flag file). Skipping prompt.")
    # <<< --- END MODIFICATION --- >>>

    # --- End License Acceptance Step ---


    # 1. Check/Create/Relaunch in Virtual Environment
    # (Keep existing venv check/relaunch logic)
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
            except Exception as e: print_error(f"Failed venv creation: {e}"); sys.exit(1)
        print_system(f"Relaunching script using Python from venv: {PYTHON_EXECUTABLE}")
        if not os.path.exists(PYTHON_EXECUTABLE): print_error(f"Python not found in venv: {PYTHON_EXECUTABLE}"); sys.exit(1)
        try: os.execv(PYTHON_EXECUTABLE, [PYTHON_EXECUTABLE] + sys.argv)
        except Exception as e: print_error(f"Failed relaunch: {e}"); sys.exit(1)


    # --- Running inside the correct venv ---
    print_system(f"Running inside the correct virtual environment: {VENV_DIR}")

    # 2. Install/Check Dependencies (Python)
    # (Keep existing Python dependency installation logic, including llama-cpp-python build)
    print_system("--- Installing/Checking Python Dependencies (Engine) ---")
    req_path = os.path.join(ENGINE_MAIN_DIR, "requirements.txt")
    if not os.path.exists(req_path): print_error(f"requirements.txt not found: {req_path}"); sys.exit(1)
    if not run_command([PIP_EXECUTABLE, "install", "-r", req_path], ENGINE_MAIN_DIR, "PIP", COLOR_SYSTEM):
         print_error("Failed install Python deps. Exiting."); sys.exit(1)
    print_system("Standard Python dependencies checked/installed.")

    # --- Custom llama-cpp-python Installation ---
    provider_env = os.getenv("PROVIDER", "llama_cpp").lower()
    print_system(f"Checking PROVIDER environment variable: {provider_env}")
    if provider_env == "llama_cpp":
        print_system(f"--- Attempting Custom llama-cpp-python Installation ---")
        if not shutil.which(GIT_CMD): print_error(f"'{GIT_CMD}' not found. Git required. Exiting."); sys.exit(1)
        print_system("Uninstalling standard llama-cpp-python (if present)...")
        run_command([PIP_EXECUTABLE, "uninstall", "llama-cpp-python", "-y"], ROOT_DIR, "PIP", COLOR_SYSTEM, check=False)
        if not os.path.exists(LLAMA_CPP_PYTHON_CLONE_PATH):
            print_system(f"Cloning '{LLAMA_CPP_PYTHON_REPO_URL}' into '{LLAMA_CPP_PYTHON_CLONE_PATH}'...")
            if not run_command([GIT_CMD, "clone", LLAMA_CPP_PYTHON_REPO_URL, LLAMA_CPP_PYTHON_CLONE_PATH], ROOT_DIR, "GIT", COLOR_SYSTEM):
                print_error("Failed clone llama-cpp-python. Exiting."); sys.exit(1)
        else: print_system(f"llama-cpp-python dir exists: '{LLAMA_CPP_PYTHON_CLONE_PATH}'. Skip clone.")
        print_system("Initializing/updating submodules (using default llama.cpp commit)...")
        if not run_command([GIT_CMD, "submodule", "update", "--init", "--recursive"], LLAMA_CPP_PYTHON_CLONE_PATH, "GIT", COLOR_SYSTEM):
            print_error("Failed update submodules. Exiting."); sys.exit(1)
        # --- Prepare environment for build ---
        print_system("Preparing environment for llama.cpp build...")
        build_env = {'FORCE_CMAKE': '1'}; cmake_args_list = []
        default_backend = 'cpu'
        if sys.platform == "darwin": default_backend = 'metal'
        elif sys.platform in ["linux", "win32"]: default_backend = 'cuda'
        llama_backend = os.getenv("LLAMA_CPP_BACKEND", default_backend).lower()
        print_system(f"Selected llama.cpp backend: {llama_backend}")
        if llama_backend == "cuda": cmake_args_list.append("-DGGML_CUDA=on")
        elif llama_backend == "metal":
            if sys.platform != "darwin": print_warning("Metal backend but not macOS.")
            cmake_args_list.append("-DGGML_METAL=on")
            if platform.machine() == "arm64": print_system("Adding Metal arm64 CMake flags."); cmake_args_list.extend(["-DCMAKE_OSX_ARCHITECTURES=arm64", "-DCMAKE_APPLE_SILICON_PROCESSOR=arm64"])
        # ... (Add other backend flags: rocm, openblas, vulkan, sycl, rpc as before) ...
        elif llama_backend == "rocm": cmake_args_list.append("-DGGML_HIPBLAS=on"); print_warning("ROCm: May need AMDGPU_TARGETS env var.")
        elif llama_backend == "openblas": cmake_args_list.append("-DGGML_BLAS=ON"); cmake_args_list.append("-DGGML_BLAS_VENDOR=OpenBLAS")
        elif llama_backend == "vulkan": cmake_args_list.append("-DGGML_VULKAN=on"); print_warning("Vulkan: Ensure Vulkan SDK installed and VULKAN_SDK env var set.")
        elif llama_backend == "sycl": cmake_args_list.append("-DGGML_SYCL=on"); print_warning("SYCL: Ensure oneAPI installed and env sourced.")
        elif llama_backend == "rpc": cmake_args_list.append("-DGGML_RPC=on")
        elif llama_backend == "cpu": print_system("Building for CPU only."); cmake_args_list = []
        else: print_warning(f"Unknown backend '{llama_backend}'. CPU only."); cmake_args_list = []
        if cmake_args_list: build_env['CMAKE_ARGS'] = " ".join(cmake_args_list)
        print_system(f"Running pip install from '{LLAMA_CPP_PYTHON_CLONE_PATH}' with build flags...")
        pip_install_command = [ PIP_EXECUTABLE, "install", ".", "--upgrade", "--no-cache-dir" ] # More aggressive rebuild flags
        if not run_command(pip_install_command, LLAMA_CPP_PYTHON_CLONE_PATH, "PIP-LLAMA", COLOR_SYSTEM, env_override=build_env):
            print_error("Failed install custom llama-cpp-python. Exiting."); sys.exit(1)
        print_system("Custom llama-cpp-python build installed.")


    # 3. Install/Check Dependencies (Node.js Backend Service)
    # (Keep existing Node backend dependency check)
    print_system("--- Installing/Checking Node Backend Dependencies ---")
    pkg_path = os.path.join(BACKEND_SERVICE_DIR, "package.json")
    if not os.path.exists(pkg_path): print_error(f"package.json not found: {pkg_path}"); sys.exit(1)
    if not run_command([NPM_CMD, "install"], BACKEND_SERVICE_DIR, "NPM-BE", COLOR_SYSTEM):
         print_error("Failed install Node BE deps. Exiting."); sys.exit(1)
    print_system("Node backend dependencies checked/installed.")


    # 4. Install/Check Dependencies (Node.js Frontend)
    # (Keep existing Node frontend dependency check)
    print_system("--- Installing/Checking Node Frontend Dependencies ---")
    pkg_path_fe = os.path.join(FRONTEND_DIR, "package.json")
    if not os.path.exists(pkg_path_fe): print_error(f"package.json not found: {pkg_path_fe}"); sys.exit(1)
    if not run_command([NPM_CMD, "install"], FRONTEND_DIR, "NPM-FE", COLOR_SYSTEM):
         print_error("Failed install Node FE deps. Exiting."); sys.exit(1)
    print_system("Node frontend dependencies checked/installed.")


    # 5. Start all services concurrently
    # (Keep existing service startup logic)
    print_system("--- Starting Services ---")
    threads = []
    threads.append(start_service_thread(start_engine_main, "EngineMainThread"))
    print_system("Waiting a few seconds for Engine to initialize...")
    time.sleep(5)
    engine_proc_ok = False
    with process_lock:
        for proc, name in running_processes:
            if name == "ENGINE" and proc.poll() is None: engine_proc_ok = True; break
    if not engine_proc_ok: print_error("Engine Main failed to stay running. Check ENGINE logs."); sys.exit(1)
    else: print_system("Engine Main running. Starting other services...")
    threads.append(start_service_thread(start_backend_service, "BackendServiceThread"))
    time.sleep(2)
    threads.append(start_service_thread(start_frontend, "FrontendThread"))
    print_system("All services starting. Press Ctrl+C to shut down.")


    # 6. Keep the main thread alive and monitor processes
    # (Keep existing monitoring loop)
    try:
        while True:
            active_process_found = False; all_processes_ok = True
            with process_lock: current_running = list(running_processes); processes_to_remove = []
            for i in range(len(current_running) - 1, -1, -1):
                 proc, name = current_running[i]
                 if proc.poll() is None: active_process_found = True
                 else: print_error(f"Service '{name}' exited unexpectedly RC={proc.poll()}."); processes_to_remove.append((proc, name)); all_processes_ok = False
            if processes_to_remove:
                 with process_lock: running_processes[:] = [p for p in running_processes if p not in processes_to_remove]
            if not all_processes_ok: print_error("Service exited. Shutting down."); break
            if not active_process_found and threads: print_system("All managed services exited."); break
            time.sleep(5)
    except KeyboardInterrupt: pass
    finally: print_system("Launcher shutting down or finished.")