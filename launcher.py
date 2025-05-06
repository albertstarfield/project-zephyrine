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

# --- License Acceptance Imports ---
import curses
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
LICENSE_DIR = os.path.join(ROOT_DIR, "licenses")
LICENSE_FLAG_FILE = os.path.join(ROOT_DIR, ".license_accepted_v1")

# --- NEW: MeloTTS Configuration ---
MELO_TTS_SUBMODULE_DIR_NAME = "MeloAudioTTS_SubEngine" # As per your structure
MELO_TTS_PATH = os.path.join(ENGINE_MAIN_DIR, MELO_TTS_SUBMODULE_DIR_NAME)
MELO_TTS_INSTALLED_FLAG_FILE = os.path.join(ROOT_DIR, ".melo_tts_installed_v1")
# --- END NEW ---

# Define paths to executables within the virtual environment
IS_WINDOWS = os.name == 'nt'

if IS_WINDOWS:
    try: import curses
    except ImportError:
        print(f"ERROR: `windows-curses` not found. Please install it:")
        _pip_path = os.path.join(VENV_DIR, 'Scripts', 'pip.exe') if os.path.exists(os.path.join(VENV_DIR, 'Scripts')) else 'pip'
        print(f"  {_pip_path} install windows-curses")
        sys.exit(1)

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
LLAMA_CPP_SUBMODULE_PATH = "vendor/llama.cpp" # Relative within the clone path
# --- NEW: Custom Llama.cpp Installation Flag ---
CUSTOM_LLAMA_CPP_INSTALLED_FLAG_FILE = os.path.join(ROOT_DIR, ".custom_llama_cpp_installed_v1")
# --- END NEW ---

# Global list to keep track of running subprocesses
running_processes = []
process_lock = threading.Lock()

# --- Helper Functions (print_colored, print_system, print_error, print_warning, stream_output, run_command, start_service_thread, cleanup_processes, signal_handler) ---
# ... (Keep these functions as they are) ...
def print_colored(prefix, message, color=None):
    now = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    print(f"[{now} | {prefix.ljust(10)}] {message.strip()}")
def print_system(message): print_colored("SYSTEM", message)
def print_error(message): print_colored("ERROR", message)
def print_warning(message): print_colored("WARNING", message)

def stream_output(pipe, name=None, color=None):
    try:
        for line in iter(pipe.readline, ''):
            if line: sys.stdout.write(line); sys.stdout.flush()
    except Exception as e:
        if 'read of closed file' not in str(e).lower():
            print_error(f"Error reading output stream for pipe {pipe}: {e}")
    finally:
        if pipe:
            try: pipe.close()
            except Exception: pass

def run_command(command, cwd, name, color=None, check=True, capture_output=False, env_override=None):
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
            shell=(IS_WINDOWS and command[0] in [NPM_CMD, 'node', GIT_CMD, PIP_EXECUTABLE, PYTHON_EXECUTABLE]), # Added PIP/PYTHON for shell on Windows
            env=current_env
        )
        stdout_thread = threading.Thread(target=stream_output, args=(process.stdout,), daemon=True)
        stdout_thread.start()
        stderr_thread = threading.Thread(target=stream_output, args=(process.stderr,), daemon=True)
        stderr_thread.start()
        process.wait()
        stdout_thread.join()
        stderr_thread.join()
        if check and process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command)
        print_system(f"Command finished successfully in '{os.path.basename(cwd)}': {' '.join(command)}")
        return True
    except FileNotFoundError:
        print_error(f"Command not found: {command[0]}. Is it installed and in PATH?")
        if command[0] == GIT_CMD: print_error("Ensure Git is installed: https://git-scm.com/downloads")
        elif command[0] in [NPM_CMD, 'node']: print_error("Ensure Node.js and npm are installed: https://nodejs.org/")
        elif command[0] == HYPERCORN_EXECUTABLE: print_error(f"Ensure '{os.path.basename(HYPERCORN_EXECUTABLE)}' is installed in the venv.")
        return False
    except subprocess.CalledProcessError as e:
        print_error(f"Command failed in '{os.path.basename(cwd)}' with exit code {e.returncode}.")
        return False
    except Exception as e:
        print_error(f"An unexpected error occurred while running command in '{os.path.basename(cwd)}': {e}")
        return False

def start_service_thread(target_func, name):
    print_system(f"Preparing to start service: {name}")
    thread = threading.Thread(target=target_func, name=name, daemon=True)
    thread.start()
    return thread

def cleanup_processes():
    print_system("\nShutting down services...")
    with process_lock: procs_to_terminate = list(running_processes); running_processes.clear()
    for proc, name in reversed(procs_to_terminate):
        if proc.poll() is None:
            print_system(f"Terminating {name} (PID: {proc.pid})...")
            try:
                proc.terminate()
                try: proc.wait(timeout=3); print_system(f"{name} terminated gracefully.")
                except subprocess.TimeoutExpired: print_system(f"{name} did not terminate, killing..."); proc.kill(); proc.wait(); print_system(f"{name} killed.")
            except Exception as e: print_error(f"Error terminating {name}: {e}")
        else: print_system(f"{name} already exited (return code: {proc.poll()}).")
atexit.register(cleanup_processes)

def signal_handler(sig, frame): print_system("\nCtrl+C received. Initiating shutdown..."); sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)


# --- Service Start Functions (start_engine_main, start_backend_service, start_frontend) ---
# ... (Keep these functions as they are) ...
def start_engine_main():
    name = "ENGINE"
    command = [HYPERCORN_EXECUTABLE, "app:app", "--bind", "127.0.0.1:11434", "--workers", "1", "--log-level", "info"]
    print_system(f"Launching Engine Main via Hypercorn: {' '.join(command)} in {ENGINE_MAIN_DIR}")
    try:
        process = subprocess.Popen(command, cwd=ENGINE_MAIN_DIR, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace', bufsize=1)
        with process_lock: running_processes.append((process, name))
        stdout_thread = threading.Thread(target=stream_output, args=(process.stdout,), daemon=True); stderr_thread = threading.Thread(target=stream_output, args=(process.stderr,), daemon=True)
        stdout_thread.start(); stderr_thread.start()
    except FileNotFoundError: print_error(f"Command failed: '{os.path.basename(HYPERCORN_EXECUTABLE)}' not found."); sys.exit(1)
    except Exception as e: print_error(f"Failed to start {name}: {e}"); sys.exit(1)

def start_backend_service():
    name = "BACKEND"; command = ["node", "server.js"]
    print_system(f"Launching Backend Service: {' '.join(command)} in {BACKEND_SERVICE_DIR}")
    try:
        process = subprocess.Popen(command, cwd=BACKEND_SERVICE_DIR, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace', bufsize=1, shell=IS_WINDOWS)
        with process_lock: running_processes.append((process, name))
        stdout_thread = threading.Thread(target=stream_output, args=(process.stdout,), daemon=True); stderr_thread = threading.Thread(target=stream_output, args=(process.stderr,), daemon=True)
        stdout_thread.start(); stderr_thread.start()
    except FileNotFoundError: print_error("Command failed: 'node' not found."); sys.exit(1)
    except Exception as e: print_error(f"Failed to start {name}: {e}")

def start_frontend():
    name = "FRONTEND"; command = [NPM_CMD, "run", "dev"]
    print_system(f"Launching Frontend (Vite): {' '.join(command)} in {FRONTEND_DIR}")
    try:
        process = subprocess.Popen(command, cwd=FRONTEND_DIR, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace', bufsize=1, shell=IS_WINDOWS)
        with process_lock: running_processes.append((process, name))
        stdout_thread = threading.Thread(target=stream_output, args=(process.stdout,), daemon=True); stderr_thread = threading.Thread(target=stream_output, args=(process.stderr,), daemon=True)
        stdout_thread.start(); stderr_thread.start()
    except FileNotFoundError: print_error(f"Command failed: '{NPM_CMD}' not found."); sys.exit(1)
    except Exception as e: print_error(f"Failed to start {name}: {e}")


# --- License Acceptance Logic (load_licenses, calculate_reading_time, display_license_prompt) ---
# ... (Keep these functions as they are) ...
def load_licenses() -> tuple[dict[str, str], str]:
    licenses_content = {}; combined_text = ""
    print_system("Loading licenses...")
    if not os.path.isdir(LICENSE_DIR): print_error(f"License directory not found: {LICENSE_DIR}"); sys.exit(1)
    for filename, description in LICENSES_TO_ACCEPT:
        filepath = os.path.join(LICENSE_DIR, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f: content = f.read(); licenses_content[filename] = content
            combined_text += f"\n\n--- LICENSE: {filename} ({description}) ---\n\n" + content; print_system(f"  Loaded: {filename}")
        except FileNotFoundError: print_error(f"License file not found: {filepath}."); sys.exit(1)
        except Exception as e: print_error(f"Error reading license file {filepath}: {e}"); sys.exit(1)
    return licenses_content, combined_text.strip()

LICENSES_TO_ACCEPT = [("APACHE_2.0.txt", "Flux1,..."), ("MIT_Deepscaler.txt", "Deepscaler"), ("MIT_Zephyrine.txt", "Project Zephyrine")] # Shortened for brevity

def calculate_reading_time(text: str) -> float:
    if not TIKTOKEN_AVAILABLE: print_warning("tiktoken not found, cannot estimate reading time."); return 0.0
    if not text: return 0.0
    WORDS_PER_MINUTE = 238; AVG_WORDS_PER_TOKEN = 0.75
    try:
        enc = tiktoken.get_encoding("cl100k_base"); tokens = enc.encode(text); num_tokens = len(tokens)
        estimated_words = num_tokens * AVG_WORDS_PER_TOKEN; estimated_minutes = estimated_words / WORDS_PER_MINUTE
        estimated_seconds = estimated_minutes * 60; print_system(f"Estimated reading time: {estimated_seconds:.2f}s ({num_tokens} tokens)")
        return estimated_seconds
    except Exception as e: print_warning(f"Failed to calculate reading time: {e}"); return 0.0

def display_license_prompt(stdscr, licenses_text_lines: list, estimated_seconds: float) -> tuple[bool, float]:
    curses.curs_set(0); stdscr.nodelay(False); stdscr.keypad(True); curses.noecho(); curses.cbreak()
    curses.start_color(); curses.init_pair(1, curses.COLOR_YELLOW, curses.COLOR_BLACK); curses.init_pair(2, curses.COLOR_WHITE, curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_BLACK, curses.COLOR_GREEN); curses.init_pair(4, curses.COLOR_BLACK, curses.COLOR_RED); curses.init_pair(5, curses.COLOR_CYAN, curses.COLOR_BLACK)
    max_y, max_x = stdscr.getmaxyx()
    if max_y < 10 or max_x < 40: curses.nocbreak(); stdscr.keypad(False); curses.echo(); curses.endwin(); print("Terminal too small."); sys.exit(1)
    TEXT_AREA_HEIGHT = max_y - 6; TEXT_START_Y = 2; INSTR_Y = max_y - 3; PROMPT_Y = max_y - 2
    ACCEPT_X = max_x // 2 - 15; REJECT_X = max_x // 2 + 5; top_line = 0; total_lines = len(licenses_text_lines)
    accepted = False; start_time = time.monotonic()
    while True:
        try:
            stdscr.clear(); header = "--- SOFTWARE LICENSES ---"; stdscr.addstr(0, max(0, (max_x - len(header)) // 2), header, curses.color_pair(5) | curses.A_BOLD)
            assoc = "Models: Flux1,... | Code: Deepscaler, Zephyrine"; stdscr.addstr(1, 0, assoc[:max_x-1], curses.color_pair(5))
            for i in range(TEXT_AREA_HEIGHT):
                line_idx = top_line + i
                if line_idx < total_lines: stdscr.addstr(TEXT_START_Y + i, 0, licenses_text_lines[line_idx][:max_x - 1].rstrip(), curses.color_pair(2))
                else: break
            instr = "Scroll: Arrows, PgUp/PgDn | 'a' to Accept, 'n' to Not Accept"; stdscr.addstr(INSTR_Y, 0, instr[:max_x-1], curses.color_pair(1))
            stdscr.addstr(PROMPT_Y, ACCEPT_X, "[ Accept (a) ]", curses.color_pair(3) | curses.A_REVERSE); stdscr.addstr(PROMPT_Y, REJECT_X, "[Not Accept (n)]", curses.color_pair(4) | curses.A_REVERSE)
            stdscr.refresh()
        except curses.error as e: print_warning(f"Curses display error: {e}"); time.sleep(0.1); max_y, max_x = stdscr.getmaxyx(); TEXT_AREA_HEIGHT=max_y-6; ACCEPT_X=max_x//2-15; REJECT_X=max_x//2+5; continue
        try: key = stdscr.getch()
        except KeyboardInterrupt: accepted = False; break
        except Exception as e: print_warning(f"Error getting input: {e}"); key = -1
        if key == ord('a') or key == ord('A'): accepted = True; break
        elif key == ord('n') or key == ord('N'): accepted = False; break
        elif key == curses.KEY_DOWN: top_line = min(top_line + 1, total_lines - TEXT_AREA_HEIGHT)
        elif key == curses.KEY_UP: top_line = max(0, top_line - 1)
        elif key == curses.KEY_NPAGE: top_line = min(max(0, total_lines - TEXT_AREA_HEIGHT), top_line + TEXT_AREA_HEIGHT)
        elif key == curses.KEY_PPAGE: top_line = max(0, top_line - TEXT_AREA_HEIGHT)
        elif key == curses.KEY_RESIZE: max_y,max_x=stdscr.getmaxyx(); TEXT_AREA_HEIGHT=max_y-6; ACCEPT_X=max_x//2-15; REJECT_X=max_x//2+5; top_line=min(max(0,total_lines-TEXT_AREA_HEIGHT),top_line)
    end_time = time.monotonic(); time_taken = end_time - start_time
    return accepted, time_taken


# --- Main Execution Logic ---
if __name__ == "__main__":
    print_system("--- Project Zephyrine Launcher ---")

    # --- License Acceptance Step ---
    # ... (Keep existing license acceptance logic) ...
    if not os.path.exists(LICENSE_FLAG_FILE):
        print_system("License agreement required for first run.")
        try:
            licenses_data, combined_license_text = load_licenses()
            estimated_reading_seconds = calculate_reading_time(combined_license_text)
            license_lines = combined_license_text.splitlines()
            accepted, time_taken = curses.wrapper(display_license_prompt, license_lines, estimated_reading_seconds)
            if not accepted: print_error("License not accepted. Exiting."); sys.exit(1)
            else:
                try:
                    with open(LICENSE_FLAG_FILE, 'w') as f: f.write(f"Accepted on: {datetime.now().isoformat()}\n")
                    print_system(f"Acceptance recorded.")
                except Exception as flag_err: print_error(f"Could not create acceptance flag file: {flag_err}"); time.sleep(2)
                print_system(f"Licenses accepted in {time_taken:.2f} seconds.")
                if estimated_reading_seconds > 10 and time_taken < (estimated_reading_seconds * 0.5):
                    print_warning("Warning: Licenses accepted very quickly."); time.sleep(3)
        except Exception as e: print_error(f"License acceptance error: {e}")
        try: curses.endwin()
        except: pass; sys.exit(1)
    else: print_system("License previously accepted.")


    # 1. Check/Create/Relaunch in Virtual Environment
    # ... (Keep existing venv check and relaunch logic) ...
    is_in_target_venv = (os.getenv('VIRTUAL_ENV') == VENV_DIR) or \
                         (hasattr(sys, 'prefix') and sys.prefix == VENV_DIR) or \
                         (hasattr(sys, 'real_prefix') and sys.real_prefix == VENV_DIR)
    if not is_in_target_venv:
        print_system(f"Not in target venv: {VENV_DIR}")
        if not os.path.exists(VENV_DIR):
            print_system(f"Venv not found. Creating at: {VENV_DIR}")
            try: subprocess.run([sys.executable, "-m", "venv", VENV_DIR], check=True, capture_output=True); print_system("Venv created.")
            except Exception as e: print_error(f"Failed create venv: {e}"); sys.exit(1)
        print_system(f"Relaunching in venv: {PYTHON_EXECUTABLE}")
        if not os.path.exists(PYTHON_EXECUTABLE): print_error(f"Python not found in venv: {PYTHON_EXECUTABLE}"); sys.exit(1)
        try: os.execv(PYTHON_EXECUTABLE, [PYTHON_EXECUTABLE] + sys.argv)
        except Exception as e: print_error(f"Failed to relaunch in venv: {e}"); sys.exit(1)

    print_system(f"Running inside venv: {VENV_DIR}")

    # 2. Install/Check Dependencies (Python)
    print_system("--- Installing/Checking Python Dependencies (Engine) ---")
    req_path = os.path.join(ENGINE_MAIN_DIR, "requirements.txt")
    if not os.path.exists(req_path): print_error(f"requirements.txt not found at {req_path}"); sys.exit(1)
    if not run_command([PIP_EXECUTABLE, "install", "-r", req_path], ENGINE_MAIN_DIR, "PIP"):
         print_error("Failed to install Python dependencies. Exiting."); sys.exit(1)
    print_system("Standard Python dependencies checked/installed.")

    # --- NEW: MeloTTS Installation ---
    if not os.path.exists(MELO_TTS_INSTALLED_FLAG_FILE):
        print_system(f"--- Installing MeloTTS from {MELO_TTS_PATH} ---")
        if not os.path.isdir(MELO_TTS_PATH):
            print_error(f"MeloTTS directory not found at: {MELO_TTS_PATH}")
            print_error("Please ensure MeloTTS is cloned or placed there.")
            sys.exit(1)

        # Step 1: pip install -e . (editable install from the MeloTTS directory)
        if not run_command([PIP_EXECUTABLE, "install", "-e", "."], MELO_TTS_PATH, "PIP-MELO"):
            print_error("Failed to install MeloTTS (editable). Exiting."); sys.exit(1)
        print_system("MeloTTS (editable) installed successfully.")

        # Step 2: python -m unidic download
        print_system("Downloading unidic dictionary for MeloTTS...")
        # Note: Using PYTHON_EXECUTABLE from venv to ensure it uses the correct Python
        if not run_command([PYTHON_EXECUTABLE, "-m", "unidic", "download"], MELO_TTS_PATH, "UNIDIC"):
            print_warning("Failed to download unidic dictionary automatically. You might need to run it manually.")
            print_warning(f"  Command: cd \"{MELO_TTS_PATH}\" && \"{PYTHON_EXECUTABLE}\" -m unidic download")
            # Decide if this is a fatal error or just a warning. For now, warning.
        else:
            print_system("unidic dictionary downloaded successfully.")

        # Create flag file on successful completion
        try:
            with open(MELO_TTS_INSTALLED_FLAG_FILE, 'w') as f:
                f.write(f"MeloTTS installed on: {datetime.now().isoformat()}\n")
            print_system("MeloTTS installation flag created.")
        except Exception as flag_err:
            print_error(f"Could not create MeloTTS installation flag file: {flag_err}")
    else:
        print_system("MeloTTS previously installed (flag file found). Skipping installation.")
    # --- END NEW ---


    # --- Custom llama-cpp-python Installation ---
    provider_env = os.getenv("PROVIDER", "llama_cpp").lower()
    if provider_env == "llama_cpp":
        if not os.path.exists(CUSTOM_LLAMA_CPP_INSTALLED_FLAG_FILE):
            print_system(f"--- Attempting Custom llama-cpp-python Installation ---")
            if not shutil.which(GIT_CMD): print_error(f"'{GIT_CMD}' command not found. Git required."); sys.exit(1)
            print_system("Uninstalling standard llama-cpp-python (if present)...")
            run_command([PIP_EXECUTABLE, "uninstall", "llama-cpp-python", "-y"], ROOT_DIR, "PIP", check=False)

            if os.path.exists(LLAMA_CPP_PYTHON_CLONE_PATH):
                 print_system(f"Cleaning existing llama-cpp-python build directory: {LLAMA_CPP_PYTHON_CLONE_PATH}")
                 try:
                     shutil.rmtree(LLAMA_CPP_PYTHON_CLONE_PATH)
                 except Exception as e:
                     print_error(f"Failed to remove existing build directory: {e}. Please remove manually and retry."); sys.exit(1)

            print_system(f"Cloning '{LLAMA_CPP_PYTHON_REPO_URL}' into '{LLAMA_CPP_PYTHON_CLONE_PATH}'...")
            if not run_command([GIT_CMD, "clone", LLAMA_CPP_PYTHON_REPO_URL, LLAMA_CPP_PYTHON_CLONE_PATH], ROOT_DIR, "GIT"):
                print_error("Failed to clone llama-cpp-python. Exiting."); sys.exit(1)

            print_system("Initializing/updating submodules...")
            # Ensure the submodule path is relative to the clone directory for the command
            if not run_command([GIT_CMD, "submodule", "update", "--init", "--recursive", LLAMA_CPP_SUBMODULE_PATH], LLAMA_CPP_PYTHON_CLONE_PATH, "GIT"):
                print_error("Failed to update submodules. Exiting."); sys.exit(1)

            build_env = {'FORCE_CMAKE': '1'}
            cmake_args_list = []; default_backend = 'cpu'
            if sys.platform == "darwin": default_backend = 'metal'
            elif sys.platform in ["linux", "win32"]: default_backend = 'cuda' # Default to CUDA if available
            llama_backend = os.getenv("LLAMA_CPP_BACKEND", default_backend).lower()
            print_system(f"Selected llama.cpp backend: {llama_backend}")

            if llama_backend == "cuda": cmake_args_list.append("-DGGML_CUDA=on")
            elif llama_backend == "metal":
                if sys.platform != "darwin": print_warning("Metal selected but not on macOS.")
                cmake_args_list.append("-DGGML_METAL=on")
                if platform.machine() == "arm64": cmake_args_list.extend(["-DCMAKE_OSX_ARCHITECTURES=arm64", "-DCMAKE_APPLE_SILICON_PROCESSOR=arm64"])
            # ... (other backend CMAKE_ARGS logic - keep as is) ...
            elif llama_backend == "rocm": cmake_args_list.append("-DGGML_HIPBLAS=on"); # ...
            elif llama_backend == "openblas": cmake_args_list.extend(["-DGGML_BLAS=ON", "-DGGML_BLAS_VENDOR=OpenBLAS"]); # ...
            elif llama_backend == "vulkan": cmake_args_list.append("-DGGML_VULKAN=on"); # ...
            elif llama_backend == "sycl": cmake_args_list.append("-DGGML_SYCL=on"); # ...
            elif llama_backend == "rpc": cmake_args_list.append("-DGGML_RPC=on"); # ...
            elif llama_backend == "cpu": print_system("Building for CPU only.");
            else: print_warning(f"Unknown backend '{llama_backend}'. Building for CPU only.");

            if cmake_args_list: build_env['CMAKE_ARGS'] = " ".join(cmake_args_list)
            print_system(f"Running pip install from '{LLAMA_CPP_PYTHON_CLONE_PATH}' with build flags...")
            pip_install_command = [ PIP_EXECUTABLE, "install", ".", "--upgrade", "--no-cache-dir" ] # Add --no-cache-dir for fresh build
            if not run_command(pip_install_command, LLAMA_CPP_PYTHON_CLONE_PATH, "PIP-LLAMA", env_override=build_env):
                print_error("Failed to install custom llama-cpp-python. Exiting."); sys.exit(1)
            print_system("Custom llama-cpp-python build installed successfully.")
            # Create flag file for custom llama.cpp
            try:
                with open(CUSTOM_LLAMA_CPP_INSTALLED_FLAG_FILE, 'w') as f:
                    f.write(f"Custom Llama CPP Python installed on: {datetime.now().isoformat()}\n")
                print_system("Custom llama-cpp-python installation flag created.")
            except Exception as flag_err:
                print_error(f"Could not create custom llama-cpp-python installation flag file: {flag_err}")
        else:
            print_system("Custom llama-cpp-python previously installed (flag file found). Skipping build.")
    # --- End Custom Llama.cpp Installation ---

    # 3. Install/Check Dependencies (Node.js Backend Service)
    # ... (Keep existing Node backend dep install) ...
    print_system("--- Installing/Checking Node Backend Dependencies ---")
    pkg_path = os.path.join(BACKEND_SERVICE_DIR, "package.json")
    if not os.path.exists(pkg_path): print_error(f"package.json not found at {pkg_path}"); sys.exit(1)
    if not run_command([NPM_CMD, "install"], BACKEND_SERVICE_DIR, "NPM-BE"): print_error("Failed Node backend deps."); sys.exit(1)
    print_system("Node backend dependencies checked/installed.")


    # 4. Install/Check Dependencies (Node.js Frontend)
    # ... (Keep existing Node frontend dep install) ...
    print_system("--- Installing/Checking Node Frontend Dependencies ---")
    pkg_path_fe = os.path.join(FRONTEND_DIR, "package.json")
    if not os.path.exists(pkg_path_fe): print_error(f"package.json not found at {pkg_path_fe}"); sys.exit(1)
    if not run_command([NPM_CMD, "install"], FRONTEND_DIR, "NPM-FE"): print_error("Failed Node frontend deps."); sys.exit(1)
    print_system("Node frontend dependencies checked/installed.")


    # 5. Start all services concurrently
    # ... (Keep existing service startup logic) ...
    print_system("--- Starting Services ---")
    threads = []; threads.append(start_service_thread(start_engine_main, "EngineMainThread"))
    print_system("Waiting for Engine to initialize..."); time.sleep(5)
    engine_proc_ok = False
    with process_lock:
        for proc, name in running_processes:
            if name == "ENGINE" and proc.poll() is None: engine_proc_ok = True; break
    if not engine_proc_ok: print_error("Engine Main (Hypercorn) failed to stay running."); sys.exit(1)
    else: print_system("Engine Main running. Starting other services...")
    threads.append(start_service_thread(start_backend_service, "BackendServiceThread")); time.sleep(2)
    threads.append(start_service_thread(start_frontend, "FrontendThread"))
    print_system("All services starting. Press Ctrl+C to shut down.")


    # 6. Keep the main thread alive and monitor processes
    # ... (Keep existing monitoring loop) ...
    try:
        while True:
            active_process_found = False; all_processes_ok = True
            with process_lock: current_running = list(running_processes); processes_to_remove = []
            for i in range(len(current_running) - 1, -1, -1):
                 proc, name = current_running[i]
                 if proc.poll() is None: active_process_found = True
                 else: print_error(f"Service '{name}' exited (code {proc.poll()})."); processes_to_remove.append((proc,name)); all_processes_ok = False
            if processes_to_remove:
                 with process_lock: running_processes[:] = [p for p in running_processes if p not in processes_to_remove]
            if not all_processes_ok: print_error("A service exited. Shutting down."); break
            if not active_process_found and threads: print_system("All managed services exited."); break
            time.sleep(5)
    except KeyboardInterrupt: pass
    finally: print_system("Launcher shutting down or finished.")