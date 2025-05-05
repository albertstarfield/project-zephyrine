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
LICENSE_DIR = os.path.join(ROOT_DIR, "licenses") # Directory for license files
LICENSE_FLAG_FILE = os.path.join(ROOT_DIR, ".license_accepted_v1") # Flag file name

# Define paths to executables within the virtual environment
IS_WINDOWS = os.name == 'nt'

# Add windows-curses handling
if IS_WINDOWS:
    try:
        import curses # Try importing the windows-curses version
    except ImportError:
        # Print directly as colorama is removed
        print(f"ERROR: `windows-curses` not found. Please install it:")
        # Try to construct the pip path even without colorama
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
LLAMA_CPP_SUBMODULE_PATH = "vendor/llama.cpp"

# Global list to keep track of running subprocesses
running_processes = []
process_lock = threading.Lock()

# --- Helper Functions ---

def print_colored(prefix, message, color=None): # Removed color usage
    """Prints a message with a timestamp and prefix (no launcher colors)."""
    now = datetime.now().strftime('%H:%M:%S.%f')[:-3] # HH:MM:SS.ms
    # Just print prefix and message, ignore the color argument
    print(f"[{now} | {prefix.ljust(10)}] {message.strip()}")

def print_system(message): print_colored("SYSTEM", message)
def print_error(message): print_colored("ERROR", message)
def print_warning(message): print_colored("WARNING", message)

def stream_output(pipe, name=None, color=None): # Removed name/color usage
    """Reads lines from a subprocess pipe and prints them directly (passthrough)."""
    try:
        for line in iter(pipe.readline, ''):
            if line:
                # Print the line exactly as received from the subprocess
                # Use sys.stdout.write for direct output without added newline
                # and flush to ensure immediate visibility
                sys.stdout.write(line)
                sys.stdout.flush()
    except Exception as e:
        # Avoid printing error if pipe is closed expectedly during cleanup
        if 'read of closed file' not in str(e).lower():
            # Use the launcher's error printer for consistency
            print_error(f"Error reading output stream for pipe {pipe}: {e}")
    finally:
        if pipe:
            try: pipe.close()
            except Exception: pass

def run_command(command, cwd, name, color=None, check=True, capture_output=False, env_override=None):
    """Runs a command, optionally checks for errors, allows passthrough output."""
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
        # Pass pipe directly to the modified stream_output (no name/color needed)
        stdout_thread = threading.Thread(target=stream_output, args=(process.stdout,), daemon=True)
        stdout_thread.start()
        stderr_thread = threading.Thread(target=stream_output, args=(process.stderr,), daemon=True)
        stderr_thread.start()
        process.wait() # Wait for the process to complete
        stdout_thread.join() # Wait for the stream readers to finish
        stderr_thread.join()

        if check and process.returncode != 0:
            # Error message was printed by stderr thread via stream_output
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

# --- Service Start Functions ---
def start_engine_main():
    """Starts the Python Engine Main service using Hypercorn."""
    name = "ENGINE" # Name used for tracking the process
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
            command, cwd=ENGINE_MAIN_DIR, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding='utf-8', errors='replace', bufsize=1
        )
        with process_lock:
            running_processes.append((process, name)) # Track process by name
        # Start threads using the pass-through stream_output
        stdout_thread = threading.Thread(target=stream_output, args=(process.stdout,), daemon=True)
        stderr_thread = threading.Thread(target=stream_output, args=(process.stderr,), daemon=True)
        stdout_thread.start()
        stderr_thread.start()
    except FileNotFoundError:
         print_error(f"Command failed: '{os.path.basename(HYPERCORN_EXECUTABLE)}' not found. In venv?")
         sys.exit(1)
    except Exception as e:
        print_error(f"Failed to start {name}: {e}")
        sys.exit(1)

def start_backend_service():
    """Starts the Node.js Backend Service."""
    name = "BACKEND"
    command = ["node", "server.js"]
    print_system(f"Launching Backend Service: {' '.join(command)} in {BACKEND_SERVICE_DIR}")
    try:
        process = subprocess.Popen(
            command, cwd=BACKEND_SERVICE_DIR, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding='utf-8', errors='replace', bufsize=1, shell=IS_WINDOWS
        )
        with process_lock:
            running_processes.append((process, name))
        stdout_thread = threading.Thread(target=stream_output, args=(process.stdout,), daemon=True)
        stderr_thread = threading.Thread(target=stream_output, args=(process.stderr,), daemon=True)
        stdout_thread.start()
        stderr_thread.start()
    except FileNotFoundError:
        print_error("Command failed: 'node' not found. Is Node.js installed?")
        sys.exit(1)
    except Exception as e:
        print_error(f"Failed to start {name}: {e}")

def start_frontend():
    """Starts the Vite Frontend Development Server."""
    name = "FRONTEND"
    command = [NPM_CMD, "run", "dev"]
    print_system(f"Launching Frontend (Vite): {' '.join(command)} in {FRONTEND_DIR}")
    try:
        process = subprocess.Popen(
            command, cwd=FRONTEND_DIR, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding='utf-8', errors='replace', bufsize=1, shell=IS_WINDOWS
        )
        with process_lock:
            running_processes.append((process, name))
        stdout_thread = threading.Thread(target=stream_output, args=(process.stdout,), daemon=True)
        stderr_thread = threading.Thread(target=stream_output, args=(process.stderr,), daemon=True)
        stdout_thread.start()
        stderr_thread.start()
    except FileNotFoundError:
        print_error(f"Command failed: '{NPM_CMD}' not found. Is Node.js/npm installed?")
        sys.exit(1)
    except Exception as e:
        print_error(f"Failed to start {name}: {e}")

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
    if not os.path.isdir(LICENSE_DIR):
        print_error(f"License directory not found: {LICENSE_DIR}")
        print_error("Please create it and place the license files inside:")
        for fname, _ in LICENSES_TO_ACCEPT: print_error(f" - {fname}")
        sys.exit(1)

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
        # Need to end curses before printing error
        curses.nocbreak(); stdscr.keypad(False); curses.echo()
        curses.endwin()
        print("Terminal window too small to display license. Please resize and relaunch.")
        sys.exit(1)
        # return False, 0.0 # Indicate failure due to small terminal - exit instead

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
        try:
            stdscr.clear()

            # Header
            header = "--- SOFTWARE LICENSES ---"
            stdscr.addstr(0, max(0, (max_x - len(header)) // 2), header, curses.color_pair(5) | curses.A_BOLD)
            assoc = "Models: Flux1, Qwen2.5VL, Qwen2, Embeddings, SD | Code: Deepscaler, Zephyrine"
            stdscr.addstr(1, 0, assoc[:max_x-1], curses.color_pair(5)) # Truncate association if needed

            # Display license text
            for i in range(TEXT_AREA_HEIGHT):
                line_idx = top_line + i
                if line_idx < total_lines:
                    # Truncate lines that are too long for the screen width
                    line_text = licenses_text_lines[line_idx][:max_x - 1].rstrip() # Also strip trailing whitespace
                    stdscr.addstr(TEXT_START_Y + i, 0, line_text, curses.color_pair(2))
                else:
                    break # Stop drawing if we run out of lines

            # Instructions
            instr = "Scroll: UP/DOWN Arrows, PgUp/PgDn | Press 'a' to Accept, 'n' to Not Accept"
            stdscr.addstr(INSTR_Y, 0, instr[:max_x-1], curses.color_pair(1))

            # Prompt buttons
            stdscr.addstr(PROMPT_Y, ACCEPT_X, "[ Accept (a) ]", curses.color_pair(3) | curses.A_REVERSE)
            stdscr.addstr(PROMPT_Y, REJECT_X, "[Not Accept (n)]", curses.color_pair(4) | curses.A_REVERSE)

            stdscr.refresh()
        except curses.error as e:
             # Ignore errors potentially caused by resizing during display
             print_warning(f"Curses display error (potential resize?): {e}")
             time.sleep(0.1) # Small pause before trying to redraw
             max_y, max_x = stdscr.getmaxyx() # Update dimensions
             TEXT_AREA_HEIGHT = max_y - 6
             ACCEPT_X = max_x // 2 - 15
             REJECT_X = max_x // 2 + 5
             continue # Retry drawing

        # Get user input
        try:
            key = stdscr.getch()
        except KeyboardInterrupt: # Handle Ctrl+C during getch()
             accepted = False
             break
        except Exception as e:
             print_warning(f"Error getting input: {e}") # Log other potential errors
             key = -1 # Treat as no input


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
             new_top = top_line + TEXT_AREA_HEIGHT
             # Clamp to ensure last line is visible
             top_line = min(max(0, total_lines - TEXT_AREA_HEIGHT), new_top)
        elif key == curses.KEY_PPAGE: # Page Up
             top_line = max(0, top_line - TEXT_AREA_HEIGHT)
        elif key == curses.KEY_RESIZE: # Handle terminal resize event explicitly
             max_y, max_x = stdscr.getmaxyx()
             TEXT_AREA_HEIGHT = max_y - 6
             ACCEPT_X = max_x // 2 - 15
             REJECT_X = max_x // 2 + 5
             # Recalculate top_line to ensure it's valid
             top_line = min(max(0, total_lines - TEXT_AREA_HEIGHT), top_line)


    end_time = time.monotonic()
    time_taken = end_time - start_time
    # Curses cleanup happens in the wrapper

    return accepted, time_taken

# --- Main Execution Logic ---
if __name__ == "__main__":
    print_system("--- Project Zephyrine Launcher ---")

    # --- License Acceptance Step ---
    license_already_accepted = os.path.exists(LICENSE_FLAG_FILE)

    if not license_already_accepted:
        print_system("License agreement required for first run.")
        try:
            licenses_data, combined_license_text = load_licenses()
            estimated_reading_seconds = calculate_reading_time(combined_license_text)
            license_lines = combined_license_text.splitlines()

            # Use curses wrapper for safe init/cleanup
            accepted, time_taken = curses.wrapper(display_license_prompt, license_lines, estimated_reading_seconds)

            if not accepted:
                # Error message printed inside wrapper if needed, or print here
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
            # curses.wrapper should handle cleanup, but just in case:
            try: curses.endwin()
            except: pass
            sys.exit(1)
    else:
        print_system("License previously accepted (found flag file). Skipping prompt.")
    # --- End License Acceptance Step ---


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
                subprocess.run([python_cmd_for_venv, "-m", "venv", VENV_DIR], check=True, capture_output=True) # Capture output to hide venv messages
                print_system("Virtual environment created successfully.")
            except subprocess.CalledProcessError as e: print_error(f"Failed create venv: {e.stderr or e.stdout or e}"); sys.exit(1)
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
    if not run_command([PIP_EXECUTABLE, "install", "-r", req_path], ENGINE_MAIN_DIR, "PIP"): # Removed COLOR_SYSTEM
         print_error("Failed to install Python dependencies from requirements.txt. Exiting.")
         sys.exit(1)
    print_system("Standard Python dependencies checked/installed.")

    # --- Custom llama-cpp-python Installation ---
    provider_env = os.getenv("PROVIDER", "llama_cpp").lower()
    print_system(f"Checking PROVIDER environment variable: {provider_env}")

    if provider_env == "llama_cpp":
        print_system(f"--- Attempting Custom llama-cpp-python Installation ---")
        if not shutil.which(GIT_CMD): print_error(f"'{GIT_CMD}' command not found. Git is required. Exiting."); sys.exit(1)
        print_system("Uninstalling standard llama-cpp-python (if present)...")
        run_command([PIP_EXECUTABLE, "uninstall", "llama-cpp-python", "-y"], ROOT_DIR, "PIP", check=False)
        if not os.path.exists(LLAMA_CPP_PYTHON_CLONE_PATH):
            print_system(f"Cloning '{LLAMA_CPP_PYTHON_REPO_URL}' into '{LLAMA_CPP_PYTHON_CLONE_PATH}'...")
            if not run_command([GIT_CMD, "clone", LLAMA_CPP_PYTHON_REPO_URL, LLAMA_CPP_PYTHON_CLONE_PATH], ROOT_DIR, "GIT"):
                print_error("Failed to clone llama-cpp-python repository. Exiting."); sys.exit(1)
        else: print_system(f"llama-cpp-python directory already exists at '{LLAMA_CPP_PYTHON_CLONE_PATH}'. Skipping clone.")

        print_system("Initializing/updating submodules within llama-cpp-python (using default llama.cpp commit)...")
        if not run_command([GIT_CMD, "submodule", "update", "--init", "--recursive"], LLAMA_CPP_PYTHON_CLONE_PATH, "GIT"):
            print_error("Failed to update submodules. Exiting."); sys.exit(1)

        # --- Prepare environment for build with Dynamic CMAKE_ARGS ---
        print_system("Preparing environment for llama.cpp build...")
        build_env = {'FORCE_CMAKE': '1'}
        cmake_args_list = []
        default_backend = 'cpu'
        if sys.platform == "darwin": default_backend = 'metal'
        elif sys.platform in ["linux", "win32"]: default_backend = 'cuda'
        llama_backend = os.getenv("LLAMA_CPP_BACKEND", default_backend).lower()
        print_system(f"Selected llama.cpp backend: {llama_backend} (Control with LLAMA_CPP_BACKEND env var)")

        # Construct CMAKE_ARGS based on selected backend
        if llama_backend == "cuda": cmake_args_list.append("-DGGML_CUDA=on")
        elif llama_backend == "metal":
            if sys.platform != "darwin": print_warning("Metal backend selected but platform is not macOS.")
            cmake_args_list.append("-DGGML_METAL=on")
            if platform.machine() == "arm64":
                print_system("Adding Metal arm64 CMake flags for Apple Silicon.")
                cmake_args_list.extend(["-DCMAKE_OSX_ARCHITECTURES=arm64", "-DCMAKE_APPLE_SILICON_PROCESSOR=arm64"])
        elif llama_backend == "rocm":
            cmake_args_list.append("-DGGML_HIPBLAS=on")
            if not os.getenv("AMDGPU_TARGETS"): print_warning("ROCm: You might need to set AMDGPU_TARGETS env var.")
        elif llama_backend == "openblas":
            cmake_args_list.append("-DGGML_BLAS=ON"); cmake_args_list.append("-DGGML_BLAS_VENDOR=OpenBLAS")
        elif llama_backend == "vulkan":
            cmake_args_list.append("-DGGML_VULKAN=on")
            if not os.getenv("VULKAN_SDK"): print_warning("Vulkan: Ensure Vulkan SDK installed and VULKAN_SDK env var set.")
        elif llama_backend == "sycl":
             cmake_args_list.append("-DGGML_SYCL=on"); print_warning("SYCL: Ensure oneAPI installed and env sourced.")
        elif llama_backend == "rpc": cmake_args_list.append("-DGGML_RPC=on")
        elif llama_backend == "cpu": print_system("Building for CPU only."); cmake_args_list = []
        else: print_warning(f"Unknown backend '{llama_backend}'. CPU only."); cmake_args_list = []

        if cmake_args_list: build_env['CMAKE_ARGS'] = " ".join(cmake_args_list)
        print_system(f"Running pip install from '{LLAMA_CPP_PYTHON_CLONE_PATH}' with build flags...")
        pip_install_command = [ PIP_EXECUTABLE, "install", ".", "--upgrade" ]
        if not run_command(pip_install_command, LLAMA_CPP_PYTHON_CLONE_PATH, "PIP-LLAMA", env_override=build_env):
            print_error("Failed to install custom llama-cpp-python build. Exiting."); sys.exit(1)
        print_system("Custom llama-cpp-python build installed successfully.")

    # --- Continue with other dependencies (Node.js Backend/Frontend) ---
    # 3. Install/Check Dependencies (Node.js Backend Service)
    print_system("--- Installing/Checking Node Backend Dependencies ---")
    pkg_path = os.path.join(BACKEND_SERVICE_DIR, "package.json")
    if not os.path.exists(pkg_path): print_error(f"package.json not found at {pkg_path}"); sys.exit(1)
    if not run_command([NPM_CMD, "install"], BACKEND_SERVICE_DIR, "NPM-BE"):
         print_error("Failed to install Node backend dependencies. Exiting."); sys.exit(1)
    print_system("Node backend dependencies checked/installed.")

    # 4. Install/Check Dependencies (Node.js Frontend)
    print_system("--- Installing/Checking Node Frontend Dependencies ---")
    pkg_path_fe = os.path.join(FRONTEND_DIR, "package.json")
    if not os.path.exists(pkg_path_fe): print_error(f"package.json not found at {pkg_path_fe}"); sys.exit(1)
    if not run_command([NPM_CMD, "install"], FRONTEND_DIR, "NPM-FE"):
         print_error("Failed to install Node frontend dependencies. Exiting."); sys.exit(1)
    print_system("Node frontend dependencies checked/installed.")


    # 5. Start all services concurrently
    print_system("--- Starting Services ---")
    threads = []
    threads.append(start_service_thread(start_engine_main, "EngineMainThread"))
    print_system("Waiting a few seconds for Engine to initialize...")
    time.sleep(5)

    engine_proc_ok = False
    with process_lock:
        for proc, name in running_processes:
            if name == "ENGINE" and proc.poll() is None: engine_proc_ok = True; break
    if not engine_proc_ok: print_error("Engine Main (Hypercorn) failed to stay running. Check logs."); sys.exit(1)
    else: print_system("Engine Main appears to be running. Starting other services...")

    threads.append(start_service_thread(start_backend_service, "BackendServiceThread"))
    time.sleep(2)
    threads.append(start_service_thread(start_frontend, "FrontendThread"))

    print_system("All services are starting up. Press Ctrl+C to shut down.")

    # 6. Keep the main thread alive and monitor processes
    try:
        while True:
            active_process_found = False; all_processes_ok = True
            with process_lock:
                current_running = list(running_processes)
                processes_to_remove = []

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