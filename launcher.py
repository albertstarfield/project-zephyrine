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
    try: import curses # Ensure curses is checked early on Windows if needed for license
    except ImportError:
        print(f"ERROR: `windows-curses` not found. Please install it:")
        # Attempt to determine pip path for instruction, actual install is later
        _pip_path_scripts = os.path.join(VENV_DIR, 'Scripts', 'pip.exe')
        _pip_path_bin = os.path.join(VENV_DIR, 'bin', 'pip') # Should not happen on windows but good fallback
        _pip_exe_in_venv = os.path.exists(_pip_path_scripts) or os.path.exists(_pip_path_bin)
        _pip_cmd = _pip_path_scripts if os.path.exists(_pip_path_scripts) else 'pip'
        if not _pip_exe_in_venv: # If venv not created yet, pip might not be in venv path
             print(f"  If you have a virtual environment at {VENV_DIR}, use '{_pip_cmd} install windows-curses'")
             print(f"  Otherwise, install it globally or ensure your Python environment has it: pip install windows-curses")
        else:
             print(f"  {_pip_cmd} install windows-curses")
        sys.exit(1)

VENV_BIN_DIR = os.path.join(VENV_DIR, "Scripts" if IS_WINDOWS else "bin")
PYTHON_EXECUTABLE = os.path.join(VENV_BIN_DIR, "python.exe" if IS_WINDOWS else "python")
PIP_EXECUTABLE = os.path.join(VENV_BIN_DIR, "pip.exe" if IS_WINDOWS else "pip")
HYPERCORN_EXECUTABLE = os.path.join(VENV_BIN_DIR, "hypercorn.exe" if IS_WINDOWS else "hypercorn")
NPM_CMD = 'npm.cmd' if IS_WINDOWS else 'npm'
GIT_CMD = 'git.exe' if IS_WINDOWS else 'git'

# --- NEW: Static Model Pool Configuration ---
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
        "filename": "flux1-schnell-q2_k.gguf",
        "url": "https://huggingface.co/lllyasviel/FLUX.1-schnell-gguf/resolve/08fc03e0dc5c1466c60fb7e108974ded1124c515/flux1-schnell-Q2_K.gguf?download=true",
        "description": "FLUX.1 Schnell Model (Image Gen related)"
    },
    {
        "filename": "LatexMind-2B-Codec-i1-GGUF-IQ4_XS.gguf",
        "url": "https://huggingface.co/mradermacher/LatexMind-2B-Codec-i1-GGUF/resolve/main/LatexMind-2B-Codec.i1-IQ4_XS.gguf?download=true",
        "description": "LatexMind Model (LaTeX/VLM)"
    },
    {
        "filename": "mxbai-embed-large-v1.gguf", # Note: URL provides -f16 variant, filename kept generic
        "url": "https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1/resolve/7130e2d16051fdf3e0157e841f8b5a8d0d5e63ef/gguf/mxbai-embed-large-v1-f16.gguf?download=true",
        "description": "MXBAI Embeddings Model"
    },
    {
        "filename": "NanoTranslator-immersive_translate-0.5B-GGUF-Q4_K_M.gguf", # Note: URL uses Q5_K_M
        "url": "https://huggingface.co/mradermacher/NanoTranslator-immersive_translate-0.5B-GGUF/resolve/main/NanoTranslator-immersive_translate-0.5B.Q5_K_M.gguf?download=true",
        "description": "NanoTranslator Model"
    },
    {
        "filename": "qwen2-math-1.5b-instruct-q5_K_M.gguf",
        "url": "https://huggingface.co/lmstudio-community/Qwen2-Math-1.5B-Instruct-GGUF/resolve/main/Qwen2-Math-1.5B-Instruct-Q5_K_M.gguf?download=true",
        "description": "Qwen2 Math Model"
    },
    {
        "filename": "Qwen2.5-1.5B-Instruct-iq3_m.gguf", # Note: URL uses Q3_K_M
        "url": "https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/Qwen2.5-1.5B-Instruct-Q3_K_M.gguf?download=true",
        "description": "Qwen2.5 Instruct Model (Fast General)"
    },
    {
        "filename": "qwen2.5-coder-3b-instruct-q5_K_M.gguf",
        "url": "https://huggingface.co/bartowski/Qwen2.5-Coder-3B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-3B-Instruct-Q5_K_M.gguf?download=true",
        "description": "Qwen2.5 Coder Model"
    },
    {
        "filename": "Qwen2.5-VL-7B-Instruct-q4_k_m.gguf", # Note: URL links to Qwen2-VL
        "url": "https://huggingface.co/bartowski/Qwen2-VL-7B-Instruct-GGUF/resolve/main/Qwen2-VL-7B-Instruct-Q4_K_M.gguf?download=true",
        "description": "Qwen2.5 VL Model"
    },
    {
        "filename": "whisper-large-v3-q8_0.gguf",
        "url": "https://huggingface.co/vonjack/whisper-large-v3-gguf/resolve/main/whisper-large-v3-q8_0.gguf?download=true",
        "description": "Whisper Large v3 Model (ASR)"
    }
]

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

# --- Helper Functions ---
def print_colored(prefix, message, color=None): # Color param currently unused
    now = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    # Basic color for prefix (ANSI escape codes)
    prefix_color = "\033[94m" # Blue for SYSTEM
    if prefix == "ERROR": prefix_color = "\033[91m" # Red
    elif prefix == "WARNING": prefix_color = "\033[93m" # Yellow
    reset_color = "\033[0m"
    # Only use colors if output is a TTY (to avoid escape codes in logs)
    if sys.stdout.isatty():
        print(f"[{now} | {prefix_color}{prefix.ljust(10)}{reset_color}] {message.strip()}")
    else:
        print(f"[{now} | {prefix.ljust(10)}] {message.strip()}")

def print_system(message): print_colored("SYSTEM", message)
def print_error(message): print_colored("ERROR", message)
def print_warning(message): print_colored("WARNING", message)

def stream_output(pipe, name=None, color=None): # name, color params unused but kept for compatibility
    try:
        for line in iter(pipe.readline, ''):
            if line:
                # Try to print with prefix if name is available, else raw
                if name:
                    # A basic color for process output
                    # Could be enhanced to use the 'color' param if passed
                    output_color = "\033[90m" # Dark Grey for process output
                    reset_color = "\033[0m"
                    if sys.stdout.isatty():
                         sys.stdout.write(f"{output_color}[{name.upper()}] {line.strip()}{reset_color}\n")
                    else:
                         sys.stdout.write(f"[{name.upper()}] {line.strip()}\n")
                else:
                    sys.stdout.write(line)
                sys.stdout.flush()
    except Exception as e:
        # Be careful with printing errors from here, can cause recursion if print_error uses this
        if 'read of closed file' not in str(e).lower() and 'Stream is closed' not in str(e): # common on process end
             print(f"[STREAM_ERROR] Error reading output stream for pipe {pipe} (process: {name}): {e}")
    finally:
        if pipe:
            try: pipe.close()
            except Exception: pass


def run_command(command, cwd, name, color=None, check=True, capture_output=False, env_override=None):
    # Use a more descriptive name for logging if available
    log_name_prefix = name if name else os.path.basename(command[0])
    print_system(f"Running command in '{os.path.basename(cwd)}' for '{log_name_prefix}': {' '.join(command)}")
    current_env = os.environ.copy()
    if env_override:
        str_env_override = {k: str(v) for k, v in env_override.items()} # Ensure all env vars are strings
        print_system(f"  with custom env for '{log_name_prefix}': {str_env_override}")
        current_env.update(str_env_override)
    
    # Ensure all command parts are strings, especially for complex commands
    command = [str(c) for c in command]

    try:
        # For Windows, shell=True can be problematic with command lists.
        # It's generally safer to use shell=True only for specific commands known to need it (like npm.cmd).
        # Python's subprocess on Windows can handle .exe/.bat/.cmd directly if they are in PATH or full path is given.
        # The original (IS_WINDOWS and command[0] in [NPM_CMD, 'node', GIT_CMD, PIP_EXECUTABLE, PYTHON_EXECUTABLE])
        # was too broad. Let's make it more specific.
        use_shell = False
        if IS_WINDOWS:
            # shell=True is often needed for .cmd, .bat files or built-in shell commands.
            # For .exe files, it's usually not needed if they are in PATH or full path is given.
            # 'npm' on Windows often resolves to 'npm.cmd', so shell=True is good.
            if command[0] == NPM_CMD or command[0].lower().endswith('.cmd') or command[0].lower().endswith('.bat'):
                use_shell = True
            # Node.js (node.exe) itself usually doesn't need shell=True
            # Git (git.exe) usually doesn't need shell=True
            # Python/Pip executables from venv usually don't need shell=True

        process = subprocess.Popen(
            command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding='utf-8', errors='replace', bufsize=1, # line buffering
            shell=use_shell,
            env=current_env
        )
        
        # Give more context to streamed output
        stdout_thread = threading.Thread(target=stream_output, args=(process.stdout, f"{log_name_prefix}-OUT"), daemon=True)
        stdout_thread.start()
        stderr_thread = threading.Thread(target=stream_output, args=(process.stderr, f"{log_name_prefix}-ERR"), daemon=True)
        stderr_thread.start()
        
        process.wait() # Wait for the process to complete
        
        stdout_thread.join(timeout=5) # Wait for threads to finish flushing output
        stderr_thread.join(timeout=5)
        
        if check and process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command)
        
        print_system(f"Command finished successfully for '{log_name_prefix}' in '{os.path.basename(cwd)}'.")
        return True
    except FileNotFoundError:
        print_error(f"Command not found for '{log_name_prefix}': {command[0]}. Is it installed and in PATH?")
        if command[0] == GIT_CMD: print_error("Ensure Git is installed: https://git-scm.com/downloads")
        elif command[0] == NPM_CMD or (command[0] == 'node' and not IS_WINDOWS): print_error("Ensure Node.js and npm are installed: https://nodejs.org/")
        elif command[0] == 'node' and IS_WINDOWS : print_error("Ensure Node.js (node.exe) is installed and in PATH: https://nodejs.org/")
        elif command[0] == HYPERCORN_EXECUTABLE: print_error(f"Ensure '{os.path.basename(HYPERCORN_EXECUTABLE)}' is installed in the venv.")
        return False
    except subprocess.CalledProcessError as e:
        print_error(f"Command failed for '{log_name_prefix}' in '{os.path.basename(cwd)}' with exit code {e.returncode}.")
        # stderr should have been streamed by the thread
        return False
    except Exception as e:
        print_error(f"An unexpected error occurred while running command for '{log_name_prefix}' in '{os.path.basename(cwd)}': {e}")
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

    for proc, name in reversed(procs_to_terminate): # Terminate in reverse order of startup
        if proc.poll() is None: # Check if process is still running
            print_system(f"Terminating {name} (PID: {proc.pid})...")
            try:
                # On Windows, terminate() is an alias for TerminateProcess() which is forceful.
                # On POSIX, terminate() sends SIGTERM.
                proc.terminate()
                try:
                    proc.wait(timeout=5) # Wait for graceful termination
                    print_system(f"{name} terminated gracefully.")
                except subprocess.TimeoutExpired:
                    print_warning(f"{name} did not terminate gracefully, killing (PID: {proc.pid})...")
                    proc.kill() # Force kill
                    proc.wait(timeout=2) # Wait for kill
                    print_system(f"{name} killed.")
            except Exception as e:
                print_error(f"Error terminating/killing {name} (PID: {proc.pid}): {e}")
        else:
            print_system(f"{name} already exited (return code: {proc.poll()}).")
atexit.register(cleanup_processes)

def signal_handler(sig, frame):
    print_system("\nCtrl+C received. Initiating shutdown...")
    # cleanup_processes() will be called by atexit
    sys.exit(0) # This will trigger atexit handlers
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
        return process # Return process to check if it started okay
    except FileNotFoundError:
        print_error(f"Command failed for {name}: '{command[0]}' not found.")
        # cleanup_processes() will run on exit
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
    # node.exe itself doesn't need shell=True, but if server.js relies on shell features or relative paths weirdly,
    # it might be needed. Generally, False is safer.
    start_service_process(command, BACKEND_SERVICE_DIR, name, use_shell_windows=False)


def start_frontend():
    name = "FRONTEND"
    command = [NPM_CMD, "run", "dev"]
    # npm.cmd usually needs shell=True on Windows
    start_service_process(command, FRONTEND_DIR, name, use_shell_windows=True)


# --- License Acceptance Logic ---
LICENSES_TO_ACCEPT = [
    ("APACHE_2.0.txt", "Primary Framework Components (Apache 2.0)"),
    ("MIT_Deepscaler.txt", "Deepscaler Model/Code Components (MIT)"),
    ("MIT_Zephyrine.txt", "Project Zephyrine Core & UI (MIT)"),
    # Add other licenses here if they have separate files and distinct user-facing components
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
            combined_text += f"\n\n{'='*20} LICENSE: {filename} ({description}) {'='*20}\n\n{content}"
            print_system(f"  Loaded: {filename}")
        except FileNotFoundError:
            print_error(f"License file not found: {filepath}. Please ensure all license files are present in the '{LICENSE_DIR}' directory.")
            sys.exit(1)
        except Exception as e:
            print_error(f"Error reading license file {filepath}: {e}")
            sys.exit(1)
    return licenses_content, combined_text.strip()


def calculate_reading_time(text: str) -> float:
    if not TIKTOKEN_AVAILABLE:
        print_warning("tiktoken not found, cannot estimate reading time. Defaulting to 60s estimate.")
        return 60.0 # Default estimate if tiktoken is not available
    if not text: return 0.0
    
    WORDS_PER_MINUTE = 200 # A conservative estimate for reading legal/technical text
    AVG_WORDS_PER_TOKEN = 0.75 # Standard estimate
    
    try:
        # Using a common encoding, cl100k_base is for gpt-3.5-turbo, gpt-4
        enc = tiktoken.get_encoding("cl100k_base")
    except Exception as e:
        print_warning(f"Failed to get tiktoken encoding 'cl100k_base': {e}. Trying 'p50k_base'.")
        try:
            enc = tiktoken.get_encoding("p50k_base") # Fallback for older models
        except Exception as e2:
            print_warning(f"Failed to get tiktoken encoding 'p50k_base': {e2}. Defaulting reading time to 60s.")
            return 60.0

    try:
        tokens = enc.encode(text)
        num_tokens = len(tokens)
        estimated_words = num_tokens * AVG_WORDS_PER_TOKEN
        estimated_minutes = estimated_words / WORDS_PER_MINUTE
        estimated_seconds = estimated_minutes * 60
        print_system(f"Estimated reading time for licenses: ~{estimated_minutes:.1f} minutes ({estimated_seconds:.0f} seconds, based on {num_tokens} tokens).")
        return estimated_seconds
    except Exception as e:
        print_warning(f"Failed to calculate reading time with tiktoken: {e}. Defaulting to 60s.")
        return 60.0

def display_license_prompt(stdscr, licenses_text_lines: list, estimated_seconds: float) -> tuple[bool, float]:
    curses.curs_set(0) # Hide cursor
    stdscr.nodelay(False) # Wait for input
    stdscr.keypad(True) # Enable special keys
    curses.noecho() # Don't echo input
    curses.cbreak() # React to keys instantly

    # Colors
    if curses.has_colors():
        curses.start_color()
        curses.init_pair(1, curses.COLOR_YELLOW, curses.COLOR_BLACK) # Instructions
        curses.init_pair(2, curses.COLOR_WHITE, curses.COLOR_BLACK)  # License text
        curses.init_pair(3, curses.COLOR_BLACK, curses.COLOR_GREEN)  # Accept button
        curses.init_pair(4, curses.COLOR_BLACK, curses.COLOR_RED)    # Reject button
        curses.init_pair(5, curses.COLOR_CYAN, curses.COLOR_BLACK)   # Header
        curses.init_pair(6, curses.COLOR_GREEN, curses.COLOR_BLACK)  # Accept text
        curses.init_pair(7, curses.COLOR_RED, curses.COLOR_BLACK)    # Reject text
    else: # Fallback for no colors
        curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_WHITE, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_WHITE, curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_WHITE, curses.COLOR_BLACK)
        curses.init_pair(5, curses.COLOR_WHITE, curses.COLOR_BLACK)
        curses.init_pair(6, curses.COLOR_WHITE, curses.COLOR_BLACK)
        curses.init_pair(7, curses.COLOR_WHITE, curses.COLOR_BLACK)


    max_y, max_x = stdscr.getmaxyx()
    if max_y < 10 or max_x < 40: # Basic terminal size check
        # Clean up curses before printing error and exiting
        curses.nocbreak(); stdscr.keypad(False); curses.echo(); curses.endwin()
        print_error("Terminal window is too small to display licenses. Please resize and try again.")
        sys.exit(1)

    TEXT_AREA_HEIGHT = max_y - 7 # Adjusted for more info lines
    TEXT_START_Y = 1
    SCROLL_INFO_Y = max_y - 5
    EST_TIME_Y = max_y - 4
    INSTR_Y = max_y - 3
    PROMPT_Y = max_y - 2

    top_line = 0
    total_lines = len(licenses_text_lines)
    accepted = False
    start_time = time.monotonic() # More precise than time.time() for measuring duration

    current_selection = 0 # 0 for Accept, 1 for Not Accept

    while True:
        try:
            stdscr.clear()
            max_y, max_x = stdscr.getmaxyx() # Update dimensions on each loop for resize handling
            
            # Recalculate layout based on potentially new dimensions
            TEXT_AREA_HEIGHT = max(1, max_y - 7)
            TEXT_START_Y = 1
            SCROLL_INFO_Y = max_y - 5
            EST_TIME_Y = max_y - 4
            INSTR_Y = max_y - 3
            PROMPT_Y = max_y - 2

            # Header
            header = "--- SOFTWARE LICENSE AGREEMENT ---"
            stdscr.addstr(0, max(0, (max_x - len(header)) // 2), header, curses.color_pair(5) | curses.A_BOLD)

            # Display license text
            for i in range(TEXT_AREA_HEIGHT):
                line_idx = top_line + i
                if line_idx < total_lines:
                    # Truncate line to fit window width
                    display_line = licenses_text_lines[line_idx][:max_x -1].rstrip()
                    stdscr.addstr(TEXT_START_Y + i, 0, display_line, curses.color_pair(2))
                else:
                    break
            
            # Scroll progress
            if total_lines > TEXT_AREA_HEIGHT:
                progress = f"Line {top_line + 1} - {min(top_line + TEXT_AREA_HEIGHT, total_lines)} of {total_lines}"
                stdscr.addstr(SCROLL_INFO_Y, max(0, max_x - len(progress) -1 ), progress, curses.color_pair(1))
            else:
                 stdscr.addstr(SCROLL_INFO_Y, 0, "All text visible.", curses.color_pair(1))


            # Estimated reading time
            est_time_str = f"Estimated minimum review time: {estimated_seconds / 60:.1f} min ({estimated_seconds:.0f} sec)"
            stdscr.addstr(EST_TIME_Y, 0, est_time_str[:max_x-1], curses.color_pair(1))
            
            # Instructions
            instr = "Scroll: UP/DOWN/PGUP/PGDN | Select: LEFT/RIGHT | Confirm: ENTER / (a/n)"
            stdscr.addstr(INSTR_Y, 0, instr[:max_x-1], curses.color_pair(1))

            # Buttons
            accept_button_text = "[ Accept (A) ]"
            reject_button_text = "[ Not Accept (N) ]"
            
            button_spacing = 4
            buttons_total_width = len(accept_button_text) + len(reject_button_text) + button_spacing
            start_buttons_x = max(0, (max_x - buttons_total_width) // 2)
            
            accept_style = curses.color_pair(3) | curses.A_REVERSE if current_selection == 0 else curses.color_pair(6)
            reject_style = curses.color_pair(4) | curses.A_REVERSE if current_selection == 1 else curses.color_pair(7)

            stdscr.addstr(PROMPT_Y, start_buttons_x, accept_button_text, accept_style)
            stdscr.addstr(PROMPT_Y, start_buttons_x + len(accept_button_text) + button_spacing, reject_button_text, reject_style)
            
            stdscr.refresh()

        except curses.error as e:
            # This can happen if terminal is resized too small during display
            # We attempt to clean up curses and then re-raise or handle
            print_warning(f"Curses display error: {e}. If persists, resize terminal.")
            time.sleep(0.1) # Brief pause
            # Try to recover by re-getting dimensions, if terminal is still too small, error will recur
            max_y, max_x = stdscr.getmaxyx()
            if max_y < 10 or max_x < 40:
                curses.nocbreak(); stdscr.keypad(False); curses.echo(); curses.endwin()
                print_error("Terminal window became too small. Please resize and try again.")
                sys.exit(1)
            continue # Retry drawing

        try:
            key = stdscr.getch()
        except KeyboardInterrupt: # Handle Ctrl+C during getch
            accepted = False
            break 
        except Exception as e: # Catch other potential getch errors
            print_warning(f"Error getting input: {e}")
            key = -1 # Treat as no input

        if key == ord('a') or key == ord('A'):
            accepted = True; break
        elif key == ord('n') or key == ord('N'):
            accepted = False; break
        elif key == curses.KEY_ENTER or key == 10 or key == 13:
            if current_selection == 0: accepted = True
            else: accepted = False
            break
        elif key == curses.KEY_LEFT:
            current_selection = 0
        elif key == curses.KEY_RIGHT:
            current_selection = 1
        elif key == curses.KEY_DOWN:
            top_line = min(top_line + 1, max(0, total_lines - TEXT_AREA_HEIGHT))
        elif key == curses.KEY_UP:
            top_line = max(0, top_line - 1)
        elif key == curses.KEY_NPAGE: # Page Down
            top_line = min(max(0, total_lines - TEXT_AREA_HEIGHT), top_line + TEXT_AREA_HEIGHT)
        elif key == curses.KEY_PPAGE: # Page Up
            top_line = max(0, top_line - TEXT_AREA_HEIGHT)
        elif key == curses.KEY_HOME:
            top_line = 0
        elif key == curses.KEY_END:
            top_line = max(0, total_lines - TEXT_AREA_HEIGHT)
        elif key == curses.KEY_RESIZE:
            # Terminal was resized, loop will re-calculate dimensions and redraw
            pass 
            
    end_time = time.monotonic()
    time_taken = end_time - start_time
    
    # Clean up curses settings before returning
    curses.nocbreak(); stdscr.keypad(False); curses.echo(); curses.endwin()
    return accepted, time_taken


# --- File Download Logic ---
# Note: tqdm and requests are imported later, after venv check and initial pip install
# This function will be called after those imports are confirmed.

def download_file_with_progress(url, destination_path, file_description, requests_session):
    """Downloads a file from a URL to a destination path with a progress bar, using a requests.Session."""
    from tqdm import tqdm # Import here as it's installed conditionally
    
    print_system(f"Downloading {file_description} from {url} to {destination_path}...")
    try:
        # Ensure parent directory for destination_path exists
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)

        response = requests_session.get(url, stream=True, timeout=(10, 300))  # (connect_timeout, read_timeout)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 8192  # 8 Kibibytes, a common chunk size for network operations

        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, desc=file_description, ascii=IS_WINDOWS, leave=False) # ascii for windows
        
        temp_destination_path = destination_path + ".tmp_download" # Temporary file for atomicity

        with open(temp_destination_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()

        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print_error(
                f"Download ERROR for {file_description}: Size mismatch (expected {total_size_in_bytes}, got {progress_bar.n}). File might be corrupted.")
            if os.path.exists(temp_destination_path): os.remove(temp_destination_path)
            return False
        
        shutil.move(temp_destination_path, destination_path) # Atomic move if possible
        print_system(f"Successfully downloaded and verified {file_description}.")
        return True

    except ImportError: # Should not happen if script structure is followed
        print_error("tqdm or requests module not found. Critical download components missing.")
        return False
    except requests_session.exceptions.RequestException as e: # Corrected exception type
        print_error(f"Failed to download {file_description} due to network/request error: {e}")
        if os.path.exists(destination_path + ".tmp_download"): # Clean up partial temporary download
            try: os.remove(destination_path + ".tmp_download")
            except Exception as rm_err: print_warning(f"Could not remove partial download {destination_path + '.tmp_download'}: {rm_err}")
        return False
    except Exception as e:
        print_error(f"An unexpected error occurred during download of {file_description}: {e}")
        if os.path.exists(destination_path + ".tmp_download"): # Clean up partial temporary download
            try: os.remove(destination_path + ".tmp_download")
            except Exception as rm_err: print_warning(f"Could not remove partial download {destination_path + '.tmp_download'}: {rm_err}")
        return False


# --- Main Execution Logic ---
if __name__ == "__main__":
    print_system("--- Project Zephyrine Launcher ---")
    print_system(f"Root directory: {ROOT_DIR}")
    print_system(f"Python version: {sys.version.split()[0]} on {platform.system()} ({platform.machine()})")

    # --- License Acceptance Step ---
    if not os.path.exists(LICENSE_FLAG_FILE):
        print_system("License agreement required for first run or if flag file is missing.")
        try:
            _, combined_license_text = load_licenses() # licenses_data dict is not used further here
            estimated_reading_seconds = calculate_reading_time(combined_license_text)
            license_lines = combined_license_text.splitlines()
            
            # Ensure curses can be initialized (especially on Windows for windows-curses)
            if IS_WINDOWS and 'curses' not in sys.modules:
                print_error("Windows-curses module not loaded. This should have been checked earlier.")
                print_error("Please ensure 'pip install windows-curses' has been run in your environment.")
                sys.exit(1)

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
                    print_error(f"Critical: Could not create license acceptance flag file '{LICENSE_FLAG_FILE}': {flag_err}")
                    print_error("Please check permissions for the application directory. Exiting.")
                    sys.exit(1) # This is critical, should not proceed without it.
                
                print_system(f"Licenses accepted by user in {time_taken:.2f} seconds.")
                # MIN_REASONABLE_TIME_FACTOR: e.g. 0.25 meaning if accepted in <25% of est. time
                MIN_REASONABLE_TIME_FACTOR = 0.1 
                if estimated_reading_seconds > 30 and time_taken < (estimated_reading_seconds * MIN_REASONABLE_TIME_FACTOR): # e.g. if est >30s and took <10% of that
                    print_warning(f"Warning: Licenses were accepted very quickly ({time_taken:.2f}s vs estimated {estimated_reading_seconds:.2f}s).")
                    print_warning("Please ensure you have understood the terms.")
                    time.sleep(3) # Brief pause for user to see warning
        
        except curses.error as e: # Catch curses errors not handled within display_license_prompt
            print_error(f"A Curses error occurred during license display: {e}")
            print_error("This might be due to an incompatible terminal or terminal size issues.")
            sys.exit(1)
        except Exception as e:
            print_error(f"An unexpected error occurred during the license acceptance process: {e}")
            # Attempt to clean up curses if it was initialized and an error occurred elsewhere
            try: curses.endwin()
            except: pass
            sys.exit(1)
    else:
        print_system(f"License previously accepted (flag file found: {LICENSE_FLAG_FILE}).")


    # 1. Check/Create/Relaunch in Virtual Environment
    is_in_target_venv = (os.getenv('VIRTUAL_ENV') == VENV_DIR) or \
                         (hasattr(sys, 'prefix') and sys.prefix == VENV_DIR) or \
                         (hasattr(sys, 'real_prefix') and sys.real_prefix == VENV_DIR) # for some venv versions

    if not is_in_target_venv:
        print_system(f"Not running in the target virtual environment: {VENV_DIR}")
        if not os.path.exists(VENV_DIR):
            print_system(f"Virtual environment not found. Creating at: {VENV_DIR}")
            try:
                # Using sys.executable ensures we use the python that started this script
                subprocess.run([sys.executable, "-m", "venv", VENV_DIR], check=True, capture_output=True, text=True)
                print_system("Virtual environment created successfully.")
            except subprocess.CalledProcessError as e:
                print_error(f"Failed to create virtual environment: {e}")
                print_error(f"Stdout: {e.stdout}")
                print_error(f"Stderr: {e.stderr}")
                sys.exit(1)
            except Exception as e: # Catch other potential errors like permission issues
                print_error(f"An unexpected error occurred while creating virtual environment: {e}")
                sys.exit(1)
        
        print_system(f"Relaunching script in the virtual environment using: {PYTHON_EXECUTABLE}")
        if not os.path.exists(PYTHON_EXECUTABLE):
            print_error(f"Python executable not found in venv: {PYTHON_EXECUTABLE}")
            print_error("The virtual environment might be corrupted or was not created correctly.")
            sys.exit(1)
        
        try:
            # os.execv replaces the current process with the new one
            os.execv(PYTHON_EXECUTABLE, [PYTHON_EXECUTABLE] + sys.argv)
        except Exception as e:
            print_error(f"Failed to relaunch script in virtual environment: {e}")
            sys.exit(1)

    print_system(f"Running inside target virtual environment: {VENV_DIR}")

    # Install tqdm and requests first, as they are needed for model downloads
    print_system("--- Installing Core Helper Utilities (tqdm, requests) ---")
    # Use ENGINE_MAIN_DIR as cwd, assuming it's a stable directory. ROOT_DIR is also fine.
    if not run_command([PIP_EXECUTABLE, "install", "tqdm", "requests"], ROOT_DIR, "PIP-UTILS"):
        print_error("Failed to install tqdm/requests. These are essential for downloading models. Exiting.")
        sys.exit(1)
    print_system("Core helper utilities (tqdm, requests) installed/checked.")

    # Now import requests, as it should be available
    try:
        import requests
    except ImportError:
        print_error("Failed to import 'requests' after attempting installation. Venv might be misconfigured. Exiting.")
        sys.exit(1)
    
    # Create a requests session for optimized downloads
    requests_session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=20) # Adjust pool sizes as needed
    requests_session.mount('http://', adapter)
    requests_session.mount('https://', adapter)


    # 2. Install/Check Dependencies (Python from requirements.txt for Engine)
    print_system("--- Installing/Checking Python Dependencies (Engine) ---")
    # Requirements file should be relative to ENGINE_MAIN_DIR or specify full path
    engine_req_path = os.path.join(ENGINE_MAIN_DIR, "requirements.txt") # Assuming it's inside ENGINE_MAIN_DIR
    if not os.path.exists(engine_req_path):
        print_error(f"Engine requirements.txt not found at {engine_req_path}")
        sys.exit(1)
    if not run_command([PIP_EXECUTABLE, "install", "-r", engine_req_path], ENGINE_MAIN_DIR, "PIP-ENGINE-REQ"):
        print_error("Failed to install Python dependencies for Engine. Exiting.")
        sys.exit(1)
    print_system("Standard Python dependencies for Engine checked/installed.")


    # --- Static Model Pool Setup ---
    print_system(f"--- Checking/Populating Static Model Pool at {STATIC_MODEL_POOL_PATH} ---")
    if not os.path.isdir(STATIC_MODEL_POOL_PATH):
        print_system(f"Static model pool directory not found. Creating: {STATIC_MODEL_POOL_PATH}")
        try:
            os.makedirs(STATIC_MODEL_POOL_PATH, exist_ok=True)
            print_system("Static model pool directory created.")
        except OSError as e:
            print_error(f"Failed to create static model pool directory '{STATIC_MODEL_POOL_PATH}': {e}")
            sys.exit(1)

    all_models_present_and_correct = True
    for model_info in MODELS_TO_DOWNLOAD:
        model_dest_path = os.path.join(STATIC_MODEL_POOL_PATH, model_info["filename"])
        if not os.path.exists(model_dest_path):
            print_warning(f"Model '{model_info['description']}' ({model_info['filename']}) not found in pool. Attempting download.")
            if not download_file_with_progress(model_info["url"], model_dest_path, model_info["description"], requests_session):
                print_error(
                    f"Failed to download '{model_info['filename']}'. Please check the URL, your internet connection, and disk space.")
                all_models_present_and_correct = False
                # Decide whether to exit or continue if a model download fails.
                # For now, let's continue and report all missing/failed downloads at the end.
            else:
                print_system(f"Model '{model_info['filename']}' downloaded successfully.")
        else:
            print_system(f"Model '{model_info['description']}' ({model_info['filename']}) already present in pool.")
            # Optional: Add size/hash verification for existing files here if desired
            # For example, fetch expected size from headers if possible and compare.

    if not all_models_present_and_correct:
        print_error("One or more required models could not be downloaded or verified. Application functionality might be impaired.")
        # Optionally, exit here if all models are strictly critical:
        # print_error("Exiting due to missing critical models.")
        # sys.exit(1)
    else:
        print_system("All required models for the static pool are present or were successfully downloaded.")
    # --- END Static Model Pool Setup ---


    # --- MeloTTS Installation & Initial Test ---
    if not os.path.exists(MELO_TTS_INSTALLED_FLAG_FILE):
        print_system(f"--- MeloTTS First-Time Setup from {MELO_TTS_PATH} ---")
        if not os.path.isdir(MELO_TTS_PATH):
            print_error(f"MeloTTS submodule directory not found at: {MELO_TTS_PATH}")
            print_error("Please ensure MeloTTS is cloned or placed there (e.g., 'git submodule update --init --recursive').")
            sys.exit(1)

        # Step 1: pip install -e . (editable install from the MeloTTS directory)
        print_system("Installing MeloTTS in editable mode...")
        if not run_command([PIP_EXECUTABLE, "install", "-e", "."], MELO_TTS_PATH, "PIP-MELO-EDITABLE"):
            print_error("Failed to install MeloTTS (editable). Check pip logs. Exiting.")
            sys.exit(1)
        print_system("MeloTTS (editable) installed successfully.")

        # Step 2: python -m unidic download (for Japanese support in MeloTTS)
        print_system("Downloading 'unidic' dictionary for MeloTTS (for Japanese)...")
        # Run this in a context where MeloTTS is importable, e.g. ROOT_DIR or MELO_TTS_PATH
        if not run_command([PYTHON_EXECUTABLE, "-m", "unidic", "download"], MELO_TTS_PATH, "UNIDIC-DOWNLOAD"):
            print_warning(
                "Failed to download 'unidic' dictionary automatically. MeloTTS might not work correctly for Japanese text-to-speech.")
            # This is not fatal for non-Japanese TTS, so we continue.
        else:
            print_system("'unidic' dictionary downloaded successfully (or already present).")

        # Step 3: Run audio_worker.py in --test-mode to trigger MeloTTS internal model downloads
        print_system("--- Running MeloTTS Initial Test (this will trigger internal model downloads) ---")
        audio_worker_script = os.path.join(ENGINE_MAIN_DIR, "audio_worker.py")
        if not os.path.exists(audio_worker_script):
            print_error(f"Critical: audio_worker.py not found at '{audio_worker_script}' for MeloTTS initial test.")
            sys.exit(1)

        test_mode_temp_dir = os.path.join(ROOT_DIR, "temp_melo_audio_test_files")
        os.makedirs(test_mode_temp_dir, exist_ok=True)
        # Use a timestamp to avoid collision if run multiple times without cleanup
        test_output_filename = f"initial_melo_test_{datetime.now().strftime('%Y%m%d%H%M%S')}.wav" # MeloTTS defaults to wav

        # Test with English (EN) first as it's common and models are smaller.
        # Other languages like 'ES', 'FR', 'ZH', 'JP', 'KR' can be tested if primary.
        test_command = [
            PYTHON_EXECUTABLE,
            audio_worker_script,
            "--test-mode",
            "--model-lang", "EN",
            "--device", "auto",      # Let MeloTTS auto-detect device (cpu/cuda)
            "--output-file", test_output_filename,
            "--temp-dir", test_mode_temp_dir # audio_worker.py should use this for its output
        ]
        # The audio_worker.py will run in its own directory context (ENGINE_MAIN_DIR)
        # and output its file to test_mode_temp_dir relative to where it's called or absolute.
        # For simplicity, make --output-file an absolute path.
        abs_test_output_file = os.path.join(test_mode_temp_dir, test_output_filename)
        test_command[test_command.index("--output-file")+1] = abs_test_output_file


        print_system(f"Executing MeloTTS test: {' '.join(test_command)}")
        if not run_command(test_command, ENGINE_MAIN_DIR, "MELO-INIT-TEST"):
            print_warning(
                "MeloTTS initial test mode run failed or returned an error. This might indicate an issue with MeloTTS setup, its internal model downloads, or device compatibility (e.g., CUDA unavailable/misconfigured if 'auto' tried to use it).")
            print_warning("The application will proceed, but Text-to-Speech functionality via MeloTTS could be impaired.")
            # Optionally, make this a fatal error if TTS is critical:
            # print_error("Exiting due to MeloTTS initialization failure.")
            # sys.exit(1)
        else:
            print_system("MeloTTS initial test mode run completed successfully.")
            if os.path.exists(abs_test_output_file):
                print_system(f"Test audio file generated at: {abs_test_output_file}")
                print_system("You can play this file to verify MeloTTS is working. It will be kept for now.")
                # To auto-delete:
                # try:
                #     os.remove(abs_test_output_file)
                #     print_system(f"Test audio file {abs_test_output_file} removed.")
                # except Exception as e:
                #     print_warning(f"Could not remove test audio file {abs_test_output_file}: {e}")
            else:
                print_warning(f"MeloTTS test mode indicated success, but test output file '{abs_test_output_file}' was not found. Check audio_worker.py logic.")


        # Create flag file on successful completion of all MeloTTS setup steps
        try:
            with open(MELO_TTS_INSTALLED_FLAG_FILE, 'w', encoding='utf-8') as f:
                f.write(f"MeloTTS installed and initially tested on: {datetime.now().isoformat()}\n")
            print_system("MeloTTS installation and initial test flag created.")
        except IOError as flag_err:
            print_error(f"Could not create MeloTTS installation flag file '{MELO_TTS_INSTALLED_FLAG_FILE}': {flag_err}")
            # This isn't fatal for running the app, but setup will re-run next time.
    else:
        print_system("MeloTTS previously installed and tested (flag file found). Skipping setup.")
    # --- END MeloTTS Installation & Initial Test ---


   # ... (all the previous script content up to the llama-cpp-python section) ...

    # --- Custom llama-cpp-python Installation ---
    provider_env = os.getenv("PROVIDER", "llama_cpp").lower()
    if provider_env == "llama_cpp":
        if not os.path.exists(CUSTOM_LLAMA_CPP_INSTALLED_FLAG_FILE):
            print_system(f"--- Custom llama-cpp-python Installation (as PROVIDER=llama_cpp) ---")
            
            if not shutil.which(GIT_CMD):
                print_error(f"'{GIT_CMD}' command not found. Git is required to clone and build custom llama-cpp-python.")
                print_error("Please install Git (https://git-scm.com/downloads) and ensure it's in your system's PATH.")
                sys.exit(1)

            # --- Conda Environment Check ---
            conda_prefix_env = os.getenv("CONDA_PREFIX")
            if conda_prefix_env:
                print_warning(f"WARNING: Conda environment detected (CONDA_PREFIX='{conda_prefix_env}').")
                print_warning("The build process for C/C++ extensions like llama-cpp-python might use Conda's compilers.")
                print_warning("While this script runs in its own venv, Conda in your PATH can influence external build tools.")
                print_warning("If you encounter build issues related to Conda and intend a pure venv/system compiler build:")
                print_warning("  1. Ensure your Conda environment has necessary build tools (e.g., 'gcc_linux-64', 'gxx_linux-64', 'libgomp' via 'conda install').")
                print_warning("  2. Or, try running this script from a shell where Conda is not activated or its paths are not dominant in your PATH.")
            # --- End Conda Environment Check ---

            print_system("Attempting to uninstall any existing standard 'llama-cpp-python'...")
            run_command([PIP_EXECUTABLE, "uninstall", "llama-cpp-python", "-y"], ROOT_DIR, "PIP-UNINSTALL-LLAMA", check=False)

            if os.path.exists(LLAMA_CPP_PYTHON_CLONE_PATH):
                 print_system(f"Previous llama-cpp-python build directory found. Cleaning: {LLAMA_CPP_PYTHON_CLONE_PATH}")
                 try: shutil.rmtree(LLAMA_CPP_PYTHON_CLONE_PATH)
                 except Exception as e:
                     print_error(f"Failed to remove existing build directory '{LLAMA_CPP_PYTHON_CLONE_PATH}': {e}.")
                     print_error("Please remove it manually and retry. Exiting."); sys.exit(1)

            print_system(f"Cloning '{LLAMA_CPP_PYTHON_REPO_URL}' into '{LLAMA_CPP_PYTHON_CLONE_PATH}'...")
            if not run_command([GIT_CMD, "clone", LLAMA_CPP_PYTHON_REPO_URL, LLAMA_CPP_PYTHON_CLONE_PATH], ROOT_DIR, "GIT-CLONE-LLAMA"):
                print_error("Failed to clone llama-cpp-python repository. Check network and Git setup. Exiting."); sys.exit(1)

            print_system("Initializing/updating llama.cpp submodule...")
            if not run_command([GIT_CMD, "submodule", "update", "--init", "--recursive"], LLAMA_CPP_PYTHON_CLONE_PATH, "GIT-SUBMODULE-LLAMA"):
                print_error("Failed to initialize/update git submodules for llama-cpp-python. Exiting."); sys.exit(1)

            build_env = {'FORCE_CMAKE': '1'}
            
            # CRITICAL FIX for linking errors with examples: Disable building llama.cpp examples and tests.
            # The python bindings do not require them.
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
            elif llama_backend == "vulkan": cmake_args_list.append("-DGGML_VULKAN=ON")
            elif llama_backend == "sycl": cmake_args_list.append("-DGGML_SYCL=ON")
            elif llama_backend == "rpc": cmake_args_list.append("-DGGML_RPC=ON")
            elif llama_backend == "cpu":
                print_system("Building llama.cpp for CPU with OpenMP (if available).")
                cmake_args_list.append("-DLLAMA_OPENMP=ON") # Default in llama.cpp, but explicit
                if sys.platform.startswith("linux"):
                     print_warning("For CPU builds on Linux, ensure 'libgomp1' (or equivalent OpenMP runtime) is installed for optimal performance. (e.g., 'sudo apt install libgomp1', or if using Conda's GCC, 'conda install libgomp').")
            else:
                print_warning(f"Unknown LLAMA_CPP_BACKEND '{llama_backend}'. Defaulting to CPU-only build with OpenMP (if available).")
                cmake_args_list.append("-DLLAMA_OPENMP=ON")
                if sys.platform.startswith("linux"):
                     print_warning("Ensure 'libgomp1' is installed for OpenMP.")
            
            if cmake_args_list:
                effective_cmake_args = " ".join(filter(None, cmake_args_list))
                if effective_cmake_args: build_env['CMAKE_ARGS'] = effective_cmake_args
            
            print_system(f"Running pip install for custom llama-cpp-python from '{LLAMA_CPP_PYTHON_CLONE_PATH}'...")
            print_system(f"  with CMAKE_ARGS: {build_env.get('CMAKE_ARGS', 'None')}")
            
            pip_install_command = [PIP_EXECUTABLE, "install", ".", "--upgrade", "--no-cache-dir", "--verbose"]
            
            if not run_command(pip_install_command, LLAMA_CPP_PYTHON_CLONE_PATH, "PIP-BUILD-LLAMA", env_override=build_env):
                print_error("Failed to build and install custom llama-cpp-python.")
                print_error("Please check the build log above for specific errors from CMake or the compiler.")
                print_error("Common issues: missing C/C++ compiler (GCC/Clang/MSVC), CMake, or backend-specific SDKs (CUDA, ROCm).")
                print_error("If related to OpenMP ('libgomp'), ensure it's installed and accessible by your compiler.")
                sys.exit(1)
            
            print_system("Custom llama-cpp-python built and installed successfully.")
            try:
                with open(CUSTOM_LLAMA_CPP_INSTALLED_FLAG_FILE, 'w', encoding='utf-8') as f:
                    f.write(f"Custom Llama CPP Python (backend: {llama_backend}) installed on: {datetime.now().isoformat()}\n")
                print_system("Custom llama-cpp-python installation flag created.")
            except IOError as flag_err:
                print_error(f"Could not create custom llama-cpp-python installation flag file: {flag_err}")
        else:
            print_system("Custom llama-cpp-python previously installed (flag file found). Skipping build.")
    else:
        print_system(f"PROVIDER is set to '{provider_env}'. Skipping custom llama-cpp-python build. Standard version will be used if listed in requirements.")
    # --- End Custom Llama.cpp Installation ---

# ... (rest of the script: Node dependencies, service starts, monitoring loop, etc.) ...


    # 3. Install/Check Dependencies (Node.js Backend Service)
    print_system("--- Installing/Checking Node.js Backend Dependencies ---")
    backend_pkg_path = os.path.join(BACKEND_SERVICE_DIR, "package.json")
    if not os.path.exists(backend_pkg_path):
        print_error(f"Backend package.json not found at {backend_pkg_path}. Cannot install dependencies.")
        sys.exit(1)
    # `npm install` is generally idempotent and safe to run.
    if not run_command([NPM_CMD, "install"], BACKEND_SERVICE_DIR, "NPM-BACKEND"):
        print_error("Failed to install Node.js backend dependencies. Check npm logs. Exiting.")
        sys.exit(1)
    print_system("Node.js backend dependencies checked/installed.")


    # 4. Install/Check Dependencies (Node.js Frontend)
    print_system("--- Installing/Checking Node.js Frontend Dependencies ---")
    frontend_pkg_path = os.path.join(FRONTEND_DIR, "package.json")
    if not os.path.exists(frontend_pkg_path):
        print_error(f"Frontend package.json not found at {frontend_pkg_path}. Cannot install dependencies.")
        sys.exit(1)
    if not run_command([NPM_CMD, "install"], FRONTEND_DIR, "NPM-FRONTEND"):
        print_error("Failed to install Node.js frontend dependencies. Check npm logs. Exiting.")
        sys.exit(1)
    print_system("Node.js frontend dependencies checked/installed.")


    # 5. Start all services concurrently
    print_system("--- Starting All Services ---")
    service_threads = [] # Renamed from 'threads' to avoid conflict with 'threading' module

    # Start Engine Main first as other services might depend on it
    service_threads.append(start_service_thread(start_engine_main, "EngineMainThread"))
    
    print_system("Waiting for Engine Main (Hypercorn) to initialize (up to 10 seconds)...")
    time.sleep(2) # Initial brief pause
    engine_ready = False
    for _ in range(8): # Check for ~8 more seconds
        with process_lock:
            for proc, name in running_processes:
                if name == "ENGINE" and proc.poll() is None: # Check if process is running
                    engine_ready = True
                    break
        if engine_ready:
            print_system("Engine Main appears to be running. Proceeding with other services.")
            break
        # Check if it exited prematurely
        engine_exited = False
        with process_lock:
            for proc, name in running_processes:
                 if name == "ENGINE" and proc.poll() is not None:
                    print_error(f"Engine Main (Hypercorn) seems to have exited prematurely with code {proc.poll()}.")
                    engine_exited = True
                    break
        if engine_exited: break
        time.sleep(1)
    
    if not engine_ready and not engine_exited: # If loop finished without engine ready and not exited
         with process_lock: # Final check
            for proc, name in running_processes:
                if name == "ENGINE" and proc.poll() is None: engine_ready = True
    
    if not engine_ready:
        print_error("Engine Main (Hypercorn) failed to start or stay running. Check logs above. Exiting.")
        # cleanup_processes() will be called by atexit
        sys.exit(1)
    
    # Start other services
    service_threads.append(start_service_thread(start_backend_service, "BackendServiceThread"))
    time.sleep(2) # Stagger starts slightly
    service_threads.append(start_service_thread(start_frontend, "FrontendThread"))

    print_system("All services are being started. Launcher will monitor them. Press Ctrl+C to shut down.")


    # 6. Keep the main thread alive and monitor processes
    try:
        while True:
            active_managed_process_found = False
            all_processes_ok = True
            
            with process_lock:
                current_procs_snapshot = list(running_processes) # Take snapshot for safe iteration
                procs_to_remove_from_monitoring = []

            if not current_procs_snapshot and service_threads: # No processes but threads were started
                print_warning("No managed processes are running, but service threads exist. This might indicate startup failures not caught earlier.")
                # This state could mean all subprocesses launched by threads failed very quickly.
                # The start_service_process itself calls sys.exit(1) on immediate failure,
                # but if the thread starts and then the Popen inside fails in a way not caught, this might be a fallback.
                # However, start_service_process should handle FileNotFoundError and general Exception.
                # This is more of a sanity check.
                all_processes_ok = False # Treat as a failure condition

            for proc, name in current_procs_snapshot:
                if proc.poll() is None: # Process is still running
                    active_managed_process_found = True
                else: # Process has exited
                    print_error(f"Service '{name}' has exited unexpectedly with return code {proc.poll()}.")
                    procs_to_remove_from_monitoring.append((proc, name))
                    all_processes_ok = False # Mark that at least one process failed

            if procs_to_remove_from_monitoring:
                with process_lock:
                    # Remove only those that were confirmed exited from the main list
                    for item_to_remove in procs_to_remove_from_monitoring:
                        if item_to_remove in running_processes:
                            running_processes.remove(item_to_remove)
            
            if not all_processes_ok:
                print_error("One or more services terminated. Initiating shutdown of remaining services.")
                break # Exit the monitoring loop, cleanup will be handled by atexit

            if not active_managed_process_found and service_threads:
                # This means all processes that were started have now exited (and presumably not due to error,
                # otherwise all_processes_ok would be False).
                # This case is unlikely if services are meant to run indefinitely.
                print_system("All managed services have finished or exited. Launcher will now shut down.")
                break

            # If no service threads were ever started (e.g. script does only setup), this loop won't run infinitely.
            if not service_threads:
                print_system("No services were configured to start. Setup complete. Exiting launcher.")
                break

            time.sleep(5) # Interval for checking process statuses

    except KeyboardInterrupt:
        print_system("\nKeyboardInterrupt received by main thread. Shutting down...")
        # atexit handler (cleanup_processes) will take care of stopping subprocesses.
    finally:
        # This block will run after the loop breaks naturally or due to an exception (like KeyboardInterrupt)
        print_system("Launcher main loop finished. Ensuring cleanup...")
        # cleanup_processes() is registered with atexit, so it will run automatically on normal exit or unhandled exception.
        # If we want to force it here, we can, but it might run twice.
        # It's generally fine as atexit is robust.