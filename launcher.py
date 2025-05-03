#!/usr/bin/env python3
import os
import sys
import subprocess
import threading
import time
import signal
import atexit
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
HYPERCORN_EXECUTABLE = os.path.join(VENV_BIN_DIR, "hypercorn.exe" if IS_WINDOWS else "hypercorn") # <-- Added
NPM_CMD = 'npm.cmd' if IS_WINDOWS else 'npm' # Use npm command based on OS

# Global list to keep track of running subprocesses
running_processes = []
process_lock = threading.Lock() # To safely append/remove

# --- Helper Functions ---
def print_colored(prefix, message, color):
    """Prints a message with a timestamp, colored prefix, and resets color."""
    now = datetime.now().strftime('%H:%M:%S.%f')[:-3] # HH:MM:SS.ms
    print(f"{color}[{now} | {prefix.ljust(10)}] {COLOR_RESET}{message.strip()}")

def print_system(message):
    print_colored("SYSTEM", message, COLOR_SYSTEM)

def print_error(message):
    print_colored("ERROR", message, COLOR_ERROR)

def run_command(command, cwd, name, color, check=True, capture_output=False):
    """Runs a command, optionally checks for errors, and can stream output."""
    print_system(f"Running command in '{os.path.basename(cwd)}': {' '.join(command)}")
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
             # Use shell=True cautiously, but often needed for npm/node on Windows
            shell=(IS_WINDOWS and command[0] in [NPM_CMD, 'node'])
        )

        stdout_thread = threading.Thread(target=stream_output, args=(process.stdout, name, color), daemon=True)
        stdout_thread.start()
        stderr_thread = threading.Thread(target=stream_output, args=(process.stderr, f"{name}-ERR", COLOR_ERROR), daemon=True)
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
        if command[0] in [NPM_CMD, 'node']:
            print_error("Ensure Node.js and npm are installed: https://nodejs.org/")
        elif command[0] == HYPERCORN_EXECUTABLE:
             print_error(f"Ensure '{os.path.basename(HYPERCORN_EXECUTABLE)}' is installed in the venv (check requirements.txt).")
        return False
    except subprocess.CalledProcessError as e:
        print_error(f"Command failed in '{os.path.basename(cwd)}' with exit code {e.returncode}: {' '.join(command)}")
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
        print_error(f"Error reading output stream for {name}: {e}")
    finally:
        if pipe:
            pipe.close()

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
        # Use slice copy for safe iteration if removing items (though we clear at end)
        procs_to_terminate = list(running_processes)
        running_processes.clear() # Clear original list

    for proc, name in reversed(procs_to_terminate): # Iterate backwards
        if proc.poll() is None:
            print_system(f"Terminating {name} (PID: {proc.pid})...")
            try:
                # Send SIGTERM (terminate) first
                proc.terminate()
                try:
                    proc.wait(timeout=3) # Wait 3 seconds
                    print_system(f"{name} terminated gracefully.")
                except subprocess.TimeoutExpired:
                    print_system(f"{name} did not terminate gracefully, killing...")
                    proc.kill() # Force kill (SIGKILL)
                    proc.wait() # Wait for kill to complete
                    print_system(f"{name} killed.")
            except Exception as e:
                print_error(f"Error terminating {name}: {e}")
        else:
            print_system(f"{name} already exited (return code: {proc.poll()}).")

# Register cleanup function to run on script exit
atexit.register(cleanup_processes)

# Handle Ctrl+C (SIGINT) for graceful shutdown
def signal_handler(sig, frame):
    print_system("\nCtrl+C received. Initiating shutdown...")
    # cleanup_processes() will be called by atexit
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)


# --- Service Start Functions ---
def start_engine_main():
    """Starts the Python Engine Main service using Hypercorn."""
    name = "ENGINE"
    color = COLOR_ENGINE
    # --- Command Changed Here ---
    # We run hypercorn directly from the venv's bin directory
    command = [
        HYPERCORN_EXECUTABLE,
        "app:app",                   # Load the 'app' object from 'app.py'
        "--bind", "127.0.0.1:11434", # Bind to localhost port 11434
        "--workers", "1"             # Number of worker processes
        # Add other hypercorn options if needed, e.g., --log-level debug
    ]
    # Ensure hypercorn is installed via requirements.txt!
    print_system(f"Launching Engine Main via Hypercorn: {' '.join(command)} in {ENGINE_MAIN_DIR}")
    try:
        process = subprocess.Popen(
            command,
            cwd=ENGINE_MAIN_DIR, # Run hypercorn from the engineMain directory
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, # Capture stderr separately
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1
            # shell=False should be okay for hypercorn even on Windows if path is correct
        )
        with process_lock:
            running_processes.append((process, name))

        # Start streaming threads
        stdout_thread = threading.Thread(target=stream_output, args=(process.stdout, name, color), daemon=True)
        # Stream stderr using a different color/prefix
        stderr_thread = threading.Thread(target=stream_output, args=(process.stderr, f"{name}-ERR", COLOR_ERROR), daemon=True)
        stdout_thread.start()
        stderr_thread.start()

        # We don't wait here, the process runs in the background.
        # The main loop will monitor it.

    except FileNotFoundError:
         print_error(f"Command failed: '{HYPERCORN_EXECUTABLE}' not found.")
         print_error("Ensure 'hypercorn' is listed in systemCore/engineMain/requirements.txt and the venv is active.")
    except Exception as e:
        print_error(f"Failed to start {name}: {e}")

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
            shell=IS_WINDOWS # May need shell=True for node on Windows
        )
        with process_lock:
            running_processes.append((process, name))
        stdout_thread = threading.Thread(target=stream_output, args=(process.stdout, name, color), daemon=True)
        stderr_thread = threading.Thread(target=stream_output, args=(process.stderr, f"{name}-ERR", COLOR_ERROR), daemon=True)
        stdout_thread.start()
        stderr_thread.start()
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
            shell=IS_WINDOWS # npm often requires shell=True on Windows
        )
        with process_lock:
            running_processes.append((process, name))
        stdout_thread = threading.Thread(target=stream_output, args=(process.stdout, name, color), daemon=True)
        stderr_thread = threading.Thread(target=stream_output, args=(process.stderr, f"{name}-ERR", COLOR_ERROR), daemon=True)
        stdout_thread.start()
        stderr_thread.start()
    except Exception as e:
        print_error(f"Failed to start {name}: {e}")


# --- Main Execution Logic ---
if __name__ == "__main__":
    print_system("--- Project Zephyrine Launcher ---")

    # 1. Check/Create/Relaunch in Virtual Environment
    # (Keep existing venv check and relaunch logic)
    is_in_target_venv = (os.getenv('VIRTUAL_ENV') == VENV_DIR) or \
                         (hasattr(sys, 'prefix') and sys.prefix == VENV_DIR) or \
                         (hasattr(sys, 'real_prefix') and sys.real_prefix == VENV_DIR) # More checks

    if not is_in_target_venv:
        print_system(f"Not running in the target virtual environment: {VENV_DIR}")
        if not os.path.exists(VENV_DIR):
            print_system(f"Virtual environment not found. Creating it at: {VENV_DIR}")
            try:
                python_cmd_for_venv = sys.executable
                print_system(f"Using Python interpreter for venv creation: {python_cmd_for_venv}")
                subprocess.run([python_cmd_for_venv, "-m", "venv", VENV_DIR], check=True)
                print_system("Virtual environment created successfully.")
            except subprocess.CalledProcessError as e:
                print_error(f"Failed to create virtual environment: {e}")
                sys.exit(1)
            except Exception as e:
                print_error(f"An unexpected error occurred during venv creation: {e}")
                sys.exit(1)

        print_system(f"Relaunching script using Python from virtual environment: {PYTHON_EXECUTABLE}")
        if not os.path.exists(PYTHON_EXECUTABLE):
             print_error(f"Python executable not found in venv: {PYTHON_EXECUTABLE}")
             print_error("Virtual environment might be corrupted or incomplete.")
             sys.exit(1)
        try:
            # Use execv to replace the current process with the one in the venv
            os.execv(PYTHON_EXECUTABLE, [PYTHON_EXECUTABLE] + sys.argv)
        except Exception as e:
            print_error(f"Failed to relaunch script in virtual environment: {e}")
            sys.exit(1)

    # --- Running inside the correct venv ---
    print_system(f"Running inside the correct virtual environment: {VENV_DIR}")

    # 2. Install/Check Dependencies (Python)
    print_system("--- Installing/Checking Python Dependencies (Engine) ---")
    req_path = os.path.join(ENGINE_MAIN_DIR, "requirements.txt")
    if not os.path.exists(req_path):
        print_error(f"requirements.txt not found at {req_path}")
        sys.exit(1)
    if not run_command([PIP_EXECUTABLE, "install", "-r", req_path], ENGINE_MAIN_DIR, "PIP", COLOR_SYSTEM):
         print_error("Failed to install Python dependencies. Exiting.")
         sys.exit(1)
    print_system("Python dependencies checked/installed.")

    # 3. Install/Check Dependencies (Node.js Backend Service)
    print_system("--- Installing/Checking Node Backend Dependencies ---")
    pkg_path = os.path.join(BACKEND_SERVICE_DIR, "package.json")
    if not os.path.exists(pkg_path):
         print_error(f"package.json not found at {pkg_path}")
         sys.exit(1)
    if not run_command([NPM_CMD, "install"], BACKEND_SERVICE_DIR, "NPM-BE", COLOR_SYSTEM):
         print_error("Failed to install Node backend dependencies. Exiting.")
         sys.exit(1)
    print_system("Node backend dependencies checked/installed.")

    # 4. Install/Check Dependencies (Node.js Frontend)
    print_system("--- Installing/Checking Node Frontend Dependencies ---")
    pkg_path_fe = os.path.join(FRONTEND_DIR, "package.json")
    if not os.path.exists(pkg_path_fe):
         print_error(f"package.json not found at {pkg_path_fe}")
         sys.exit(1)
    if not run_command([NPM_CMD, "install"], FRONTEND_DIR, "NPM-FE", COLOR_SYSTEM):
         print_error("Failed to install Node frontend dependencies. Exiting.")
         sys.exit(1)
    print_system("Node frontend dependencies checked/installed.")

    # 5. Start all services concurrently
    print_system("--- Starting Services ---")
    threads = []
    # Start Engine first if others depend on its API
    threads.append(start_service_thread(start_engine_main, "EngineMainThread"))
    time.sleep(2) # Give engine a moment to bind port
    threads.append(start_service_thread(start_backend_service, "BackendServiceThread"))
    time.sleep(1)
    threads.append(start_service_thread(start_frontend, "FrontendThread"))

    print_system("All services are starting up. Press Ctrl+C to shut down.")

    # 6. Keep the main thread alive and monitor processes
    try:
        while True:
            active_process_found = False
            with process_lock: # Lock for safe checking/modification
                # Iterate safely if removing items (though we don't remove here, good practice)
                current_running = list(running_processes)

            for i in range(len(current_running) - 1, -1, -1):
                 proc, name = current_running[i]
                 if proc.poll() is None: # Process is still running
                     active_process_found = True
                 else: # Process has exited
                     print_error(f"Service '{name}' exited unexpectedly with code {proc.poll()}.")
                     # Remove from the main list so cleanup doesn't try to kill it again
                     with process_lock:
                        # Find the specific tuple to remove
                        running_processes[:] = [p for p in running_processes if p[0].pid != proc.pid]


            if not active_process_found and threads:
                print_system("All managed service processes seem to have exited.")
                break # Exit the main loop

            time.sleep(5) # Check process status periodically

    except KeyboardInterrupt:
        # Handled by signal handler, just pass here to allow cleanup
        pass
    finally:
        # Cleanup is handled by atexit registration
        print_system("Launcher shutting down or finished.")