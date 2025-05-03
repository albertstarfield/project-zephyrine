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
    colorama.init() # Initialize colorama for Windows compatibility
    # Define colors using colorama
    COLOR_RESET = colorama.Style.RESET_ALL
    COLOR_ENGINE = colorama.Fore.CYAN + colorama.Style.BRIGHT
    COLOR_BACKEND = colorama.Fore.MAGENTA + colorama.Style.BRIGHT
    COLOR_FRONTEND = colorama.Fore.GREEN + colorama.Style.BRIGHT
    COLOR_SYSTEM = colorama.Fore.YELLOW + colorama.Style.BRIGHT
    COLOR_ERROR = colorama.Fore.RED + colorama.Style.BRIGHT
except ImportError:
    print("Warning: colorama not found. Logs will not be colored.")
    print("Install it using: pip install colorama")
    # Define colors as empty strings if colorama is not available
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

PYTHON_EXECUTABLE = os.path.join(VENV_DIR, "bin", "python") if os.name != 'nt' else os.path.join(VENV_DIR, "Scripts", "python.exe")
PIP_EXECUTABLE = os.path.join(VENV_DIR, "bin", "pip") if os.name != 'nt' else os.path.join(VENV_DIR, "Scripts", "pip.exe")

# Global list to keep track of running subprocesses
running_processes = []
process_lock = threading.Lock() # To safely append to the list

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
        # Use Popen for potentially long-running commands or if streaming needed
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace', # Handle potential decoding errors
            bufsize=1, # Line-buffered
            # Use shell=True cautiously, needed for things like 'npm' on Windows sometimes
            shell=(os.name == 'nt' and command[0] in ['npm', 'node'])
        )

        # Stream stdout
        stdout_thread = threading.Thread(target=stream_output, args=(process.stdout, name, color), daemon=True)
        stdout_thread.start()

        # Stream stderr (using error color)
        stderr_thread = threading.Thread(target=stream_output, args=(process.stderr, f"{name}-ERR", COLOR_ERROR), daemon=True)
        stderr_thread.start()

        process.wait() # Wait for the command to complete
        stdout_thread.join()
        stderr_thread.join()

        if check and process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command)

        print_system(f"Command finished in '{os.path.basename(cwd)}': {' '.join(command)}")
        return True # Indicate success

    except FileNotFoundError:
        print_error(f"Command not found: {command[0]}. Is it installed and in PATH?")
        if command[0] == 'npm':
            print_error("Ensure Node.js and npm are installed: https://nodejs.org/")
        return False # Indicate failure
    except subprocess.CalledProcessError as e:
        print_error(f"Command failed in '{os.path.basename(cwd)}' with exit code {e.returncode}: {' '.join(command)}")
        # Error output was already streamed by stderr_thread
        return False # Indicate failure
    except Exception as e:
        print_error(f"An unexpected error occurred while running command in '{os.path.basename(cwd)}': {e}")
        return False # Indicate failure

def stream_output(pipe, name, color):
    """Reads lines from a subprocess pipe and prints them with prefix and color."""
    try:
        for line in iter(pipe.readline, ''):
            if line:
                print_colored(name, line, color)
    except Exception as e:
        # Handle potential errors during reading (e.g., if process closes abruptly)
        print_error(f"Error reading output stream for {name}: {e}")
    finally:
        if pipe:
            pipe.close()

def start_service_thread(target_func, name):
    """Starts a service in a daemon thread."""
    print_system(f"Starting service: {name}")
    thread = threading.Thread(target=target_func, name=name, daemon=True)
    thread.start()
    return thread

def cleanup_processes():
    """Attempts to terminate all tracked subprocesses."""
    print_system("\nShutting down services...")
    with process_lock:
        # Iterate backwards to avoid issues if removal happens (though not strictly needed here)
        for proc, name in reversed(running_processes):
            if proc.poll() is None: # Check if process is still running
                print_system(f"Terminating {name} (PID: {proc.pid})...")
                try:
                    # Try graceful termination first
                    proc.terminate()
                    try:
                        # Wait a short time for graceful shutdown
                        proc.wait(timeout=2)
                        print_system(f"{name} terminated gracefully.")
                    except subprocess.TimeoutExpired:
                        print_system(f"{name} did not terminate gracefully, killing...")
                        proc.kill() # Force kill
                        proc.wait() # Wait for kill
                        print_system(f"{name} killed.")
                except Exception as e:
                    print_error(f"Error terminating {name}: {e}")
            else:
                print_system(f"{name} already exited (return code: {proc.poll()}).")
        running_processes.clear() # Clear the list after attempting termination

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
    """Starts the Python Engine Main service."""
    name = "ENGINE"
    color = COLOR_ENGINE
    command = [PYTHON_EXECUTABLE, "app.py"] # Use venv python
    print_system(f"Launching Engine Main: {' '.join(command)} in {ENGINE_MAIN_DIR}")
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

        # Start streaming threads
        stdout_thread = threading.Thread(target=stream_output, args=(process.stdout, name, color), daemon=True)
        stderr_thread = threading.Thread(target=stream_output, args=(process.stderr, f"{name}-ERR", COLOR_ERROR), daemon=True)
        stdout_thread.start()
        stderr_thread.start()

        # Wait for process to complete (optional, services usually run indefinitely)
        # process.wait()
        # stdout_thread.join()
        # stderr_thread.join()
        # print_system(f"{name} process finished.") # Usually won't print if service runs forever

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
            shell=(os.name == 'nt') # May need shell=True for node on Windows
        )
        with process_lock:
            running_processes.append((process, name))

        # Start streaming threads
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
    # Use 'npm.cmd' on Windows, 'npm' otherwise
    npm_cmd = 'npm.cmd' if os.name == 'nt' else 'npm'
    command = [npm_cmd, "run", "dev"]
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
            shell=(os.name == 'nt') # npm often requires shell=True on Windows
        )
        with process_lock:
            running_processes.append((process, name))

        # Start streaming threads
        stdout_thread = threading.Thread(target=stream_output, args=(process.stdout, name, color), daemon=True)
        stderr_thread = threading.Thread(target=stream_output, args=(process.stderr, f"{name}-ERR", COLOR_ERROR), daemon=True)
        stdout_thread.start()
        stderr_thread.start()

    except Exception as e:
        print_error(f"Failed to start {name}: {e}")


# --- Main Execution Logic ---
if __name__ == "__main__":
    print_system("--- Project Zephyrine Launcher ---")

    # 1. Check if running inside the desired virtual environment
    is_in_target_venv = (os.getenv('VIRTUAL_ENV') == VENV_DIR) or (hasattr(sys, 'prefix') and sys.prefix == VENV_DIR)

    if not is_in_target_venv:
        print_system(f"Not running in the target virtual environment: {VENV_DIR}")

        # 2. Check if venv directory exists
        if not os.path.exists(VENV_DIR):
            print_system(f"Virtual environment not found. Creating it at: {VENV_DIR}")
            try:
                # Find a suitable python3 executable
                python_cmd = sys.executable # Use the python that's running this script
                if 'venv' not in python_cmd: # Prefer system python if possible for creating venv
                   try:
                       result = subprocess.run(['which', 'python3'], capture_output=True, text=True, check=True)
                       python_cmd = result.stdout.strip()
                   except (FileNotFoundError, subprocess.CalledProcessError):
                       print_warning("Could not find 'python3' via 'which', using current executable.")

                print_system(f"Using Python interpreter: {python_cmd}")
                subprocess.run([python_cmd, "-m", "venv", VENV_DIR], check=True)
                print_system("Virtual environment created successfully.")
            except subprocess.CalledProcessError as e:
                print_error(f"Failed to create virtual environment: {e}")
                sys.exit(1)
            except Exception as e:
                print_error(f"An unexpected error occurred during venv creation: {e}")
                sys.exit(1)

        # 3. Relaunch the script using the venv's Python interpreter
        print_system(f"Relaunching script using Python from virtual environment: {PYTHON_EXECUTABLE}")
        if not os.path.exists(PYTHON_EXECUTABLE):
             print_error(f"Python executable not found in venv: {PYTHON_EXECUTABLE}")
             print_error("Virtual environment might be corrupted or incomplete.")
             sys.exit(1)
        try:
            # Pass original arguments, replacing the current process
            os.execv(PYTHON_EXECUTABLE, [PYTHON_EXECUTABLE] + sys.argv)
        except Exception as e:
            print_error(f"Failed to relaunch script in virtual environment: {e}")
            sys.exit(1)

    else:
        # 4. Now running inside the correct virtual environment
        print_system(f"Running inside the correct virtual environment: {VENV_DIR}")

        # 5. Install dependencies (Python)
        print_system("--- Installing/Checking Python Dependencies ---")
        req_path = os.path.join(ENGINE_MAIN_DIR, "requirements.txt")
        if not os.path.exists(req_path):
            print_error(f"requirements.txt not found at {req_path}")
            sys.exit(1)
        if not run_command([PIP_EXECUTABLE, "install", "-r", req_path], ENGINE_MAIN_DIR, "PIP", COLOR_SYSTEM):
             print_error("Failed to install Python dependencies. Exiting.")
             sys.exit(1)
        print_system("Python dependencies checked/installed.")

        # 6. Install dependencies (Node.js Backend Service)
        print_system("--- Installing/Checking Node Backend Dependencies ---")
        pkg_path = os.path.join(BACKEND_SERVICE_DIR, "package.json")
        if not os.path.exists(pkg_path):
             print_error(f"package.json not found at {pkg_path}")
             sys.exit(1)
        # Use 'npm.cmd' on Windows, 'npm' otherwise
        npm_cmd = 'npm.cmd' if os.name == 'nt' else 'npm'
        if not run_command([npm_cmd, "install"], BACKEND_SERVICE_DIR, "NPM-BE", COLOR_SYSTEM):
             print_error("Failed to install Node backend dependencies. Exiting.")
             sys.exit(1)
        print_system("Node backend dependencies checked/installed.")


        # 7. Install dependencies (Node.js Frontend)
        print_system("--- Installing/Checking Node Frontend Dependencies ---")
        pkg_path_fe = os.path.join(FRONTEND_DIR, "package.json")
        if not os.path.exists(pkg_path_fe):
             print_error(f"package.json not found at {pkg_path_fe}")
             sys.exit(1)
        # Use 'npm.cmd' on Windows, 'npm' otherwise
        npm_cmd = 'npm.cmd' if os.name == 'nt' else 'npm'
        if not run_command([npm_cmd, "install"], FRONTEND_DIR, "NPM-FE", COLOR_SYSTEM):
             print_error("Failed to install Node frontend dependencies. Exiting.")
             sys.exit(1)
        print_system("Node frontend dependencies checked/installed.")

        # 8. Start all services concurrently
        print_system("--- Starting Services ---")
        threads = []
        threads.append(start_service_thread(start_engine_main, "EngineMainThread"))
        time.sleep(1) # Small delay between starts if needed
        threads.append(start_service_thread(start_backend_service, "BackendServiceThread"))
        time.sleep(1)
        threads.append(start_service_thread(start_frontend, "FrontendThread"))

        print_system("All services are starting up. Press Ctrl+C to shut down.")

        # 9. Keep the main thread alive (or wait for threads, though they are daemons)
        # Services run indefinitely, so just wait for interruption
        try:
            while True:
                # Check if any service process has exited unexpectedly
                with process_lock:
                    for i in range(len(running_processes) - 1, -1, -1):
                         proc, name = running_processes[i]
                         if proc.poll() is not None: # Process has exited
                             print_error(f"Service '{name}' exited unexpectedly with code {proc.poll()}.")
                             # Optionally decide if this should trigger a full shutdown
                             # For now, just report it. The thread streaming its output will stop.
                             # Remove it from the list to avoid repeated checks/termination attempts
                             del running_processes[i]

                if not running_processes and threads: # If all processes are gone but threads were started
                    print_system("All service processes seem to have exited.")
                    break # Exit the main loop

                time.sleep(5) # Check status periodically

        except KeyboardInterrupt:
            # Handled by signal handler, just pass here
            pass
        finally:
            # Cleanup is handled by atexit
            print_system("Launcher finished.")