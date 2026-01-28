# stella_icarus_utils.py

import os
import re
import time
import importlib.util
import threading
import sys
import subprocess
import json
import random
import math
import queue
from typing import List, Tuple, Callable, Optional, Any, Dict
import datetime
from loguru import logger

# --- Configuration Import with Fallbacks ---
try:
    from CortexConfiguration import (
        ENABLE_STELLA_ICARUS_HOOKS, STELLA_ICARUS_HOOK_DIR, STELLA_ICARUS_CACHE_DIR,
        ENABLE_STELLA_ICARUS_DAEMON, STELLA_ICARUS_ADA_DIR, ALR_DEFAULT_EXECUTABLE_NAME, STELLA_ICARUS_PICORESPONSEHOOKCACHE_HOOK_DIR,
        ADA_DAEMON_RETRY_DELAY_SECONDS # NEW: Import retry delay
    )
except ImportError:
    logger.critical("StellaIcarusUtils: Failed to import configuration. All features will be disabled.")
    ENABLE_STELLA_ICARUS_HOOKS = False
    STELLA_ICARUS_HOOK_DIR = "./StellaIcarus"
    STELLA_ICARUS_CACHE_DIR = "./StellaIcarus_Cache"
    ENABLE_STELLA_ICARUS_DAEMON = False
    STELLA_ICARUS_ADA_DIR = "./StellaIcarus_Ada"
    ALR_DEFAULT_EXECUTABLE_NAME = "stella_greeting"
    ADA_DAEMON_RETRY_DELAY_SECONDS = 30 # NEW: Fallback value

class StellaIcarusHookManager:
    def __init__(self):
        """
        Initializes the Hook Manager.
        """
        # 1. Initialize instance variables
        self.hooks: List[Tuple[re.Pattern, Callable[[re.Match, str, str], Optional[str]], str]] = []
        self.hook_load_errors: List[str] = []
        self.is_enabled = ENABLE_STELLA_ICARUS_HOOKS

        # 2. Early exit if the feature is disabled
        if not self.is_enabled:
            logger.info("StellaIcarusHookManager: Hooks are disabled by configuration.")
            return

        # 3. Initial Load
        self.load_hooks()

    def reload_hooks(self):
        """
        Clears existing hooks and re-scans directories to hot-reload changes.
        """
        logger.info("StellaIcarusHookManager: 🔄 Triggering Hot Reload...")
        self.hooks.clear()
        self.hook_load_errors.clear()
        self.load_hooks()
        logger.success(f"StellaIcarusHookManager: Hot Reload Complete. Active Hooks: {len(self.hooks)}")

    def load_hooks(self):
        """
        Discovers, validates, and dynamically loads all Python-based hooks
        from the configured directories.
        """
        # --- 1. Load from Main Directory ---
        if not os.path.isdir(STELLA_ICARUS_HOOK_DIR):
            logger.error(f"StellaIcarusHookManager: Hook directory '{STELLA_ICARUS_HOOK_DIR}' not found.")
            self.hook_load_errors.append(f"Hook directory not found: {STELLA_ICARUS_HOOK_DIR}")
        else:
            logger.info(f"StellaIcarusHookManager: Loading hooks from '{STELLA_ICARUS_HOOK_DIR}'...")
            self._scan_and_load_directory(STELLA_ICARUS_HOOK_DIR, "stella_hook_")

        # --- 2. Load from PicoResponse Cache Directory ---
        # Use config or fallback
        pico_cache_dir = getattr(sys.modules[__name__], 'STELLA_ICARUS_PICORESPONSEHOOKCACHE_HOOK_DIR',
                                 os.path.join(STELLA_ICARUS_HOOK_DIR, "picoResponseHookCache"))

        if os.path.isdir(pico_cache_dir):
            logger.info(f"StellaIcarusHookManager: Scanning pico cache dir '{pico_cache_dir}'...")
            self._scan_and_load_directory(pico_cache_dir, "stella_hook_pico_")

        # Summary
        if not self.hooks and not self.hook_load_errors:
            logger.warning("StellaIcarusHookManager: No hooks found in any directory.")

    def _scan_and_load_directory(self, directory: str, module_prefix: str):
        """Helper to scan a specific directory and load valid hooks."""
        for filename in os.listdir(directory):
            if filename.endswith(".py") and not filename.startswith("_"):
                module_name = f"{module_prefix}{filename[:-3]}"
                file_path = os.path.join(directory, filename)

                try:
                    # Dynamically load the python file
                    spec = importlib.util.spec_from_file_location(module_name, file_path)
                    if spec is None or spec.loader is None:
                        continue

                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    # Validation
                    pattern_attr = getattr(module, "PATTERN", None)
                    handler_func = getattr(module, "handler", None)
                    is_jit_reliable = getattr(module, "IS_JIT_COMPILED", False)

                    if pattern_attr is None or handler_func is None:
                        logger.warning(f"  Skipping '{filename}': Missing PATTERN or handler.")
                        continue

                    if not callable(handler_func):
                        logger.warning(f"  Skipping '{filename}': Handler not callable.")
                        continue

                    # Compile Regex
                    compiled_pattern: re.Pattern
                    if isinstance(pattern_attr, str):
                        compiled_pattern = re.compile(pattern_attr, re.IGNORECASE)
                    elif isinstance(pattern_attr, re.Pattern):
                        compiled_pattern = pattern_attr
                    else:
                        continue

                    # Register
                    self.hooks.append((compiled_pattern, handler_func, module_name))

                    log_msg = f"  Loaded Hook: '{module_name}'"
                    if not is_jit_reliable:
                        log_msg += " (Interpreted)"
                    logger.info(log_msg)

                except Exception as e:
                    logger.error(f"  Error loading hook '{filename}': {e}")
                    self.hook_load_errors.append(f"Error in {filename}: {e}")

    def check_and_execute(self, user_input: str, session_id: str) -> Optional[str]:
        if not self.is_enabled or not self.hooks:
            return None

        for pattern, handler, module_name in self.hooks:
            match = pattern.match(user_input)
            if match:
                hook_start_time = time.perf_counter_ns()
                try:
                    response = handler(match, user_input, session_id)
                    hook_end_time = time.perf_counter_ns()
                    duration_us = (hook_end_time - hook_start_time) / 1000.0
                    logger.debug(f"StellaIcarusHook '{module_name}' matched. Handler duration: {duration_us:.3f} µs.")

                    if response is not None and isinstance(response, str):
                        return response
                except Exception as e:
                    logger.error(f"StellaIcarusHook '{module_name}' execution error: {e}")
        return None

    def try_hooks(self, user_input: str, session_id: str) -> Optional[str]:
        if not self.is_enabled or not self.hooks:
            return None

        for pattern, handler, module_name in self.hooks:
            match = pattern.match(user_input)
            if match:
                hook_start_time = time.perf_counter_ns()
                try:
                    response = handler(match, user_input, session_id)
                    hook_end_time = time.perf_counter_ns()
                    duration_us = (hook_end_time - hook_start_time) / 1000.0
                    logger.debug(f"StellaIcarusHook '{module_name}' matched. Handler duration: {duration_us:.3f} µs.")

                    if response is not None and isinstance(response, str):
                        return response
                except Exception as e:
                    logger.error(f"StellaIcarusHook '{module_name}' execution error: {e}")
        return None


# --- NEW: Stella Icarus Ada Daemon Manager (Refactored) ---
class StellaIcarusAdaDaemonManager:
    """Discovers, builds, runs, and manages multiple Ada daemon projects."""

    def __init__(self):
        self.is_enabled = ENABLE_STELLA_ICARUS_DAEMON
        self.ada_projects: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self.data_queue = queue.Queue(maxsize=1000)  # For aggregating data from all daemons

    def _discover_ada_projects(self):
        """Scans the STELLA_ICARUS_ADA_DIR for valid Ada projects."""
        if not self.is_enabled or not os.path.isdir(STELLA_ICARUS_ADA_DIR):
            return

        # Avoid rediscovering if already populated (prevents duplicates)
        if self.ada_projects:
            return

        logger.info(f"Discovering Ada projects in '{STELLA_ICARUS_ADA_DIR}'...")
        for item in os.listdir(STELLA_ICARUS_ADA_DIR):
            project_path = os.path.join(STELLA_ICARUS_ADA_DIR, item)
            
            if os.path.isdir(project_path):
                has_alire_toml = os.path.exists(os.path.join(project_path, "alire.toml"))
                gpr_files = [f for f in os.listdir(project_path) if f.endswith(".gpr")]

                if has_alire_toml or gpr_files:
                    project_name = item
                    
                    # [FIX 2] DYNAMIC NAMING
                    # Use the directory name as the binary name.
                    # 'avionics_daemon' folder -> 'avionics_daemon' binary
                    executable_name = project_name
                    
                    # Handle Windows extension
                    if os.name == 'nt':
                        executable_name += ".exe"

                    self.ada_projects.append({
                        "name": project_name,
                        "path": project_path,
                        "executable_name": executable_name, # <--- CORRECTED
                        "process": None,
                        "thread": None,
                        "stop_event": threading.Event()
                    })
                    logger.info(f"  Discovered Ada project: '{project_name}' -> expecting binary '{executable_name}'")

    def build_all(self):
        """Builds all discovered Ada projects using 'alr build' with verbose error logging."""
        if not self.is_enabled: return

        logger.info("--- Building all discovered StellaIcarus Ada projects... ---")
        for project in self.ada_projects:
            logger.info(f"Building '{project['name']}' in '{project['path']}'...")
            try:
                # Capture BOTH stdout and stderr to catch all compiler messages
                process = subprocess.run(
                    ["alr", "build"],
                    cwd=project["path"],
                    capture_output=True, text=True, check=False, timeout=300
                )
                
                if process.returncode == 0:
                    logger.success(f"  ✅ Successfully built '{project['name']}'.")
                else:
                    logger.error(f"  ❌ Failed to build '{project['name']}'. RC: {process.returncode}")
                    
                    # --- IMPROVED ERROR LOGGING ---
                    # Combine streams to preserve order of error messages
                    output_stream = (process.stdout or "") + "\n" + (process.stderr or "")
                    
                    if not output_stream.strip():
                        logger.error("     [NO OUTPUT CAPTURED] - Check Alire installation.")
                    
                    for line in output_stream.splitlines():
                        line = line.strip()
                        if not line: continue
                        
                        # Make errors pop out in red (Critical)
                        if "error:" in line.lower() or "exception" in line.lower():
                            logger.critical(f"     🔥 {line}")
                        elif "warning:" in line.lower():
                            logger.warning(f"     ⚠️ {line}")
                        else:
                            # Log normal build info as debug/info so it doesn't clutter unless needed
                            logger.info(f"     [BUILD] {line}")

            except FileNotFoundError:
                logger.error("  ❌ Build failed: 'alr' command not found. Is Alire installed and in PATH?")
                break
            except subprocess.TimeoutExpired as e:
                logger.error(f"  ❌ Build timed out for '{project['name']}'.")
                # Try to print what happened before it froze
                if e.stdout: logger.error(f"Last Output:\n{e.stdout.decode()}")
                if e.stderr: logger.error(f"Last Errors:\n{e.stderr.decode()}")
            except Exception as e:
                logger.error(f"  ❌ Unexpected error building '{project['name']}': {e}")
        
        logger.info("--- Finished building Ada projects. ---")

    def _run_daemon_thread(self, project: Dict[str, Any]):
        """
        Target function for each daemon's management thread.
        MODIFIED: Now includes a high-availability retry loop on process failure.
        """
        thread_name = f"AdaDaemon-{project['name']}"
        executable_path = os.path.join(project["path"], "bin", project["executable_name"])
        stop_event = project["stop_event"]

        if not os.path.exists(executable_path):
            logger.error(f"[{thread_name}] Executable not found, thread will exit permanently: {executable_path}")
            return

        # --- MODIFICATION START: High-Availability Loop ---
        while not stop_event.is_set():
            logger.info(f"[{thread_name}] Attempting to start daemon process: {executable_path}")
            process = None
            try:
                process = subprocess.Popen(
                    [executable_path],
                    cwd=project["path"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    stdin=subprocess.PIPE,
                    text=True, encoding='utf-8', errors='replace'
                )
                with self._lock:
                    project["process"] = process

                # --- (The existing stdout/stderr monitoring logic goes here) ---
                # Communicate through STDIO (why did i forgot about it you can communicate through stdio for the Ada daemons smh smh smh smh)
                def send_command(self, daemon_name: str, command: dict):
                    """Sends a JSON command to the specific Ada daemon via Stdin Pipe."""
                    for project in self.ada_projects:
                        if project["name"] == daemon_name and project["process"]:
                            try:
                                msg = json.dumps(command) + "\n"
                                project["process"].stdin.write(msg)
                                project["process"].stdin.flush()
                                logger.debug(f"Sent to {daemon_name}: {msg.strip()}")
                            except Exception as e:
                                logger.error(f"Failed to write to {daemon_name}: {e}")
                
                def log_stderr():
                    if process and process.stderr:
                        for line in iter(process.stderr.readline, ''):
                            logger.warning(f"[{thread_name} STDERR] {line.strip()}")

                stderr_thread = threading.Thread(target=log_stderr, daemon=True)
                stderr_thread.start()

                if process.stdout:
                    for line in iter(process.stdout.readline, ''):
                        if stop_event.is_set(): break
                        line = line.strip()
                        if line:
                            try:
                                data = json.loads(line)
                                payload = {
                                    "source_daemon": project["name"],
                                    "timestamp_py": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                    "data": data
                                }
                                self.data_queue.put(payload, timeout=1.0)
                            except json.JSONDecodeError:
                                logger.warning(f"[{thread_name}] Received non-JSON output: {line}")
                            except queue.Full:
                                logger.warning(f"[{thread_name}] Data queue is full. Discarding message.")

                # Wait for the process to finish to get its return code
                process.wait()

            except Exception as e:
                logger.error(f"[{thread_name}] Unhandled exception in daemon runner: {e}")
            finally:
                # This block runs after the process has terminated, either cleanly or by crashing.
                if process:
                    logger.warning(
                        f"[{thread_name}] Daemon process terminated unexpectedly (RC: {process.returncode}).")

                with self._lock:
                    project["process"] = None

            # If the stop event was set, break the loop cleanly. Otherwise, it was a failure.
            if stop_event.is_set():
                logger.info(f"[{thread_name}] Stop event received. Exiting management thread.")
                break

            # --- MODIFICATION: Log INOP Error and Wait Before Retrying ---
            logger.critical("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            logger.critical(f"!! [INOP ERROR] Ada Daemon '{project['name']}' has failed!                 !!")
            logger.critical(
                f"!! The system will attempt to restart it in {ADA_DAEMON_RETRY_DELAY_SECONDS} seconds.            !!")
            logger.critical("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

            # Wait for the specified delay, but allow the stop_event to interrupt the wait
            stop_event.wait(timeout=ADA_DAEMON_RETRY_DELAY_SECONDS)
        # --- MODIFICATION END ---

        logger.info(f"[{thread_name}] Thread finished.")

    def start_all(self):
        """Discovers and starts all Ada daemons, each in its own thread."""
        if not self.is_enabled:
            logger.info("StellaIcarus Ada Daemon feature is disabled.")
            return

        self._discover_ada_projects()
        
        if not self.ada_projects:
            self._discover_ada_projects()

        for project in self.ada_projects:
            thread = threading.Thread(target=self._run_daemon_thread, args=(project,), daemon=True)
            project["thread"] = thread
            thread.start()

    def stop_all(self):
        """Stops all running Ada daemon threads and processes."""
        if not self.is_enabled: return

        logger.info("Stopping all StellaIcarus Ada daemons...")
        for project in self.ada_projects:
            try:
                if project.get("stop_event"):
                    project["stop_event"].set()

                proc = project.get("process")
                if proc and proc.poll() is None:
                    logger.debug(f"Terminating process for '{project['name']}' (PID: {proc.pid})")
                    proc.terminate()

                thread = project.get("thread")
                if thread and thread.is_alive():
                    thread.join(timeout=5.0)
                    if thread.is_alive():
                        logger.warning(f"Thread for '{project['name']}' did not stop in time.")
            except Exception as e:
                logger.error(f"Error stopping daemon '{project['name']}': {e}")
        logger.info("All StellaIcarus Ada daemons have been signaled to stop.")

    def get_data_from_queue(self) -> Optional[Dict[str, Any]]:
        """Non-blocking read from the central data queue."""
        try:
            return self.data_queue.get_nowait()
        except queue.Empty:
            return None