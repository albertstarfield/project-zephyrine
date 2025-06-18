# stella_icarus_utils.py

import os
import re
import time
import importlib.util
import threading
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
        ENABLE_STELLA_ICARUS_DAEMON, STELLA_ICARUS_ADA_DIR, ALR_DEFAULT_EXECUTABLE_NAME
    )
except ImportError:
    logger.critical("StellaIcarusUtils: Failed to import configuration. All features will be disabled.")
    ENABLE_STELLA_ICARUS_HOOKS = False
    STELLA_ICARUS_HOOK_DIR = "./StellaIcarus_default_hooks"
    STELLA_ICARUS_CACHE_DIR = "./StellaIcarus_Cache_default"
    ENABLE_STELLA_ICARUS_DAEMON = False
    STELLA_ICARUS_ADA_DIR = "./StellaIcarus"
    ALR_DEFAULT_EXECUTABLE_NAME = "stella_greeting"


class StellaIcarusHookManager:
    # ... (The existing StellaIcarusHookManager class remains here, completely unchanged) ...
    def __init__(self):
        self.hooks: List[Tuple[re.Pattern, Callable[[re.Match, str, str], Optional[str]], str]] = []
        self.hook_load_errors: List[str] = []
        self.is_enabled = ENABLE_STELLA_ICARUS_HOOKS

        if not self.is_enabled:
            logger.info("StellaIcarusHookManager: Hooks are disabled by configuration.")
            return

        if not os.path.isdir(STELLA_ICARUS_HOOK_DIR):
            logger.error(
                f"StellaIcarusHookManager: Hook directory '{STELLA_ICARUS_HOOK_DIR}' not found. No hooks loaded.")
            self.hook_load_errors.append(f"Hook directory not found: {STELLA_ICARUS_HOOK_DIR}")
            return

        logger.info(f"StellaIcarusHookManager: Loading hooks from '{STELLA_ICARUS_HOOK_DIR}'...")
        for filename in os.listdir(STELLA_ICARUS_HOOK_DIR):
            if filename.endswith(".py") and not filename.startswith("_"):
                module_name = f"stella_hook_{filename[:-3]}"
                file_path = os.path.join(STELLA_ICARUS_HOOK_DIR, filename)
                try:
                    spec = importlib.util.spec_from_file_location(module_name, file_path)
                    if spec is None or spec.loader is None:
                        raise ImportError(f"Could not create spec for module {module_name} at {file_path}")

                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    pattern_attr = getattr(module, "PATTERN", None)
                    handler_func = getattr(module, "handler", None)

                    if pattern_attr is None or handler_func is None:
                        logger.warning(f"  Skipping '{filename}': Missing PATTERN or handler function.")
                        self.hook_load_errors.append(f"Missing PATTERN/handler in {filename}")
                        continue

                    compiled_pattern: re.Pattern
                    if isinstance(pattern_attr, str):
                        compiled_pattern = re.compile(pattern_attr)
                    elif isinstance(pattern_attr, re.Pattern):
                        compiled_pattern = pattern_attr
                    else:
                        logger.warning(f"  Skipping '{filename}': PATTERN is not a string or compiled regex.")
                        self.hook_load_errors.append(f"Invalid PATTERN type in {filename}")
                        continue

                    if not callable(handler_func):
                        logger.warning(f"  Skipping '{filename}': 'handler' is not callable.")
                        self.hook_load_errors.append(f"Non-callable handler in {filename}")
                        continue

                    self.hooks.append((compiled_pattern, handler_func, module_name))
                    logger.info(
                        f"  Loaded StellaIcarusHook: '{module_name}' with pattern: '{compiled_pattern.pattern}'")

                except Exception as e:
                    logger.error(f"  Error loading StellaIcarusHook from '{filename}': {e}")
                    self.hook_load_errors.append(f"Error loading {filename}: {e}")

        if not self.hooks and not self.hook_load_errors:
            logger.info("StellaIcarusHookManager: No hook files found in the directory.")
        elif self.hooks:
            logger.success(f"StellaIcarusHookManager: Successfully loaded {len(self.hooks)} hook(s).")
        if self.hook_load_errors:
            logger.error(
                f"StellaIcarusHookManager: Encountered {len(self.hook_load_errors)} error(s) during hook loading.")

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

        logger.info(f"Discovering Ada projects in '{STELLA_ICARUS_ADA_DIR}'...")
        for item in os.listdir(STELLA_ICARUS_ADA_DIR):
            project_path = os.path.join(STELLA_ICARUS_ADA_DIR, item)
            if os.path.isdir(project_path):
                has_alire_toml = os.path.exists(os.path.join(project_path, "alire.toml"))
                gpr_files = [f for f in os.listdir(project_path) if f.endswith(".gpr")]

                if has_alire_toml or gpr_files:
                    project_name = item
                    executable_name = ALR_DEFAULT_EXECUTABLE_NAME  # Or parse from GPR if needed
                    self.ada_projects.append({
                        "name": project_name,
                        "path": project_path,
                        "executable_name": f"{executable_name}.exe" if os.name == 'nt' else executable_name,
                        "process": None,
                        "thread": None,
                        "stop_event": threading.Event()
                    })
                    logger.info(f"  Discovered Ada project: '{project_name}'")

    def build_all(self):
        """Builds all discovered Ada projects using 'alr build'."""
        if not self.is_enabled: return

        logger.info("--- Building all discovered StellaIcarus Ada projects... ---")
        for project in self.ada_projects:
            logger.info(f"Building '{project['name']}' in '{project['path']}'...")
            try:
                # Using shell=True can be a security risk if project['path'] is not trusted.
                # For this controlled environment, it's simpler.
                process = subprocess.run(
                    ["alr", "build"],
                    cwd=project["path"],
                    capture_output=True, text=True, check=False, timeout=300
                )
                if process.returncode == 0:
                    logger.success(f"  ✅ Successfully built '{project['name']}'.")
                else:
                    logger.error(f"  ❌ Failed to build '{project['name']}'. RC: {process.returncode}")
                    logger.error(f"     STDOUT: {process.stdout.strip()}")
                    logger.error(f"     STDERR: {process.stderr.strip()}")
            except FileNotFoundError:
                logger.error("  ❌ Build failed: 'alr' command not found. Is Alire installed and in the PATH?")
                # Stop trying to build other projects if alr is missing
                break
            except subprocess.TimeoutExpired:
                logger.error(f"  ❌ Build timed out for '{project['name']}'.")
            except Exception as e:
                logger.error(f"  ❌ An unexpected error occurred during build of '{project['name']}': {e}")
        logger.info("--- Finished building Ada projects. ---")

    def _run_daemon_thread(self, project: Dict[str, Any]):
        """
        Target function for each daemon's management thread.
        Tries to run the real Ada binary and pipes its output to the central queue.
        """
        thread_name = f"AdaDaemon-{project['name']}"
        executable_path = os.path.join(project["path"], "bin", project["executable_name"])
        stop_event = project["stop_event"]

        if not os.path.exists(executable_path):
            logger.error(f"[{thread_name}] Executable not found, thread will exit: {executable_path}")
            return  # Exit thread if binary doesn't exist

        logger.info(f"[{thread_name}] Starting daemon process: {executable_path}")
        try:
            process = subprocess.Popen(
                [executable_path],
                cwd=project["path"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True, encoding='utf-8', errors='replace'
            )
            project["process"] = process

            # Monitor stderr for errors in a separate thread
            def log_stderr():
                if process.stderr:
                    for line in iter(process.stderr.readline, ''):
                        logger.warning(f"[{thread_name} STDERR] {line.strip()}")

            stderr_thread = threading.Thread(target=log_stderr, daemon=True)
            stderr_thread.start()

            # Main loop to read stdout and push to the queue
            if process.stdout:
                for line in iter(process.stdout.readline, ''):
                    if stop_event.is_set():
                        break
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            payload = {
                                "source_daemon": project["name"],
                                "timestamp_py": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                "data": data
                            }
                            self.data_queue.put(payload)
                        except json.JSONDecodeError:
                            logger.warning(f"[{thread_name}] Received non-JSON output: {line}")
                        except queue.Full:
                            logger.warning(f"[{thread_name}] Data queue is full. Discarding message.")

        except Exception as e:
            logger.error(f"[{thread_name}] Error running daemon process: {e}")
        finally:
            if project.get("process") and project["process"].poll() is None:
                logger.info(f"[{thread_name}] Daemon process ended or is being cleaned up.")
                project["process"].terminate()
            logger.info(f"[{thread_name}] Thread finished.")

    def start_all(self):
        """Discovers and starts all Ada daemons, each in its own thread."""
        if not self.is_enabled:
            logger.info("StellaIcarus Ada Daemon feature is disabled.")
            return

        self._discover_ada_projects()
        if not self.ada_projects:
            logger.info("No Ada projects discovered to start.")
            return

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