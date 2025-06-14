# stella_icarus_utils.py

import os
import re
import time
import importlib.util
import importlib.machinery
from typing import List, Tuple, Callable, Optional, Any, Dict

from loguru import logger

# Attempt to import from CortexConfiguration. If this util is in a different location
# relative to config.py than app.py, this import might need adjustment
# or config values might need to be passed to the manager.
# Assuming it's in the same directory as app.py and config.py for this example.
try:
    from CortexConfiguration import *
except ImportError:
    logger.critical("StellaIcarusUtils: Failed to import configuration. Hooks will likely be disabled or use defaults.")
    # Define fallbacks so the class can at least be defined
    ENABLE_STELLA_ICARUS_HOOKS = False
    STELLA_ICARUS_HOOK_DIR = "./StellaIcarus_default_hooks"
    STELLA_ICARUS_CACHE_DIR = "./StellaIcarus_Cache_default"


class StellaIcarusHookManager:
    def __init__(self):
        self.hooks: List[Tuple[re.Pattern, Callable[[re.Match, str, str], Optional[str]], str]] = []
        self.hook_load_errors: List[str] = []
        self.is_enabled = ENABLE_STELLA_ICARUS_HOOKS  # Store enabled status

        if not self.is_enabled:
            logger.info("StellaIcarusHookManager: Hooks are disabled by configuration.")
            return

        if not os.path.isdir(STELLA_ICARUS_HOOK_DIR):
            logger.error(
                f"StellaIcarusHookManager: Hook directory '{STELLA_ICARUS_HOOK_DIR}' not found. No hooks loaded.")
            self.hook_load_errors.append(f"Hook directory not found: {STELLA_ICARUS_HOOK_DIR}")
            return

        # Ensure Numba cache directory exists if Numba is to be used in hooks
        if os.path.exists(STELLA_ICARUS_CACHE_DIR):
            logger.info(f"StellaIcarusHookManager: Numba cache directory '{STELLA_ICARUS_CACHE_DIR}' targeted.")
            os.environ['NUMBA_CACHE_DIR'] = STELLA_ICARUS_CACHE_DIR
        else:
            try:
                os.makedirs(STELLA_ICARUS_CACHE_DIR, exist_ok=True)
                logger.info(f"StellaIcarusHookManager: Created Numba cache directory '{STELLA_ICARUS_CACHE_DIR}'.")
                os.environ['NUMBA_CACHE_DIR'] = STELLA_ICARUS_CACHE_DIR
            except OSError as e:
                logger.error(
                    f"StellaIcarusHookManager: Could not create Numba cache directory '{STELLA_ICARUS_CACHE_DIR}': {e}")

        logger.info(f"StellaIcarusHookManager: Loading hooks from '{STELLA_ICARUS_HOOK_DIR}'...")
        for filename in os.listdir(STELLA_ICARUS_HOOK_DIR):
            if filename.endswith(".py") and not filename.startswith("_"):
                module_name = f"stella_hook_{filename[:-3]}"
                file_path = os.path.join(STELLA_ICARUS_HOOK_DIR, filename)
                try:
                    spec = importlib.util.spec_from_file_location(module_name, file_path)
                    if spec is None or spec.loader is None:  # type: ignore
                        raise ImportError(f"Could not create spec for module {module_name} at {file_path}")

                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)  # type: ignore

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
                        compiled_pattern = pattern_attr  # type: ignore
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
        """
        Checks the user input against loaded hooks and executes the handler of the first match.
        Returns the handler's response string or None if no hook matches or handles.
        """
        if not self.is_enabled or not self.hooks:
            return None

        for pattern, handler, module_name in self.hooks:
            match = pattern.match(user_input)  # Consider re.search if pattern isn't anchored
            if match:
                hook_start_time = time.perf_counter_ns()
                try:
                    response = handler(match, user_input, session_id)
                    hook_end_time = time.perf_counter_ns()
                    duration_ns = hook_end_time - hook_start_time
                    duration_us = duration_ns / 1000.0
                    logger.debug(
                        f"StellaIcarusHook '{module_name}' matched. Handler duration: {duration_us:.3f} µs ({duration_ns} ns).")

                    if response is not None:
                        if not isinstance(response, str):
                            logger.error(
                                f"StellaIcarusHook '{module_name}' handler returned non-string type: {type(response)}. Hook skipped.")
                            return None
                        return response
                except Exception as e:
                    hook_end_time = time.perf_counter_ns()  # type: ignore
                    duration_ns = hook_end_time - hook_start_time  # type: ignore
                    duration_us = duration_ns / 1000.0  # type: ignore
                    logger.error(f"StellaIcarusHook '{module_name}' execution error after {duration_us:.3f} µs: {e}")
                    logger.exception(f"StellaIcarusHook '{module_name}' Traceback:")
                    return None  # Skip on error
        return None  # No hook matched