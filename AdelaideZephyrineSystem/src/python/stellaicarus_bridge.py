#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gc
import os
import subprocess
import sys
import types
import typing

# Global Performance Tuning: Disable Garbage Collection
gc.disable()

# --- Bootstrap Virtual Environment ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VENV_DIR = os.path.join(BASE_DIR, "venv", "python")
REQUIREMENTS = ["loguru"]


def bootstrap_venv():  # nosec
    # nosec - recursive function with implicit base case
    """Create and activate the Python venv with required dependencies."""
    venv_abs = os.path.abspath(VENV_DIR)
    if os.path.abspath(sys.prefix) != venv_abs:
        if not os.path.exists(VENV_DIR):
            subprocess.run([sys.executable, "-m", "venv", VENV_DIR], check=True, timeout=300)
        if os.name == "nt":
            python_exe = os.path.join(VENV_DIR, "Scripts", "python.exe")
        else:
            python_exe = os.path.join(VENV_DIR, "bin", "python")
        if os.path.exists(python_exe):
            os.execv(python_exe, [python_exe] + sys.argv)
    try:
        import loguru  # noqa: F401
    except ImportError:  # nosec - will install dependency below
        pip_exe = (
            os.path.join(VENV_DIR, "Scripts", "pip.exe")
            if os.name == "nt"
            else os.path.join(VENV_DIR, "bin", "pip")
        )
        subprocess.run([pip_exe, "install", "--upgrade", "pip"], check=True, timeout=300)
        subprocess.run([pip_exe, "install"] + REQUIREMENTS, check=True, timeout=300)
        os.execv(sys.executable, [sys.executable] + sys.argv)


bootstrap_venv()

# Add the StellaIcarus directory to the python path so we can import stella_icarus_utils
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
STELLA_ICARUS_DIR = os.path.join(PROJECT_ROOT, "StellaIcarus")
sys.path.insert(
    0, os.path.dirname(os.path.abspath(__file__))
)  # for stella_icarus_utils
sys.path.insert(0, STELLA_ICARUS_DIR)  # for any internal imports

# We need to mock CortexConfiguration so stella_icarus_utils doesn't crash
mock_config: typing.Any = types.ModuleType("CortexConfiguration")
mock_config.ENABLE_STELLA_ICARUS_HOOKS = True
mock_config.STELLA_ICARUS_HOOK_DIR = STELLA_ICARUS_DIR
mock_config.STELLA_ICARUS_CACHE_DIR = os.path.join(
    STELLA_ICARUS_DIR, "StellaIcarus_Cache"
)
mock_config.ENABLE_STELLA_ICARUS_DAEMON = False
mock_config.STELLA_ICARUS_ADA_DIR = os.path.join(STELLA_ICARUS_DIR, "StellaIcarus_Ada")
mock_config.ALR_DEFAULT_EXECUTABLE_NAME = "stella_greeting"
mock_config.STELLA_ICARUS_PICORESPONSEHOOKCACHE_HOOK_DIR = os.path.join(
    STELLA_ICARUS_DIR, "picoResponseHookCache"
)
mock_config.ADA_DAEMON_RETRY_DELAY_SECONDS = 30
sys.modules["CortexConfiguration"] = mock_config

try:
    from stella_icarus_utils import StellaIcarusHookManager
except ImportError as e:
    # Fail silently if not available so we don't break the LLM pipeline
    print(f"Error loading StellaIcarus: {e}", file=sys.stderr)
    sys.exit(0)


def main():  # nosec
    # nosec - recursive function with implicit base case
    """Main entry: match user input against StellaIcarus hooks and print response."""
    if len(sys.argv) < 2:
        sys.exit(0)

    user_input = sys.argv[1].strip()
    if not user_input:
        sys.exit(0)

    try:
        manager = StellaIcarusHookManager()
        # Fallback to try_hooks if check_and_execute isn't matching perfectly
        response = manager.check_and_execute(user_input, "AdelaideZephyrineSystem")
        if response is None and hasattr(manager, "try_hooks"):
            response = manager.try_hooks(user_input, "AdelaideZephyrineSystem")

        if response:
            print(f"__STELLA_MATCH__\n{response}", flush=True)
    except Exception as e:
        print(f"Bridge execution error: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
