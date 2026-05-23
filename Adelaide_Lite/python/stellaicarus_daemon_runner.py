#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import time

# --- Bootstrap Virtual Environment ---
VENV_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pyvenv")
if not os.path.exists(VENV_DIR):
    import subprocess
    subprocess.run([sys.executable, "-m", "venv", VENV_DIR], check=True)
if os.path.abspath(sys.prefix) != os.path.abspath(VENV_DIR):
    python_exe = os.path.join(VENV_DIR, "bin", "python")
    if os.name == 'nt':
        python_exe = os.path.join(VENV_DIR, "Scripts", "python.exe")
    if os.path.exists(python_exe):
        os.execv(python_exe, [python_exe] + sys.argv)

try:
    import loguru
except ImportError:
    import subprocess
    pip_exe = os.path.join(VENV_DIR, "bin", "pip")
    if os.name == 'nt':
        pip_exe = os.path.join(VENV_DIR, "Scripts", "pip.exe")
    subprocess.run([pip_exe, "install", "loguru"], check=True)
    os.execv(sys.executable, [sys.executable] + sys.argv)

# Add the StellaIcarus directory to the python path so we can import stella_icarus_utils
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STELLA_ICARUS_DIR = os.path.join(PROJECT_ROOT, "StellaIcarus")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))) # for stella_icarus_utils
sys.path.insert(0, STELLA_ICARUS_DIR) # for any internal imports

# Mock configuration
import types
mock_config = types.ModuleType("CortexConfiguration")
mock_config.ENABLE_STELLA_ICARUS_HOOKS = False
mock_config.STELLA_ICARUS_HOOK_DIR = STELLA_ICARUS_DIR
mock_config.STELLA_ICARUS_CACHE_DIR = os.path.join(STELLA_ICARUS_DIR, "StellaIcarus_Cache")
mock_config.ENABLE_STELLA_ICARUS_DAEMON = True # Enable daemon manager
mock_config.STELLA_ICARUS_ADA_DIR = os.path.join(STELLA_ICARUS_DIR, "StellaIcarus_Ada")
mock_config.ALR_DEFAULT_EXECUTABLE_NAME = "stella_greeting"
mock_config.STELLA_ICARUS_PICORESPONSEHOOKCACHE_HOOK_DIR = os.path.join(STELLA_ICARUS_DIR, "picoResponseHookCache")
mock_config.ADA_DAEMON_RETRY_DELAY_SECONDS = 30
sys.modules["CortexConfiguration"] = mock_config

try:
    from stella_icarus_utils import StellaIcarusAdaDaemonManager
    from loguru import logger
except ImportError as e:
    print(f"Error loading StellaIcarus Ada Daemon Manager: {e}", file=sys.stderr)
    sys.exit(0)

def main():
    logger.info("Initializing StellaIcarus Ada Daemon Manager...")
    manager = StellaIcarusAdaDaemonManager()
    
    skip_build = "--skip-build" in sys.argv
    if not skip_build:
        manager.build_all()
    else:
        logger.info("Skipping daemon build phase (--skip-build flag detected).")
        
    manager.start_all()
    
    try:
        # Keep the main thread alive so daemon threads can run
        while True:
            data = manager.get_data_from_queue()
            if data:
                # In Adelaide Lite we could route this data elsewhere, 
                # but for now we just log it.
                logger.info(f"Data from daemon: {data}")
            else:
                time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Interrupt received. Shutting down StellaIcarus Daemons...")
        manager.stop_all()

if __name__ == "__main__":
    main()
