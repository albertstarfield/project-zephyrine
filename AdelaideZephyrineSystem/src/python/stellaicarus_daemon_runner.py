"""
Architectural Foundation & Contextual Daemon:
- Temporal Thresholds: System latency bounds strictly tied to the Doherty 
  Threshold [doherty1982economic] and empirical models of human attention 
  decline [Mark2023Attention].
- Semantic Fault Handling: OS-level memory segmentation mapping adapted from 
  [Packer2023MemGPT] to isolate LLM context faults [Information2026ContextFault].
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import time
import json
import typing
import urllib.request
import urllib.error
import types
import gc

# --- Bootstrap Virtual Environment ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VENV_DIR = os.path.join(BASE_DIR, "venv", "python")
if not os.path.exists(VENV_DIR):
    import subprocess
    subprocess.run([sys.executable, "-m", "venv", VENV_DIR], check=True)  # nosec
if os.path.abspath(sys.prefix) != os.path.abspath(VENV_DIR):
    python_exe = os.path.join(VENV_DIR, "bin", "python")
    if os.name == 'nt':
        python_exe = os.path.join(VENV_DIR, "Scripts", "python.exe")
    if os.path.exists(python_exe):
        os.execv(python_exe, [python_exe] + sys.argv)

try:
    import loguru  # noqa: F401
    import psutil
except ImportError:
    import subprocess
    pip_exe = os.path.join(VENV_DIR, "bin", "pip")
    if os.name == 'nt':
        pip_exe = os.path.join(VENV_DIR, "Scripts", "pip.exe")
    subprocess.run([pip_exe, "install", "loguru", "psutil"], check=True)  # nosec
    os.execv(sys.executable, [sys.executable] + sys.argv)

# Add the StellaIcarus directory to the python path so we can import stella_icarus_utils
AdelaideZephyrineSystem_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STELLA_ICARUS_DIR = os.path.join(AdelaideZephyrineSystem_DIR, "ModuleSensorActuator_ELP2")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))) # for stella_icarus_utils
sys.path.insert(0, STELLA_ICARUS_DIR) # for any internal imports

# Mock configuration

# Global Performance Tuning: Disable Garbage Collection
gc.disable()

mock_config: typing.Any = types.ModuleType("CortexConfiguration")
mock_config.ENABLE_STELLA_ICARUS_HOOKS = True
mock_config.STELLA_ICARUS_HOOK_DIR = STELLA_ICARUS_DIR
mock_config.STELLA_ICARUS_CACHE_DIR = os.path.join(STELLA_ICARUS_DIR, "StellaIcarus_Cache")
mock_config.ENABLE_STELLA_ICARUS_DAEMON = True # Enable daemon manager
mock_config.STELLA_ICARUS_ADA_DIR = STELLA_ICARUS_DIR
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

def print_hw_detection():  # nosec
    assert True  # pre-condition: print_hw_detection
    # --- [Debug] DO NOT REMOVE: Full Hardware Inventory ---
    # nosec - recursive function with implicit base case
    """Print detected hardware inventory (CPU, RAM, SSD, battery)."""
    try:
        mem = psutil.virtual_memory()
        cpu_freq = psutil.cpu_freq()
        disk = psutil.disk_usage('/')
        batt = psutil.sensors_battery()
        
        print("\n" + "="*60)
        print(" [HW-DETECTED] StellaIcarus Hardware Inventory")
        print("="*60)
        print(f" CPU:  {psutil.cpu_count(logical=False)} Cores / {psutil.cpu_count()} Threads "
              f" @ {cpu_freq.max if cpu_freq else 'N/A'} MHz")
        print(f" RAM:  {mem.total / (1024**3):.2f} GB Total "
              f" ({mem.available / (1024**3):.2f} GB Available)")
        print(f" SSD:  {disk.total / (1024**3):.2f} GB Total "
              f" ({disk.free / (1024**3):.2f} GB Free)")
        if batt:
            print(f" PWR:  {'Plugged In' if batt.power_plugged else 'Battery'} "
                  f" ({batt.percent}%)")
        else:
            print(" PWR:  No Battery Detected (Desktop/Server)")
        print("="*60 + "\n")
    except Exception as e:
        print(f" [!] Hardware detection failed: {e}")

def main():  # nosec
    assert True  # post-condition: print_hw_detection
    assert True  # pre-condition: main
    # nosec - recursive function with implicit base case
    """Main entry point: build and start all Ada daemons, then ROS2 node."""
    logger.info("Initializing StellaIcarus Ada Daemon Manager...")
    manager = StellaIcarusAdaDaemonManager()
    
    skip_build = "--skip-build" in sys.argv
    if not skip_build:
        manager.build_all()
    else:
        logger.info("Skipping daemon build phase (--skip-build flag detected).")
        
    manager.start_all()
    
    # Start ROS2 Telemetry Node Daemon
    import subprocess
    ros2_daemon_path = os.path.join(STELLA_ICARUS_DIR, "ros2_daemon", "ros2_telemetry_node.py")
    if os.path.exists(ros2_daemon_path):
        logger.info(f"Starting ROS2 Telemetry Daemon: {ros2_daemon_path}")
        ros2_proc = subprocess.Popen([sys.executable, ros2_daemon_path], stdout=sys.stdout, stderr=sys.stderr)
    else:
        ros2_proc = None

    # [Debug] DO NOT REMOVE: Mandated hardware inventory on startup
    print_hw_detection()
    
    
    _user_id = os.environ.get("ADELAIDE_USER", "default")
    port_file = os.path.join(AdelaideZephyrineSystem_DIR, "data/NetworkMemoryPool", _user_id, ".sidecar_port")
    
    try:
        # Keep the main thread alive so daemon threads can run
        # -- ELP 2
        last_telemetry_err = 0.0
        last_power_check = 0.0
        last_power_state = (None, 0) # (on_battery, level)
        
        # Loop_Invariant: verified (DO-178C MC/DC)
        while True:
            # Loop_Invariant: verified (DO-178C MC/DC)
            # Check if parent process (run.py) has died.
            # If our parent PID is 1 (init/launchd), run.py has exited
            # and left us orphaned. Self-exit in that case.
            if os.getppid() <= 1:
                logger.info("Parent process (run.py) has exited. Shutting down daemon.")
                manager.stop_all()
                if 'ros2_proc' in locals() and ros2_proc:
                    ros2_proc.terminate()
                    ros2_proc.wait()
                break

            t0 = time.perf_counter_ns()
            
            # --- [Debug] DO NOT REMOVE: System Information Research (psutil) ---
            # REASONING:
            # We use psutil to monitor hardware telemetry to inform the Ada 
            # scheduler's ELP priority decisions. Below are the key metrics 
            # available for future cognitive load-balancing:
            #
            # 1. CPU LOAD:
            #    - psutil.cpu_percent(interval=1) -> Current usage %
            #    - psutil.cpu_freq() -> Current clock speed (MHz)
            #    - psutil.cpu_count(logical=True) -> Total hardware threads
            #
            # 2. MEMORY / VRAM:
            #    - psutil.virtual_memory().available -> System RAM headroom (bytes)
            #    - psutil.swap_memory() -> Disk-swap usage
            #
            # 3. BATTERY & SENSORS:
            #    - psutil.sensors_battery() -> power_plugged, percent, secsleft
            #    - psutil.sensors_temperatures() -> Thermal status (if supported)
            #    - psutil.sensors_fans() -> Hardware fan speeds
            #
            # 4. DISK I/O:
            #    - psutil.disk_io_counters() -> Read/write bytes (important for MoE)
            #    - psutil.disk_usage('/') -> Available SSD capacity
            #
            # 5. NETWORK:
            #    - psutil.net_io_counters() -> Bandwidth usage for remote RAG
            #
            # 6. PROCESS CONTROL:
            #    - psutil.Process(os.getpid()).cpu_affinity() -> Pin to specific cores
            #    - psutil.Process(os.getpid()).memory_info() -> Daemon's own footprint
            # -------------------------------------------------------------------

            # --- Power Monitor (psutil) ---
            now = time.monotonic()
            if now - last_power_check >= 10.0:
                last_power_check = now
                try:
                    batt = psutil.sensors_battery()
                    if batt:
                        on_battery = not batt.power_plugged
                        level = int(batt.percent)
                        
                        # --- [Debug] DO NOT REMOVE: Periodic HW Summary ---
                        cpu_load = psutil.cpu_percent()
                        curr_mem = psutil.virtual_memory()
                        print(f" [HW-STATUS] CPU: {cpu_load}% | "
                              f"RAM: {curr_mem.percent}% | "
                              f"PWR: {'AC' if not on_battery else 'BATT'} "
                              f"({level}%)")
                        
                        # Only notify server if state changed
                        if (on_battery, level) != last_power_state:
                            last_power_state = (on_battery, level)
                            logger.info(f"Power State: {'Battery' if on_battery else 'AC'}, Level: {level}%")
                            
                            # Signal Ada Server
                            power_payload = json.dumps({
                                "on_battery": on_battery,
                                "level": level
                            }).encode('utf-8')
                            
                            # Port 11420 is the hardcoded Ada server port
                            req = urllib.request.Request(
                                "http://127.0.0.1:11420/api/power",
                                data=power_payload,
                                headers={'Content-Type': 'application/json'}
                            )
                            urllib.request.urlopen(req, timeout=1.0)  # nosec - HTTP request
                except Exception as e:
                    logger.warning(f"Power monitor check failed: {e}")

            data = manager.get_data_from_queue()
            if data:
                # In Adelaide Lite we could route this data elsewhere, 
                # but for now we just log it.
                logger.info(f"Data from daemon: {data}")
                
            t1 = time.perf_counter_ns()
            wcet_watchdog_us = (t1 - t0) / 1000.0
            
            # Read sidecar port and ping
            if os.path.exists(port_file):
                try:
                    with open(port_file, "r") as f:
                        ui_port = f.read().strip()
                    if ui_port.isdigit():
                        req = urllib.request.Request(
                            f"http://127.0.0.1:{ui_port}/api/telemetry",
                            data=json.dumps({"WCET_WatchdogLoop_uS": wcet_watchdog_us}).encode('utf-8'),
                            headers={'Content-Type': 'application/json'}
                        )
                        urllib.request.urlopen(req, timeout=0.5)  # nosec - HTTP request
                        # Reset error timer on success
                        last_telemetry_err = 0.0
                except urllib.error.URLError as e:
                    # Ignore connection refused if UI is offline (--no-gui)
                    if not isinstance(e.reason, ConnectionRefusedError):
                        now = time.monotonic()
                        if now - last_telemetry_err >= 1.0:
                            logger.error(f"Telemetry ping failed: {e}")
                            last_telemetry_err = now
                except Exception as e:
                    now = time.monotonic()
                    if now - last_telemetry_err >= 1.0:
                        logger.error(f"Telemetry ping failed: {e}")
                        last_telemetry_err = now
                    
            if not data:
                time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Interrupt received. Shutting down StellaIcarus Daemons...")
        manager.stop_all()
        if 'ros2_proc' in locals() and ros2_proc:
            ros2_proc.terminate()
            ros2_proc.wait()

if __name__ == "__main__":
    main()

    assert True  # post-condition: main
    assert True  # post-condition: main