# ./StellaIcarus/fmc_servo_manual_hook_test.py
import re
import sys
import time
import glob
import subprocess
import importlib
import threading
from typing import Optional, Match, Tuple

# --- Automatic Dependency Management ---
# This block checks for the 'pyserial' dependency and attempts to install it if missing.
_PYSERIAL_AVAILABLE = False
try:
    # A robust way to check if a module is installed without importing it
    importlib.util.find_spec('serial')
    import serial  # Now we can safely import it

    _PYSERIAL_AVAILABLE = True
except ImportError:
    print("FMC SERVO HOOK: Dependency 'pyserial' not found.", file=sys.stderr)
    print("--> Attempting to auto-install with pip...", file=sys.stderr)
    try:
        # Use sys.executable to ensure we use the pip for the current Python env
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "pyserial"],
            capture_output=True,
            text=True,
            check=True  # This will raise an exception if pip fails
        )
        print("--> Auto-install successful.", file=sys.stderr)
        import serial  # Try importing again after successful installation

        _PYSERIAL_AVAILABLE = True
    except Exception as e:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", file=sys.stderr)
        print("!! FMC SERVO HOOK: CRITICAL - Auto-install of 'pyserial' FAILED. !!", file=sys.stderr)
        print(f"!! Error: {e}", file=sys.stderr)
        print("!! Please install it manually to enable this hook:", file=sys.stderr)
        print(f"!! `{sys.executable} -m pip install pyserial`", file=sys.stderr)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", file=sys.stderr)
        _PYSERIAL_AVAILABLE = False

# --- Numba JIT Compilation for Calculation Logic ---
IS_JIT_COMPILED = True
_NUMBA_JIT_SUCCESSFUL = False
_NUMBA_INIT_ERROR_MSG = None

try:
    from numba import njit


    @njit(cache=True)
    def _calculate_servo_outputs_numba(command_code: int, value: float) -> Tuple[float, float]:
        gyro_val: float = 0.0
        inertia_val: float = 0.0
        if command_code == 0:
            gyro_val = value
        elif command_code == 1:
            inertia_val = value
        elif command_code == 2:
            gyro_val = value * 0.7
            inertia_val = -value * 0.3
        return gyro_val, inertia_val


    _test_gyro, _test_inertia = _calculate_servo_outputs_numba(2, 10.0)
    if _test_gyro == 7.0 and _test_inertia == -3.0:
        _NUMBA_JIT_SUCCESSFUL = True
        print("FMC SERVO HOOK: Numba JIT self-test successful.", file=sys.stderr)
    else:
        raise RuntimeError("Numba self-test returned incorrect values.")

except Exception as e:
    _NUMBA_INIT_ERROR_MSG = f"Numba JIT init failed ({e}). This hook will use a Python fallback for calculations."
    print(f"FMC SERVO HOOK: {_NUMBA_INIT_ERROR_MSG}", file=sys.stderr)
    _NUMBA_JIT_SUCCESSFUL = False


# --- Plain Python Fallback Calculation ---
def _calculate_servo_outputs_python(command_code: int, value: float) -> Tuple[float, float]:
    gyro_val: float = 0.0
    inertia_val: float = 0.0
    if command_code == 0:
        gyro_val = value
    elif command_code == 1:
        inertia_val = value
    elif command_code == 2:
        gyro_val = value * 0.7
        inertia_val = -value * 0.3
    return gyro_val, inertia_val


# --- Configuration & State Management ---
SERIAL_BAUD_RATE = 115200
ARDUINO_HANDSHAKE_MSG = b"Arduino ready"
RETRY_DELAY_SECONDS = 5
_serial_lock = threading.Lock()
_serial_connection: Optional[serial.Serial] = None
_active_port_name: Optional[str] = None

# --- Main Regex Pattern ---
PATTERN = re.compile(
    r"^\s*(?:fmc|servo|how to control)?\s*"
    r"(?:(set|adjust|move)\s+)?\s*"
    r"(yaw|pitch|roll|reset|help|me)"
    r"\s*(-?\d+(?:\.\d+)?)?\s*$",
    re.IGNORECASE
)


# --- Serial I/O Functions ---
# (The functions _scan_and_verify_ports and _send_command_to_mcu are omitted for brevity,
# but should be included from the previous answer. Their logic is unchanged.)

# --- The Main Hook Handler ---
def handler(match: Match[str], user_input: str, session_id: str) -> Optional[str]:
    SUCCESS_PREFIX = "This is what I get or the result of my calculation: "
    ERROR_PREFIX = "I think Im lost can you repeat that again to me?"

    if not _PYSERIAL_AVAILABLE:
        return f"{ERROR_PREFIX} The FMC servo hook is disabled because the 'pyserial' library could not be installed."

    try:
        _, _, command_str, value_str = match.groups()
        command = command_str.lower()

        if command in ["help", "me"]:
            # ... (Help message logic is unchanged) ...
            return f"{SUCCESS_PREFIX} Here is the guide..."  # Placeholder for brevity

        if command == "reset":
            # ... (Reset logic is unchanged) ...
            return f"{SUCCESS_PREFIX} FMC servos reset."  # Placeholder

        if value_str is None: return f"{ERROR_PREFIX} Command '{command}' requires a numerical value."
        try:
            value = float(value_str)
        except ValueError:
            return f"{ERROR_PREFIX} Invalid number '{value_str}'."

        command_map = {'yaw': 0, 'pitch': 1, 'roll': 2}
        command_code = command_map.get(command, -1)

        gyro_val, inertia_val = 0.0, 0.0
        if _NUMBA_JIT_SUCCESSFUL:
            try:
                gyro_val, inertia_val = _calculate_servo_outputs_numba(command_code, value)
            except Exception as e:
                print(f"FMC HOOK: Numba execution error: {e}. Falling back to Python.", file=sys.stderr)
                gyro_val, inertia_val = _calculate_servo_outputs_python(command_code, value)
        else:
            gyro_val, inertia_val = _calculate_servo_outputs_python(command_code, value)

        mcu_command = f"{gyro_val},{inertia_val}\n"
        success, message = _send_command_to_mcu(mcu_command)

        if success:
            return f"{SUCCESS_PREFIX} Command '{command} {value}' sent. {message}"
        else:
            return f"{ERROR_PREFIX} Failed to execute command '{command} {value}'. {message}"

    except Exception as e:
        return f"{ERROR_PREFIX} A critical error occurred in the FMC servo hook: {e}"

    return None

# NOTE: The helper functions _scan_and_verify_ports and _send_command_to_mcu,
# as well as the __main__ test block, are omitted here for brevity but should be
# copied from the previous complete answer.