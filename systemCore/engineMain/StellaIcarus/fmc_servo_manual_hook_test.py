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
    r"^\s*"  # Start of string, optional whitespace

    # --- Optional Trigger Phrases ---
    # Matches common prefixes for commands.
    r"(?:fmc|servo|flight\s+control|manual\s+override|how to control|command|input)?\s*"

    # --- The Command Verb (Group 1: Optional, but helps with clarity) ---
    # Matches the action the user wants to perform.
    r"(?P<verb>set|adjust|move|input|command|trim|deflect|execute|engage|request)\s+"

    # --- The Control Axis/Surface (Group 2: The Core Command) ---
    # This is the primary command string.
    r"(?P<command>"
    r"yaw|pitch|roll"  # Primary axes
    r"|rudder|elevator|aileron"  # Control surfaces
    r"|trim" # Trim control
    r"|reset|center|zero"  # Reset commands
    r"|help|status|query|me"  # Help and query commands
    r")"

    # --- Optional Preposition/Connector Words ---
    r"\s*(?:to|at|by|value)?\s*"

    # --- The Numerical Value (Group 3: Optional, for commands that need it) ---
    # Captures a signed integer or float.
    r"(?P<value>-?\d+(?:\.\d+)?)?\s*"

    # --- Optional Units ---
    # Matches common units but doesn't capture them, as we assume degrees/percentage.
    r"(?:deg(?:rees)?|%|percent|units)?\s*$",

    re.IGNORECASE
)


# --- The Main Hook Handler ---
def _scan_and_verify_ports() -> Optional[str]:
    """
    Scans for available serial ports and attempts to handshake with an Arduino.

    This function iterates through potential serial ports, opens a connection,
    waits for the Arduino's "Arduino ready" handshake message, and if successful,
    returns the name of the verified port. This is crucial for establishing a
    reliable connection without hardcoding the port name.

    Returns:
        The name of the verified serial port (e.g., '/dev/cu.usbmodem14201' or 'COM3'),
        or None if no suitable device is found.
    """
    global _active_port_name, _serial_connection

    # Use glob to find potential serial ports in a cross-platform way.
    # On macOS/Linux, they often look like /dev/tty.* or /dev/cu.*
    # On Windows, they are COM ports.
    if sys.platform.startswith('darwin') or sys.platform.startswith('linux'):
        # Prioritize 'cu' on macOS as it's for call-out devices
        potential_ports = glob.glob('/dev/cu.usbmodem*') + glob.glob('/dev/ttyACM*') + glob.glob('/dev/tty.usbmodem*')
    elif sys.platform.startswith('win'):
        potential_ports = [f'COM{i}' for i in range(1, 257)]
    else:
        print("FMC HOOK: Unsupported operating system for serial port scanning.", file=sys.stderr)
        return None

    if not potential_ports:
        print("FMC HOOK: No potential serial ports found.", file=sys.stderr)
        return None

    print(f"FMC HOOK: Scanning potential ports: {potential_ports}", file=sys.stderr)
    for port in potential_ports:
        print(f"--> Checking port '{port}'...", file=sys.stderr)
        try:
            # Open the serial port with a timeout.
            # The 'with' statement ensures the port is closed even if errors occur.
            with serial.Serial(port, SERIAL_BAUD_RATE, timeout=3) as ser:
                time.sleep(2)  # Give the Arduino time to reset after connection

                # Read a few lines to find the handshake message.
                # The Arduino sends other debug messages before the final "ready" signal.
                for _ in range(5):
                    line = ser.readline()
                    if ARDUINO_HANDSHAKE_MSG in line:
                        print(f"--> Handshake successful on '{port}'. This is the one.", file=sys.stderr)
                        # We found the correct port, now establish the persistent connection.
                        # We must close the temporary 'with' connection first.
                        _active_port_name = port
                        _serial_connection = serial.Serial(port, SERIAL_BAUD_RATE, timeout=1)
                        return port  # Success
            print(f"--> Handshake failed on '{port}'.", file=sys.stderr)
        except (OSError, serial.SerialException) as e:
            # This can happen if the port is in use or doesn't exist.
            print(f"--> Could not open port '{port}': {e}", file=sys.stderr)
            continue  # Try the next port

    return None  # No suitable port found


def _send_command_to_mcu(command: str) -> Tuple[bool, str]:
    """
    Sends a command string to the connected microcontroller (MCU).

    This function manages the serial connection, ensuring it's open before
    sending a command. It uses a thread lock to prevent multiple requests
    from trying to use the serial port at the same time. It waits for the
    MCU's 'OK' response to confirm the command was received.

    Args:
        command: The command string to send (e.g., "7.0,-3.0\n").

    Returns:
        A tuple of (success_boolean, message_string).
    """
    global _serial_connection, _active_port_name

    with _serial_lock:  # Ensure only one thread can access the serial port at a time
        # --- Connection Management ---
        # If there is no active connection, try to establish one.
        if _serial_connection is None or not _serial_connection.is_open:
            print("FMC HOOK: No active serial connection. Attempting to scan and connect...", file=sys.stderr)
            if _scan_and_verify_ports() is None:
                return False, "Connection failed: No verified Arduino device found."

        # --- Command Sending and Response Handling ---
        try:
            if not _serial_connection.is_open:
                # This can happen if the device was unplugged.
                return False, "Connection lost: Serial port is not open."

            # Flush any old data in the buffers
            _serial_connection.reset_input_buffer()
            _serial_connection.reset_output_buffer()

            # Send the command, encoded as bytes.
            _serial_connection.write(command.encode('utf-8'))

            # Wait for the confirmation "OK\n" from the Arduino.
            response = _serial_connection.readline().decode('utf-8').strip()

            if response == "OK":
                return True, f"MCU acknowledged command on port '{_active_port_name}'."
            else:
                # This could happen if the Arduino is busy or sends an error.
                return False, f"MCU communication error: Expected 'OK', got '{response}'."

        except serial.SerialException as e:
            print(f"FMC HOOK: Serial communication error: {e}. Closing connection.", file=sys.stderr)
            # If a major error occurs, close the connection so it can be re-established next time.
            if _serial_connection:
                _serial_connection.close()
            _serial_connection = None
            _active_port_name = None
            return False, f"Serial communication failed: {e}"
        except Exception as e:
            # Catch any other unexpected errors.
            return False, f"An unexpected error occurred during serial communication: {e}"

def handler(match: Match[str], user_input: str, session_id: str) -> Optional[str]:
    SUCCESS_PREFIX = "This is what I get or the result of my calculation: "
    ERROR_PREFIX = "I think Im lost can you repeat that again to me?"

    if not _PYSERIAL_AVAILABLE:
        return f"{ERROR_PREFIX} The FMC servo hook is disabled because the 'pyserial' library is not available."

    try:
        # --- Use named groups for robust extraction ---
        command_verb = match.group("verb")      # e.g., "set"
        command_str = match.group("command")    # e.g., "yaw"
        value_str = match.group("value")        # e.g., "-45.5" or None

        command = command_str.lower()

        # --- Normalize commands (map synonyms to a base command) ---
        if command in ["rudder"]: command = "yaw"
        if command in ["elevator"]: command = "pitch"
        if command in ["aileron"]: command = "roll"
        if command in ["center", "zero"]: command = "reset"
        if command in ["status", "query"]: command = "help"

        # Handle help commands
        if command in ["help", "me"]:
            return (f"{SUCCESS_PREFIX} Flight Management Computer (FMC) ready. "
                    "Available commands: YAW, PITCH, ROLL, TRIM, RESET. "
                    "Example: 'set pitch to -15' or 'trim roll 5'.")

        # Handle reset command
        if command == "reset":
            success, message = _send_command_to_mcu("0.0,0.0\n")
            if success: return f"{SUCCESS_PREFIX} FMC servos centered. {message}"
            else: return f"{ERROR_PREFIX} Failed to center servos. {message}"

        # For all other commands, a value is now required.
        if value_str is None:
            return f"{ERROR_PREFIX} Command '{command}' requires a numerical value (e.g., 'set {command} 10')."
        try:
            value = float(value_str)
        except ValueError:
            return f"{ERROR_PREFIX} Invalid number '{value_str}' provided for command '{command}'."

        # --- Map command to calculation code ---
        # We can now handle 'trim' as a variation of 'roll' for calculation purposes.
        command_map = {'yaw': 0, 'pitch': 1, 'roll': 2, 'trim': 2}
        command_code = command_map.get(command, -1)

        if command_code == -1:
            return f"{ERROR_PREFIX} Unrecognized control command '{command}'."

        # Calculation logic remains the same
        gyro_val, inertia_val = 0.0, 0.0
        if IS_JIT_COMPILED:
            try:
                gyro_val, inertia_val = _calculate_servo_outputs_numba(command_code, value)
            except Exception as e:
                print(f"FMC HOOK: Numba execution error: {e}. Falling back to Python.", file=sys.stderr)
                gyro_val, inertia_val = _calculate_servo_outputs_python(command_code, value)
        else:
            gyro_val, inertia_val = _calculate_servo_outputs_python(command_code, value)

        # Sending command to MCU remains the same
        mcu_command = f"{gyro_val},{inertia_val}\n"
        success, message = _send_command_to_mcu(mcu_command)

        if success:
            return f"{SUCCESS_PREFIX} Command '{command_verb} {command} {value}' acknowledged. {message}"
        else:
            return f"{ERROR_PREFIX} Failed to execute command '{command} {value}'. {message}"

    except Exception as e:
        return f"{ERROR_PREFIX} A critical error occurred in the FMC servo hook: {e}"
