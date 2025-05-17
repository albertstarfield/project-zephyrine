# ./StellaIcarus/basic_math_hook.py
import re
import sys
import time
from typing import Optional, Match, Tuple

# --- Numba Import & Fallback ---
_NUMBA_JIT_SUCCESSFUL_FIRST_TIME = False
_NUMBA_INIT_ERROR_MSG = None

try:
    from numba import njit, NumbaWarning
    import warnings

    warnings.simplefilter('ignore', category=NumbaWarning)  # Broader Numba warning suppression


    @njit(cache=True)  # Try to cache the real one
    def _perform_calc_numba_internal(n1: float, op_code: int, n2: float) -> Tuple[float, int]:
        res: float = 0.0;
        err: int = 0
        if op_code == 0:
            res = n1 + n2  # '+'
        elif op_code == 1:
            res = n1 - n2  # '-'
        elif op_code == 2:
            res = n1 * n2  # '*' or 'x' or 'X'
        elif op_code == 3:  # '/'
            if n2 == 0.0:
                err = 1  # Div by zero
            else:
                res = n1 / n2
        else:
            err = 2  # Unknown op
        return res, err


    # Perform a quick self-test at import time to see if JIT is likely to work
    # This doesn't guarantee all calls will work, but catches basic environment issues.
    _test_res, _test_err = _perform_calc_numba_internal(1.0, 0, 2.0)  # Test '+'
    if _test_err == 0 and _test_res == 3.0:
        _NUMBA_JIT_SUCCESSFUL_FIRST_TIME = True
        # print("StellaMathHook: Numba JIT self-test for _perform_calc_numba_internal successful.", file=sys.stderr)
    else:
        _NUMBA_INIT_ERROR_MSG = f"Numba self-test failed (err:{_test_err}, res:{_test_res}). Defaulting to Python math."
        # print(f"StellaMathHook: {_NUMBA_INIT_ERROR_MSG}", file=sys.stderr)
        _NUMBA_JIT_SUCCESSFUL_FIRST_TIME = False

except ImportError as e_imp:
    _NUMBA_INIT_ERROR_MSG = f"Numba library not found ({e_imp}). Using Python math."
    # print(f"StellaMathHook: {_NUMBA_INIT_ERROR_MSG}", file=sys.stderr)
    _NUMBA_JIT_SUCCESSFUL_FIRST_TIME = False
except Exception as e_jit_init:  # Catch other Numba errors during initial JIT attempt
    _NUMBA_INIT_ERROR_MSG = f"Numba JIT init error for _perform_calc_numba_internal ({e_jit_init}). Using Python math."
    # print(f"StellaMathHook: {_NUMBA_INIT_ERROR_MSG}", file=sys.stderr)
    _NUMBA_JIT_SUCCESSFUL_FIRST_TIME = False


# --- Plain Python Fallback Calculation ---
def _perform_calc_python(n1: float, op_char: str, n2: float) -> Tuple[float, int]:
    res: float = 0.0;
    err: int = 0
    if op_char == '+':
        res = n1 + n2
    elif op_char == '-':
        res = n1 - n2
    elif op_char.lower() == 'x' or op_char == '*':
        res = n1 * n2
    elif op_char == '/':
        if n2 == 0.0:
            err = 1  # Div by zero
        else:
            res = n1 / n2
    else:
        err = 2  # Unknown op
    return res, err


# --- Regex Pattern ---
PATTERN = re.compile(
    r"^\s*(?:calc(?:ulate)?|compute|eval(?:uate)?)\s+"
    r"(-?\d+(?:\.\d+)?)\s*"  # num1 (group 1)
    r"([+\-*/xX])\s*"  # operator (group 2)
    r"(-?\d+(?:\.\d+)?)\s*$",  # num2 (group 3)
    re.IGNORECASE
)


def handler(match: Match[str], user_input: str, session_id: str) -> Optional[str]:
    SUCCESS_PREFIX = "StellaCalc: "
    ERROR_PREFIX = "StellaError: "

    try:
        num1_str, operator_str, num2_str = match.groups()

        try:
            num1 = float(num1_str)
            num2 = float(num2_str)
        except ValueError:
            return f"{ERROR_PREFIX}Invalid number format in '{user_input}'. Use valid numbers."

        calc_result: float = 0.0
        error_code: int = 0  # 0=success, 1=div_zero, 2=unknown_op, 3=numba_exec_error

        if _NUMBA_JIT_SUCCESSFUL_FIRST_TIME:
            op_code_map = {'+': 0, '-': 1, '*': 2, 'x': 2, 'X': 2, '/': 3}
            op_code_for_numba = op_code_map.get(operator_str, -1)  # Default to an invalid code

            if op_code_for_numba != -1:
                try:
                    calc_result, error_code = _perform_calc_numba_internal(num1, op_code_for_numba, num2)
                    if error_code != 0:  # Numba function signaled an error
                        pass  # Handled below by error_code checks
                except Exception as e_numba_exec:  # Catch runtime errors from Numba call itself
                    # This is where "No module named '<dynamic>'" or similar might be caught
                    # print(f"StellaMathHook: Numba execution error for '{user_input}': {e_numba_exec}", file=sys.stderr)
                    error_code = 3  # Custom error code for Numba execution failure
            else:
                error_code = 2  # Unknown operator for Numba path (should be caught by regex)

        if not _NUMBA_JIT_SUCCESSFUL_FIRST_TIME or error_code == 3:  # Fallback to Python if Numba init failed or Numba exec failed
            if error_code == 3 and _NUMBA_JIT_SUCCESSFUL_FIRST_TIME:  # Log if Numba exec failed after successful init
                print(f"StellaMathHook: Numba JIT was OK, but execution failed for '{user_input}'. Falling to Python.",
                      file=sys.stderr)
            elif not _NUMBA_JIT_SUCCESSFUL_FIRST_TIME:
                print(
                    f"StellaMathHook: Numba JIT not active/failed init ({_NUMBA_INIT_ERROR_MSG}). Using Python for '{user_input}'.",
                    file=sys.stderr)

            calc_result, error_code = _perform_calc_python(num1, operator_str, num2)

        # --- Handle result and error_code ---
        if error_code == 1:
            return f"{ERROR_PREFIX}Division by zero is mathematically undefined."
        elif error_code == 2:
            return f"{ERROR_PREFIX}Operator '{operator_str}' is not recognized."
        elif error_code == 0:  # Success
            if calc_result == int(calc_result):
                return f"{SUCCESS_PREFIX}{int(calc_result)}"
            else:
                return f"{SUCCESS_PREFIX}{calc_result:.6g}"
        else:  # Includes error_code 3 from Numba failure or other unexpected codes
            return f"{ERROR_PREFIX}Internal calculation processing error (code {error_code}). Input: '{user_input}'"

    except Exception as e_handler_main:
        # print(f"StellaMathHook: Main handler Python error for '{user_input}': {e_handler_main}", file=sys.stderr)
        # This catches errors in the Python logic of the handler itself (e.g., regex group access)
        return f"{ERROR_PREFIX}Could not process calculation input '{user_input}'. Hook setup error: {type(e_handler_main).__name__}."


# --- Self-test block (as before) ---
if __name__ == "__main__":
    # ... (your existing test cases) ...
    # Add print for _NUMBA_JIT_SUCCESSFUL_FIRST_TIME and _NUMBA_INIT_ERROR_MSG here
    print(f"Numba JIT initially successful: {_NUMBA_JIT_SUCCESSFUL_FIRST_TIME}")
    if _NUMBA_INIT_ERROR_MSG:
        print(f"Numba Init Error Message: {_NUMBA_INIT_ERROR_MSG}")

    # ... (rest of your test cases from the previous full hook code)
    test_cases = [
        ("calc 1 + 2", "StellaCalc: 3"), ("calculate 10 - 5.5", "StellaCalc: 4.5"),
        ("  compute   -3 * 4 ", "StellaCalc: -12"), ("eval 100 / 4", "StellaCalc: 25"),
        ("evaluate 10 / 0", "StellaError: Division by zero is mathematically undefined."),
        ("calc 10 x 3", "StellaCalc: 30"), ("calc 1 / 3", "StellaCalc: 0.333333"),
    ]
    for i, (test_input, expected_output) in enumerate(test_cases):
        print(f"\n--- Test Case {i + 1} ---");
        print(f"Input: '{test_input}'")
        match_obj = PATTERN.match(test_input)
        if match_obj:
            print(f"Regex Matched. Groups: {match_obj.groups()}")
            actual_output = handler(match_obj, test_input, "test_session_001")
            print(f"Handler Output: '{actual_output}'")
            if actual_output == expected_output:
                print("Result: PASS")
            else:
                print(f"Result: FAIL (Expected: '{expected_output}')")
        else:
            print("Regex Did Not Match.")
            if expected_output is None:
                print("Result: PASS (Correctly not matched)")
            else:
                print(f"Result: FAIL (Expected a match and output: '{expected_output}')")