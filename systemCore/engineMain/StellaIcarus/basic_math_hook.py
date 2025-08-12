# ./StellaIcarus/basic_math_hook.py
import re
import sys
import time
from typing import Optional, Match, Tuple

# --- Numba JIT Compilation & Self-Test ---
# This block attempts to import Numba, JIT-compile the core logic, and
# run a self-test to verify its functionality at startup.

_NUMBA_AVAILABLE = False
_NUMBA_JIT_ERROR = None

try:
    from numba import njit


    # The core, high-performance calculation logic.
    # The @njit decorator compiles this function to machine code for maximum speed.
    @njit(cache=True, fastmath=True)
    def _perform_calc_numba(n1: float, op_code: int, n2: float) -> Tuple[float, int]:
        """
        Performs the basic arithmetic operation using Numba's JIT compiler.
        Returns a tuple of (result, error_code).
        Error codes: 0 = success, 1 = division by zero, 2 = unknown operator.
        """
        result: float = 0.0
        error_code: int = 0

        if op_code == 0:  # Addition
            result = n1 + n2
        elif op_code == 1:  # Subtraction
            result = n1 - n2
        elif op_code == 2:  # Multiplication
            result = n1 * n2
        elif op_code == 3:  # Division
            if n2 == 0.0:
                error_code = 1  # Division by zero
            else:
                result = n1 / n2
        else:
            error_code = 2  # Unknown operator

        return result, error_code


    # --- Startup Self-Test ---
    # We run a simple calculation to confirm the JIT-compiled function works as expected.
    # If this fails, the hook will gracefully fall back to pure Python.
    _test_result, _test_error = _perform_calc_numba(10.0, 0, 5.0)  # Test 10 + 5
    if _test_error == 0 and _test_result == 15.0:
        _NUMBA_AVAILABLE = True
        print("✅ StellaMathHook: Numba JIT self-test PASSED. High-performance mode enabled.", file=sys.stderr)
    else:
        _NUMBA_JIT_ERROR = f"Numba self-test FAILED (err:{_test_error}, res:{_test_result})."
        _NUMBA_AVAILABLE = False

except ImportError:
    _NUMBA_JIT_ERROR = "Numba library not found."
    _NUMBA_AVAILABLE = False
except Exception as e:
    _NUMBA_JIT_ERROR = f"Numba JIT compilation failed: {e}"
    _NUMBA_AVAILABLE = False

IS_JIT_COMPILED = _NUMBA_AVAILABLE

if not _NUMBA_AVAILABLE:
    print(f"⚠️ StellaMathHook: Numba JIT disabled. Reason: {_NUMBA_JIT_ERROR}. Falling back to standard Python.",
          file=sys.stderr)


# --- Pure Python Fallback Logic ---
# This version is used if Numba is unavailable or fails at runtime.
def _perform_calc_python(n1: float, op_char: str, n2: float) -> Tuple[float, int]:
    """
    Pure Python fallback for the arithmetic operation.
    Returns a tuple of (result, error_code).
    """
    result: float = 0.0
    error_code: int = 0

    op = op_char.lower()
    if op == '+':
        result = n1 + n2
    elif op == '-':
        result = n1 - n2
    elif op in ('*', 'x'):
        result = n1 * n2
    elif op == '/':
        if n2 == 0.0:
            error_code = 1
        else:
            result = n1 / n2
    else:
        error_code = 2

    return result, error_code


# --- Regex Pattern ---
# This compiled regex pattern is the "gate" for this hook.
# It captures simple arithmetic expressions like "calc 10.5 * -2".
PATTERN = re.compile(
    r"^\s*"  # Start of the string, optional whitespace

    # --- Trigger Phrases (Group 1: Optional) ---
    # Matches common command words or questions.
    r"(?:"  # Start of non-capturing group for trigger words
    r"(?:what\s+is|what's|tell\s+me)\s+"  # "what is", "what's", "tell me"
    r"|(?:"  # Nested group for calculation verbs
    r"calc(?:ulate)?|compute|eval(?:uate)?|solve"
    r")\s+"
    r")?"  # End of non-capturing group, make the whole phrase optional

    # --- The First Number (Group 'num1') ---
    r"(?P<num1>-?\d+(?:\.\d+)?)\s*"  # A signed integer or float

    # --- The Operator Phrase (Group 'operator') ---
    r"(?:"  # Start of non-capturing group for operator words/symbols
    r"(?P<operator>[+\-*/xX])"  # Direct symbols: +, -, *, /, x, X
    r"|"  # OR
    r"(?:plus|add|added\s+to)\s+"  # Words for addition
    r"|"
    r"(?:minus|subtract|subtracted\s+by)\s+"  # Words for subtraction
    r"|"

    r"(?:times|multiplied\s+by)\s+"  # Words for multiplication
    r"|"

    r"(?:divided\s+by)\s+"  # Words for division
    r")\s*"  # End of operator group

    # --- The Second Number (Group 'num2') ---
    r"(?P<num2>-?\d+(?:\.\d+)?)"  # A signed integer or float

    # --- Optional Ending ---
    r"\s*\??\s*$",  # Optional whitespace, an optional question mark, and end of string

    re.IGNORECASE
)


# --- Main Handler Function ---
# This is the entry point called by the AI core when the PATTERN matches.
def handler(match: Match[str], user_input: str, session_id: str) -> Optional[str]:
    """
    Handles the matched user input, normalizes the detected operator (symbol or word),
    performs the calculation using the fastest available method (Numba JIT or Python),
    and returns a formatted string response.
    """
    SUCCESS_PREFIX = "This is what I get or the result of my calculation: "
    ERROR_PREFIX = "I think Im lost can you repeat that again to me?"

    try:
        # --- Step 1: Extract captured data from the regex match ---
        # The regex uses named groups, so we access them by name for clarity.
        num1_str = match.group("num1")
        operator_symbol = match.group("operator")  # This might be None if a word was used
        num2_str = match.group("num2")

        # --- Step 2: Normalize the operator ---
        # We need a single character ('+', '-', '*', '/') to pass to the calculation functions.
        operator_str: Optional[str] = None

        if operator_symbol:
            # If a symbol like '+' or 'x' was directly matched, use it.
            operator_str = operator_symbol
        else:
            # If no symbol was matched, it means a word ("plus", "times", etc.) was used.
            # We check the full text of the match to determine the operator.
            full_match_text = match.group(0).lower()
            if "plus" in full_match_text or "add" in full_match_text:
                operator_str = '+'
            elif "minus" in full_match_text or "subtract" in full_match_text:
                operator_str = '-'
            elif "times" in full_match_text or "multiplied" in full_match_text:
                operator_str = '*'
            elif "divided" in full_match_text:
                operator_str = '/'

        if not operator_str:
            # This is a safety net; the regex should not allow a match without an operator.
            return f"{ERROR_PREFIX} I understood the numbers but not the operation."

        # --- Step 3: Convert numbers and handle potential errors ---
        num1 = float(num1_str)
        num2 = float(num2_str)

    except (ValueError, IndexError):
        # This catches errors if regex groups are missing or numbers are invalid.
        return f"{ERROR_PREFIX} I couldn't understand the numbers in your request."

    # --- Step 4: Perform the calculation using the best available method ---
    calc_result: float = 0.0
    error_code: int = 0  # 0=success, 1=div_zero, 2=unknown_op, 3=numba_runtime_error

    # Primary Path: Use Numba JIT if it's available and passed its self-test.
    if IS_JIT_COMPILED:
        op_code_map = {'+': 0, '-': 1, '*': 2, 'x': 2, 'X': 2, '/': 3}
        op_code = op_code_map.get(operator_str.lower(), -1)

        try:
            # Execute the high-speed, JIT-compiled function
            calc_result, error_code = _perform_calc_numba(num1, op_code, num2)
        except Exception as e_numba_runtime:
            # Critical Fallback: If the JIT'd function itself crashes at runtime
            print(
                f"CRITICAL: StellaMathHook Numba runtime error for '{user_input}': {e_numba_runtime}. Falling back to Python.",
                file=sys.stderr)
            error_code = 3  # Signal a Numba runtime failure

    # Fallback Path: Use Pure Python if Numba is not available or if it just crashed.
    if not IS_JIT_COMPILED or error_code == 3:
        calc_result, error_code = _perform_calc_python(num1, operator_str, num2)

    # --- Step 5: Format the final response based on the calculation outcome ---
    if error_code == 0:  # Success
        # Return as an integer if it's a whole number for cleaner output.
        if calc_result == int(calc_result):
            return f"{SUCCESS_PREFIX}{int(calc_result)}"
        else:
            # Use .6g formatting to remove unnecessary trailing zeros from floats.
            return f"{SUCCESS_PREFIX}{calc_result:.6g}"

    elif error_code == 1:
        return f"{ERROR_PREFIX} I can't divide by zero, that's a one-way trip to a black hole."
    elif error_code == 2:
        # This error is now less likely due to the robust regex, but it's a good safeguard.
        return f"{ERROR_PREFIX} I don't recognize the operator '{operator_str}'."
    else:
        # This catches the Numba runtime error (code 3) or any other unexpected codes.
        return f"{ERROR_PREFIX} I encountered an internal calculation error while processing '{user_input}'."


# --- Self-test block for direct execution ---
if __name__ == "__main__":
    print("--- Running StellaIcarus Math Hook Self-Test ---")

    test_cases = [
        ("calc 1 + 2", "This is what I get or the result of my calculation: 3"),
        ("calculate 10 - 5.5", "This is what I get or the result of my calculation: 4.5"),
        ("  compute   -3 * 4 ", "This is what I get or the result of my calculation: -12"),
        ("eval 100 / 4", "This is what I get or the result of my calculation: 25"),
        ("evaluate 10 / 0",
         "I think Im lost can you repeat that again to me? I can't divide by zero, that's a one-way trip to a black hole."),
        ("calc 10 x 3", "This is what I get or the result of my calculation: 30"),
        ("calc 1 / 3", "This is what I get or the result of my calculation: 0.333333"),
        ("calc nine + 1", None),  # Should not match
    ]

    for i, (test_input, expected_output) in enumerate(test_cases):
        print(f"\n--- Test Case {i + 1} ---")
        print(f"Input: '{test_input}'")
        match_obj = PATTERN.match(test_input)

        if match_obj:
            print(f"Regex Matched. Groups: {match_obj.groupdict()}")
            actual_output = handler(match_obj, test_input, "test_session_001")
            print(f"Handler Output: '{actual_output}'")
            if actual_output == expected_output:
                print("✅ Result: PASS")
            else:
                print(f"❌ Result: FAIL (Expected: '{expected_output}')")
        else:
            print("Regex Did Not Match.")
            if expected_output is None:
                print("✅ Result: PASS (Correctly not matched)")
            else:
                print(f"❌ Result: FAIL (Expected a match)")