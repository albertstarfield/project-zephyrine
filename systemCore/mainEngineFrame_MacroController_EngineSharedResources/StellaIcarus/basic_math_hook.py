# ./StellaIcarus/basic_math_hook.py
#Use this as an example to make other StellaIcarus
import ctypes
import os
import platform
import re
import subprocess
import sys
from typing import Optional, Match

# ==========================================
# CONFIGURATION & GLOBAL STATE
# ==========================================
LIB_EXT = ".dll" if platform.system() == "Windows" else ".so"
LIB_NAME = f"stella_math_core{LIB_EXT}"
SOURCE_FILE = "stella_math_core.cpp"

_ENGINE_MODE = "PYTHON" # Default fallback
_C_LIB = None
_NUMBA_FUNC = None

# ==========================================
# TIER 1: C++ NATIVE ENGINE (The "Trickshot")
# ==========================================
cpp_source = """
extern "C" {
    // 0=Add, 1=Sub, 2=Mul, 3=Div
    void fast_calc(double a, int op, double b, double* res, int* err) {
        *err = 0;
        switch(op) {
            case 0: *res = a + b; break;
            case 1: *res = a - b; break;
            case 2: *res = a * b; break;
            case 3: 
                if (b == 0.0) *err = 1; 
                else *res = a / b; 
                break;
            default: *err = 2;
        }
    }
    int health() { return 777; }
}
"""

def _setup_cpp():
    global _C_LIB, _ENGINE_MODE
    try:
        lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), LIB_NAME))
        src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), SOURCE_FILE))

        # Compile if missing
        if not os.path.exists(lib_path):
            with open(src_path, "w") as f: f.write(cpp_source)
            flags = ["g++", "-O3", "-shared", "-fPIC", "-march=native", src_path, "-o", lib_path]
            subprocess.check_call(flags, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Load
        lib = ctypes.CDLL(lib_path)
        lib.fast_calc.argtypes = [ctypes.c_double, ctypes.c_int, ctypes.c_double, 
                                  ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int)]
        
        if lib.health() == 777:
            _C_LIB = lib
            _ENGINE_MODE = "CPP_O3"
            print("✅ StellaMath: C++ Engine Active (Tier 1)", file=sys.stderr)
            return True
    except Exception:
        return False
    return False

# ==========================================
# TIER 2: NUMBA JIT (The "Speedster")
# ==========================================
def _setup_numba():
    global _NUMBA_FUNC, _ENGINE_MODE
    try:
        from numba import njit
        
        # We define this inside to ensure Numba is available
        @njit(cache=True, fastmath=True)
        def jit_calc(a, op, b):
            # Returns (result, error_code)
            if op == 0: return a + b, 0
            if op == 1: return a - b, 0
            if op == 2: return a * b, 0
            if op == 3:
                if b == 0.0: return 0.0, 1
                return a / b, 0
            return 0.0, 2

        # Warmup compilation
        jit_calc(1.0, 0, 1.0)
        
        _NUMBA_FUNC = jit_calc
        _ENGINE_MODE = "NUMBA_JIT"
        print("✅ StellaMath: Numba Engine Active (Tier 2)", file=sys.stderr)
        return True
    except ImportError:
        pass
    except Exception as e:
        print(f"⚠️ StellaMath: Numba failed ({e})", file=sys.stderr)
    return False

# ==========================================
# INITIALIZATION SEQUENCE
# ==========================================
# Try C++ first. If fails, try Numba. If fails, stay Python.
if not _setup_cpp():
    if not _setup_numba():
        print("⚠️ StellaMath: Using Pure Python (Tier 3)", file=sys.stderr)


# ==========================================
# TIER 3: PURE PYTHON (The Fallback)
# ==========================================
def _calc_python(a, op, b):
    # 0=Add, 1=Sub, 2=Mul, 3=Div
    if op == 0: return a + b, 0
    if op == 1: return a - b, 0
    if op == 2: return a * b, 0
    if op == 3:
        if b == 0.0: return 0.0, 1
        return a / b, 0
    return 0.0, 2


# ==========================================
# FLOATING REGEX (Context Aware)
# ==========================================
PATTERN = re.compile(
    r"(?is)"
    r".*?"  # Eat prefix (e.g. "Hey Zephy, ")
    r"(?P<n1>-?\d+(?:\.\d+)?)"
    r"\s*"
    r"(?P<op>" 
        r"[\+\-\*\/xX]" 
        r"|"
        r"(?:plus|add|added\s+to|minus|subtract|subtracted\s+by|times|multiplied\s+by|divided\s+by)"
    r")"
    r"\s*"
    r"(?P<n2>-?\d+(?:\.\d+)?)"
    r".*"   # Eat suffix (e.g. ". Only number.")
)

# ==========================================
# HANDLER
# ==========================================
def handler(match: Match[str], user_input: str, session_id: str) -> Optional[str]:
    # 1. Parse & Normalize
    try:
        n1 = float(match.group("n1"))
        n2 = float(match.group("n2"))
        raw_op = match.group("op").strip().lower()
        
        # Map everything to Integers [0,1,2,3] for C++/Numba compatibility
        op_code = -1
        if raw_op in ['+', 'plus', 'add', 'added to']: op_code = 0
        elif raw_op in ['-', 'minus', 'subtract', 'subtracted by']: op_code = 1
        elif raw_op in ['*', 'x', 'times', 'multiplied by']: op_code = 2
        elif raw_op in ['/', 'divided by']: op_code = 3
        
        if op_code == -1: return None # Regex shouldn't allow this, but safety first

    except ValueError:
        return "I couldn't parse those numbers."

    # 2. Execute based on Mode
    res = 0.0
    err = 0
    
    if _ENGINE_MODE == "CPP_O3":
        c_res = ctypes.c_double()
        c_err = ctypes.c_int()
        _C_LIB.fast_calc(n1, op_code, n2, ctypes.byref(c_res), ctypes.byref(c_err))
        res, err = c_res.value, c_err.value
        
    elif _ENGINE_MODE == "NUMBA_JIT":
        res, err = _NUMBA_FUNC(n1, op_code, n2)
        
    else: # Python
        res, err = _calc_python(n1, op_code, n2)

    # 3. Format
    if err == 1: return "I can't divide by zero."
    if err == 2: return "Unknown calculation error."
    
    # Clean output (5.0 -> 5)
    final_val = int(res) if res.is_integer() else res
    
    # We return the simple prefix you requested
    return f"This is what I get or the result of my calculation: {final_val}"

# Self Test
if __name__ == "__main__":
    print(f"Current Engine: {_ENGINE_MODE}")
    print(handler(PATTERN.match("Calculate 10 + 20."), "", ""))