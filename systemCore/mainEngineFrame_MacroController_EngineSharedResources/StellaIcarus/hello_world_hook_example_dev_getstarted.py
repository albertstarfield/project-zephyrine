# ./StellaIcarus/hello_world_hook_example_dev_getstarted.py

"""
STELLA ICARUS HOOK: DEVELOPER GETTING STARTED
=============================================
This is a template for creating high-performance, deterministic extensions ("Hooks") 
for the Zephy/Adelaide AI system using the "Trickshot" Architecture.

DOCUMENTATION REFERENCES:
-------------------------
- Architecture Overview: https://[your-internal-docs]/arch/trickshot
- C++ Embedding Guide:   https://[your-internal-docs]/hooks/cpp-embedding
- Regex Standards:       https://[your-internal-docs]/hooks/regex-best-practices

WHAT IS TRICKSHOT?
------------------
Trickshot is a hybrid architecture that allows Python script hooks to contain embedded C++ code.
1. **Dynamic Recompilation:** On system startup (or first run), the Python script checks for a compiled .so/.dll.
2. **AOT Optimization:** If missing, it invokes the system compiler (g++) with `-O3 -march=native`.
   This optimizes the machine code specifically for the CPU responding to the user.
3. **Zero-Latency:** Once loaded, Python calls the C++ functions via `ctypes` with nanosecond latency.

HOW TO USE THIS TEMPLATE:
1. Copy this file.
2. Rename it to `your_feature_hook.py`.
3. Modify the C++ Source, Regex PATTERN, and handler function.
"""

import ctypes
import os
import platform
import re
import subprocess
import sys
import time
from typing import Optional, Match

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
# Define unique names for your compiled library to avoid collisions with other hooks.
LIB_EXT = ".dll" if platform.system() == "Windows" else ".so"
LIB_NAME = f"hello_trickshot_core{LIB_EXT}"
SOURCE_FILE = "hello_trickshot_core.cpp"

# ==============================================================================
# 2. THE C++ ENGINE (The "Trickshot" Core)
# ==============================================================================
# Write your high-performance logic here. 
# Keep functions `extern "C"` to make them callable from Python.
# Use simple types (int, double, char*) for easiest interfacing.

cpp_source_code = """
extern "C" {
    // A simple function to demonstrate logic offloading.
    // In a real scenario, this could be complex math, encryption, or signal processing.
    // Returns a "magic number" calculation based on the input length.
    
    int compute_magic_hash(int input_val) {
        // Simulation of heavy work optimized by -O3
        // If this were Python, this loop might take ms. In C++, it's instant.
        long long accumulator = input_val;
        
        #pragma GCC ivdep 
        for(int i=0; i<1000; i++) {
            accumulator = (accumulator * 33) ^ i;
        }
        
        return (int)(accumulator % 1000);
    }

    // Health check to ensure the library loaded the correct version.
    int health_check() { return 200; }
}
"""

# ==============================================================================
# 3. DYNAMIC COMPILATION LOADER
# ==============================================================================
_LIB_HANDLE = None
_IS_OPTIMIZED = False

def _compile_and_load():
    """
    Standard Trickshot Loader. 
    Checks for the library, compiles it if missing using host-specific optimizations, 
    and loads it into memory.
    """
    global _LIB_HANDLE, _IS_OPTIMIZED
    
    # Get absolute paths relative to THIS file location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    lib_path = os.path.join(script_dir, LIB_NAME)
    src_path = os.path.join(script_dir, SOURCE_FILE)

    # A. Compilation Step
    if not os.path.exists(lib_path):
        try:
            print(f"[*] Trickshot Dev: Compiling {SOURCE_FILE}...", file=sys.stderr)
            with open(src_path, "w") as f:
                f.write(cpp_source_code)
            
            # Flags explanation:
            # -O3: Aggressive optimization
            # -shared -fPIC: Create a dynamic library
            # -march=native: Generate instructions specific to THIS machine's CPU features (AVX, etc)
            cmd = ["g++", "-O3", "-shared", "-fPIC", "-march=native", src_path, "-o", lib_path]
            
            subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"[+] Trickshot Dev: Compilation successful.", file=sys.stderr)
        except Exception as e:
            print(f"[-] Trickshot Dev: Compilation failed: {e}. Logic will fail.", file=sys.stderr)
            return

    # B. Loading Step
    try:
        lib = ctypes.CDLL(lib_path)
        
        # Define Argument/Return types for safety
        lib.compute_magic_hash.argtypes = [ctypes.c_int]
        lib.compute_magic_hash.restype = ctypes.c_int
        
        if lib.health_check() == 200:
            _LIB_HANDLE = lib
            _IS_OPTIMIZED = True
            # print("[+] Trickshot Dev: Library loaded.", file=sys.stderr) # Uncomment for debug
    except Exception as e:
        print(f"[-] Trickshot Dev: Load failed: {e}", file=sys.stderr)

# Initialize on module import
_compile_and_load()


# ==============================================================================
# 4. REGEX TRIGGER (The Gatekeeper)
# ==============================================================================
# Define the pattern that activates this hook.
# Use (?is) for case-insensitivity and dot-matching-newlines.
# Use non-greedy matching (.*?) for prefixes/suffixes.

PATTERN = re.compile(
    r"(?is)"
    r".*?"                          # Consume conversational prefix
    r"\b(?:test|try|run)\s+"        # Action verbs
    r"trickshot\s+hello\s+world"    # Specific trigger phrase
    r"(?:\s+(?P<val>\d+))?"         # Optional capture group: a number
    r".*"                           # Consume suffix
)


# ==============================================================================
# 5. THE HANDLER (The Bridge)
# ==============================================================================
def handler(match: Match[str], user_input: str, session_id: str) -> Optional[str]:
    """
    Called by the AI Core when PATTERN matches the user input.
    
    Args:
        match: The RegEx match object (contains captured groups).
        user_input: The raw string the user typed.
        session_id: Context ID for logging.
        
    Returns:
        str: The response to send to the user (bypassing the LLM).
        None: If something went wrong, return None to fall back to the LLM.
    """
    
    # 1. Parse Input
    # Extract the number captured in the regex group 'val'. Default to 42 if not found.
    input_val_str = match.group("val")
    input_val = int(input_val_str) if input_val_str else 42

    # 2. Check Engine Status
    if not _IS_OPTIMIZED:
        return "Hello World! (Note: C++ Trickshot engine is offline/compilation failed)."

    # 3. Execution Timing
    start_t = time.perf_counter()
    
    # --- CALL C++ FUNCTION ---
    # This executes in nanoseconds
    magic_result = _LIB_HANDLE.compute_magic_hash(input_val)
    # -------------------------
    
    end_t = time.perf_counter()
    duration_ns = (end_t - start_t) * 1e9

    # 4. Format Response
    response = (
        f"Hello World from the Stella Icarus Trickshot Hook!\n\n"
        f"I received your input value: **{input_val}**\n"
        f"I processed it through the C++ O3 Engine.\n"
        f"**Magic Hash Result:** {magic_result}\n"
        f"**Compute Latency:** {duration_ns:.2f} ns"
    )
    
    return response


# ==============================================================================
# 6. SELF-TEST (Developer Verification)
# ==============================================================================
if __name__ == "__main__":
    print("\n--- Stella Icarus Hook Developer Test ---")
    
    test_inputs = [
        "Please run trickshot hello world",
        "Test trickshot hello world 12345",
        "Can you run trickshot hello world 999 for me?",
        "Ignore this line"
    ]
    
    for text in test_inputs:
        print(f"\nUser Input: '{text}'")
        m = PATTERN.match(text)
        
        if m:
            print(">> Regex Matched!")
            print(f">> Captures: {m.groupdict()}")
            result = handler(m, text, "dev_test_session")
            print(f">> Result:\n{result}")
        else:
            print(">> No Match (LLM would handle this normally)")