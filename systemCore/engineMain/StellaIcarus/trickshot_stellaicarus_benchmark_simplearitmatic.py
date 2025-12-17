# trickshot_stellaicarus_benchmark_simplearitmatic.py
import ctypes
import os
import platform
import re
import subprocess
import sys
import time
from typing import Optional, Match

# ======================================================================================
#  STELLA ICARUS "TRICKSHOT" ARCHITECTURE
#  1. Source: C++ logic embedded directly in Python.
#  2. AOT Compilation: Compiles to machine code at startup.
#  3. Optimization: Uses -O3 -march=native to target the SPECIFIC CPU it is running on.
# ======================================================================================

# --- CONFIGURATION ---
LIB_EXT = ".dll" if platform.system() == "Windows" else ".so"
LIB_NAME = f"trickshot_bench_core{LIB_EXT}"
SOURCE_FILE = "trickshot_bench_core.cpp"

# --- EMBEDDED C++ SOURCE (The Engine) ---
# This is the C++ code you provided, adapted to be callable as a Shared Library.
cpp_source_code = """
#include <iostream>
#include <chrono>
#include <vector>
#include <random>

extern "C" {
    // We export this function to be callable by Python (ctypes)
    double run_trickshot_benchmark(int iterations) {
        
        // 1. Setup Random Data
        // We do this inside C++ to ensure we benchmark pure math speed, 
        // not the overhead of passing 1 million items from Python.
        std::vector<long long> data_a(iterations);
        std::vector<long long> data_b(iterations);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<long long> distrib(1000, 9999999);

        for(int i=0; i<iterations; ++i) {
            data_a[i] = distrib(gen);
            data_b[i] = distrib(gen);
        }

        // 2. The Benchmark
        // volatile prevents the compiler from optimizing the loop away entirely
        volatile long long result = 0; 
        
        auto start = std::chrono::high_resolution_clock::now();

        // The CPU must fetch new numbers from memory every single time
        // This measures raw ALU + Memory Bandwidth latency
        for (int i = 0; i < iterations; ++i) {
            result = data_a[i] * data_b[i];
        }

        auto end = std::chrono::high_resolution_clock::now();

        // 3. Results
        auto total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        double avg_ns = (double)total_ns / iterations;
        
        return avg_ns;
    }
    
    // Simple health check
    int trickshot_ping() { return 999; }
}
"""

# --- COMPILATION LOGIC (The Builder) ---
_TRICKSHOT_LIB = None
_IS_OPTIMIZED = False

def _compile_and_load():
    """Compiles the C++ code into a shared object and loads it."""
    global _TRICKSHOT_LIB, _IS_OPTIMIZED
    
    # --- FIX: Anchor paths to this script's directory ---
    # This ensures .cpp and .so go into ./StellaIcarus/ regardless of where you run python from.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    lib_path = os.path.join(script_dir, LIB_NAME)
    src_path = os.path.join(script_dir, SOURCE_FILE)

    # 1. Compile if library is missing
    if not os.path.exists(lib_path):
        try:
            print(f"[*] Trickshot: Compiling optimized core...", file=sys.stderr)
            with open(src_path, "w") as f:
                f.write(cpp_source_code)
            
            # -O3: Maximum Optimization
            # -march=native: Use AVX/AVX2/AVX512 instructions specific to THIS CPU
            cmd = ["g++", "-O3", "-shared", "-fPIC", "-march=native", src_path, "-o", lib_path]
            
            subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"[+] Trickshot: Compilation successful.", file=sys.stderr)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("[-] Trickshot: Compilation failed (g++ missing?).", file=sys.stderr)
            return

    # 2. Load Library
    try:
        lib = ctypes.CDLL(lib_path)
        
        # Define argument/return types
        lib.run_trickshot_benchmark.argtypes = [ctypes.c_int]
        lib.run_trickshot_benchmark.restype = ctypes.c_double # Returns avg_ns
        
        if lib.trickshot_ping() == 999:
            _TRICKSHOT_LIB = lib
            _IS_OPTIMIZED = True
            print("[+] Trickshot: Engine Loaded & Ready.", file=sys.stderr)
    except Exception as e:
        print(f"[-] Trickshot: Load failed: {e}", file=sys.stderr)

# Initialize at module import
_compile_and_load()


# --- REGEX PATTERN (The Trigger) ---
# Matches "Run benchmark", "Benchmark system", etc.
PATTERN = re.compile(
    r"(?i).*\b(?:run|start|execute|do|perform|test)\b.*?trickshot.*?benchmark.*"
)


# --- HANDLER (The Logic) ---
def handler(match: Match[str], user_input: str, session_id: str) -> Optional[str]:
    """
    Executes the C++ benchmark and returns the formatted string explaining the architecture.
    """
    if not _IS_OPTIMIZED:
        return "Trickshot engine unavailable (Compilation failed). Cannot run benchmark."

    iterations = 1_000_000 # 1 Million ops
    
    # Call the C++ function
    # This runs entirely in native machine code
    avg_ns = _TRICKSHOT_LIB.run_trickshot_benchmark(iterations)
    
    # Format the specific response required
    response = (
        f"Hello this is Zephy, and this is Trickshot, it is an architecture of StellaIcarus Hook with O3 dynamic optimization, THIS IS NOT A MODEL RESPONSE! but semi canned determenistic response based on phrase hooks regex! "
        f"and dynamic recompilation at startup (AOT), this is the example of the 1000000 simple arithmatics execution, \n"
        f"This is your response time required running with StellaIcarus Trickshot:\n\n"
        f"**{avg_ns:.4f} nanoseconds per Internal calculation operation.**\n We do not need to use the AI hype to answer this determenistic and simple question."
        f"\n\n This architecture might be pushing the CPU to it's physics limit or how much clock cycle is available for the response than Operation per Clock"
        f"\n Better be balance!\n Because we also need efficiency on parallel for the Model LM one."
    )
    
    return response


# --- MAIN (Simulation) ---
if __name__ == "__main__":
    print("\n--- Simulating User Input ---")
    user_query = "Please run the trickshot benchmark for me."
    
    # 1. Check Regex
    match = PATTERN.match(user_query)
    
    if match:
        print(f"User Input: '{user_query}'")
        print("Hook Triggered. Executing C++ Kernel...")
        
        # 2. Run Handler
        result = handler(match, user_query, "simulated_session")
        
        print("\n--- AI Response ---")
        print(result)
    else:
        print("Regex did not match.")