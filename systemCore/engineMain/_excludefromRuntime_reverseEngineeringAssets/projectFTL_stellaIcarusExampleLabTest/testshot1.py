import ctypes
import os
import subprocess
import time
import random
import sys
import platform



#Project FTL Trickshot1


# --- CONFIGURATION ---
LIB_NAME = "libfastvector.so"
SOURCE_FILE = "fastvector.cpp"
BATCH_SIZE = 1_000_000 
LATENCY_ITERS = 100_000

# --- STEP 1: EMBEDDED C++ SOURCE CODE ---
cpp_source_code = """
#include <iostream>

extern "C" {
    // 1. Batch Function (The Heavy Lifter)
    void vector_multiply(long long* a, long long* b, long long* result, int size) {
        // 'ivdep' tells compiler to ignore vector dependencies
        #pragma GCC ivdep
        for (int i = 0; i < size; i++) {
            result[i] = a[i] * b[i];
        }
    }

    // 2. Single Function (The Lightweight)
    long long single_multiply(long long a, long long b) {
        return a * b;
    }
    
    // 3. Health Check
    int health_check() {
        return 42;
    }
}
"""

def get_compile_flags():
    """Detects CPU and returns optimal flags."""
    system = platform.system()
    machine = platform.machine()
    
    # Base flags: Max optimization, shared lib, position independent
    flags = ["g++", "-O3", "-shared", "-fPIC"]
    
    # --- AUTO-DETECT CPU ARCHITECTURE ---
    # -march=native tells the compiler: "Optimize exactly for THIS cpu"
    # It enables AVX2, AVX-512, NEON, etc. automatically.
    flags.append("-march=native")

    # Extra specific tuning if needed (usually -march=native covers it)
    print(f"[*] Detected Host: {system} ({machine})")
    
    return flags

def compile_cpp():
    print(f"[*] Compiling {SOURCE_FILE} for this specific CPU...")
    
    # Write source
    with open(SOURCE_FILE, "w") as f:
        f.write(cpp_source_code)
    
    cmd = get_compile_flags()
    cmd.extend([SOURCE_FILE, "-o", LIB_NAME])
    
    print(f"    Command: {' '.join(cmd)}")
    
    try:
        subprocess.check_call(cmd)
        print(f"[+] Compilation successful.\n")
    except subprocess.CalledProcessError:
        print("[-] Compilation failed! Ensure g++ or clang is installed.")
        sys.exit(1)

def load_library():
    """
    Tries to load the library. 
    If it fails (wrong arch, corrupt, missing symbols), it recompiles and retries.
    """
    lib_path = os.path.abspath(LIB_NAME)
    if not os.path.exists(lib_path):
        compile_cpp()
        
    try:
        # Load once. If this fails, we MUST exit. We cannot re-load in the same process.
        lib = ctypes.CDLL(lib_path)
        
        # Check symbols
        _ = lib.vector_multiply
        if lib.health_check() != 42:
            raise OSError("Integrity check failed")
            
        return lib
        
    except (OSError, AttributeError):
        print("[!] Library is corrupt. Recompiling...")
        compile_cpp()
        print("[-] Please run the script again. (OS cache cleared on restart)")
        sys.exit(1) # Force restart to clear the "Zombie" cache
    
    # Strategy: Try to load -> Verify Symbols -> Run Check
    for attempt in range(2):
        try:
            if not os.path.exists(lib_path):
                raise OSError("Library file missing")
                
            print(f"[*] Attempting to load {LIB_NAME} (Attempt {attempt+1})...")
            lib = ctypes.CDLL(lib_path)
            
            # --- SYMBOLIC CHECK ---
            # We explicitly look for our functions. If missing, it throws AttributeError
            try:
                _ = lib.vector_multiply
                _ = lib.single_multiply
                check = lib.health_check
            except AttributeError:
                print("[!] Library loaded but symbols are missing. Corrupt build?")
                raise OSError("Missing symbols")

            # --- RUNTIME CHECK ---
            if check() != 42:
                print("[!] Health check returned wrong value!")
                raise OSError("Runtime check failed")

            print("[+] Library Integrity Verified. Ready to race.\n")
            return lib

        except OSError as e:
            print(f"[!] Load failed: {e}")
            if attempt == 0:
                print("[*] Triggering Auto-Recompile...")
                compile_cpp()
            else:
                print("[-] Fatal Error: Could not load library even after recompile.")
                sys.exit(1)

# --- MAIN EXECUTION ---

# 1. Load the robust library
c_lib = load_library()

# 2. Configure Types
c_lib.vector_multiply.argtypes = [ctypes.POINTER(ctypes.c_longlong), ctypes.POINTER(ctypes.c_longlong), ctypes.POINTER(ctypes.c_longlong), ctypes.c_int]
c_lib.single_multiply.argtypes = [ctypes.c_longlong, ctypes.c_longlong]
c_lib.single_multiply.restype = ctypes.c_longlong


# --- SETUP DATA ---
print(f"[*] Generating {BATCH_SIZE:,} random integers...")
py_a = [random.randint(100, 99999) for _ in range(BATCH_SIZE)]
py_b = [random.randint(100, 99999) for _ in range(BATCH_SIZE)]

ArrayType = ctypes.c_longlong * BATCH_SIZE
c_a = ArrayType(*py_a)
c_b = ArrayType(*py_b)
c_res = ArrayType()

print("-" * 60)
print(f"TEST 1: THROUGHPUT (Processing {BATCH_SIZE:,} items)")
print("-" * 60)

# RACE 1: PYTHON
start_py = time.perf_counter()
# Standard List Comprehension
_ = [x * y for x, y in zip(py_a, py_b)]
end_py = time.perf_counter()
py_batch_time = (end_py - start_py) * 1000

# RACE 2: C++
start_c = time.perf_counter()
c_lib.vector_multiply(c_a, c_b, c_res, BATCH_SIZE)
end_c = time.perf_counter()
c_batch_time = (end_c - start_c) * 1000

print(f"Python Batch: {py_batch_time:.4f} ms")
print(f"C++ Batch:    {c_batch_time:.4f} ms")
print(f">> SPEEDUP:   {py_batch_time / c_batch_time:.1f}x FASTER")


print("\n" + "-" * 60)
print(f"TEST 2: LATENCY (Overhead of {LATENCY_ITERS} single calls)")
print("-" * 60)

# RACE 3: LATENCY
val_a, val_b = 12345, 67890

start_py_single = time.perf_counter()
for _ in range(LATENCY_ITERS):
    _ = val_a * val_b
end_py_single = time.perf_counter()
avg_py_ns = ((end_py_single - start_py_single) / LATENCY_ITERS) * 1e9

start_c_single = time.perf_counter()
for _ in range(LATENCY_ITERS):
    _ = c_lib.single_multiply(val_a, val_b)
end_c_single = time.perf_counter()
avg_c_ns = ((end_c_single - start_c_single) / LATENCY_ITERS) * 1e9

print(f"Python Single: {avg_py_ns:.1f} ns/op")
print(f"C++ Single:    {avg_c_ns:.1f} ns/op")

overhead = avg_c_ns - avg_py_ns
if overhead > 0:
    print(f">> BRIDGE TAX: It costs {overhead:.1f} ns just to switch from Python to C++.")
else:
    print(f">> RESULT: C++ is faster even with overhead (Rare!)")