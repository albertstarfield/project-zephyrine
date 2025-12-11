import ctypes
import os
import subprocess
import time
import random
import sys
import platform

# --- CONFIGURATION ---
LIB_NAME = "libfastvector.so"
SOURCE_FILE = "fastvector.cpp"
ITERATIONS = 1_000_000  # 1 Million Data Points

# --- STEP 1: EMBEDDED C++ SOURCE CODE ---
# We write this to a file so g++ can compile it.
cpp_source_code = """
#include <iostream>

extern "C" {
    // A vectorized multiplication function: Result[i] = A[i] * B[i]
    // We use pointers to access the memory Python allocated.
    void vector_multiply(long long* a, long long* b, long long* result, int size) {
        
        // The compiler will auto-vectorize this loop with -O3
        #pragma omp simd
        for (int i = 0; i < size; i++) {
            result[i] = a[i] * b[i];
        }
    }
}
"""

def compile_cpp():
    print(f"[*] Compiling {SOURCE_FILE} with -O3 optimization...")
    
    # Write source to disk
    with open(SOURCE_FILE, "w") as f:
        f.write(cpp_source_code)
    
    # Determine command based on OS (macOS/Linux)
    # -O3 : Maximum optimization
    # -shared : Make a library, not an app
    # -fPIC : Position Independent Code (needed for libraries)
    cmd = ["g++", "-O3", "-shared", "-fPIC", SOURCE_FILE, "-o", LIB_NAME]
    
    # Check for Apple Silicon specifics (optional, but good for M1/M2/M3)
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        cmd.insert(1, "-mcpu=apple-m1")

    try:
        subprocess.check_call(cmd)
        print(f"[+] Compilation successful: {LIB_NAME} created.")
    except subprocess.CalledProcessError:
        print("[-] Compilation failed! Do you have g++ installed?")
        sys.exit(1)

# --- STEP 2: LOAD LIBRARY ---
if not os.path.exists(LIB_NAME):
    compile_cpp()
else:
    print(f"[*] Found existing {LIB_NAME}, skipping compilation.")

# Load the DLL
# We use abspath because ctypes can be picky about relative paths
lib_path = os.path.abspath(LIB_NAME)
try:
    c_lib = ctypes.CDLL(lib_path)
except OSError:
    # If loading fails, maybe the architecture changed? Recompile.
    print("[!] Load failed. Recompiling...")
    compile_cpp()
    c_lib = ctypes.CDLL(lib_path)

# Define arguments: (Array A, Array B, Array Result, Size)
# POINTER(c_longlong) is roughly equivalent to long long* in C
c_lib.vector_multiply.argtypes = [
    ctypes.POINTER(ctypes.c_longlong), 
    ctypes.POINTER(ctypes.c_longlong), 
    ctypes.POINTER(ctypes.c_longlong), 
    ctypes.c_int
]

# --- STEP 3: GENERATE DATA ---
print(f"[*] Generating {ITERATIONS:,} random integers in Python...")
# Create standard Python lists
py_a = [random.randint(100, 99999) for _ in range(ITERATIONS)]
py_b = [random.randint(100, 99999) for _ in range(ITERATIONS)]

# Convert to C-types Arrays (This takes a moment, but is one-time setup)
# (ctypes.c_longlong * Size) creates an array TYPE of that size
ArrayType = ctypes.c_longlong * ITERATIONS
c_a = ArrayType(*py_a)
c_b = ArrayType(*py_b)
c_res = ArrayType() # Empty result array

print("[*] Data ready. Starting The Race.")
print("-" * 50)

# --- RACE 1: PYTHON ---
print(">>> Running Pure Python Loop...")
start_py = time.perf_counter()

# The most common way to do this in raw Python
py_result = [x * y for x, y in zip(py_a, py_b)]

end_py = time.perf_counter()
py_time = (end_py - start_py) * 1000 # ms
print(f"    Python Time: {py_time:.4f} ms")


# --- RACE 2: C++ (VIA CTYPES) ---
print(">>> Running C++ Optimized Library...")
start_c = time.perf_counter()

# We pass the pointers to the arrays
c_lib.vector_multiply(c_a, c_b, c_res, ITERATIONS)

end_c = time.perf_counter()
c_time = (end_c - start_c) * 1000 # ms
print(f"    C++ Time:    {c_time:.4f} ms")


# --- RESULTS ---
print("-" * 50)
if c_time > 0:
    speedup = py_time / c_time
    print(f"SPEEDUP: {speedup:.2f}x")
else:
    print("C++ was too fast to measure accurately!")

# Verify correctness (check first element)
if c_res[0] == py_result[0]:
    print(f"Verification: MATCH ( {py_a[0]} * {py_b[0]} = {c_res[0]} )")
else:
    print("Verification: FAILED (Math mismatch!)")