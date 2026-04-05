import ctypes
import time
import os

# 1. Load the C++ Shared Library
# We get the absolute path to make sure Python finds it
lib_path = os.path.abspath("./libfastmath.so")
c_lib = ctypes.CDLL(lib_path)

# 2. Tell Python what the C++ function looks like
# arguments: (long long, long long, int) -> result: long long
c_lib.heavy_computation.argtypes = [ctypes.c_longlong, ctypes.c_longlong, ctypes.c_int]
c_lib.heavy_computation.restype = ctypes.c_longlong

# --- THE BENCHMARK ---
a = 3868143
b = 1483460
iterations = 1_000_000

print(f"Running {iterations} iterations...")
print("-" * 40)

# TEST A: Pure Python
start_py = time.perf_counter()
result_py = 0
for _ in range(iterations):
    result_py = a * b
end_py = time.perf_counter()
py_duration = (end_py - start_py) * 1_000_000 # Convert to microseconds

print(f"Python Time: {py_duration:.2f} \u00b5s")

# TEST B: Hybrid (Python calling C++)
start_c = time.perf_counter()
# The loop happens INSIDE C++, so we only cross the bridge once
result_c = c_lib.heavy_computation(a, b, iterations)
end_c = time.perf_counter()
c_duration = (end_c - start_c) * 1_000_000 # Convert to microseconds

print(f"C++ Time:    {c_duration:.2f} \u00b5s")
print("-" * 40)

# Check who won
speedup = py_duration / c_duration
print(f"Speedup:     {speedup:.1f}x FASTER")