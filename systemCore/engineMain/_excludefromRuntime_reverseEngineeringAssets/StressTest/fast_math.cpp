// fast_math.cpp
#include <iostream>

extern "C" {
    // We create a function that does the HEAVY lifting inside C++
    // This function performs the multiplication 'iters' times.
    long long heavy_computation(long long a, long long b, int iters) {
        volatile long long result = 0; // 'volatile' prevents compiler optimization cheating
        for (int i = 0; i < iters; i++) {
            result = a * b;
        }
        return result;
    }
}