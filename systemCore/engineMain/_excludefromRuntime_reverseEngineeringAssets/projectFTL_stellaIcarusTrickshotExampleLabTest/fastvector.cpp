
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
