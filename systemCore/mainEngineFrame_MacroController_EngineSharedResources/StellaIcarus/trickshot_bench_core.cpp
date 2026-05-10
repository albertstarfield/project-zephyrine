
#include <iostream>
#include <chrono>
#include <vector>
#include <random>

extern "C" {
    // We export this function to be callable by Python (ctypes)
    double run_trickshot_benchmark(int iterations) {
        
        // 1. Setup Random Data
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
        volatile long long result = 0; 
        
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < iterations; ++i) {
            result = data_a[i] * data_b[i];
        }

        auto end = std::chrono::high_resolution_clock::now();

        // 3. Results - CHANGED TO PICOSECONDS
        // We cast directly to picoseconds here to keep the raw unit logic inside C++
        auto total_ps = std::chrono::duration_cast<std::chrono::picoseconds>(end - start).count();
        double avg_ps = (double)total_ps / iterations;
        
        return avg_ps;
    }
    
    int trickshot_ping() { return 999; }
}
