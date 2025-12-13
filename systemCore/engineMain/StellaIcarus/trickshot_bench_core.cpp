
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
