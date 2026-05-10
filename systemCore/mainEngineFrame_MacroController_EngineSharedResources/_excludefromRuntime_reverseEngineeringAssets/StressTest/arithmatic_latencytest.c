#include <iostream>
#include <chrono>
#include <vector>
#include <random>

int main() {
    int iterations = 1000000;
    
    // 1. Setup Random Data (We don't time this part)
    std::cout << "Generating " << iterations << " random number pairs..." << std::endl;
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

    // The CPU must fetch new numbers from memory every single time
    for (int i = 0; i < iterations; ++i) {
        result = data_a[i] * data_b[i];
    }

    auto end = std::chrono::high_resolution_clock::now();

    // 3. Results
    auto total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    double avg_ns = (double)total_ns / iterations;

    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "Average time per RANDOM multiplication: " << avg_ns << " nanoseconds" << std::endl;
    
    return 0;
}