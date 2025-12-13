
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
