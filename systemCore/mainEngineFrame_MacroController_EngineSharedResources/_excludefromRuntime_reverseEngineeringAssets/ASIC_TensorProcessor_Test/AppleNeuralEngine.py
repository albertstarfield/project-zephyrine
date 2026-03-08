import torch
import torch.nn as nn
import coremltools as ct
import numpy as np
import time

# 1. Define a "Pseudo-4D" Linear Operation
# ANE loves 1x1 Convolutions. They are mathematically identical to Linear layers.
class ANESignalTest(nn.Module):
    def __init__(self):
        super(ANESignalTest, self).__init__()
        # 1024 channels in, 1024 channels out. 
        # This is a massive matrix multiplication for the ASIC.
        self.heavy_op = nn.Conv2d(1024, 1024, kernel_size=1)

    def forward(self, x):
        return self.heavy_op(x)

# 2. Setup the "Windshear" Input (Rank-4 Noise)
# Format: (Batch, Channels, Height, Width) -> (1, 1024, 1, 1024)
# We use 1024 to stay within the ANE's preferred cache alignment.
example_input = torch.randn(1, 1024, 1, 1024)
model = ANESignalTest().eval()

# 3. Trace and Convert
traced_model = torch.jit.trace(model, example_input)

print("--- Starting Conversion to ANE ---")
mlmodel = ct.convert(
    traced_model,
    inputs=[ct.TensorType(shape=example_input.shape, name="noise_input")],
    # Force float16 because ANE doesn't do float32
    compute_precision=ct.precision.FLOAT16, 
    # Use ANE_ONLY to verify it CAN run there. If it fails, the ASIC said 'No.'
    compute_units=ct.ComputeUnit.CPU_AND_NE
)

# 4. Save and Test
model_path = "ane_stress_test.mlpackage"
mlmodel.save(model_path)

# 5. The Prediction Run (The actual 'Drift')
print(f"Model saved. Running prediction on ANE...")
try:
    # Warm up
    noise_data = np.random.rand(1, 1024, 1, 1024).astype(np.float16)
    
    start_time = time.time()
    for _ in range(100): # Run 100 times to see it on the Activity Monitor
        mlmodel.predict({"noise_input": noise_data})
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 100
    print(f"Success! Average ANE Latency: {avg_time:.6f} seconds")
    print("Check Activity Monitor -> CPU Load (Neural Engine graph) now!")
    
except Exception as e:
    print(f"ANE REJECTED THE DRIFT: {e}")
