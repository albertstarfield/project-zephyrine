import torch
import torch.nn as nn
import coremltools as ct
import numpy as np

class ANESequenceDrift(nn.Module):
    def __init__(self):
        super().__init__()
        # Matches the input channels exactly
        self.lane_op = nn.Conv2d(512, 512, kernel_size=1, bias=False)

    def forward(self, x):
        # Repeat to keep the ASIC busy
        for _ in range(20): 
            x = self.lane_op(x)
        return x

# Input Rank-4: (Batch, Channels, Height, Width) -> (1, 512, 1, 2048)
example_input = torch.randn(1, 512, 1, 2048)
model = ANESequenceDrift().eval()
traced_model = torch.jit.trace(model, example_input)

print("--- Initializing Sequence Stretch (2048 Context) ---")
mlmodel = ct.convert(
    traced_model,
    inputs=[ct.TensorType(shape=example_input.shape, name="context_input")],
    compute_precision=ct.precision.FLOAT16,
    compute_units=ct.ComputeUnit.CPU_AND_NE
)

print("Running 1000 iterations. Watch the NPU bar now!")
noise_data = np.random.rand(1, 512, 1, 2048).astype(np.float16)
for i in range(1000):
    mlmodel.predict({"context_input": noise_data})
    if i % 100 == 0: print(f"Lap {i}/1000...")
