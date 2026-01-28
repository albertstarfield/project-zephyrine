import mmap
import os
import numpy as np
import gguf # pip install gguf
from llama_cpp import Llama # pip install llama-cpp-python

# 1. Configuration
model_path = "/Users/albertstarfield/Documents/misc/AdaptiveSystem/project-zephyrine/systemCore/engineMain/staticmodelpool/Qwen3LowLatency.gguf"
target_tensor_name = "blk.0.attn_q.weight" # Example tensor to patch

# 2. Safety Check: Load GGUF metadata to find offsets
# We use the official GGUF reader to parse the header structure
reader = gguf.GGUFReader(model_path)
target_tensor = None

for tensor in reader.tensors:
    if tensor.name == target_tensor_name:
        target_tensor = tensor
        break

if not target_tensor:
    raise ValueError(f"Tensor {target_tensor_name} not found in model!")

print(f"Target found: {target_tensor.name}")
print(f"Data Offset: {target_tensor.data_offset}")
print(f"Size in bytes: {target_tensor.n_bytes}")

# 3. The "In-Memory Patching" Trick
# We open the file, but we MMAP it with MAP_PRIVATE (Copy-on-Write)
# This ensures writes stay in RAM and never hit the disk.

with open(model_path, "r+b") as f:
    # mmap the file. 
    # access=mmap.ACCESS_COPY is CRITICAL. It creates a copy-on-write mapping.
    # Modifications are visible to this process but NOT written to disk.
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_COPY)

    # 4. Perform the Patch
    # Let's say we want to inject noise or zero out specific weights.
    # Note: You must respect the quantization format (e.g., Q4_K, Q8_0).
    # Writing raw float32s into a Q4_K block will destroy the model.
    # For this example, we'll zero out the first 100 bytes (safest visual test).
    
    start_addr = target_tensor.data_offset
    patch_size = 100 # Patching first 100 bytes
    
    print(f"Patching memory at offset {start_addr}...")
    
    # Generate your patch data (bytes)
    # real scenario: convert your float weights -> quantized bytes -> write here
    patch_data = b'\x00' * patch_size 
    
    # WRITE TO MEMORY
    mm.seek(start_addr)
    mm.write(patch_data)
    
    print("Patch applied to RAM. Disk file is untouched.")

    # 5. Inference with Patched Model
    # llama-cpp-python can load directly from a memory buffer? 
    # Standard Llama(...) takes a path, but we can pass the model_path 
    # and rely on the OS file cache if we didn't use mmap, 
    # BUT since we modified a private mmap, we need a library that accepts 
    # a pointer or we need to keep this mmap alive and pass it.
    
    # TRICK: Most high-level bindings don't accept a raw buffer easily.
    # Ideally, you would use C++ (llama.cpp) for this.
    # However, for Python, we can't easily pass 'mm' to Llama().
    
    # ALTERNATIVE: Patching *after* loading.
    # If the above mmap trick is too hard to pass to the engine,
    # we use the "Post-Load Pointer Patch" (See below).
