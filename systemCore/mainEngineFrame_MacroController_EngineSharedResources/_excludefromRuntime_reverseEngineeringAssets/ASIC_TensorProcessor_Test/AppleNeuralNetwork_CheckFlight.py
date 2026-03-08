import coremltools as ct

# Load your generated model
model = ct.models.MLModel("ANE_Noise_Drift.mlpackage")

# Get the compute plan (New in 2025/2026)
# This will show you exactly what hardware CoreML assigned
compute_plan = model.get_compute_plan()

print("\n--- HARDWARE ASSIGNMENT LOG ---")
for i, layer in enumerate(model.get_spec().neuralNetwork.layers):
    device = compute_plan.get_device_usage(layer.name)
    status = "✅ ANE" if "NeuralEngine" in str(device) else "❌ CPU/GPU"
    print(f"Layer {i} [{layer.name}]: {status}")
