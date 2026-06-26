import time
t0 = time.time()
import onnxruntime as ort  # noqa: E402
ort.InferenceSession("vad_component/silero_vad.onnx", providers=['CPUExecutionProvider'])
print(f"Load time: {time.time()-t0:.3f}s")
