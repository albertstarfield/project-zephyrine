import subprocess
import json
import os
import sys

# ================= CONFIGURATION =================
# Update these paths to match your actual file locations
# Based on your logs, I tried to guess the model dir, but please verify.
MODEL_DIR = "/Users/albertstarfield/Documents/misc/AI/project-zephyrine/systemCore/engineMain/staticmodelpool"
MODEL_FILENAME = "Qwen3-VL-ImageDescripter_v2.gguf"
MMPROJ_FILENAME = "Qwen3-VL-ImageDescripter_v2_mmproj.gguf"

# You must have a real image file to test. 
# Create a dummy one or point to an existing one.
TEST_IMAGE_PATH = "./_excludefromRuntime_reverseEngineeringAssets/test_image.png" 
# =================================================

def debug_vision_task():
    # 1. Setup Paths
    model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
    mmproj_path = os.path.join(MODEL_DIR, MMPROJ_FILENAME)
    python_exe = sys.executable
    worker_script = "llama_worker.py"

    # 2. Validation
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found: {model_path}")
        return
    if not os.path.exists(mmproj_path):
        print(f"❌ Error: Projector file not found: {mmproj_path}")
        return
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"❌ Error: Test image not found: {TEST_IMAGE_PATH}")
        print("   -> Please copy any PNG/JPG file to this folder and name it 'test_image.png'")
        return

    # 3. Prepare the JSON Payload (what Cortex normally sends via stdin)
    payload = {
        "mmproj_path": mmproj_path,
        "image_path": os.path.abspath(TEST_IMAGE_PATH),
        "prompt": "Describe this image.",
        "kwargs": {
            "temperature": 0.1
        },
        # The worker expects these if they aren't provided via CLI args sometimes, 
        # but for 'vision' task logic we wrote, it relies on the dictionary keys above.
        "messages": [] 
    }

    # 4. Prepare the Command arguments
    cmd = [
        python_exe,
        worker_script,
        "--model-path", model_path,
        "--task-type", "vision",
        "--n-gpu-layers", "-1", # Force Metal/GPU
        "--verbose" 
    ]

    print(f"\n🚀 Launching Worker Command:\n{' '.join(cmd)}\n")
    print(f"📨 Sending JSON Payload:\n{json.dumps(payload, indent=2)}\n")
    print("-" * 60)

    # 5. Run Subprocess
    try:
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Send payload and wait
        stdout, stderr = process.communicate(input=json.dumps(payload))
        return_code = process.returncode

        print(f"🏁 Process Finished with Exit Code: {return_code}")
        
        print("\n--- 🛑 WORKER STDERR (Logs/C++ Output) ---")
        print(stderr)
        
        print("\n--- ✅ WORKER STDOUT (JSON Result) ---")
        print(stdout)
        
        # 6. Verification
        if stdout.strip():
            try:
                data = json.loads(stdout)
                if "result" in data and "choices" in data["result"]:
                    content = data["result"]["choices"][0]["message"]["content"]
                    print(f"\n✨ SUCCESS! Extracted Content: {content}")
                elif "error" in data:
                    print(f"\n❌ WORKER RETURNED ERROR JSON: {data['error']}")
                else:
                    print(f"\n⚠️ Unknown JSON Structure: {data.keys()}")
            except json.JSONDecodeError:
                print("\n❌ STDOUT is not valid JSON!")
        else:
            print("\n❌ STDOUT is Empty!")

    except Exception as e:
        print(f"\n🔥 Exception running subprocess: {e}")

if __name__ == "__main__":
    debug_vision_task()