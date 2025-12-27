import subprocess
import os
import sys

# ================= CONFIGURATION =================
MODEL_DIR = "/Users/albertstarfield/Documents/misc/AI/project-zephyrine/systemCore/engineMain/staticmodelpool"
# Use a general-purpose model for this test
MODEL_FILENAME = "Qwen3LowLatency.gguf" 
# =================================================

def test_lmexec_command():
    model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)

    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found at: {model_path}")
        return

    # Your base_cmd list, ready for execution
    cmd = [
        "LMExec",
        "--model", model_path,
        "--n-gpu-layers", "-1",
        "--simple-io",
        "--offline",
        "--no-warmup",
        "-no-cnv",
        "-c", "16384",
        "--mmap",
        "--cpu-strict", "1",
        "-fa", "off",
        "-ot", ".ffn_.*_exps.=CPU",
        "-p", "Go is a high-level, general-purpose programming language that is statically-typed and compiled. It is known for the simplicity of its syntax and the efficiency of development that it enables through the inclusion of a large standard library supplying many needs for common projects.[12] It was designed at Google[13] in 2007 by Robert Griesemer, Rob Pike, and Ken Thompson, and publicly announced in November 2009.[4] It is syntactically similar to C, but also has garbage collection, structural typing,[7] and CSP-style concurrency.[14] It is often referred to as Golang to avoid ambiguity and because of its former domain name, golang.org",
        "--temp", "0.5",
        "-n", "1024" # Limit output to 128 tokens
    ]

    print(f"🚀 Executing Command:\n{' '.join(cmd)}\n")
    print("-" * 60)

    try:
        # Run the command and capture output
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60 # Add a timeout to prevent hanging
        )

        print(f"🏁 Process Finished with Exit Code: {process.returncode}")
        
        print("\n--- 🛑 STDERR (Logs/Errors from LMExec) ---")
        print(process.stderr if process.stderr else "[No stderr output]")
        
        print("\n--- ✅ STDOUT (Generated Text) ---")
        print(process.stdout if process.stdout else "[No stdout output]")

        if process.returncode != 0:
            print("\n❌ FAILED: The LMExec binary returned a non-zero exit code. Check STDERR above for clues.")
        elif not process.stdout.strip():
            print("\n⚠️ WARNING: The command ran successfully but produced no text. This could be a prompt or model issue.")
        else:
            print("\n✨ SUCCESS: The command executed and generated text.")

    except FileNotFoundError:
        print("\n❌ CRITICAL ERROR: The 'LMExec' command was not found.")
        print("   Make sure your conda environment is activated and the binary is in your PATH.")
    except Exception as e:
        print(f"\n🔥 An exception occurred while running the subprocess: {e}")

if __name__ == "__main__":
    test_lmexec_command()
