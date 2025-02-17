import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import logging
import platform
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def detect_device():
    """Detects the available device (CPU, CUDA, MPS, ROCm, etc.)."""
    try:
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu"
        elif hasattr(torch, "hpu") and torch.hpu.is_available():
            return "hpu"
        elif hasattr(torch, "dml") and torch.dml.is_available():
            return "dml"
        elif "ROCM_PATH" in os.environ:
            logging.warning("ROCm environment detected. Double-check correct PyTorch installation.")
            try:
                import subprocess
                result = subprocess.run(['rocminfo'], capture_output=True, text=True, check=False)
                if result.returncode == 0 and "gfx" in result.stdout:
                    return "rocm"
            except (FileNotFoundError, Exception) as e:
                logging.warning(f"ROCm check failed: {e}")
        elif "ZE_ENABLE_SYSMAN" in os.environ:
            logging.warning("Intel oneAPI environment detected. Double-check installation.")
            return "oneapi"
        try:
            import pkgutil
            if pkgutil.find_loader('pyopencl'):
                logging.warning("PyOpenCL detected, but not a guarantee of usability.")
                return "opencl"
        except ImportError:
            pass
        try:
            import torch_xla  # Basic TPU check
            logging.warning("torch_xla detected, but not a guarantee of usability.")
            return 'tpu'
        except ImportError:
            pass

        return "cpu"
    except Exception as e:
        logging.exception(f"Device detection error: {e}")
        return "cpu"

def load_model_and_tokenizer(device):
    """Loads the AWQ model and tokenizer."""
    try:
        model_name = "elysiantech/gemma-2b-awq-4bit"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)  # Always good practice

        # Load the AWQ model.  device_map="auto" should work.
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True, # For Custom code
             # No need for torch_dtype here; AWQ handles it
        )
        return model, tokenizer

    except Exception as e:
        logging.exception(f"Error loading model/tokenizer: {e}")
        sys.exit(1)

def chat(model, tokenizer, device):
    """Chat interaction loop."""
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ("exit", "quit", "bye"):
                break

            inputs = tokenizer(user_input, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.generate(**inputs, streamer=streamer, max_new_tokens=512,
                                         do_sample=True, top_p=0.95, top_k=60, temperature=0.7)

            # Streamer handles output; this is just a fallback
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if not streamer:
                print(f"Bot: {output_text}")

        except KeyboardInterrupt:
            print("\nExiting.")
            break
        except Exception as e:
            logging.error(f"Generation error: {e}")
            print("Bot: Sorry, I had a problem.")

def main():
    """Main function."""
    print("-" * 30)
    print(f"Platform: {platform.system()} {platform.release()} ({platform.architecture()[0]})")
    device = detect_device()
    print(f"Using device: {device}")
    print("-" * 30)

    model, tokenizer = load_model_and_tokenizer(device)
    chat(model, tokenizer, device)

if __name__ == "__main__":
    main()