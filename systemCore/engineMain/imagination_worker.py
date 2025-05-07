# imagination_worker.py

import sys
import os
import json
import time
import traceback
import argparse
import base64
from io import BytesIO
from typing import Optional

try:
    from PIL import Image
except ImportError:
    # log_worker is not defined yet, print directly to stderr
    print("[IMAGINATION_WORKER|ERROR] Pillow (PIL) not found in imagination_worker environment.", file=sys.stderr,
          flush=True)
    sys.exit(1)

STABLE_DIFFUSION_CPP_AVAILABLE = False
try:
    from stable_diffusion_cpp import StableDiffusion  # Assuming SDLogLevel is not used or needed

    STABLE_DIFFUSION_CPP_AVAILABLE = True
except ImportError:
    print("[IMAGINATION_WORKER|ERROR] stable-diffusion-cpp library not found.", file=sys.stderr, flush=True)
    # Main function will check STABLE_DIFFUSION_CPP_AVAILABLE and exit if False


# --- Basic Logging to stderr ---
def log_worker(level, message):
    print(f"[IMAGINATION_WORKER|{level}] {message}", file=sys.stderr, flush=True)


def main():
    parser = argparse.ArgumentParser(description="Stable Diffusion Image Generation Worker (FLUX Optimized)")
    parser.add_argument("--model-dir", required=True, help="Base directory where model files are located.")
    parser.add_argument("--diffusion-model-name", default="flux1-schnell.gguf",
                        help="Filename of the main FLUX diffusion GGUF model.")
    parser.add_argument("--clip-l-name", default="flux1-clip_l.gguf", help="Filename of the FLUX CLIP L GGUF model.")
    parser.add_argument("--t5xxl-name", default="flux1-t5xxl.gguf", help="Filename of the FLUX T5 XXL GGUF model.")
    parser.add_argument("--vae-name", default="flux1-ae.gguf", help="Filename of the FLUX VAE GGUF model.")
    parser.add_argument("--w-device", default="default",
                        help="Device hint for stable-diffusion-cpp (e.g., 'cpu', 'cuda:0', 'mps', 'default').")
    parser.add_argument("--rng-type", default="std_default", choices=["std_default", "cuda"],
                        help="RNG type for stable-diffusion-cpp.")
    parser.add_argument("--n-threads", type=int, default=0,
                        help="Number of threads for CPU operations (0 for auto/default, positive for specific count).")

    parser.add_argument("--test-mode", action="store_true", help="Run in test mode with fixed parameters.")
    parser.add_argument("--test-prompt", default="a realistic RTX goofy doraemon blue smile cat holding a sign says 'Hello'",
                        help="Prompt for test mode.")
    parser.add_argument("--test-output-file", default="zephyImagination_test_image.png",
                        help="Output filename for test mode image.")

    args = parser.parse_args()

    if not STABLE_DIFFUSION_CPP_AVAILABLE:
        log_worker("CRITICAL", "stable-diffusion-cpp library failed to import. Worker cannot run.")
        print(json.dumps({"error": "stable-diffusion-cpp library not available in worker environment"}), flush=True)
        sys.exit(1)

    log_worker("INFO",
               f"Starting Imagination Worker. Device Hint: {args.w_device}, TestMode: {args.test_mode}, RNG Type: {args.rng_type}, n_threads_arg: {args.n_threads}")

    # --- Construct full model paths ---
    try:
        diffusion_model_path_arg = os.path.join(args.model_dir, args.diffusion_model_name)
        clip_l_path_arg = os.path.join(args.model_dir, args.clip_l_name)
        t5xxl_path_arg = os.path.join(args.model_dir, args.t5xxl_name)
        vae_path_arg = os.path.join(args.model_dir, args.vae_name)

        paths_to_check = {
            "FLUX Diffusion Model": diffusion_model_path_arg,
            "FLUX CLIP L Model": clip_l_path_arg,
            "FLUX T5 XXL Model": t5xxl_path_arg,
            "FLUX VAE Model": vae_path_arg
        }
        for name, path_to_check in paths_to_check.items():
            if not os.path.exists(path_to_check):
                raise FileNotFoundError(f"{name} not found at: {path_to_check}")
            log_worker("DEBUG", f"Found {name} at: {path_to_check}")

    except FileNotFoundError as fe:
        log_worker("ERROR", f"Model file not found: {fe}")
        print(json.dumps({"error": f"Model file missing: {fe}"}), flush=True)
        sys.exit(1)
    except Exception as path_err:
        log_worker("ERROR", f"Error with model paths: {path_err}")
        print(json.dumps({"error": f"Error with model paths: {path_err}"}), flush=True)
        sys.exit(1)

    # --- Load Stable Diffusion Model ---
    sd_instance: Optional[StableDiffusion] = None
    try:
        load_start_time = time.monotonic()
        log_worker("INFO", "Initializing StableDiffusion instance for FLUX model...")

        rng_type_map = {"std_default": 0, "cuda": 1}
        rng_type_val = rng_type_map.get(args.rng_type.lower(), 0)

        # --- Determine n_threads to pass to the library ---
        n_threads_for_constructor: int
        if args.n_threads > 0:
            n_threads_for_constructor = args.n_threads
            log_worker("DEBUG", f"User specified n_threads: {n_threads_for_constructor}")
        else:  # args.n_threads is 0 (auto) or was not provided
            cpu_count = os.cpu_count()
            if cpu_count is not None and cpu_count > 0:
                # Use all available logical cores as a common auto strategy
                # Some libraries prefer cpu_count // 2
                n_threads_for_constructor = cpu_count
                log_worker("DEBUG", f"Auto-detected n_threads using os.cpu_count(): {n_threads_for_constructor}")
            else:
                n_threads_for_constructor = 4  # A sensible fallback if os.cpu_count() fails
                log_worker("WARNING",
                           f"os.cpu_count() returned unusable value or failed. Falling back to n_threads={n_threads_for_constructor}")

        log_worker("INFO", f"Passing n_threads={n_threads_for_constructor} to StableDiffusion constructor.")
        # --- End n_threads determination ---

        sd_instance = StableDiffusion(
            diffusion_model_path=diffusion_model_path_arg,
            clip_l_path=clip_l_path_arg,
            t5xxl_path=t5xxl_path_arg,
            vae_path=vae_path_arg,
            vae_decode_only=True,
            n_threads=n_threads_for_constructor,  # Pass the determined integer
            rng_type=rng_type_val,
            # log_level is removed, assuming library default or env var control
        )

        load_duration = time.monotonic() - load_start_time
        log_worker("INFO", f"StableDiffusion FLUX model loaded in {load_duration:.2f}s.")
    except TypeError as te:
        log_worker("ERROR", f"StableDiffusion model loading failed due to TypeError: {te}")
        log_worker("ERROR",
                   "This often means a parameter was None when an int/bool was expected, or a parameter name is incorrect.")
        log_worker("ERROR",
                   "Check constructor parameters for StableDiffusion, especially n_threads, against library documentation.")
        log_worker("ERROR", traceback.format_exc())
        print(json.dumps({"error": f"Worker failed to load StableDiffusion model (TypeError): {te}"}), flush=True)
        sys.exit(1)
    except Exception as e:
        log_worker("ERROR", f"StableDiffusion model loading failed: {e}")
        log_worker("ERROR", traceback.format_exc())
        print(json.dumps({"error": f"Worker failed to load StableDiffusion model: {e}"}), flush=True)
        sys.exit(1)

    # --- Test Mode ---
    if args.test_mode:
        log_worker("INFO", "Running in TEST MODE with FLUX parameters.")
        try:
            log_worker("INFO", f"Test prompt: '{args.test_prompt}'")
            output_pil_images = sd_instance.txt_to_img(
                prompt=args.test_prompt,
                negative_prompt="Bad Morphed Graphic or Body",
                sample_method="euler",
                sample_steps=20,
                cfg_scale=1.0,
                width=768,
                height=448,
                seed=-1,
            )

            if output_pil_images and len(output_pil_images) > 0:
                first_image: Image.Image = output_pil_images[0]
                # Save test image in the current worker directory for inspection
                output_file_path = os.path.join(os.getcwd(), args.test_output_file)
                first_image.save(output_file_path)
                log_worker("INFO", f"Test image generated and saved to {output_file_path}")
                print(json.dumps({"result": {"status": "Test image generated", "file": output_file_path}}), flush=True)
            else:
                log_worker("ERROR", "Test mode: txt_to_img returned no images.")
                print(json.dumps({"error": "Test mode: No image generated"}), flush=True)
        except Exception as e:
            log_worker("ERROR", f"Test mode execution failed: {e}")
            log_worker("ERROR", traceback.format_exc())
            print(json.dumps({"error": f"Test mode execution error: {e}"}), flush=True)
        sys.exit(0)

    # --- Standard Mode (Read request from stdin) ---
    result_payload = {"error": "Worker did not receive valid input or process request."}
    try:
        log_worker("INFO", "Standard Mode: Waiting for request data on stdin...")
        input_json_str = sys.stdin.read()
        if not input_json_str: raise ValueError("Received empty input from stdin.")
        log_worker("DEBUG", f"Received raw input string (len={len(input_json_str)}): {input_json_str[:200]}...")
        request_data = json.loads(input_json_str)
        log_worker("INFO", "Request data JSON parsed successfully.")

        task_type = request_data.get("task_type", "txt2img").lower()
        prompt = request_data.get("prompt")
        negative_prompt = request_data.get("negative_prompt", "")
        num_images = int(request_data.get("n", 1))
        size_str = request_data.get("size", "512x512")
        try:
            width, height = map(int, size_str.split('x'))
        except:
            width, height = 512, 512
            log_worker("WARNING", f"Invalid size '{size_str}', defaulting to {width}x{height}")

        is_flux_model = "flux" in args.diffusion_model_name.lower()
        cfg_scale = float(request_data.get("cfg_scale", 1.0 if is_flux_model else 7.0))
        sample_steps = int(request_data.get("sample_steps", 4 if is_flux_model else 20))
        sample_method_str = str(request_data.get("sample_method", "euler" if is_flux_model else "euler_a")).lower()
        seed = int(request_data.get("seed", -1))
        response_format = request_data.get("response_format", "b64_json").lower()
        input_image_b64 = request_data.get("input_image_b64")  # For img2img

        if not prompt or not isinstance(prompt, str):
            raise ValueError("Missing or invalid 'prompt' in request.")

        log_worker("INFO",
                   f"Task: {task_type}, Prompt: '{prompt[:50]}...', N: {num_images}, Size: {width}x{height}, Steps: {sample_steps}, CFG: {cfg_scale}, Method: {sample_method_str}")

        generated_pil_images: list[Image.Image] = []
        task_start_time = time.monotonic()

        if task_type == "txt2img":
            log_worker("INFO", "Executing txt_to_img with FLUX parameters...")
            generated_pil_images = sd_instance.txt_to_img(
                prompt=prompt,
                negative_prompt=negative_prompt,
                sample_method=sample_method_str,
                sample_steps=sample_steps,
                cfg_scale=cfg_scale,  # Ensure library uses this name or guidance_scale
                width=width,
                height=height,
                seed=seed,
            )
        elif task_type == "img2img":
            log_worker("WARNING", "img2img task is a STUB in this worker.")
            if not input_image_b64: raise ValueError("Missing 'input_image_b64' for img2img task.")
            result_payload = {"error": "img2img task is not yet fully implemented."}
            # generated_pil_images would be empty here or from a stub call
        else:
            raise ValueError(f"Unsupported task_type: {task_type}")

        task_duration = time.monotonic() - task_start_time
        log_worker("INFO",
                   f"Image generation task ({task_type}) completed in {task_duration:.2f}s. Generated {len(generated_pil_images)} image(s).")

        if not generated_pil_images and task_type == "txt2img" and "error" not in result_payload:
            raise RuntimeError(f"{task_type} call returned no images unexpectedly.")

        if "error" not in result_payload:  # Only process if no error (like img2img stub) was set
            output_data_list = []
            if response_format == "b64_json":
                for i, pil_img in enumerate(generated_pil_images):
                    try:
                        buffered = BytesIO()
                        pil_img.save(buffered, format="PNG")
                        img_str_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                        output_data_list.append({"b64_json": img_str_b64})
                    except Exception as enc_err:
                        log_worker("ERROR", f"Failed to encode image {i} to base64 PNG: {enc_err}")
                log_worker("INFO", f"Encoded {len(output_data_list)} images to b64_json.")
            elif response_format == "url":
                log_worker("WARNING", "response_format 'url' requested, but worker returns b64_json as fallback.")
                for i, pil_img in enumerate(generated_pil_images):
                    try:
                        buffered = BytesIO();
                        pil_img.save(buffered, format="PNG");
                        img_str_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                        output_data_list.append(
                            {"b64_json": img_str_b64, "comment": "URL format requested, returning b64_json."})
                    except:
                        pass  # Best effort for fallback
            else:
                raise ValueError(f"Unsupported response_format requested by client: {response_format}")

            result_payload = {"created": int(time.time()), "data": output_data_list}

    except json.JSONDecodeError as e:
        log_worker("ERROR", f"JSON Decode Error: {e}"); result_payload = {"error": f"Worker JSON Decode Error: {e}"}
    except ValueError as e:
        log_worker("ERROR", f"Input Validation Error: {e}"); result_payload = {"error": f"Worker Input Error: {e}"}
    except RuntimeError as e:
        log_worker("ERROR", f"Runtime Error (Image Gen): {e}\n{traceback.format_exc()}"); result_payload = {
            "error": f"Worker Image Gen Error: {e}"}
    except Exception as e:
        log_worker("ERROR", f"Unexpected Error: {e}\n{traceback.format_exc()}"); result_payload = {
            "error": f"Worker Unexpected Error: {e}"}

    # --- Write Result/Error to stdout ---
    try:
        output_json = json.dumps(result_payload)
        log_worker("DEBUG", f"Sending output JSON (len={len(output_json)}): {output_json[:200]}...")
        print(output_json, flush=True)
        log_worker("INFO", "Result/Error sent to stdout.")
    except Exception as e_print:
        log_worker("ERROR", f"Failed to serialize/write result to stdout: {e_print}")
        final_error_payload = {"error": f"Worker critical error: Failed to write result: {e_print}"}
        try:
            print(json.dumps(final_error_payload), flush=True)
        except:
            pass
        if "error" not in result_payload and sys.exc_info()[0] is None:
            sys.exit(1)

    log_worker("INFO", "Imagination Worker process finished.")
    sys.exit(0)


if __name__ == "__main__":
    main()