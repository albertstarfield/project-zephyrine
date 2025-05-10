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
    if hasattr(Image.SAVE, "AVIF"):
        PIL_AVIF_PLUGIN_POSSIBLY_AVAILABLE = True
        print("[IMAGINATION_WORKER|INFO] Pillow AVIF save format detected as potentially available.", file=sys.stderr, flush=True)
    else:
        try:
            from pillow_avif import AvifImagePlugin # noqa
            PIL_AVIF_PLUGIN_POSSIBLY_AVAILABLE = True
            print("[IMAGINATION_WORKER|INFO] Pillow AVIF plugin registered and potentially available.", file=sys.stderr, flush=True)
        except ImportError:
            PIL_AVIF_PLUGIN_POSSIBLY_AVAILABLE = False
            print("[IMAGINATION_WORKER|WARNING] pillow-avif-plugin not found. AVIF conversion will be skipped.", file=sys.stderr, flush=True)

except ImportError:
    print("[IMAGINATION_WORKER|ERROR] Pillow (PIL) not found in imagination_worker environment.", file=sys.stderr, flush=True)
    sys.exit(1)

STABLE_DIFFUSION_CPP_AVAILABLE = False
try:
    from stable_diffusion_cpp import StableDiffusion # Assuming SDLogLevel is not used or needed

    STABLE_DIFFUSION_CPP_AVAILABLE = True
except ImportError:
    print("[IMAGINATION_WORKER|ERROR] stable-diffusion-cpp library not found.", file=sys.stderr, flush=True)

def log_worker(level, message):
    print(f"[IMAGINATION_WORKER|{level}] {message}", file=sys.stderr, flush=True)

def main():
    parser = argparse.ArgumentParser(description="Stable Diffusion Image Generation Worker (FLUX Optimized)")
    parser.add_argument("--model-dir", required=True, help="Base directory where model files are located.")
    parser.add_argument("--diffusion-model-name", default="flux1-schnell.gguf", help="Filename of the main FLUX diffusion GGUF model.")
    parser.add_argument("--clip-l-name", default="flux1-clip_l.gguf", help="Filename of the FLUX CLIP L GGUF model.")
    parser.add_argument("--t5xxl-name", default="flux1-t5xxl.gguf", help="Filename of the FLUX T5 XXL GGUF model.")
    parser.add_argument("--vae-name", default="flux1-ae.gguf", help="Filename of the FLUX VAE GGUF model.")
    parser.add_argument("--w-device", default="default", help="Device hint for stable-diffusion-cpp (e.g., 'cpu', 'cuda:0', 'mps', 'default').")
    parser.add_argument("--rng-type", default="std_default", choices=["std_default", "cuda"], help="RNG type for stable-diffusion-cpp.")
    parser.add_argument("--n-threads", type=int, default=0, help="Number of threads for CPU operations (0 for auto/default, positive for specific count).")
    parser.add_argument("--test-mode", action="store_true", help="Run in test mode with fixed parameters.")
    parser.add_argument("--test-prompt", default="a realistic RTX goofy doraemon blue smile cat holding a sign says 'Hello'", help="Prompt for test mode.")
    parser.add_argument("--test-output-file", default="zephyImagination_test_image.png", help="Output filename for test mode image.")

    args = parser.parse_args()

    if not STABLE_DIFFUSION_CPP_AVAILABLE:
        log_worker("CRITICAL", "stable-diffusion-cpp library failed to import. Worker cannot run.")
        print(json.dumps({"error": "stable-diffusion-cpp library not available in worker environment"}), flush=True)
        sys.exit(1)

    log_worker("INFO", f"Starting Imagination Worker. PID: {os.getpid()}. Device Hint: {args.w_device}, TestMode: {args.test_mode}, RNG Type: {args.rng_type}, n_threads_arg: {args.n_threads}")
    if PIL_AVIF_PLUGIN_POSSIBLY_AVAILABLE:
        log_worker("INFO", "AVIF plugin seems available, AVIF conversion will be attempted.")
    else:
        log_worker("WARNING", "AVIF plugin not available or not detected, AVIF conversion will be skipped.")

    try:
        diffusion_model_path_arg = os.path.join(args.model_dir, args.diffusion_model_name)
        clip_l_path_arg = os.path.join(args.model_dir, args.clip_l_name)
        t5xxl_path_arg = os.path.join(args.model_dir, args.t5xxl_name)
        vae_path_arg = os.path.join(args.model_dir, args.vae_name)
        paths_to_check = { "FLUX Diffusion Model": diffusion_model_path_arg, "FLUX CLIP L Model": clip_l_path_arg, "FLUX T5 XXL Model": t5xxl_path_arg, "FLUX VAE Model": vae_path_arg }
        for name, path_to_check in paths_to_check.items():
            if not os.path.exists(path_to_check): raise FileNotFoundError(f"{name} not found at: {path_to_check}")
            log_worker("DEBUG", f"Found {name} at: {path_to_check}")
    except FileNotFoundError as fe:
        log_worker("ERROR", f"Model file not found: {fe}")
        print(json.dumps({"error": f"Model file missing: {fe}"}), flush=True); sys.exit(1)
    except Exception as path_err:
        log_worker("ERROR", f"Error with model paths: {path_err}")
        print(json.dumps({"error": f"Error with model paths: {path_err}"}), flush=True); sys.exit(1)

    sd_instance: Optional[StableDiffusion] = None
    try:
        load_start_time = time.monotonic()
        log_worker("INFO", "Initializing StableDiffusion instance for FLUX model...")
        rng_type_map = {"std_default": 0, "cuda": 1}; rng_type_val = rng_type_map.get(args.rng_type.lower(), 0)
        n_threads_for_constructor: int
        if args.n_threads > 0: n_threads_for_constructor = args.n_threads; log_worker("DEBUG", f"User specified n_threads: {n_threads_for_constructor}")
        else:
            cpu_count = os.cpu_count()
            if cpu_count is not None and cpu_count > 0: n_threads_for_constructor = cpu_count; log_worker("DEBUG", f"Auto-detected n_threads using os.cpu_count(): {n_threads_for_constructor}")
            else: n_threads_for_constructor = 4; log_worker("WARNING", f"os.cpu_count() failed. Falling back to n_threads={n_threads_for_constructor}")
        log_worker("INFO", f"Passing n_threads={n_threads_for_constructor} to StableDiffusion constructor.")

        # *** Add more logging from stable-diffusion-cpp if possible ***
        # Check if the library has a way to set its internal log level, e.g., via an environment variable
        # or a parameter during StableDiffusion instantiation like `log_level=sd.SDLogLevel.DEBUG` (hypothetical).
        # If it prints to C++ stdout/stderr, that should be captured.
        sd_instance = StableDiffusion(
            diffusion_model_path=diffusion_model_path_arg, clip_l_path=clip_l_path_arg,
            t5xxl_path=t5xxl_path_arg, vae_path=vae_path_arg, vae_decode_only=True,
            n_threads=n_threads_for_constructor, rng_type=rng_type_val,
        )
        load_duration = time.monotonic() - load_start_time
        log_worker("INFO", f"StableDiffusion FLUX model loaded in {load_duration:.2f}s.")
    except TypeError as te:
        log_worker("ERROR", f"StableDiffusion model loading failed due to TypeError: {te}\n{traceback.format_exc()}"); print(json.dumps({"error": f"Worker model load TypeError: {te}"}), flush=True); sys.exit(1)
    except Exception as e:
        log_worker("ERROR", f"StableDiffusion model loading failed: {e}\n{traceback.format_exc()}"); print(json.dumps({"error": f"Worker model load error: {e}"}), flush=True); sys.exit(1)

    if args.test_mode:
        log_worker("INFO", "Running in TEST MODE with FLUX parameters.")
        try:
            log_worker("INFO", f"Test prompt: '{args.test_prompt}'")
            log_worker("INFO", "Calling txt_to_img for test mode...") # <<< Added log
            output_pil_images = sd_instance.txt_to_img(
                prompt=args.test_prompt, negative_prompt="Bad Morphed Graphic or Body",
                sample_method="euler", sample_steps=5, cfg_scale=1.0,
                width=768, height=448, seed=-1,
            )
            log_worker("INFO", f"txt_to_img (test mode) call returned. Images: {len(output_pil_images) if output_pil_images else 0}") # <<< Added log
            if output_pil_images and len(output_pil_images) > 0:
                first_image: Image.Image = output_pil_images[0]
                output_file_path = os.path.join(os.getcwd(), args.test_output_file)
                first_image.save(output_file_path); log_worker("INFO", f"Test image generated and saved to {output_file_path}")
                print(json.dumps({"result": {"status": "Test image generated", "file": output_file_path}}), flush=True)
            else:
                log_worker("ERROR", "Test mode: txt_to_img returned no images."); print(json.dumps({"error": "Test mode: No image generated"}), flush=True)
        except Exception as e:
            log_worker("ERROR", f"Test mode execution failed: {e}\n{traceback.format_exc()}"); print(json.dumps({"error": f"Test mode execution error: {e}"}), flush=True)
        sys.exit(0)

    result_payload = {"error": "Worker did not receive valid input or process request."}
    try:
        log_worker("INFO", "Standard Mode: Waiting for request data on stdin...")
        input_json_str = sys.stdin.read()
        if not input_json_str: raise ValueError("Received empty input from stdin.")
        log_worker("DEBUG", f"Received raw input string (len={len(input_json_str)}): {input_json_str[:200]}...")
        request_data = json.loads(input_json_str); log_worker("INFO", "Request data JSON parsed successfully.")

        task_type = request_data.get("task_type", "txt2img").lower()
        prompt = request_data.get("prompt")
        negative_prompt = request_data.get("negative_prompt", ""); num_images = int(request_data.get("n", 1))
        size_str = request_data.get("size", "768x448"); width, height = 768, 448 # Default for FLUX
        try: width, height = map(int, size_str.split('x'))
        except ValueError: log_worker("WARNING", f"Invalid size '{size_str}', defaulting to {width}x{height}")
        is_flux_model = "flux" in args.diffusion_model_name.lower()
        cfg_scale = float(request_data.get("cfg_scale", 1.0 if is_flux_model else 7.0))
        sample_steps = int(request_data.get("sample_steps", 4 if is_flux_model else 4))
        sample_method_str = str(request_data.get("sample_method", "euler" if is_flux_model else "euler_a")).lower()
        seed = int(request_data.get("seed", -1)); response_format = request_data.get("response_format", "b64_json").lower()
        input_image_b64 = request_data.get("input_image_b64")
        if not prompt or not isinstance(prompt, str): raise ValueError("Missing or invalid 'prompt' in request.")

        log_worker("INFO", f"Task: {task_type}, Prompt: '{prompt[:50]}...', N: {num_images}, Size: {width}x{height}, Steps: {sample_steps}, CFG: {cfg_scale}, Method: {sample_method_str}, Seed: {seed}")

        generated_pil_images: list[Image.Image] = []
        task_start_time = time.monotonic()

        # *** More specific logging around the actual generation call ***
        if task_type == "txt2img":
            log_worker("INFO", "Executing txt_to_img...")
            # If stable-diffusion-cpp has a verbose option or progress callback for txt_to_img, use it here.
            # For example (hypothetical):
            # def progress_callback(step, total_steps, _):
            #     log_worker("DEBUG", f"txt2img progress: step {step}/{total_steps}")
            #     return True # Continue
            # generated_pil_images = sd_instance.txt_to_img(..., progress_callback=progress_callback)
            generated_pil_images = sd_instance.txt_to_img(
                prompt=prompt, negative_prompt=negative_prompt, sample_method=sample_method_str,
                sample_steps=sample_steps, cfg_scale=cfg_scale, width=width, height=height, seed=seed,
            )
            log_worker("INFO", f"txt_to_img call returned. Images generated: {len(generated_pil_images) if generated_pil_images else 0}")
        elif task_type == "img2img":
            if not input_image_b64: raise ValueError("Missing 'input_image_b64' for img2img task.")
            try:
                image_bytes = base64.b64decode(input_image_b64)
                input_pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
                log_worker("INFO", f"Input image for img2img decoded. Size: {input_pil_image.size}")
            except Exception as img_decode_err: raise ValueError(f"Failed to decode 'input_image_b64': {img_decode_err}")
            strength = float(request_data.get("strength", 0.8))
            log_worker("INFO", f"Executing img_to_img... Strength: {strength}")
            generated_pil_images = sd_instance.img_to_img(
                image=input_pil_image, prompt=prompt, negative_prompt=negative_prompt,
                sample_method=sample_method_str, sample_steps=sample_steps,
                cfg_scale=cfg_scale, strength=strength, seed=seed,
            )
            log_worker("INFO", f"img_to_img call returned. Images generated: {len(generated_pil_images) if generated_pil_images else 0}")
        else:
            raise ValueError(f"Unsupported task_type: {task_type}")

        task_duration = time.monotonic() - task_start_time
        log_worker("INFO", f"Image generation task ({task_type}) completed in {task_duration:.2f}s. Generated {len(generated_pil_images)} image(s).")

        if "error" in result_payload and result_payload["error"] == "Worker did not receive valid input or process request.":
            # If it's still the default initial error, and we have images, clear it
            # so the success path can be taken.
            if generated_pil_images:
                log_worker("DEBUG", "Clearing default initial error message as images were generated.")
                result_payload = {}  # Or result_payload.pop("error", None)

        if not generated_pil_images and "error" not in result_payload:
            raise RuntimeError(f"{task_type} call returned no images unexpectedly.")

        if "error" not in result_payload:
            output_data_list = []
            if response_format == "b64_json":
                log_worker("DEBUG", f"Starting encoding for {len(generated_pil_images)} image(s)...")
                for i, pil_img in enumerate(generated_pil_images):
                    image_item_data = {}
                    log_worker("TRACE", f"Encoding image {i} to PNG...")
                    try:
                        png_buffered = BytesIO(); pil_img.save(png_buffered, format="PNG")
                        img_str_b64_png = base64.b64encode(png_buffered.getvalue()).decode('utf-8')
                        image_item_data["b64_json"] = img_str_b64_png
                        log_worker("TRACE", f"Image {i} PNG encoding complete.")
                        if PIL_AVIF_PLUGIN_POSSIBLY_AVAILABLE:
                            log_worker("TRACE", f"Attempting AVIF encoding for image {i}...")
                            try:
                                avif_buffered = BytesIO(); pil_img.save(avif_buffered, format="AVIF", quality=80)
                                img_str_b64_avif = base64.b64encode(avif_buffered.getvalue()).decode('utf-8')
                                image_item_data["b64_avif"] = img_str_b64_avif
                                log_worker("TRACE", f"Image {i} AVIF encoding successful.")
                            except Exception as avif_err:
                                log_worker("WARNING", f"Failed to encode image {i} to AVIF: {avif_err}. AVIF omitted.")
                                image_item_data["b64_avif"] = None
                        else: image_item_data["b64_avif"] = None; log_worker("TRACE", f"Skipping AVIF for image {i}, plugin N/A.")
                        output_data_list.append(image_item_data)
                    except Exception as enc_err:
                        log_worker("ERROR", f"Failed to encode image {i}: {enc_err}")
                        output_data_list.append({"error": f"Encoding failed for image {i}: {enc_err}"})
                log_worker("INFO", f"Processed {len(output_data_list)} images for b64_json response.")
            elif response_format == "url":
                log_worker("WARNING", "response_format 'url' requested, but worker returns b64_json as fallback.")
                for i, pil_img in enumerate(generated_pil_images):
                    image_item_data = {"comment": "URL format requested, b64_json returned."}
                    try:
                        png_buffered = BytesIO(); pil_img.save(png_buffered, format="PNG")
                        image_item_data["b64_json"] = base64.b64encode(png_buffered.getvalue()).decode('utf-8')
                        if PIL_AVIF_PLUGIN_POSSIBLY_AVAILABLE:
                            try: avif_buffered = BytesIO(); pil_img.save(avif_buffered, format="AVIF", quality=80); image_item_data["b64_avif"] = base64.b64encode(avif_buffered.getvalue()).decode('utf-8')
                            except Exception: image_item_data["b64_avif"] = None
                        else: image_item_data["b64_avif"] = None
                        output_data_list.append(image_item_data)
                    except Exception: output_data_list.append({"error": f"Encoding fallback failed for image {i}"})
            else: raise ValueError(f"Unsupported response_format requested: {response_format}")
            result_payload = {"created": int(time.time()), "data": output_data_list}

    except json.JSONDecodeError as e: log_worker("ERROR", f"JSON Decode Error: {e}"); result_payload = {"error": f"Worker JSON Decode Error: {e}"}
    except ValueError as e: log_worker("ERROR", f"Input Validation Error: {e}"); result_payload = {"error": f"Worker Input Error: {e}"}
    except RuntimeError as e: log_worker("ERROR", f"Runtime Error (Image Gen): {e}\n{traceback.format_exc()}"); result_payload = {"error": f"Worker Image Gen Error: {e}"}
    except Exception as e: log_worker("ERROR", f"Unexpected Error: {e}\n{traceback.format_exc()}"); result_payload = {"error": f"Worker Unexpected Error: {e}"}

    try:
        output_json = json.dumps(result_payload)
        log_worker("DEBUG", f"Sending output JSON (len={len(output_json)}): {output_json[:200]}...")
        print(output_json, flush=True)
        log_worker("INFO", "Result/Error sent to stdout.")
    except Exception as e_print:
        log_worker("ERROR", f"Failed to serialize/write result to stdout: {e_print}")
        # Attempt to print a simpler, specific error message if the main payload fails
        final_error_payload_str = ""  # Initialize
        try:
            final_error_payload = {
                "error": f"Worker critical error: Failed to write result: {str(e_print)}"
            }
            final_error_payload_str = json.dumps(final_error_payload)
            print(final_error_payload_str, flush=True)
            log_worker("DEBUG", "Printed fallback critical error JSON to stdout.")
        except Exception as final_json_err:
            # If even dumping the simple error JSON fails, write a raw string.
            log_worker("ERROR", f"Failed to even serialize the fallback critical error JSON: {final_json_err}")
            raw_err_msg = (
                f'{{"error": "Worker critical error: Failed to write result AND also failed to serialize '
                f'final error message: {str(final_json_err).replace("\"", "'")}"}}\n'
            )
            # Using sys.stdout.write for direct output without print's overhead/formatting
            sys.stdout.write(raw_err_msg)
            sys.stdout.flush()
            log_worker("DEBUG", "Printed raw string critical error to stdout.")

        # Exit with error if the original result was not an error itself,
        # but we failed to print it.
        if "error" not in result_payload and sys.exc_info()[0] is None:
            # sys.exc_info()[0] is None checks if there's no *active* exception being handled
            # by an outer try/except that this finally block might be part of.
            # If we're here because e_print happened, then an exception *is* active.
            # The goal is to exit(1) if the *original task* was a success but printing failed.
            log_worker("CRITICAL",
                       "Original task was successful but printing its result to stdout failed. Exiting with error.")
            sys.exit(1)  # Indicate failure if we couldn't send a non-error result

    log_worker("INFO", f"Imagination Worker PID {os.getpid()} process finished.")
    exit_code_final = 0
    # Check if the payload *that was supposed to be sent* (or the fallback) indicates a critical error
    # This logic might be tricky because result_payload might not have been the one attempted if e_print occurred
    # A simpler check: if e_print occurred, it's a form of critical failure for the worker's output.
    if 'e_print' in locals() and e_print is not None:  # Check if the e_print exception occurred
        log_worker("DEBUG", "Setting exit code to 1 due to failure writing original result to stdout.")
        exit_code_final = 1
    elif "error" in result_payload and "Worker critical error" in result_payload.get("error", ""):
        log_worker("DEBUG", "Setting exit code to 1 due to critical error in payload.")
        exit_code_final = 1

    sys.exit(exit_code_final)

if __name__ == "__main__":
    main()