# imagination_worker.py

import sys
import os

# --- Explicitly add the script's directory to sys.path ---
# This helps ensure 'from CortexConfiguration import ...' can find config.py
# if imagination_worker.py and config.py are in the same directory (e.g., engineMain).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Using print directly here as log_worker isn't defined yet.
print(f"[IMAGINATION_WORKER|DEBUG] Worker SCRIPT_DIR: {SCRIPT_DIR}", file=sys.stderr, flush=True)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
    print(f"[IMAGINATION_WORKER|INFO] Added {SCRIPT_DIR} to sys.path for worker.", file=sys.stderr, flush=True)
else:
    print(f"[IMAGINATION_WORKER|DEBUG] {SCRIPT_DIR} already in sys.path for worker.", file=sys.stderr, flush=True)
# print(f"[IMAGINATION_WORKER|DEBUG] Worker sys.path: {sys.path}", file=sys.stderr, flush=True) # Optional: very verbose

# --- LOAD DOTENV FOR THE WORKER PROCESS ---
try:
    from dotenv import load_dotenv

    if load_dotenv():  # Will load .env from CWD or find_dotenv()
        print("[IMAGINATION_WORKER|INFO] .env file processed by worker using load_dotenv().", file=sys.stderr,
              flush=True)
    else:
        print(
            "[IMAGINATION_WORKER|INFO] .env file not found by worker's load_dotenv(), or it was empty. Using existing env vars or defaults.",
            file=sys.stderr, flush=True)
except ImportError:
    print("[IMAGINATION_WORKER|WARNING] python-dotenv not installed, .env file will not be loaded directly by worker.",
          file=sys.stderr, flush=True)
except Exception as e_dotenv:
    print(f"[IMAGINATION_WORKER|ERROR] Error during load_dotenv() in worker: {e_dotenv}", file=sys.stderr, flush=True)

import json
import time
import traceback
import argparse
import base64
from io import BytesIO
from typing import Optional, List
import gc  # For garbage collection
import numpy as np  # For noise generation

# --- Pillow and AVIF Plugin Check ---
try:
    from PIL import Image

    if hasattr(Image.SAVE, "AVIF"):  # Check if AVIF is already registered
        PIL_AVIF_PLUGIN_POSSIBLY_AVAILABLE = True
        print("[IMAGINATION_WORKER|INFO] Pillow AVIF save format detected as potentially available.", file=sys.stderr,
              flush=True)
    else:  # If not, try to import and register the plugin
        try:
            from pillow_avif import AvifImagePlugin  # noqa F401 (plugin registers on import)

            PIL_AVIF_PLUGIN_POSSIBLY_AVAILABLE = True
            print("[IMAGINATION_WORKER|INFO] Pillow AVIF plugin dynamically registered and potentially available.",
                  file=sys.stderr, flush=True)
        except ImportError:
            PIL_AVIF_PLUGIN_POSSIBLY_AVAILABLE = False
            print("[IMAGINATION_WORKER|WARNING] pillow-avif-plugin not found. AVIF conversion will be skipped.",
                  file=sys.stderr, flush=True)
except ImportError:
    print("[IMAGINATION_WORKER|ERROR] Pillow (PIL) not found in imagination_worker environment.", file=sys.stderr,
          flush=True)
    sys.exit(1)

# --- Stable Diffusion CPP Check ---
STABLE_DIFFUSION_CPP_AVAILABLE = False
try:
    from stable_diffusion_cpp import StableDiffusion

    STABLE_DIFFUSION_CPP_AVAILABLE = True
except ImportError:
    print("[IMAGINATION_WORKER|ERROR] stable-diffusion-cpp library not found.", file=sys.stderr, flush=True)
    # Further checks in main will exit if this is False

# --- Config Import with Fallbacks ---
# Define fallbacks FIRST, at the module level
REFINEMENT_MODEL_ENABLED_DEFAULT = False
REFINEMENT_MODEL_NAME_DEFAULT = "sd-refinement.gguf"
REFINEMENT_PROMPT_PREFIX_DEFAULT = "Masterpiece, Amazing, 4k, "
REFINEMENT_PROMPT_SUFFIX_DEFAULT = ", highly detailed, sharp focus, intricate details, best quality, award winning photography"
REFINEMENT_STRENGTH_DEFAULT = 0.369420
REFINEMENT_CFG_SCALE_DEFAULT = 7.0
REFINEMENT_SAMPLE_METHOD_DEFAULT = "dpmpp2mv2"
REFINEMENT_ADD_NOISE_STRENGTH_DEFAULT = 0.4

# Assign defaults initially
REFINEMENT_MODEL_ENABLED = REFINEMENT_MODEL_ENABLED_DEFAULT
REFINEMENT_MODEL_NAME = REFINEMENT_MODEL_NAME_DEFAULT
REFINEMENT_PROMPT_PREFIX = REFINEMENT_PROMPT_PREFIX_DEFAULT
REFINEMENT_PROMPT_SUFFIX = REFINEMENT_PROMPT_SUFFIX_DEFAULT
REFINEMENT_STRENGTH = REFINEMENT_STRENGTH_DEFAULT
REFINEMENT_CFG_SCALE = REFINEMENT_CFG_SCALE_DEFAULT
REFINEMENT_SAMPLE_METHOD = REFINEMENT_SAMPLE_METHOD_DEFAULT
REFINEMENT_ADD_NOISE_STRENGTH = REFINEMENT_ADD_NOISE_STRENGTH_DEFAULT

try:
    # Attempt to import from CortexConfiguration, potentially overriding the defaults above
    from CortexConfiguration import (
        REFINEMENT_MODEL_ENABLED as cfg_REFINEMENT_MODEL_ENABLED,
        REFINEMENT_MODEL_NAME as cfg_REFINEMENT_MODEL_NAME,
        REFINEMENT_PROMPT_PREFIX as cfg_REFINEMENT_PROMPT_PREFIX,
        REFINEMENT_PROMPT_SUFFIX as cfg_REFINEMENT_PROMPT_SUFFIX,
        REFINEMENT_STRENGTH as cfg_REFINEMENT_STRENGTH,
        REFINEMENT_CFG_SCALE as cfg_REFINEMENT_CFG_SCALE,
        REFINEMENT_SAMPLE_METHOD as cfg_REFINEMENT_SAMPLE_METHOD,
        REFINEMENT_ADD_NOISE_STRENGTH as cfg_REFINEMENT_ADD_NOISE_STRENGTH
    )

    # If import succeeds, use the values from CortexConfiguration.py
    REFINEMENT_MODEL_ENABLED = cfg_REFINEMENT_MODEL_ENABLED
    REFINEMENT_MODEL_NAME = cfg_REFINEMENT_MODEL_NAME
    REFINEMENT_PROMPT_PREFIX = cfg_REFINEMENT_PROMPT_PREFIX
    REFINEMENT_PROMPT_SUFFIX = cfg_REFINEMENT_PROMPT_SUFFIX
    REFINEMENT_STRENGTH = cfg_REFINEMENT_STRENGTH
    REFINEMENT_CFG_SCALE = cfg_REFINEMENT_CFG_SCALE
    REFINEMENT_SAMPLE_METHOD = cfg_REFINEMENT_SAMPLE_METHOD
    REFINEMENT_ADD_NOISE_STRENGTH = cfg_REFINEMENT_ADD_NOISE_STRENGTH
    print("[IMAGINATION_WORKER|INFO] Successfully imported refinement settings from CortexConfiguration.py.", file=sys.stderr,
          flush=True)
except ImportError as e_config_import:
    print(
        f"[IMAGINATION_WORKER|WARNING] Could not import settings from CortexConfiguration.py: {e_config_import}. Using internal defaults for refinement.",
        file=sys.stderr, flush=True)
    # Defaults assigned above are already in place if import fails


def log_worker(level, message):
    print(f"[IMAGINATION_WORKER|{level}] {message}", file=sys.stderr, flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Stable Diffusion Image Generation Worker (FLUX + Optional Refinement)")
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
    parser.add_argument("--test-prompt", default="photo of a majestic cat king sitting on a throne, epic fantasy art",
                        help="Prompt for test mode.")
    parser.add_argument("--test-output-file", default="zephyImagination_test_image.png",
                        help="Output filename for test mode image.")

    args = parser.parse_args()

    # Use a local variable for refinement status within main, initialized from module-level config
    current_refinement_enabled_status = REFINEMENT_MODEL_ENABLED

    if not STABLE_DIFFUSION_CPP_AVAILABLE:
        log_worker("CRITICAL", "stable-diffusion-cpp library failed to import. Worker cannot run.")
        print(json.dumps({"error": "stable-diffusion-cpp library not available in worker environment"}), flush=True)
        sys.exit(1)

    log_worker("INFO",
               f"Starting Imagination Worker (FLUX + Refiner). PID: {os.getpid()}. Device Arg (from cmd): {args.w_device}, TestMode: {args.test_mode}, RNG: {args.rng_type}, Threads: {args.n_threads}")
    log_worker("INFO", f"Initial Refinement Stage Enabled (from CortexConfiguration/default): {current_refinement_enabled_status}")
    if current_refinement_enabled_status:
        log_worker("INFO", f"Refinement Noise Strength (from CortexConfiguration/default): {REFINEMENT_ADD_NOISE_STRENGTH}")

    if PIL_AVIF_PLUGIN_POSSIBLY_AVAILABLE:
        log_worker("INFO", "AVIF plugin seems available, AVIF conversion will be attempted if enabled by format.")
    else:
        log_worker("WARNING", "AVIF plugin not available or not detected, AVIF conversion will be skipped.")

    n_threads_for_sd_constructor: int
    if args.n_threads > 0:
        n_threads_for_sd_constructor = args.n_threads
        log_worker("DEBUG", f"User specified n_threads: {n_threads_for_sd_constructor}")
    else:
        cpu_count = os.cpu_count()
        if cpu_count is not None and cpu_count > 0:
            n_threads_for_sd_constructor = cpu_count
            log_worker("DEBUG", f"Auto-detected n_threads using os.cpu_count(): {n_threads_for_sd_constructor}")
        else:
            n_threads_for_sd_constructor = 4
            log_worker("WARNING",
                       f"os.cpu_count() failed or returned 0. Falling back to n_threads={n_threads_for_sd_constructor}")
    log_worker("INFO", f"Using n_threads={n_threads_for_sd_constructor} for StableDiffusion instances.")
    rng_type_map = {"std_default": 0, "cuda": 1}
    rng_type_val = rng_type_map.get(args.rng_type.lower(), 0)

    sd_flux_instance: Optional[StableDiffusion] = None
    try:
        log_worker("INFO", "Loading Stage 1: FLUX Model...")
        flux_diffusion_path = os.path.join(args.model_dir, args.diffusion_model_name)
        flux_clip_l_path = os.path.join(args.model_dir, args.clip_l_name)
        flux_t5xxl_path = os.path.join(args.model_dir, args.t5xxl_name)
        flux_vae_path = os.path.join(args.model_dir, args.vae_name)

        paths_to_check_flux = {
            "FLUX Diffusion Model": flux_diffusion_path, "FLUX CLIP L Model": flux_clip_l_path,
            "FLUX T5 XXL Model": flux_t5xxl_path, "FLUX VAE Model": flux_vae_path
        }
        for name, path_to_check in paths_to_check_flux.items():
            if not os.path.exists(path_to_check):
                raise FileNotFoundError(f"{name} not found at: {path_to_check}")
            log_worker("DEBUG", f"Found {name} at: {path_to_check}")

        sd_flux_instance = StableDiffusion(
            diffusion_model_path=flux_diffusion_path,
            clip_l_path=flux_clip_l_path,
            t5xxl_path=flux_t5xxl_path,
            vae_path=flux_vae_path,
            vae_decode_only=True,
            n_threads=n_threads_for_sd_constructor,
            rng_type=rng_type_val
            # w_device removed as it caused TypeError
        )
        log_worker("INFO", "Stage 1: FLUX Model loaded successfully.")
    except FileNotFoundError as fe_flux:
        log_worker("CRITICAL",
                   f"Failed to load Stage 1 FLUX model (File Not Found): {fe_flux}\n{traceback.format_exc()}")
        print(json.dumps({"error": f"Worker critical error: FLUX model file missing: {fe_flux}"}), flush=True)
        sys.exit(1)
    except Exception as e_flux_load:
        log_worker("CRITICAL",
                   f"Failed to load Stage 1 FLUX model (Other Error): {e_flux_load}\n{traceback.format_exc()}")
        print(json.dumps({"error": f"Worker critical error: Failed to load FLUX model: {e_flux_load}"}), flush=True)
        sys.exit(1)

    if args.test_mode:
        log_worker("INFO", "Running in TEST MODE.")
        current_prompt_for_test = args.test_prompt
        flux_sample_steps_for_test = 20
        sd_refiner_instance_test: Optional[StableDiffusion] = None

        if current_refinement_enabled_status:
            try:
                log_worker("INFO", "[Test Mode] Loading Stage 2: Refinement Model...")
                refinement_model_path_test = os.path.join(args.model_dir, REFINEMENT_MODEL_NAME)
                if not os.path.exists(refinement_model_path_test):
                    raise FileNotFoundError(f"Refinement model for test not found at {refinement_model_path_test}")
                sd_refiner_instance_test = StableDiffusion(
                    model_path=refinement_model_path_test,
                    n_threads=n_threads_for_sd_constructor, rng_type=rng_type_val
                )
                log_worker("INFO", "[Test Mode] Stage 2: Refinement Model loaded.")
            except Exception as e_ref_test_load:
                log_worker("ERROR",
                           f"[Test Mode] Failed to load Refinement model: {e_ref_test_load}. Refinement will be skipped.")
                sd_refiner_instance_test = None  # Ensure it's None if load fails

        log_worker("INFO", f"[Test Mode] Stage 1: Generating FLUX image with prompt: '{current_prompt_for_test}'")
        stage1_test_images = sd_flux_instance.txt_to_img(
            prompt=current_prompt_for_test, negative_prompt="Bad Morphed Graphic, ugly, disfigured",
            sample_method="euler", sample_steps=flux_sample_steps_for_test, cfg_scale=1.0,
            width=768, height=448, seed=-1,
        )
        if not stage1_test_images:
            log_worker("ERROR", "[Test Mode] Stage 1 FLUX generation failed to produce an image.")
            sys.exit(1)

        final_test_image: Image.Image = stage1_test_images[0]
        log_worker("INFO", f"[Test Mode] Stage 1 FLUX image generated. Size: {final_test_image.size}")

        if current_refinement_enabled_status and sd_refiner_instance_test:
            log_worker("INFO", "[Test Mode] Stage 2: Refining image...")
            stage1_image_for_refiner_test = final_test_image.convert("RGB")
            if REFINEMENT_ADD_NOISE_STRENGTH > 0:
                try:
                    log_worker("DEBUG",
                               f"[Test Mode] Adding noise (strength {REFINEMENT_ADD_NOISE_STRENGTH}) to Stage 1 image before refinement.")
                    img_array_test = np.array(stage1_image_for_refiner_test).astype(np.float32)
                    noise_test = np.random.normal(0, REFINEMENT_ADD_NOISE_STRENGTH, img_array_test.shape)
                    noisy_img_array_test = np.clip(img_array_test + noise_test, 0, 255)
                    stage1_image_for_refiner_test = Image.fromarray(noisy_img_array_test.astype(np.uint8), 'RGB')
                    log_worker("INFO", f"[Test Mode] Noise added to Stage 1 image.")
                except Exception as noise_err_test:
                    log_worker("WARNING",
                               f"[Test Mode] Failed to add noise: {noise_err_test}. Using original Stage 1 image for refinement.")

            refiner_prompt_for_test = REFINEMENT_PROMPT_PREFIX + current_prompt_for_test + REFINEMENT_PROMPT_SUFFIX
            refiner_steps_for_test = flux_sample_steps_for_test * 2
            log_worker("INFO",
                       f"[Test Mode] Refiner prompt: '{refiner_prompt_for_test[:100]}...', Steps: {refiner_steps_for_test}, Strength: {REFINEMENT_STRENGTH}")
            try:
                refined_test_images = sd_refiner_instance_test.img_to_img(
                    image=stage1_image_for_refiner_test, prompt=refiner_prompt_for_test,
                    negative_prompt="blurry, low quality, noisy, grain", sample_method=REFINEMENT_SAMPLE_METHOD,
                    sample_steps=refiner_steps_for_test, cfg_scale=REFINEMENT_CFG_SCALE,
                    strength=REFINEMENT_STRENGTH, seed=-1, width=768, height=448
                )
                if refined_test_images:
                    final_test_image = refined_test_images[0]
                    log_worker("INFO", f"[Test Mode] Stage 2 Refinement successful. Size: {final_test_image.size}")
                else:
                    log_worker("WARNING", "[Test Mode] Refinement failed or returned no image. Using Stage 1 image.")
            except Exception as e_ref_test_gen:
                log_worker("ERROR",
                           f"[Test Mode] Refinement generation error: {e_ref_test_gen}. Using Stage 1 image.\n{traceback.format_exc()}")

        output_file_path_test = os.path.join(os.getcwd(), args.test_output_file)
        try:
            final_test_image.save(output_file_path_test)
            log_worker("INFO", f"Test image (final) saved to {output_file_path_test}")
            print(json.dumps({"result": {"status": "Test image generated", "file": output_file_path_test}}), flush=True)
        except Exception as save_err:
            log_worker("ERROR", f"[Test Mode] Failed to save test image: {save_err}")
            print(json.dumps({"error": f"Test mode image save error: {save_err}"}), flush=True)
        sys.exit(0)

    result_payload = {"error": "Worker did not receive valid input or process request."}
    try:
        log_worker("INFO", "Standard Mode: Waiting for request data on stdin...")
        input_json_str = sys.stdin.read()
        if not input_json_str: raise ValueError("Received empty input from stdin.")
        request_data = json.loads(input_json_str)
        log_worker("INFO", f"Request data JSON parsed: {str(request_data)[:200]}...")

        task_type = request_data.get("task_type", "txt2img").lower()
        original_prompt_from_request = request_data.get("prompt")
        negative_prompt_stage1 = request_data.get("negative_prompt", "Bad Morphed Graphic, ugly, disfigured, blurry")
        size_str = request_data.get("size", "768x448")
        width, height = 768, 448
        try:
            width, height = map(int, size_str.split('x'))
        except ValueError:
            log_worker("WARNING", f"Invalid size '{size_str}', using {width}x{height}")

        cfg_scale_flux = float(request_data.get("cfg_scale", 1.0))
        sample_steps_flux = int(request_data.get("sample_steps", 4))
        sample_method_flux = str(request_data.get("sample_method", "euler")).lower()
        seed = int(request_data.get("seed", -1))
        response_format = request_data.get("response_format", "b64_json").lower()
        input_image_b64_for_stage1 = request_data.get("input_image_b64")

        if not original_prompt_from_request: raise ValueError("Missing 'prompt'.")

        log_worker("INFO",
                   f"[Stage 1] Task: {task_type}, Original Prompt: '{original_prompt_from_request[:50]}...', Steps: {sample_steps_flux}, Size: {width}x{height}")

        stage1_pil_images_list: List[Image.Image] = []
        stage1_execution_start_time = time.monotonic()

        if task_type == "txt2img":
            log_worker("INFO", "[Stage 1] Executing FLUX txt_to_img...")
            stage1_pil_images_list = sd_flux_instance.txt_to_img(
                prompt=original_prompt_from_request, negative_prompt=negative_prompt_stage1,
                sample_method=sample_method_flux, sample_steps=sample_steps_flux,
                cfg_scale=cfg_scale_flux, width=width, height=height, seed=seed,
            )
        elif task_type == "img2img":
            if not input_image_b64_for_stage1: raise ValueError("Missing 'input_image_b64' for Stage 1 img2img task.")
            img_bytes_stage1 = base64.b64decode(input_image_b64_for_stage1)
            input_pil_image_stage1 = Image.open(BytesIO(img_bytes_stage1)).convert("RGB")
            strength_stage1 = float(request_data.get("strength", 0.65))
            log_worker("INFO", f"[Stage 1] Executing FLUX img_to_img... Strength: {strength_stage1}")
            stage1_pil_images_list = sd_flux_instance.img_to_img(
                image=input_pil_image_stage1, prompt=original_prompt_from_request,
                negative_prompt=negative_prompt_stage1,
                sample_method=sample_method_flux, sample_steps=sample_steps_flux,
                cfg_scale=cfg_scale_flux, strength=strength_stage1, seed=seed,
                width=width, height=height
            )
        else:
            raise ValueError(f"Unsupported task_type for Stage 1: {task_type}")

        log_worker("INFO",
                   f"[Stage 1] FLUX generation took {time.monotonic() - stage1_execution_start_time:.2f}s. Images generated: {len(stage1_pil_images_list)}")

        if not stage1_pil_images_list:
            raise RuntimeError("Stage 1 FLUX generation failed to produce any image.")
        stage1_output_pil_image = stage1_pil_images_list[0]

        final_output_image: Image.Image = stage1_output_pil_image
        sd_refiner_instance_main: Optional[StableDiffusion] = None

        if current_refinement_enabled_status:
            log_worker("INFO", "[Transition] Unloading FLUX model to free memory for Refiner...")
            del sd_flux_instance
            sd_flux_instance = None
            gc.collect()
            log_worker("INFO", "[Transition] FLUX model instance deleted. Waiting briefly for resources to release...")
            time.sleep(1.5)  # Slightly adjusted wait time
            try:
                log_worker("INFO", f"[Stage 2] Loading Refinement Model ({REFINEMENT_MODEL_NAME})...")
                refinement_model_path_main = os.path.join(args.model_dir, REFINEMENT_MODEL_NAME)
                if not os.path.exists(refinement_model_path_main):
                    raise FileNotFoundError(f"Refinement model for main task not found at {refinement_model_path_main}")
                sd_refiner_instance_main = StableDiffusion(
                    model_path=refinement_model_path_main,
                    n_threads=n_threads_for_sd_constructor, rng_type=rng_type_val
                )
                log_worker("INFO", "[Stage 2] Refinement Model loaded successfully.")
            except Exception as e_ref_main_load:
                log_worker("ERROR",
                           f"[Stage 2] Failed to load Refinement model: {e_ref_main_load}. Using Stage 1 image.\n{traceback.format_exc()}")
                sd_refiner_instance_main = None  # Ensure it's None if load fails

        if current_refinement_enabled_status and sd_refiner_instance_main:
            log_worker("INFO", "[Stage 2] Starting Refinement Process...")
            refinement_execution_start_time = time.monotonic()

            stage1_image_for_refiner = stage1_output_pil_image.convert("RGB")
            if REFINEMENT_ADD_NOISE_STRENGTH > 0:
                try:
                    log_worker("DEBUG",
                               f"[Stage 2 Prep] Adding noise (strength {REFINEMENT_ADD_NOISE_STRENGTH}) to Stage 1 image before refinement.")
                    img_array = np.array(stage1_image_for_refiner).astype(np.float32)
                    noise = np.random.normal(0, REFINEMENT_ADD_NOISE_STRENGTH, img_array.shape)
                    noisy_img_array = np.clip(img_array + noise, 0, 255)
                    stage1_image_for_refiner = Image.fromarray(noisy_img_array.astype(np.uint8), 'RGB')
                    log_worker("INFO",
                               f"[Stage 2 Prep] Noise (strength {REFINEMENT_ADD_NOISE_STRENGTH}) added to Stage 1 image.")
                except Exception as noise_err:
                    log_worker("WARNING",
                               f"[Stage 2 Prep] Failed to add noise: {noise_err}. Using original Stage 1 image for refinement.")

            refiner_input_prompt = REFINEMENT_PROMPT_PREFIX + original_prompt_from_request + REFINEMENT_PROMPT_SUFFIX
            refiner_num_sample_steps = sample_steps_flux * 2
            # Ensure refiner steps are reasonable, e.g., at least 10-15, max 50 for speed
            refiner_num_sample_steps = min(max(refiner_num_sample_steps, 10), 50)

            refiner_negative_prompt = "blurry, low resolution, bad quality, artifacts, watermark, signature, text, deformed, disfigured, extra limbs, bad anatomy, ugly, jpeg artifacts, lowres"

            log_worker("INFO",
                       f"[Stage 2] Refiner Prompt: '{refiner_input_prompt[:100]}...', Steps: {refiner_num_sample_steps}, Strength: {REFINEMENT_STRENGTH}, CFG: {REFINEMENT_CFG_SCALE}")
            try:
                refined_pil_images_list = sd_refiner_instance_main.img_to_img(
                    image=stage1_image_for_refiner, prompt=refiner_input_prompt,
                    negative_prompt=refiner_negative_prompt, sample_method=REFINEMENT_SAMPLE_METHOD,
                    sample_steps=refiner_num_sample_steps, cfg_scale=REFINEMENT_CFG_SCALE,
                    strength=REFINEMENT_STRENGTH, seed=seed + 1 if seed != -1 else -1,  # Vary seed
                    width=width, height=height
                )
                log_worker("INFO",
                           f"[Stage 2] Refinement process took {time.monotonic() - refinement_execution_start_time:.2f}s. Images refined: {len(refined_pil_images_list)}")
                if refined_pil_images_list:
                    final_output_image = refined_pil_images_list[0]
                    log_worker("INFO", "[Stage 2] Refinement successful. Using refined image as final output.")
                else:
                    log_worker("WARNING",
                               "[Stage 2] Refinement process returned no image. Using Stage 1 FLUX image as final output.")
            except Exception as refine_err_main:
                log_worker("ERROR",
                           f"[Stage 2] Refinement process failed: {refine_err_main}. Using Stage 1 FLUX image.\n{traceback.format_exc()}")
        else:
            if current_refinement_enabled_status and not sd_refiner_instance_main:
                log_worker("INFO", "Refinement stage skipped (Refiner model failed to load). Using Stage 1 image.")
            elif not current_refinement_enabled_status:
                log_worker("INFO", "Refinement stage skipped (disabled by config/default). Using Stage 1 image.")

        output_data_list_final = []
        if response_format == "b64_json":
            current_image_item_data = {}
            try:
                png_byte_buffer = BytesIO()
                final_output_image.save(png_byte_buffer, format="PNG")
                current_image_item_data["b64_json"] = base64.b64encode(png_byte_buffer.getvalue()).decode('utf-8')
                if PIL_AVIF_PLUGIN_POSSIBLY_AVAILABLE:
                    try:
                        avif_byte_buffer = BytesIO()
                        final_output_image.save(avif_byte_buffer, format="AVIF", quality=80)
                        current_image_item_data["b64_avif"] = base64.b64encode(avif_byte_buffer.getvalue()).decode(
                            'utf-8')
                    except Exception as avif_e:
                        current_image_item_data["b64_avif"] = None
                        log_worker("WARNING",
                                                                               f"AVIF encoding failed: {avif_e}")
                else:
                    current_image_item_data["b64_avif"] = None
                output_data_list_final.append(current_image_item_data)
            except Exception as final_encode_error:
                log_worker("ERROR", f"Final image encoding (PNG/AVIF) failed: {final_encode_error}")
                output_data_list_final.append({"error": f"Final image encoding failed: {final_encode_error}"})
        elif response_format == "url":
            log_worker("WARNING", "response_format 'url' requested, but worker returns b64_json as fallback.")
            current_image_item_data = {"comment": "URL format requested, returning b64_json as fallback."}
            try:
                png_byte_buffer = BytesIO()
                final_output_image.save(png_byte_buffer, format="PNG")
                current_image_item_data["b64_json"] = base64.b64encode(png_byte_buffer.getvalue()).decode('utf-8')
                if response_format == "b64_json":
                    current_image_item_data = {}
                    try:
                        # Create the standard PNG version for immediate VLM use
                        png_byte_buffer = BytesIO()
                        final_output_image.save(png_byte_buffer, format="PNG")
                        current_image_item_data["b64_json"] = base64.b64encode(png_byte_buffer.getvalue()).decode(
                            'utf-8')

                        # Now, create the efficient AVIF version for storage
                        if PIL_AVIF_PLUGIN_POSSIBLY_AVAILABLE:
                            try:
                                avif_byte_buffer = BytesIO()
                                # *** THIS IS THE ONLY CHANGE NEEDED ***
                                # Change the quality from 80 to the requested 40
                                final_output_image.save(avif_byte_buffer, format="AVIF", quality=40)
                                current_image_item_data["b64_avif"] = base64.b64encode(
                                    avif_byte_buffer.getvalue()).decode('utf-8')
                                log_worker("INFO", "Successfully encoded final image to AVIF with quality=40.")
                            except Exception as avif_e:
                                current_image_item_data["b64_avif"] = None
                                log_worker("WARNING", f"AVIF encoding failed with quality=40: {avif_e}")
                        else:
                            current_image_item_data["b64_avif"] = None  # Indicate AVIF is not available

                        output_data_list_final.append(current_image_item_data)
                    except Exception as final_encode_error:
                        log_worker("ERROR", f"Final image encoding (PNG) failed: {final_encode_error}")
                        output_data_list_final.append({"error": f"Final image encoding failed: {final_encode_error}"})
                else:
                    current_image_item_data["b64_avif"] = None
                output_data_list_final.append(current_image_item_data)
            except:
                output_data_list_final.append({"error": "Encoding fallback for URL format failed."})
        else:
            raise ValueError(f"Unsupported response_format: {response_format}")

        if not output_data_list_final or "error" in output_data_list_final[0]:
            error_detail = output_data_list_final[0].get("error",
                                                         "Unknown error during final image processing.") if output_data_list_final else "No image data processed."
            result_payload = {"error": f"Failed to prepare final image output: {error_detail}"}
        else:
            result_payload = {"created": int(time.time()), "data": output_data_list_final}

        log_worker("DEBUG",
                   f"WORKER: Successfully prepared result_payload. Keys: {list(result_payload.keys())}. Data items: {len(result_payload.get('data', []))}")

    except json.JSONDecodeError as e:
        log_worker("ERROR", f"JSON Decode Error: {e}"); result_payload = {"error": f"Worker JSON Decode Error: {e}"}
    except ValueError as e:
        log_worker("ERROR", f"Input Validation Error: {e}"); result_payload = {"error": f"Worker Input Error: {e}"}
    except RuntimeError as e:
        log_worker("ERROR", f"Runtime Error (Image Gen): {e}\n{traceback.format_exc()}")
        result_payload = {
            "error": f"Worker Image Gen Error: {e}"}
    except Exception as e:
        log_worker("ERROR", f"Unexpected Error: {e}\n{traceback.format_exc()}")
        result_payload = {
            "error": f"Worker Unexpected Error: {e}"}

    try:
        output_json_str_final = json.dumps(result_payload)
        log_worker("DEBUG", f"Sending output JSON (len={len(output_json_str_final)}): {output_json_str_final[:200]}...")
        print(output_json_str_final, flush=True)
        log_worker("INFO", "Result/Error sent to stdout.")
    except Exception as e_print_final:
        log_worker("ERROR", f"Failed to serialize/write result to stdout: {e_print_final}")
        final_error_payload_str_print = ""
        try:
            final_error_payload_print_dict = {
                "error": f"Worker critical error: Failed to write result: {str(e_print_final)}"}
            final_error_payload_str_print = json.dumps(final_error_payload_print_dict)
            print(final_error_payload_str_print, flush=True)
            log_worker("DEBUG", "Printed fallback critical error JSON to stdout.")
        except Exception as final_json_err_print:
            log_worker("ERROR", f"Failed to even serialize the fallback critical error JSON: {final_json_err_print}")
            raw_err_msg_print = (
                f'{{"error": "Worker critical error: Failed to write result AND also failed to serialize '
                f'final error message: {str(final_json_err_print).replace("\"", "'")}"}}\n'
            )
            sys.stdout.write(raw_err_msg_print)
            sys.stdout.flush()
            log_worker("DEBUG", "Printed raw string critical error to stdout.")
        if "error" not in result_payload:
            log_worker("CRITICAL",
                       "Original task was successful but printing its result to stdout failed. Exiting with error code 1.")
            sys.exit(1)

    log_worker("INFO", f"Imagination Worker PID {os.getpid()} process finished.")
    final_exit_code = 0
    if 'e_print_final' in locals() and e_print_final is not None and "error" not in result_payload:
        final_exit_code = 1
    elif "error" in result_payload:
        if "Worker critical error" in result_payload.get("error", ""):
            final_exit_code = 1
        else:
            final_exit_code = 0
    sys.exit(final_exit_code)


if __name__ == "__main__":
    main()