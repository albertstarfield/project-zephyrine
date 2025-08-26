# audioProcessorCortex_backbone_provider.py
import sys
import os
import json
import time
import traceback
import gc
import numpy as np
import soundfile as sf
import tempfile
import hashlib
import torch.compiler  # <-- Add this import
import requests # Needed for catching network exceptions
from huggingface_hub.utils import HfHubHTTPError # Specific HF network error
import threading  # For the monitoring thread
from CortexConfiguration import *
import shutil # Needed for deleting the cache directory

# --- Psutil for memory monitoring ---
PSUTIL_AVAILABLE = False
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    # log_thread_worker is not defined yet, will log in initialize_models if needed
    pass

# --- PyTorch and torchaudio imports ---
TORCH_AUDIO_AVAILABLE = False
torch = None
torchaudio = None
try:
    import torch
    import torchaudio
    import torch.compiler

    TORCH_AUDIO_AVAILABLE = True
except ImportError:
    pass

# --- Basic Logging to stderr (defined early) ---
_worker_pid_for_log = os.getpid()  # Get PID early for logging


def log_thread_worker(level, message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')} AUDIO_THREAD_WORKER(PID:{_worker_pid_for_log})|{level}] {message}",
          file=sys.stderr, flush=True)


# --- ChatterboxTTS Imports (dynamically imported in initialize_models) ---
CHATTERBOX_TTS_AVAILABLE_THREAD = False
ChatterboxTTS_class_ref_thread = None
# --- PyWhisperCpp Imports (dynamically imported in initialize_models) ---
PYWHISPERCPP_AVAILABLE_THREAD = False
WhisperModel_thread = None

# Global model instances (loaded once per worker process)
chatterbox_model_instance_global = None
whisper_model_instance_global = None
current_device_global = "cpu"  # Default, will be set by worker_config

# --- Memory Monitoring Thread Control ---
memory_monitor_thread = None
stop_memory_monitor_event = threading.Event()
MEMORY_MONITOR_INTERVAL_SECONDS = 30  # Log memory usage this often
PERIODIC_CACHE_CLEAR_INTERVAL_SECONDS = 5  # Attempt cache clear this often
last_cache_clear_time = 0  # Keep track of when cache was last cleared


def _find_and_patch_config(model_object):
    """
    Attempts to find and patch the configuration of a loaded ChatterboxTTS model.
    Returns True on success, False on failure.
    """
    log_thread_worker("DEBUG", "Attempting to find and patch model configuration...")
    potential_paths = [('t3', 'model', 'config'), ('model', 'config'), ('config',)]

    underlying_hf_model_config = None
    for path_tuple in potential_paths:
        current_obj = model_object
        try:
            for attr in path_tuple:
                current_obj = getattr(current_obj, attr)
            underlying_hf_model_config = current_obj
            path_str = "model_object" + "".join([f".{a}" for a in path_tuple])
            log_thread_worker("INFO", f"[Manual Patch] Found model config at path: {path_str}")
            break
        except AttributeError:
            continue

    if underlying_hf_model_config:
        log_thread_worker("INFO", "[Manual Patch] Forcing attention implementation to 'eager'.")
        underlying_hf_model_config.attn_implementation = "eager"
        log_thread_worker("INFO", "[Manual Patch] Forcing `output_attentions=True`.")
        underlying_hf_model_config.output_attentions = True
        log_thread_worker("INFO", "[Manual Patch] Model configuration successfully patched.")
        return True
    else:
        log_thread_worker("CRITICAL", "Could not locate a valid '.config' object in the loaded model's structure.")
        log_thread_worker("CRITICAL",
                          "This is a structural incompatibility, not a download error. The model files are likely intact, but the patching code needs to be updated.")
        return False

def _try_clear_pytorch_caches(device_type_str):
    """Helper function to attempt clearing PyTorch caches for a given device type."""
    if not TORCH_AUDIO_AVAILABLE or not torch:
        return

    cleared_something = False
    if device_type_str == "cuda" and hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        log_thread_worker("DEBUG", "[CacheClear] Attempted torch.cuda.empty_cache().")
        cleared_something = True
    elif device_type_str == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()  # type: ignore
            log_thread_worker("DEBUG", "[CacheClear] Attempted torch.mps.empty_cache().")
            cleared_something = True
        except Exception as e_mps_clear:
            log_thread_worker("WARNING", f"[CacheClear] torch.mps.empty_cache() failed: {e_mps_clear}")
    elif device_type_str == "vulkan":
        log_thread_worker("DEBUG",
                          "[CacheClear] Vulkan device: No direct PyTorch empty_cache() API. Relies on driver/OS for explicit freeing.")
        # For Vulkan, an explicit gc.collect() is the main thing we can do from Python
        # if Python objects were holding references to Vulkan resources managed by PyTorch.

    # Always perform gc.collect() as part of this routine,
    # as it helps ensure Python releases its references before PyTorch tries to free memory.
    gc.collect()
    log_thread_worker("DEBUG", "[CacheClear] Performed gc.collect().")


# --- torch.compile Artifact Cache ---
# The orchestrator should provide a base cache directory.
# Example: <base_cache_dir>/tts_chatterbox/<device>/<torch_version>/<model_hash_or_id>/artifact.ptc
COMPILE_CACHE_BASE_DIR = None  # To be set from worker_config

def get_compile_artifact_path(model_name_key: str, model_identifier: str, device: str):
    """Generates a unique, persistent path for torch.compile artifacts."""
    if not COMPILE_CACHE_BASE_DIR:
        return None

    torch_version_str = torch.__version__.replace('.', '_').replace('+', '_') if TORCH_AUDIO_AVAILABLE and torch else "unknown_torch"
    # Create a stable hash for the model identifier
    model_hash = hashlib.md5(model_identifier.encode()).hexdigest()[:12]

    # Path: <base>/<model_key>/<device>/<torch_ver>/<model_hash>/
    artifact_dir = os.path.join(
        COMPILE_CACHE_BASE_DIR,
        model_name_key,
        device.replace(':', '_'),  # Sanitize device string
        torch_version_str,
        model_hash
    )
    os.makedirs(artifact_dir, exist_ok=True)
    return artifact_dir # Return the directory

def monitor_memory_usage():
    """
    Thread function to periodically log memory usage and attempt cache clearing.
    """
    global last_cache_clear_time
    process = None
    if PSUTIL_AVAILABLE:
        process = psutil.Process(os.getpid())
    else:
        log_thread_worker("WARNING", "[MemoryMonitor] psutil not available. RAM monitoring disabled in worker.")

    log_thread_worker("INFO", "[MemoryMonitor] Starting memory and periodic cache clearing thread.")
    last_cache_clear_time = time.time()  # Initialize

    while not stop_memory_monitor_event.is_set():
        current_time_monitor = time.time()
        try:
            # --- Periodic Cache Clearing ---
            if (current_time_monitor - last_cache_clear_time) >= PERIODIC_CACHE_CLEAR_INTERVAL_SECONDS:
                log_thread_worker("INFO",
                                  f"[CacheClear] Interval reached. Attempting periodic cache clear for device: {current_device_global}.")
                _try_clear_pytorch_caches(current_device_global)
                last_cache_clear_time = current_time_monitor  # Reset timer after attempting clear

            # --- Memory Logging ---
            if process:
                mem_info = process.memory_info()
                rss_mb = mem_info.rss / (1024 * 1024)
                vms_mb = mem_info.vms / (1024 * 1024)
                log_thread_worker("DEBUG", f"[MemoryMonitor] RAM: RSS={rss_mb:.2f}MB, VMS={vms_mb:.2f}MB")

            if TORCH_AUDIO_AVAILABLE and torch:
                if current_device_global == "cuda" and hasattr(torch, "cuda") and torch.cuda.is_available():
                    allocated_vram_mb = torch.cuda.memory_allocated(0) / (1024 * 1024)
                    reserved_vram_mb = torch.cuda.memory_reserved(0) / (1024 * 1024)
                    log_thread_worker("DEBUG",
                                      f"[MemoryMonitor] CUDA VRAM (Dev 0): Allocated={allocated_vram_mb:.2f}MB, Reserved={reserved_vram_mb:.2f}MB")
                elif current_device_global == "mps" and hasattr(torch, "mps"):
                    if hasattr(torch.mps, "current_allocated_memory"):
                        try:
                            allocated_mps_mb = torch.mps.current_allocated_memory() / (1024 * 1024)  # type: ignore
                            log_thread_worker("DEBUG",
                                              f"[MemoryMonitor] MPS Memory: Current PyTorch Allocated={allocated_mps_mb:.2f}MB")
                        except Exception as e_mps_mem_curr:
                            log_thread_worker("TRACE",
                                              f"[MemoryMonitor] Error getting MPS current_allocated_memory: {e_mps_mem_curr}")
                    if hasattr(torch.mps, "driver_allocated_memory"):
                        try:
                            driver_allocated_mps_mb = torch.mps.driver_allocated_memory() / (
                                        1024 * 1024)  # type: ignore
                            log_thread_worker("DEBUG",
                                              f"[MemoryMonitor] MPS Memory: Driver Allocated={driver_allocated_mps_mb:.2f}MB")
                        except Exception as e_mps_mem_drv:
                            log_thread_worker("TRACE",
                                              f"[MemoryMonitor] Error getting MPS driver_allocated_memory: {e_mps_mem_drv}")
                elif current_device_global == "vulkan":
                    log_thread_worker("DEBUG",
                                      "[MemoryMonitor] Vulkan device: Detailed PyTorch VRAM stats not readily available via simple API.")

            # Determine next sleep time
            # This ensures the loop runs roughly every MEMORY_MONITOR_INTERVAL_SECONDS for logging,
            # and also respects the cache clear interval.
            time_to_next_log_cycle = MEMORY_MONITOR_INTERVAL_SECONDS - (
                        time.time() - getattr(monitor_memory_usage, 'last_log_time_attr',
                                              time.time() - MEMORY_MONITOR_INTERVAL_SECONDS))
            monitor_memory_usage.last_log_time_attr = time.time()  # type: ignore

            time_to_next_cache_clear_cycle = PERIODIC_CACHE_CLEAR_INTERVAL_SECONDS - (
                        time.time() - last_cache_clear_time)

            sleep_duration = min(max(0.1, time_to_next_log_cycle),
                                 max(0.1, time_to_next_cache_clear_cycle))

            stop_memory_monitor_event.wait(timeout=sleep_duration)

        except Exception as e_mem_mon_loop:
            log_thread_worker("ERROR",
                              f"[MemoryMonitor] Error in monitoring/cache clear loop: {e_mem_mon_loop}\n{traceback.format_exc()}")
            if not stop_memory_monitor_event.is_set():
                # Avoid busy-looping on error
                time.sleep(max(MEMORY_MONITOR_INTERVAL_SECONDS, PERIODIC_CACHE_CLEAR_INTERVAL_SECONDS))
    log_thread_worker("INFO", "[MemoryMonitor] Memory monitoring and periodic cache clear thread stopped.")




def initialize_models(worker_config):
    global CHATTERBOX_TTS_AVAILABLE_THREAD, ChatterboxTTS_class_ref_thread
    global PYWHISPERCPP_AVAILABLE_THREAD, WhisperModel_thread
    global chatterbox_model_instance_global, whisper_model_instance_global
    global current_device_global, TORCH_AUDIO_AVAILABLE, PSUTIL_AVAILABLE, torch

    if not PSUTIL_AVAILABLE:
        log_thread_worker("WARNING", "psutil library not found during init. Detailed RAM monitoring will be disabled.")

    if torch is None or torchaudio is None:
        TORCH_AUDIO_AVAILABLE = False

    requested_device = worker_config.get("device", "cpu").lower()

    if requested_device == "cuda" and TORCH_AUDIO_AVAILABLE and hasattr(torch, "cuda") and torch.cuda.is_available():
        current_device_global = "cuda"
    elif requested_device == "mps" and TORCH_AUDIO_AVAILABLE and hasattr(torch, "backends") and hasattr(torch.backends,
                                                                                                        "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        current_device_global = "mps"
    elif requested_device == "vulkan" and TORCH_AUDIO_AVAILABLE and hasattr(torch,
                                                                            "vulkan") and torch.vulkan.is_available():
        current_device_global = "vulkan"
    else:
        if requested_device not in ["cpu", "auto"]:
            log_thread_worker("WARNING",
                              f"Requested device '{requested_device}' not available or supported, falling back to CPU.")
        current_device_global = "cpu"

    log_thread_worker("INFO", f"Worker process effective PyTorch device set to: {current_device_global}")

    # --- Set the compile cache directory ---
    model_dir = worker_config.get("model_dir")
    if model_dir and os.path.isdir(model_dir):
        COMPILE_CACHE_BASE_DIR = os.path.join(model_dir, "torch_compile_cache")
        log_thread_worker("INFO", f"Torch compile cache directory set to model directory: {COMPILE_CACHE_BASE_DIR}")
    else:
        # Fallback to the temporary directory if model_dir is not provided or invalid.
        temp_dir = worker_config.get("temp_dir", tempfile.gettempdir())
        COMPILE_CACHE_BASE_DIR = os.path.join(temp_dir, "torch_compile_cache")
        log_thread_worker("WARNING",
                          f"Model directory not available. Falling back to temp compile cache: {COMPILE_CACHE_BASE_DIR}")

    # Ensure the directory exists.
    try:
        os.makedirs(COMPILE_CACHE_BASE_DIR, exist_ok=True)
    except OSError as e_mkdir:
        log_thread_worker("ERROR",
                          f"Could not create compile cache directory '{COMPILE_CACHE_BASE_DIR}': {e_mkdir}. Caching will be disabled.")
        COMPILE_CACHE_BASE_DIR = None

    # --- TTS Model Initialization ---
    if worker_config.get("enable_tts", False):
        is_tts_init_successful = False  # Assume failure until all steps pass
        try:
            if not TORCH_AUDIO_AVAILABLE:
                raise ImportError("PyTorch/Torchaudio not available for TTS.")

            from chatterbox.tts import ChatterboxTTS as ImportedChatterboxTTS_module_class_thread
            ChatterboxTTS_class_ref_thread = ImportedChatterboxTTS_module_class_thread

            effective_tts_device_for_chatterbox = current_device_global
            if current_device_global == "vulkan":
                log_thread_worker("WARNING",
                                  "Primary device is Vulkan. ChatterboxTTS likely requires CPU/CUDA/MPS. Attempting to load on CPU as fallback.")
                effective_tts_device_for_chatterbox = "cpu"

            log_thread_worker("INFO", f"Loading ChatterboxTTS model on: {effective_tts_device_for_chatterbox}...")

            original_tts_model = ChatterboxTTS_class_ref_thread.from_pretrained(
                device=effective_tts_device_for_chatterbox
            )

            max_retries = 2
            retry_delay_seconds = 15
            original_tts_model = None
            last_exception = None

            for attempt in range(max_retries):
                try:
                    log_thread_worker("INFO",
                                      f"Attempting to download/load ChatterboxTTS model (Attempt {attempt + 1}/{max_retries})...")

                    # Call the download/load function. If this succeeds, we are done.
                    original_tts_model = ChatterboxTTS_class_ref_thread.from_pretrained(
                        device=effective_tts_device_for_chatterbox
                    )

                    log_thread_worker("INFO", "ChatterboxTTS model loaded successfully.")
                    last_exception = None  # Clear any previous network errors
                    break  # Success! Exit the loop.

                except (requests.exceptions.RequestException, HfHubHTTPError) as e:
                    last_exception = e
                    log_thread_worker("WARNING",
                                      f"A network error occurred during download: {e}. Retrying in {retry_delay_seconds}s...")
                    time.sleep(retry_delay_seconds)
                    continue  # Go to the next attempt

                except Exception as e:
                    # Any other exception is considered fatal.
                    last_exception = e
                    log_thread_worker("CRITICAL", f"An unrecoverable error occurred during model initialization: {e}")
                    log_thread_worker("CRITICAL", f"FULL TRACEBACK:\n{traceback.format_exc()}")
                    break  # Exit the loop immediately.

            if original_tts_model is None or last_exception is not None:
                raise RuntimeError("Failed to load ChatterboxTTS model after all retries.") from last_exception
            # --- END OF FINAL LOOP ---

            log_thread_worker("INFO", "Attempting to apply torch.compile() to the model...")

            # STEP 1: Check for torch.compile availability first.
            if not hasattr(torch, "compile"):
                log_thread_worker("WARNING",
                                  f"torch.compile() feature not found in your PyTorch version ({torch.__version__}).")
                log_thread_worker("WARNING",
                                  "Model will run in 'eager' mode (slower). Upgrade to PyTorch 2.0+ to enable compilation.")
                chatterbox_model_instance_global = original_tts_model  # Assign the uncompiled model
            else:
                # STEP 2: If compile is available, prepare the cache path.
                log_thread_worker("INFO",
                                  f"torch.compile() is available (PyTorch version {torch.__version__}). Preparing cache...")

                cache_path = get_compile_artifact_path(
                    "chatterbox_tts",
                    original_tts_model.__class__.__name__,
                    effective_tts_device_for_chatterbox
                )

                if not cache_path:
                    log_thread_worker("WARNING",
                                      "Could not generate a valid cache path because COMPILE_CACHE_BASE_DIR was not set. Caching will be disabled.")
                    chatterbox_model_instance_global = original_tts_model  # Assign the uncompiled model
                else:
                    # STEP 3: If compile and cache path are ready, proceed.
                    os.environ["PYTORCH_COMPILE_CACHE"] = cache_path
                    log_thread_worker("INFO", f"Set PYTORCH_COMPILE_CACHE to '{cache_path}' for persistence.")

                    compile_options = {"mode": "reduce-overhead", "fullgraph": False}

                    try:
                        log_thread_worker("INFO",
                                          "Compiling the model's 'generate' method. This may be slow on the first run...")
                        compile_start_time = time.time()
                        original_tts_model.generate = torch.compile(original_tts_model.generate, **compile_options)
                        compile_duration = time.time() - compile_start_time
                        log_thread_worker("INFO", f"torch.compile() finished in {compile_duration:.2f} seconds.")
                        chatterbox_model_instance_global = original_tts_model  # Assign the COMPILED model
                    except Exception as e_compile:
                        log_thread_worker("WARNING",
                                          f"torch.compile() failed with an error: {e}. The model will run in eager mode.")
                        log_thread_worker("WARNING", f"Full compile error: {traceback.format_exc()}")
                        chatterbox_model_instance_global = original_tts_model  # Assign the UNCOMPILED model on failure

            # Warmup is now even more important, as it triggers the actual compilation/cache load.
            log_thread_worker("INFO", "Warming up the compiled ChatterboxTTS model...")
            dummy_text = "Compiled model warm-up sequence."
            dummy_prompt_path_for_warmup = worker_config.get("dummy_tts_prompt_path_for_warmup")
            if not dummy_prompt_path_for_warmup or not os.path.exists(dummy_prompt_path_for_warmup):
                log_thread_worker("WARNING", "No valid dummy prompt path for warmup. Warmup may be incomplete.")

            _ = chatterbox_model_instance_global.generate(dummy_text, audio_prompt_path=dummy_prompt_path_for_warmup)
            log_thread_worker("INFO", "Compiled ChatterboxTTS model warmed up successfully.")

            is_tts_init_successful = True

        except Exception as e_init_tts:
            log_thread_worker("CRITICAL",
                              f"A critical error occurred during the ChatterboxTTS initialization sequence: {e_init_tts}")
            # We already logged the full traceback inside the loop if it happened there.
            if "FULL TRACEBACK" not in str(e_init_tts):
                log_thread_worker("CRITICAL", f"FULL TRACEBACK:\n{traceback.format_exc()}")

        finally:
            # This block guarantees the final state is set correctly.
            if is_tts_init_successful:
                log_thread_worker("INFO",
                                  "ChatterboxTTS initialization sequence completed successfully. TTS is AVAILABLE.")
                CHATTERBOX_TTS_AVAILABLE_THREAD = True
            else:
                log_thread_worker("CRITICAL",
                                  "ChatterboxTTS initialization FAILED. TTS will be UNAVAILABLE in this worker.")
                CHATTERBOX_TTS_AVAILABLE_THREAD = False
                chatterbox_model_instance_global = None
                gc.collect()

    # --- ASR Model Initialization ---
    if worker_config.get("enable_asr", False):
        is_asr_init_successful = False  # Assume failure
        try:
            from pywhispercpp.model import Model as ImportedWhisperModel_thread
            WhisperModel_thread = ImportedWhisperModel_thread

            model_dir = worker_config.get("model_dir")
            whisper_model_name = worker_config.get("whisper_model_name")
            if not model_dir or not whisper_model_name:
                raise ValueError("Missing model_dir or whisper_model_name for ASR initialization.")

            full_model_path_asr = os.path.join(model_dir, whisper_model_name)
            if not os.path.exists(full_model_path_asr):
                raise FileNotFoundError(f"Whisper model GGUF file not found: {full_model_path_asr}")

            log_thread_worker("INFO", f"Loading Whisper model: {full_model_path_asr}...")
            whisper_model_instance_global = WhisperModel_thread(
                model=full_model_path_asr, print_realtime=False, print_progress=False
            )
            log_thread_worker("INFO", "Whisper model loaded successfully.")

            log_thread_worker("INFO", "Warming up Whisper ASR model...")
            temp_wav_path_asr_warmup = None
            try:
                asr_warmup_temp_dir = worker_config.get("temp_dir", tempfile.gettempdir())
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=asr_warmup_temp_dir,
                                                 prefix="asr_warmup_") as tmp_asr_wav_f:
                    temp_wav_path_asr_warmup = tmp_asr_wav_f.name
                sr_dummy_asr, dur_dummy_asr = 16000, 0.1
                silence_data_asr = np.zeros(int(dur_dummy_asr * sr_dummy_asr), dtype=np.int16)
                sf.write(temp_wav_path_asr_warmup, silence_data_asr, sr_dummy_asr, format='WAV', subtype='PCM_16')
                _ = whisper_model_instance_global.transcribe(temp_wav_path_asr_warmup, language="en")
                log_thread_worker("INFO", "Whisper ASR model warmed up.")
            finally:
                if temp_wav_path_asr_warmup and os.path.exists(temp_wav_path_asr_warmup):
                    os.remove(temp_wav_path_asr_warmup)

            is_asr_init_successful = True

        except Exception as e_init_asr:
            log_thread_worker("CRITICAL", f"A critical error occurred during Whisper ASR initialization: {e_init_asr}")
            log_thread_worker("CRITICAL", f"FULL TRACEBACK:\n{traceback.format_exc()}")

        finally:
            if is_asr_init_successful:
                log_thread_worker("INFO", "Whisper ASR initialization completed successfully. ASR is AVAILABLE.")
                PYWHISPERCPP_AVAILABLE_THREAD = True
            else:
                log_thread_worker("CRITICAL",
                                  "Whisper ASR initialization FAILED. ASR will be UNAVAILABLE in this worker.")
                PYWHISPERCPP_AVAILABLE_THREAD = False
                whisper_model_instance_global = None
                gc.collect()



def process_tts_chunk(params):
    if not CHATTERBOX_TTS_AVAILABLE_THREAD or not chatterbox_model_instance_global:
        return {"error": "TTS model not initialized or not available in this worker."}
    if not TORCH_AUDIO_AVAILABLE:
        return {"error": "PyTorch/Torchaudio components not available for TTS chunk processing."}

    text_to_synthesize = params.get("text_to_synthesize")
    voice_prompt_path = params.get("voice_prompt_path")
    exaggeration = params.get("exaggeration")
    cfg_weight = params.get("cfg_weight")
    output_audio_chunk_path = params.get("output_audio_chunk_path")

    if not text_to_synthesize: return {"error": "Missing 'text_to_synthesize' for TTS."}
    if not voice_prompt_path: return {"error": "Missing 'voice_prompt_path' for TTS."}
    if not os.path.exists(voice_prompt_path): return {"error": f"Voice prompt not found: {voice_prompt_path}"}
    if not output_audio_chunk_path: return {"error": "Missing 'output_audio_chunk_path' for TTS."}

    try:
        log_thread_worker("INFO",
                          f"Synthesizing TTS chunk: '{text_to_synthesize[:50]}...' to {output_audio_chunk_path}")
        wav_tensor = chatterbox_model_instance_global.generate(
            text_to_synthesize, audio_prompt_path=voice_prompt_path,
            exaggeration=exaggeration, cfg_weight=cfg_weight
        )

        if wav_tensor.ndim == 1:
            wav_tensor = wav_tensor.unsqueeze(0).repeat(2, 1)
        elif wav_tensor.ndim == 2 and wav_tensor.shape[0] == 1:
            wav_tensor = wav_tensor.repeat(2, 1)
        elif wav_tensor.ndim == 2 and wav_tensor.shape[0] > 2:
            wav_tensor = wav_tensor[:2, :]

        output_dir = os.path.dirname(output_audio_chunk_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        torchaudio.save(
            output_audio_chunk_path, wav_tensor.cpu(),
            chatterbox_model_instance_global.sr, format="wav",
            encoding="PCM_S", bits_per_sample=16
        )
        duration_samples = wav_tensor.shape[1] if wav_tensor.ndim == 2 else wav_tensor.shape[0]
        duration_ms_calc = (duration_samples / chatterbox_model_instance_global.sr) * 1000
        log_thread_worker("INFO", f"TTS chunk saved: {output_audio_chunk_path} (Duration: {duration_ms_calc:.2f} ms)")
        return {"result": {
            "status": "success", "audio_chunk_path": output_audio_chunk_path,
            "sample_rate": chatterbox_model_instance_global.sr, "duration_ms": duration_ms_calc
        }}
    except Exception as e:
        log_thread_worker("ERROR", f"TTS chunk processing failed: {e}\n{traceback.format_exc()}")
        return {"error": f"TTS chunk generation critical error: {str(e)}"}


def process_asr_chunk(params):
    if not PYWHISPERCPP_AVAILABLE_THREAD or not whisper_model_instance_global:
        return {"error": "ASR model not initialized or not available in this worker."}

    input_audio_chunk_path = params.get("input_audio_chunk_path")
    language = params.get("language", "auto")

    if not input_audio_chunk_path: return {"error": "Missing 'input_audio_chunk_path' for ASR."}
    if not os.path.exists(input_audio_chunk_path): return {
        "error": f"Input audio chunk not found: {input_audio_chunk_path}"}

    try:
        log_thread_worker("INFO", f"Transcribing ASR chunk: {input_audio_chunk_path} (Lang: '{language}')")
        transcribe_params_dict = {'language': language.lower() if language else "auto", 'translate': False}
        segments_list = whisper_model_instance_global.transcribe(input_audio_chunk_path, **transcribe_params_dict)
        transcribed_text_chunk = "".join(seg.text for seg in segments_list).strip()
        log_thread_worker("DEBUG", f"ASR chunk transcribed. Text: '{transcribed_text_chunk[:70]}...'")
        return {"result": {"text": transcribed_text_chunk}}
    except Exception as e:
        log_thread_worker("ERROR", f"ASR chunk transcription failed: {e}\n{traceback.format_exc()}")
        return {"error": f"ASR chunk transcription critical error: {str(e)}"}


def worker_loop(pipe_conn, worker_config):
    global memory_monitor_thread, stop_memory_monitor_event, last_cache_clear_time
    global chatterbox_model_instance_global, whisper_model_instance_global
    global CHATTERBOX_TTS_AVAILABLE_THREAD, PYWHISPERCPP_AVAILABLE_THREAD
    global current_device_global, TORCH_AUDIO_AVAILABLE, PSUTIL_AVAILABLE, torch

    # Step 1: Perform all model initializations immediately.
    # The entire worker's functionality depends on this block completing successfully.
    # If any part of this fails, the worker process will exit.
    try:
        initialize_models(worker_config)
        log_thread_worker("INFO", "Model initialization sequence completed.")
    except Exception as e_init_main_call:
        log_thread_worker("CRITICAL",
                          f"Model initialization sequence critically failed and the worker cannot start: {e_init_main_call}\n{traceback.format_exc()}")
        # Attempt to signal the catastrophic failure back to the orchestrator.
        try:
            pipe_conn.send({"status": "failed_init", "error": str(e_init_main_call)})
        except Exception:
            # If the pipe is broken, we can't do anything else.
            pass
        # Exit the worker process entirely. It is not functional.
        return

    # Step 2: Send a definitive "ready" signal to the orchestrator.
    # This signal is sent ONLY if the initialization above succeeded.
    try:
        is_ready_payload = {
            "status": "ready",
            "task_id": "worker_ready_signal",
            "tts_available": CHATTERBOX_TTS_AVAILABLE_THREAD,
            "asr_available": PYWHISPERCPP_AVAILABLE_THREAD
        }
        pipe_conn.send(is_ready_payload)
        log_thread_worker("INFO", f"Worker is ready. Sent signal to orchestrator: {is_ready_payload}")
    except Exception as e_send_ready:
        log_thread_worker("CRITICAL",
                          f"Could not send the 'ready' signal to the orchestrator: {e_send_ready}. The worker will now exit as it cannot communicate.")
        return

    # Step 3: Start the background memory monitoring thread.
    log_thread_worker("INFO", "Starting background memory monitor and the main command processing loop.")
    if (PSUTIL_AVAILABLE or (
            TORCH_AUDIO_AVAILABLE and torch and (current_device_global in ["cuda", "mps", "vulkan"]))) and \
            (memory_monitor_thread is None or not memory_monitor_thread.is_alive()):
        stop_memory_monitor_event.clear()
        memory_monitor_thread = threading.Thread(target=monitor_memory_usage, daemon=True, name="MemoryMonitorThread")
        memory_monitor_thread.start()
    else:
        log_thread_worker("INFO",
                          "Memory monitoring prerequisites not met or thread already running. Monitor will not be started.")

    # Step 4: Enter the main command processing loop.
    active = True
    while active:
        try:
            # Block and wait for a command from the orchestrator.
            if not pipe_conn.poll(None):  # Wait indefinitely
                # This case is unlikely but handles a closed pipe gracefully.
                log_thread_worker("WARNING", "Pipe poll returned False, suggesting it was closed. Exiting loop.")
                active = False
                continue

            message = pipe_conn.recv()
            command = message.get("command")
            task_id = message.get("task_id", "unknown_task")
            log_thread_worker("DEBUG", f"Received command: '{command}' for task_id: {task_id}")

            response_payload = {"error": f"Unknown command received: {command}", "task_id": task_id}

            if command == "shutdown":
                log_thread_worker("INFO", "Shutdown command received. Exiting gracefully.")
                response_payload = {"result": "shutdown_ack", "task_id": task_id}
                active = False
            elif command == "tts_chunk":
                response_payload = process_tts_chunk(message.get("params", {}))
                response_payload["task_id"] = task_id
            elif command == "asr_chunk":
                response_payload = process_asr_chunk(message.get("params", {}))
                response_payload["task_id"] = task_id

            pipe_conn.send(response_payload)
            log_thread_worker("DEBUG", f"Sent response for task_id: {task_id}, command: '{command}'")

        except (EOFError, BrokenPipeError):
            log_thread_worker("ERROR", "Orchestrator closed the pipe connection. Exiting worker loop.")
            active = False
        except KeyboardInterrupt:
            log_thread_worker("INFO", "KeyboardInterrupt received in worker loop. Exiting.")
            active = False
        except Exception as e_loop:
            current_task_id_err = "unknown_at_error"
            if 'message' in locals() and isinstance(message, dict):
                current_task_id_err = message.get("task_id", "unknown_at_error")

            log_thread_worker("CRITICAL",
                              f"Unhandled exception in worker_loop for task '{current_task_id_err}': {e_loop}\n{traceback.format_exc()}")
            try:
                pipe_conn.send({"error_critical_loop": f"Worker loop unhandled exception: {str(e_loop)}",
                                "task_id": current_task_id_err})
            except Exception as e_send_crit_err:
                log_thread_worker("ERROR",
                                  f"Failed to send critical loop error message back to orchestrator: {e_send_crit_err}")
            active = False

    # Step 5: Clean up resources after exiting the loop.
    log_thread_worker("INFO", "Worker loop finished. Stopping memory monitor and cleaning up resources.")

    # Stop the memory monitor thread gracefully.
    if memory_monitor_thread and memory_monitor_thread.is_alive():
        stop_memory_monitor_event.set()
        # Give the thread a moment to stop.
        memory_monitor_thread.join(timeout=5)
        if memory_monitor_thread.is_alive():
            log_thread_worker("WARNING", "Memory monitoring thread did not stop gracefully after timeout.")

    # Explicitly clean up global model instances to release memory.
    log_thread_worker("DEBUG", "Performing final cleanup of global model instances.")
    if chatterbox_model_instance_global:
        del chatterbox_model_instance_global
        chatterbox_model_instance_global = None
        log_thread_worker("TRACE", "ChatterboxTTS global instance deleted.")
    if whisper_model_instance_global:
        del whisper_model_instance_global
        whisper_model_instance_global = None
        log_thread_worker("TRACE", "Whisper ASR global instance deleted.")

    # Perform a final garbage collection and attempt to clear PyTorch caches.
    gc.collect()
    log_thread_worker("DEBUG", "gc.collect() called after model deletion.")
    _try_clear_pytorch_caches(current_device_global)

    log_thread_worker("INFO", "Worker process cleanup complete. Terminating.")


if __name__ == "__main__":
    # This script is intended to be launched by the orchestrator using multiprocessing.Process.
    # Direct execution will not have the necessary pipe_conn and worker_config.
    log_thread_worker("CRITICAL", "This script (`audioProcessorCortex_backbone_provider.py`) is designed to be launched "
                                  "as a child process by an orchestrator using Python's `multiprocessing` module. "
                                  "It expects a pipe connection and a configuration dictionary to be passed "
                                  "programmatically when `worker_loop` is called as the target of a new process. "
                                  "Direct execution is not supported for normal operation.")
    sys.exit(1)