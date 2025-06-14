# audio_thread_worker.py
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
import threading  # For the monitoring thread

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


def get_compile_artifact_path(model_name_key: str, model_identifier: str, device: str, backend: str):
    """Generates a path for torch.compile artifacts."""
    if not COMPILE_CACHE_BASE_DIR:
        return None

    torch_version_str = torch.__version__.replace('.', '_') if TORCH_AUDIO_AVAILABLE and torch else "unknown_torch"
    # Create a somewhat stable hash for the model identifier (e.g., model class name or a version string)
    model_hash = hashlib.md5(model_identifier.encode()).hexdigest()[:8]

    # Path: <base>/<model_key>/<device>/<backend>/<torch_ver>/<model_hash>/artifact.ptc
    artifact_dir = os.path.join(
        COMPILE_CACHE_BASE_DIR,
        model_name_key,
        device.replace(':', '_'),  # Sanitize device string for path
        backend,
        torch_version_str,
        model_hash
    )
    os.makedirs(artifact_dir, exist_ok=True)
    return os.path.join(artifact_dir, "artifact.ptc")

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
    global current_device_global, TORCH_AUDIO_AVAILABLE

    if not PSUTIL_AVAILABLE:
        log_thread_worker("WARNING",
                          "psutil library not found during init. Detailed RAM monitoring will be disabled in worker.")

    if torch is None or torchaudio is None: TORCH_AUDIO_AVAILABLE = False

    requested_device = worker_config.get("device", "cpu").lower()

    # Determine effective device for PyTorch operations
    if requested_device == "cuda" and TORCH_AUDIO_AVAILABLE and hasattr(torch, "cuda") and torch.cuda.is_available():
        current_device_global = "cuda"
    elif requested_device == "mps" and TORCH_AUDIO_AVAILABLE and hasattr(torch, "backends") and hasattr(torch.backends,
                                                                                                        "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():  # type: ignore
        current_device_global = "mps"
    elif requested_device == "vulkan" and TORCH_AUDIO_AVAILABLE and hasattr(torch,
                                                                            "vulkan") and torch.vulkan.is_available():  # type: ignore
        current_device_global = "vulkan"
        log_thread_worker("INFO",
                          "Vulkan device type requested. Underlying libraries (e.g., TTS model) must support it or have fallbacks.")
    else:
        if requested_device not in ["cpu", "auto"]:
            log_thread_worker("WARNING",
                              f"Requested device '{requested_device}' not available or not fully supported by worker's checks, falling back to CPU.")
        current_device_global = "cpu"

    log_thread_worker("INFO", f"Worker process effective PyTorch device set to: {current_device_global}")

    # --- MPS Memory Fraction Limit ---
    if current_device_global == "mps" and TORCH_AUDIO_AVAILABLE and torch and hasattr(torch.mps,
                                                                                      "set_per_process_memory_fraction"):
        try:
            memory_fraction_limit = worker_config.get("mps_memory_fraction_limit", 0.75)  # Default to 75%
            if 0.1 <= memory_fraction_limit <= 1.0:  # Sanity check the fraction
                torch.mps.set_per_process_memory_fraction(memory_fraction_limit)  # type: ignore
                log_thread_worker("INFO",
                                  f"[MemoryConfig] Set MPS memory fraction limit to {memory_fraction_limit * 100:.0f}%.")
            else:
                log_thread_worker("WARNING",
                                  f"[MemoryConfig] Invalid mps_memory_fraction_limit ({memory_fraction_limit}) in worker_config. Must be between 0.1 and 1.0. Using PyTorch default.")
        except Exception as e_mps_frac:
            log_thread_worker("WARNING", f"[MemoryConfig] Failed to set MPS memory fraction: {e_mps_frac}")

    # --- TTS Model Initialization ---
    if worker_config.get("enable_tts", False):
        if not TORCH_AUDIO_AVAILABLE:
            log_thread_worker("WARNING",
                              "PyTorch/Torchaudio not available for TTS. ChatterboxTTS cannot be initialized.")
            CHATTERBOX_TTS_AVAILABLE_THREAD = False
        else:
            # Determine the device to pass to ChatterboxTTS
            effective_tts_device_for_chatterbox = current_device_global
            if current_device_global == "vulkan":
                # ChatterboxTTS (based on PyTorch) is unlikely to directly support "vulkan" as a device string.
                # It will likely try to use CPU or error. Forcing CPU is safer here.
                log_thread_worker("WARNING",
                                  "Primary device is Vulkan. ChatterboxTTS likely requires CPU/CUDA/MPS. Attempting to load ChatterboxTTS on CPU as fallback for Vulkan.")
                effective_tts_device_for_chatterbox = "cpu"

            try:
                from chatterbox.tts import ChatterboxTTS as ImportedChatterboxTTS_module_class_thread
                ChatterboxTTS_class_ref_thread = ImportedChatterboxTTS_module_class_thread
                CHATTERBOX_TTS_AVAILABLE_THREAD = True  # Assume import success initially
                if ChatterboxTTS_class_ref_thread:
                    log_thread_worker("INFO",
                                      f"Loading ChatterboxTTS model on: {effective_tts_device_for_chatterbox}...")
                    # Step 1: Load the original model
                    original_tts_model = ChatterboxTTS_class_ref_thread.from_pretrained(
                        device=effective_tts_device_for_chatterbox
                    )
                    log_thread_worker("INFO", "ChatterboxTTS original model loaded.")
                    chatterbox_model_instance_global = original_tts_model  # Default to uncompiled

                    # Step 2: Apply torch.compile if configured and possible
                    if worker_config.get("use_torch_compile_tts", False) and hasattr(torch, 'compiler'):
                        log_thread_worker("INFO", "[TorchCompile] torch.compile enabled for TTS.")
                        # Determine artifact path (needs model identifier)
                        # Using class name as a simple identifier. A version string would be better.
                        model_id_for_cache = ChatterboxTTS_class_ref_thread.__name__
                        compile_backend = worker_config.get("tts_compile_backend", "inductor")

                        artifact_path = None
                        if COMPILE_CACHE_BASE_DIR:  # Only try caching if base dir is set
                            artifact_path = get_compile_artifact_path("tts_chatterbox", model_id_for_cache,
                                                                      effective_tts_device_for_chatterbox,
                                                                      compile_backend)

                        loaded_from_cache = False
                        if artifact_path and os.path.exists(artifact_path):
                            try:
                                log_thread_worker("INFO", f"[TorchCompile] Loading artifacts from: {artifact_path}")
                                with open(artifact_path, "rb") as f:
                                    artifact_bytes = f.read()
                                torch.compiler.load_cache_artifacts(artifact_bytes)  # type: ignore
                                log_thread_worker("INFO", "[TorchCompile] Artifacts loaded successfully.")
                                loaded_from_cache = True
                            except Exception as e:
                                log_thread_worker("WARNING",
                                                  f"[TorchCompile] Failed to load artifacts from {artifact_path}: {e}. Will recompile.")
                                # Potentially delete corrupted artifact: os.remove(artifact_path)

                        try:
                            log_thread_worker("INFO",
                                              f"[TorchCompile] Compiling ChatterboxTTS model with backend '{compile_backend}'...")
                            # Make sure the model is on the correct device BEFORE compiling
                            model_to_compile = original_tts_model.to(effective_tts_device_for_chatterbox)

                            compiled_tts_model = torch.compile(model_to_compile, backend=compile_backend,
                                                               fullgraph=False)  # Try fullgraph=True for more speed if it works
                            chatterbox_model_instance_global = compiled_tts_model  # Use compiled model
                            log_thread_worker("INFO", "[TorchCompile] Model compiled successfully.")

                            if artifact_path and not loaded_from_cache:  # Save if newly compiled
                                log_thread_worker("INFO",
                                                  f"[TorchCompile] Saving new compile artifacts to: {artifact_path}")
                                try:
                                    new_artifact_bytes = torch.compiler.save_cache_artifacts()  # type: ignore
                                    with open(artifact_path, "wb") as f:
                                        f.write(new_artifact_bytes)
                                    log_thread_worker("INFO", "[TorchCompile] New artifacts saved.")
                                except Exception as e_save:
                                    log_thread_worker("WARNING",
                                                      f"[TorchCompile] Failed to save artifacts to {artifact_path}: {e_save}")
                        except Exception as e_compile:
                            log_thread_worker("ERROR",
                                              f"[TorchCompile] Compilation failed: {e_compile}. Using uncompiled model.")
                            chatterbox_model_instance_global = original_tts_model  # Fallback
                    else:
                        if not hasattr(torch, 'compiler'):
                            log_thread_worker("INFO",
                                              "[TorchCompile] torch.compiler module not available. Skipping compilation.")
                        else:
                            log_thread_worker("INFO",
                                              "[TorchCompile] use_torch_compile_tts is false. Using uncompiled model.")
                    # TTS Warmup
                    try:
                        dummy_text = "Chatterbox TTS warm-up sequence initiated."
                        dummy_prompt_path_for_warmup = worker_config.get("dummy_tts_prompt_path_for_warmup")
                        if dummy_prompt_path_for_warmup and os.path.exists(dummy_prompt_path_for_warmup):
                            _ = chatterbox_model_instance_global.generate(dummy_text,
                                                                          audio_prompt_path=dummy_prompt_path_for_warmup)
                            log_thread_worker("INFO", "ChatterboxTTS model warmed up using provided dummy prompt.")
                        else:
                            log_thread_worker("WARNING",
                                              "ChatterboxTTS: No valid dummy prompt path for warmup provided in worker_config. Warmup may be incomplete or skipped.")
                    except Exception as e_warmup_tts_gen:
                        log_thread_worker("WARNING", f"ChatterboxTTS warmup generation call failed: {e_warmup_tts_gen}")
                else:
                    CHATTERBOX_TTS_AVAILABLE_THREAD = False
                    log_thread_worker("ERROR",
                                      "ChatterboxTTS class reference is None after successful import attempt (should not happen).")
            except ImportError as e_cb_module_imp:
                log_thread_worker("WARNING", f"ChatterboxTTS library import failed: {e_cb_module_imp}.")
                CHATTERBOX_TTS_AVAILABLE_THREAD = False
            except Exception as e_cb_model_load:  # Catch other errors during from_pretrained
                log_thread_worker("ERROR",
                                  f"Failed to load ChatterboxTTS model: {e_cb_model_load}\n{traceback.format_exc()}")
                CHATTERBOX_TTS_AVAILABLE_THREAD = False

    # --- ASR Model Initialization ---
    if worker_config.get("enable_asr", False):
        log_thread_worker("INFO",
                          "Initializing ASR (pywhispercpp: uses CPU or its own ggml GPU backend if compiled for it).")
        try:
            from pywhispercpp.model import Model as ImportedWhisperModel_thread
            WhisperModel_thread = ImportedWhisperModel_thread
            PYWHISPERCPP_AVAILABLE_THREAD = True  # Assume import success
            if WhisperModel_thread:
                model_dir = worker_config.get("model_dir")
                whisper_model_name = worker_config.get("whisper_model_name")
                if not model_dir or not whisper_model_name:
                    log_thread_worker("ERROR", "Missing model_dir or whisper_model_name for ASR initialization.")
                    PYWHISPERCPP_AVAILABLE_THREAD = False
                else:
                    full_model_path_asr = os.path.join(model_dir, whisper_model_name)
                    if not os.path.exists(full_model_path_asr):
                        log_thread_worker("ERROR", f"Whisper model GGUF file not found: {full_model_path_asr}")
                        PYWHISPERCPP_AVAILABLE_THREAD = False
                    else:
                        log_thread_worker("INFO", f"Loading Whisper model: {full_model_path_asr}...")
                        whisper_model_instance_global = WhisperModel_thread(
                            model=full_model_path_asr, print_realtime=False, print_progress=False
                        )
                        log_thread_worker("INFO", "Whisper model loaded successfully.")
                        # ASR Warmup
                        try:
                            temp_wav_path_asr_warmup = None
                            asr_warmup_temp_dir = worker_config.get("temp_dir",
                                                                    tempfile.gettempdir())  # Use configured temp or system temp
                            if not os.path.isdir(asr_warmup_temp_dir): asr_warmup_temp_dir = tempfile.gettempdir()

                            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=asr_warmup_temp_dir,
                                                             prefix="asr_warmup_") as tmp_asr_wav_f:
                                temp_wav_path_asr_warmup = tmp_asr_wav_f.name
                            sr_dummy_asr, dur_dummy_asr = 16000, 0.1
                            silence_data_asr = np.zeros(int(dur_dummy_asr * sr_dummy_asr), dtype=np.int16)
                            sf.write(temp_wav_path_asr_warmup, silence_data_asr, sr_dummy_asr, format='WAV',
                                     subtype='PCM_16')
                            _ = whisper_model_instance_global.transcribe(temp_wav_path_asr_warmup, language="en")
                            log_thread_worker("INFO", "Whisper ASR model warmed up.")
                        except Exception as e_warmup_asr_gen:
                            log_thread_worker("WARNING",
                                              f"Whisper ASR warmup transcription call failed: {e_warmup_asr_gen}\n{traceback.format_exc()}")
                        finally:
                            if temp_wav_path_asr_warmup and os.path.exists(temp_wav_path_asr_warmup):
                                try:
                                    os.remove(temp_wav_path_asr_warmup)
                                except Exception as e_rm_asr_warmup:
                                    log_thread_worker("WARNING",
                                                      f"Could not remove dummy ASR warmup file '{temp_wav_path_asr_warmup}': {e_rm_asr_warmup}")
            else:  # WhisperModel_thread is None
                PYWHISPERCPP_AVAILABLE_THREAD = False
                log_thread_worker("ERROR",
                                  "WhisperModel class reference is None after successful import attempt (should not happen).")
        except ImportError as e_wh_module_imp:
            log_thread_worker("WARNING", f"pywhispercpp library import failed: {e_wh_module_imp}.")
            PYWHISPERCPP_AVAILABLE_THREAD = False
        except Exception as e_wh_model_load:  # Catch other errors during Model()
            log_thread_worker("ERROR", f"Failed to load Whisper model: {e_wh_model_load}\n{traceback.format_exc()}")
            PYWHISPERCPP_AVAILABLE_THREAD = False


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
    try:
        initialize_models(worker_config)
    except Exception as e_init_main_call:
        log_thread_worker("CRITICAL",
                          f"Model initialization sequence itself raised an unhandled exception: {e_init_main_call}\n{traceback.format_exc()}")
        try:
            pipe_conn.send({
                               "error_critical_init": f"Worker model initialization sequence critically failed: {str(e_init_main_call)}"})
        except Exception:
            pass
        return

    log_thread_worker("INFO", "Worker models initialized (if configured). Starting main loop and memory monitor.")

    # Start memory monitor only if prerequisites are met and it's not already running (though it shouldn't be)
    if (PSUTIL_AVAILABLE or (
            TORCH_AUDIO_AVAILABLE and torch and (current_device_global in ["cuda", "mps", "vulkan"]))) and \
            (memory_monitor_thread is None or not memory_monitor_thread.is_alive()):
        stop_memory_monitor_event.clear()
        memory_monitor_thread = threading.Thread(target=monitor_memory_usage, daemon=True, name="MemoryMonitorThread")
        memory_monitor_thread.start()
    else:
        if not (PSUTIL_AVAILABLE or (
                TORCH_AUDIO_AVAILABLE and torch and (current_device_global in ["cuda", "mps", "vulkan"]))):
            log_thread_worker("INFO", "Memory monitoring prerequisites not met. Monitor disabled.")
        elif memory_monitor_thread and memory_monitor_thread.is_alive():
            log_thread_worker("WARNING",
                              "Memory monitor thread unexpectedly found to be alive before starting. Not restarting.")

    active = True
    while active:
        try:
            if not pipe_conn.readable and not pipe_conn.writable:
                log_thread_worker("WARNING", "Pipe seems closed from orchestrator side. Exiting worker loop.")
                active = False;
                break

            message = pipe_conn.recv()

            if message is None:
                log_thread_worker("INFO",
                                  "Received None message (pipe likely closed by orchestrator). Exiting worker loop.")
                active = False;
                break

            command = message.get("command")
            params = message.get("params", {})
            task_id = message.get("task_id", "unknown_task")
            log_thread_worker("DEBUG", f"Received command: '{command}' for task_id: {task_id}")

            response_payload = {"error": f"Unknown command received: {command}", "task_id": task_id}

            if command == "shutdown":
                log_thread_worker("INFO", "Shutdown command received. Exiting gracefully.")
                response_payload = {"result": "shutdown_ack", "task_id": task_id}
                active = False
            elif command == "tts_chunk":
                response_payload = process_tts_chunk(params)
                response_payload["task_id"] = task_id
            elif command == "asr_chunk":
                response_payload = process_asr_chunk(params)
                response_payload["task_id"] = task_id

            pipe_conn.send(response_payload)
            log_thread_worker("DEBUG", f"Sent response for task_id: {task_id}, command: '{command}'")

        except EOFError:
            log_thread_worker("ERROR", "Orchestrator closed the pipe (EOFError). Exiting worker loop.")
            active = False
        except BrokenPipeError:
            log_thread_worker("ERROR",
                              "Pipe connection to orchestrator is broken (BrokenPipeError). Exiting worker loop.")
            active = False
        except KeyboardInterrupt:
            log_thread_worker("INFO", "KeyboardInterrupt received in worker loop. Exiting.")
            active = False
        except Exception as e_loop:
            log_thread_worker("CRITICAL", f"Unhandled exception in worker_loop: {e_loop}\n{traceback.format_exc()}")
            current_task_id_err = message.get("task_id",
                                              "unknown_at_error") if 'message' in locals() else "unknown_at_error"
            try:
                pipe_conn.send({"error_critical_loop": f"Worker loop unhandled exception: {str(e_loop)}",
                                "task_id": current_task_id_err})
            except Exception as e_send_crit_err:
                log_thread_worker("ERROR",
                                  f"Failed to send critical loop error message back to orchestrator: {e_send_crit_err}")
            active = False

    log_thread_worker("INFO", "Worker loop finished. Stopping memory monitor and cleaning up.")

    if memory_monitor_thread and memory_monitor_thread.is_alive():
        stop_memory_monitor_event.set()
        monitor_join_timeout = max(MEMORY_MONITOR_INTERVAL_SECONDS, PERIODIC_CACHE_CLEAR_INTERVAL_SECONDS) + 2
        memory_monitor_thread.join(timeout=monitor_join_timeout)
        if memory_monitor_thread.is_alive():
            log_thread_worker("WARNING",
                              "[MemoryMonitor] Memory monitoring thread did not stop gracefully after timeout.")

    log_thread_worker("DEBUG", "Performing final cleanup of global model instances.")
    global chatterbox_model_instance_global, whisper_model_instance_global
    if chatterbox_model_instance_global:
        del chatterbox_model_instance_global;
        chatterbox_model_instance_global = None
        log_thread_worker("TRACE", "ChatterboxTTS global instance deleted.")
    if whisper_model_instance_global:
        del whisper_model_instance_global;
        whisper_model_instance_global = None
        log_thread_worker("TRACE", "Whisper ASR global instance deleted.")

    # Explicit gc.collect() after model deletion, before final cache clear
    gc.collect()
    log_thread_worker("DEBUG", "gc.collect() called after model deletion.")

    _try_clear_pytorch_caches(current_device_global)  # Final cache clear

    log_thread_worker("INFO", "Worker process cleanup complete. Terminating.")


if __name__ == "__main__":
    # This script is intended to be launched by the orchestrator using multiprocessing.Process.
    # Direct execution will not have the necessary pipe_conn and worker_config.
    log_thread_worker("CRITICAL", "This script (`audio_thread_worker.py`) is designed to be launched "
                                  "as a child process by an orchestrator using Python's `multiprocessing` module. "
                                  "It expects a pipe connection and a configuration dictionary to be passed "
                                  "programmatically when `worker_loop` is called as the target of a new process. "
                                  "Direct execution is not supported for normal operation.")
    sys.exit(1)