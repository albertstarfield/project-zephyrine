# audio_thread_worker.py
import sys
import os
import json
import time
import traceback
import argparse
import base64  # For TTS output if we decide to return b64 directly from chunk worker (less likely)
import io
import shutil
from typing import Dict, Any

import numpy as np
import soundfile as sf
import tempfile  # May not be needed if paths are passed directly
import subprocess
import gc

# --- PyTorch and torchaudio imports ---
TORCH_AUDIO_AVAILABLE = False
torch = None
torchaudio = None
F_torchaudio = None
try:
    import torch
    import torchaudio
    import torchaudio.functional as F_torchaudio

    TORCH_AUDIO_AVAILABLE = True
except ImportError:
    pass  # Errors will be handled in main based on task


# --- Basic Logging to stderr (defined early) ---
def log_thread_worker(level, message):
    # Add PID to distinguish logs if multiple workers run concurrently
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')} AUDIO_THREAD_WORKER(PID:{os.getpid()})|{level}] {message}",
          file=sys.stderr, flush=True)


# --- ChatterboxTTS Imports (for TTS chunks) ---
CHATTERBOX_TTS_AVAILABLE_THREAD = False
ChatterboxTTS_class_ref_thread = None
try:
    if TORCH_AUDIO_AVAILABLE:
        from chatterbox.tts import ChatterboxTTS as ImportedChatterboxTTS_module_class_thread

        ChatterboxTTS_class_ref_thread = ImportedChatterboxTTS_module_class_thread
        CHATTERBOX_TTS_AVAILABLE_THREAD = True
    else:
        raise ImportError("Torch/Torchaudio not available for ChatterboxTTS in thread worker.")
except ImportError as e_cb_thread:
    log_thread_worker("WARNING", f"ChatterboxTTS library not found/imported in thread worker: {e_cb_thread}.")
    CHATTERBOX_TTS_AVAILABLE_THREAD = False

# --- PyWhisperCpp Imports (for ASR chunks) ---
PYWHISPERCPP_AVAILABLE_THREAD = False
WhisperModel_thread = None
try:
    from pywhispercpp.model import Model as ImportedWhisperModel_thread

    WhisperModel_thread = ImportedWhisperModel_thread
    PYWHISPERCPP_AVAILABLE_THREAD = True
except ImportError as e_whisper_thread:
    log_thread_worker("WARNING", f"pywhispercpp library not found in thread worker: {e_whisper_thread}.")
    PYWHISPERCPP_AVAILABLE_THREAD = False


# --- Config Import for ASR defaults (if needed directly, though usually passed) ---
# Generally, paths and model names will be passed as args by the orchestrator audio_worker.py
# For this worker, we might not need direct config import if all params are explicit args.
# However, if some defaults are still desired from config:
# try:
#     from config import WHISPER_DEFAULT_LANGUAGE # Example
# except ImportError:
#     WHISPER_DEFAULT_LANGUAGE = "auto"


def _get_pytorch_device_thread_worker(requested_device_str: str) -> str:
    # Simplified device getter for this worker, as main audio_worker handles complex detection
    log_thread_worker("INFO", f"ThreadWorker: Requested PyTorch device: '{requested_device_str}'")
    if not TORCH_AUDIO_AVAILABLE or not torch: return "cpu"

    if requested_device_str.lower() == "cuda" and torch.cuda.is_available(): return "cuda"
    if requested_device_str.lower() == "mps" and hasattr(torch.backends,
                                                         "mps") and torch.backends.mps.is_available(): return "mps"
    # Add Vulkan if needed, but keep it simple for worker

    # Auto logic if "auto" is passed or specific fails
    if requested_device_str.lower() == "auto" or requested_device_str.lower() not in ["cpu", "cuda", "mps"]:
        if torch.cuda.is_available(): return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return "mps"
    return "cpu"


def main_thread_worker():
    parser = argparse.ArgumentParser(description="Audio Thread Worker (Single Chunk TTS/ASR)")
    parser.add_argument("--task-type", required=True, choices=["tts_chunk", "asr_chunk"])
    parser.add_argument("--device", default="auto", help="PyTorch device for TTS (cpu, cuda, mps)")
    parser.add_argument("--model-dir", help="Base directory for ASR GGUF models (passed by orchestrator)")
    parser.add_argument("--temp-dir", default=".",
                        help="Base directory for temporary output files (e.g., TTS audio chunk)")

    # ASR Chunk Args
    parser.add_argument("--input-audio-chunk-path", help="Path to the temporary WAV audio chunk for ASR")
    parser.add_argument("--whisper-model-name", help="Filename of the Whisper GGUF model for ASR")
    parser.add_argument("--language", default="auto", help="Language for ASR")

    # TTS Chunk Args
    parser.add_argument("--text-to-synthesize", help="Text phrase for TTS chunk")
    parser.add_argument("--chatterbox-model-id", help="Model ID for ChatterboxTTS (informational)")
    parser.add_argument("--voice-prompt-path", help="Path to the voice prompt WAV for ChatterboxTTS")
    parser.add_argument("--exaggeration", type=float, help="Exaggeration for ChatterboxTTS")
    parser.add_argument("--cfg-weight", type=float, help="CFG weight for ChatterboxTTS")
    parser.add_argument("--output-audio-chunk-path", help="Full path where the TTS audio chunk (WAV) should be saved")

    args = parser.parse_args()
    result_payload: Dict[str, Any] = {"error": f"ThreadWorker failed task: {args.task_type}"}

    # --- TTS CHUNK TASK ---
    if args.task_type == "tts_chunk":
        log_thread_worker("INFO", f"TTS Chunk Task started. Device: '{args.device}'")
        if not TORCH_AUDIO_AVAILABLE or not torch or not torchaudio:
            result_payload = {"error": "PyTorch/Torchaudio not available for TTS chunk."}
            log_thread_worker("CRITICAL", result_payload["error"])
        elif not CHATTERBOX_TTS_AVAILABLE_THREAD or not ChatterboxTTS_class_ref_thread:
            result_payload = {"error": "ChatterboxTTS library not available for TTS chunk."}
            log_thread_worker("CRITICAL", result_payload["error"])
        elif not args.text_to_synthesize:
            result_payload = {"error": "Missing --text-to-synthesize for TTS chunk."}
            log_thread_worker("ERROR", result_payload["error"])
        elif not args.voice_prompt_path or not os.path.exists(args.voice_prompt_path):
            result_payload = {"error": f"Missing or invalid --voice-prompt-path: {args.voice_prompt_path}"}
            log_thread_worker("ERROR", result_payload["error"])
        elif not args.output_audio_chunk_path:
            result_payload = {"error": "Missing --output-audio-chunk-path for TTS chunk."}
            log_thread_worker("ERROR", result_payload["error"])
        else:
            chatterbox_model_instance_thread = None
            try:
                effective_device = _get_pytorch_device_thread_worker(args.device)
                log_thread_worker("INFO",
                                  f"Loading ChatterboxTTS model ('{args.chatterbox_model_id}') on device: {effective_device}")
                chatterbox_model_instance_thread = ChatterboxTTS_class_ref_thread.from_pretrained(
                    device=effective_device)
                log_thread_worker("INFO", "ChatterboxTTS model loaded for chunk.")

                log_thread_worker("INFO", f"Synthesizing text chunk: '{args.text_to_synthesize[:50]}...'")
                wav_tensor = chatterbox_model_instance_thread.generate(
                    args.text_to_synthesize,
                    audio_prompt_path=args.voice_prompt_path,
                    exaggeration=args.exaggeration,
                    cfg_weight=args.cfg_weight
                )
                if wav_tensor.ndim == 1: wav_tensor = wav_tensor.unsqueeze(0)
                if wav_tensor.shape[0] == 1: wav_tensor = wav_tensor.repeat(2, 1)  # Ensure Stereo for consistency

                # Save as WAV to the specified output path
                # Ensure parent directory of output_audio_chunk_path exists
                os.makedirs(os.path.dirname(args.output_audio_chunk_path), exist_ok=True)
                torchaudio.save(args.output_audio_chunk_path, wav_tensor.cpu(), chatterbox_model_instance_thread.sr,
                                format="wav")

                log_thread_worker("INFO", f"TTS chunk generated and saved to: {args.output_audio_chunk_path}")
                result_payload = {"result": {
                    "status": "success",
                    "audio_chunk_path": args.output_audio_chunk_path,
                    "sample_rate": chatterbox_model_instance_thread.sr,
                    "duration_ms": (wav_tensor.shape[1] / chatterbox_model_instance_thread.sr) * 1000
                }}
            except Exception as e_tts_chunk:
                result_payload = {"error": f"TTS chunk generation failed: {str(e_tts_chunk)}"}
                log_thread_worker("ERROR", f"TTS chunk failed: {e_tts_chunk}")
                log_thread_worker("ERROR", traceback.format_exc())
            finally:
                if chatterbox_model_instance_thread: del chatterbox_model_instance_thread; log_thread_worker("DEBUG",
                                                                                                             "ChatterboxTTS chunk model deleted.")
                gc.collect()
                if TORCH_AUDIO_AVAILABLE and torch and effective_device == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    log_thread_worker("DEBUG", "CUDA cache cleared for TTS chunk.")

    # --- ASR CHUNK TASK ---
    elif args.task_type == "asr_chunk":
        log_thread_worker("INFO", "ASR Chunk Task started.")
        if not PYWHISPERCPP_AVAILABLE_THREAD or not WhisperModel_thread:
            result_payload = {"error": "pywhispercpp not available for ASR chunk."}
            log_thread_worker("CRITICAL", result_payload["error"])
        elif not args.input_audio_chunk_path or not os.path.exists(args.input_audio_chunk_path):
            result_payload = {"error": f"Missing or invalid --input-audio-chunk-path: {args.input_audio_chunk_path}"}
            log_thread_worker("ERROR", result_payload["error"])
        elif not args.whisper_model_name:
            result_payload = {"error": "Missing --whisper-model-name for ASR chunk."}
            log_thread_worker("ERROR", result_payload["error"])
        elif not args.model_dir or not os.path.isdir(args.model_dir):  # model_dir is for ASR models
            result_payload = {"error": f"Invalid --model-dir for ASR models: {args.model_dir}"}
            log_thread_worker("ERROR", result_payload["error"])
        else:
            whisper_model_instance_thread = None
            try:
                full_model_path_asr_chunk = os.path.join(args.model_dir, args.whisper_model_name)
                if not os.path.exists(full_model_path_asr_chunk):
                    raise FileNotFoundError(
                        f"Whisper model '{args.whisper_model_name}' not found in '{args.model_dir}'")

                log_thread_worker("INFO", f"Loading Whisper model for ASR chunk: {full_model_path_asr_chunk}")
                whisper_model_instance_thread = WhisperModel_thread(model=full_model_path_asr_chunk,
                                                                    print_realtime=False, print_progress=False)

                log_thread_worker("INFO",
                                  f"Transcribing audio chunk: {args.input_audio_chunk_path} with lang: '{args.language}'")
                transcribe_params_chunk = {'language': args.language.lower(), 'translate': False}

                segments_chunk = whisper_model_instance_thread.transcribe(args.input_audio_chunk_path,
                                                                          **transcribe_params_chunk)
                transcribed_text_chunk = "".join(seg.text for seg in segments_chunk).strip()

                log_thread_worker("DEBUG", f"ASR chunk transcribed. Text snippet: '{transcribed_text_chunk[:70]}...'")
                result_payload = {"result": {"text": transcribed_text_chunk}}
            except Exception as e_asr_chunk:
                result_payload = {"error": f"ASR chunk transcription failed: {str(e_asr_chunk)}"}
                log_thread_worker("ERROR", f"ASR chunk failed: {e_asr_chunk}")
                log_thread_worker("ERROR", traceback.format_exc())
            finally:
                if whisper_model_instance_thread: del whisper_model_instance_thread; log_thread_worker("DEBUG",
                                                                                                       "Whisper ASR chunk model deleted.")
                gc.collect()  # Whisper.cpp is C++, but Python wrapper object cleanup
                # If whisper.cpp uses GPU via ggml, specific cache clearing might be backend dependent
                # For Metal on Mac, OS manages it. For CUDA, if whisper.cpp uses PyTorch CUDA context, it might benefit.
                # For now, relying on process exit to clear GPU resources used by whisper.cpp.

    else:
        result_payload = {"error": f"Unknown task_type for thread worker: {args.task_type}"}
        log_thread_worker("ERROR", result_payload["error"])

    try:
        print(json.dumps(result_payload), flush=True)
    except Exception as e_print_final:
        log_thread_worker("CRITICAL", f"Failed to serialize/print final result: {e_print_final}")
        print(json.dumps({"error": f"ThreadWorker critical error: {str(e_print_final)}"}), flush=True)
        sys.exit(1)

    log_thread_worker("INFO", f"Thread Worker finished task: {args.task_type}.")
    sys.exit(0)


if __name__ == "__main__":
    if not TORCH_AUDIO_AVAILABLE and not PYWHISPERCPP_AVAILABLE_THREAD:
        log_thread_worker("CRITICAL",
                          "Neither PyTorch/Torchaudio (for TTS) nor PyWhisperCpp (for ASR) are available. This worker cannot function.")
        sys.exit(1)
    main_thread_worker()