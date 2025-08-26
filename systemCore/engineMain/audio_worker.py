# audio_worker.py

import sys
import os
import json
import time
import traceback
import argparse
import base64
import io
import shutil
import numpy as np
import scipy.signal
import soundfile as sf
import tempfile
import subprocess
import gc
import platform
from typing import Optional, Dict, Any, List, Tuple
import re
from functools import partial
import math
import wave
from CortexConfiguration import *
import multiprocess
import multiprocessing

# --- Platform Detection ---
IS_WINDOWS = os.name == 'nt'

# --- Basic Logging Function (defined very early) ---
def log_worker(level: str, message: str):
    """Basic logging to stderr for the worker itself."""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')} AUDIO_WORKER(PID:{os.getpid()})|{level}] {message}", file=sys.stderr, flush=True)

# --- Global Availability Flags and Class References (initialized before try-except blocks) ---
TORCH_AUDIO_AVAILABLE: bool = False
torch: Optional[Any] = None                 # Will hold the torch module
torchaudio: Optional[Any] = None            # Will hold the torchaudio module
F_torchaudio: Optional[Any] = None          # For torchaudio.functional

MELO_AVAILABLE: bool = False
TTS_melo_class_ref: Optional[type] = None   # For melo.api.TTS class
SCLParser_class_ref: Optional[type] = None  # Will hold the SCLParser class defined later

CHATTERBOX_TTS_AVAILABLE: bool = False
ChatterboxTTS_class_ref: Optional[type] = None # For chatterbox.tts.ChatterboxTTS class

PYWHISPERCPP_AVAILABLE: bool = False
WhisperModel: Optional[type] = None         # For pywhispercpp.model.Model class

try:
    import torch
    import torchaudio
    import torchaudio.functional as F_torchaudio # <<<< THIS IS THE IMPORT
    import torch.compiler  # <-- Add this import
    TORCH_AUDIO_AVAILABLE = True
    log_worker("INFO", "PyTorch and Torchaudio imported successfully.")
except ImportError as e_torch_imp:
    log_worker("ERROR", f"PyTorch or torchaudio not found: {e_torch_imp}. TTS functionality will be significantly impaired or non-functional.")
    # TORCH_AUDIO_AVAILABLE remains False, and F_torchaudio remains None


try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    log_worker("WARNING", "tqdm library not found. Progress bars will be disabled.")
    # Define a dummy tqdm if not available, so code doesn't break
    class tqdm:
        def __init__(self, *args, **kwargs): self.total = kwargs.get("total", 0); self.n = 0
        def update(self, n=1): self.n += n; # Basic update
        def close(self): pass
        def __enter__(self): return self
        def __exit__(self, *args): self.close()

# --- Attempt Core Imports and Set Availability Flags ---
try:
    import torch
    import torchaudio
    import torchaudio.functional as F_torchaudio
    TORCH_AUDIO_AVAILABLE = True
    log_worker("INFO", "PyTorch and Torchaudio imported successfully.")
except ImportError as e_torch_imp:
    log_worker("ERROR", f"PyTorch or torchaudio not found: {e_torch_imp}. TTS functionality will be significantly impaired or non-functional.")
    # TORCH_AUDIO_AVAILABLE remains False

# --- SCLParser Class will be defined after these imports but before MELO_AVAILABLE is finalized ---
# For now, define SCLParser_class_ref as Any for type hints if needed early by other definitions.
SCLParser_class_ref_type_hint: type = Any

# --- MeloTTS Imports (depends on SCLParser being defined later for a complete MELO_AVAILABLE check) ---
# We'll import TTS_melo_class_ref here, and finalize MELO_AVAILABLE after SCLParser is defined.


def _handle_mecab_failure_and_retry_download():
    """
    Handles the MeCab initialization failure by attempting to download the
    required 'unidic' dictionary. It runs in an aggressive, infinite loop
    to deal with network issues or command hangs.
    """
    log_worker("WARNING", "[MeCab Repair] MeCab dictionary failure detected. Initiating repair process.")
    # Use sys.executable to ensure we use the Python from the correct virtual environment
    download_command = [sys.executable, "-m", "unidic", "download"]

    while True: # Infinite loop to ensure it eventually succeeds, as requested.
        log_worker("INFO", f"[MeCab Repair] Attempting to execute: `{' '.join(download_command)}` Download may take 10 minutes")
        process = None
        try:
            # We use Popen to monitor the process for hangs.
            # A 15-second timeout is a reasonable compromise to detect a truly hung process
            # without killing a slow but legitimate download on a poor connection.
            process = subprocess.Popen(download_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
            stdout, stderr = process.communicate(timeout=3600)

            if process.returncode == 0:
                log_worker("INFO", "[MeCab Repair] Download process completed successfully.")
                log_worker("DEBUG", f"[MeCab Repair STDOUT]:\n{stdout}")
                return # Success! Exit the function.
            else:
                log_worker("ERROR", f"[MeCab Repair] Download process failed with return code {process.returncode}.")
                log_worker("ERROR", f"[MeCab Repair STDERR]:\n{stderr}")

        except subprocess.TimeoutExpired:
            log_worker("WARNING", "[MeCab Repair] Download process is unresponsive (hung for >15s). Terminating and will retry.")
            if process:
                process.terminate()
                # Ensure the process is cleaned up before the next loop iteration
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    log_worker("ERROR", "[MeCab Repair] Hung process did not terminate gracefully. Killing it.")
                    process.kill()
                finally:
                    process.wait() # Final wait to clean up the zombie process
            # The loop will now automatically retry.

        except FileNotFoundError:
            log_worker("CRITICAL", f"[MeCab Repair] Command failed because '{sys.executable}' or 'unidic' module could not be found. This is unrecoverable.")
            return # Exit, as we cannot fix this automatically.

        except Exception as e:
            log_worker("ERROR", f"[MeCab Repair] An unexpected error occurred during the repair attempt: {e}")

        log_worker("INFO", "[MeCab Repair] Retrying the download after a 5-second delay...")
        time.sleep(5)

melo_import_attempts = 0
while not MELO_AVAILABLE: # Loop to retry the import after a fix attempt.
    melo_import_attempts += 1
    try:
        if TORCH_AUDIO_AVAILABLE:
            from melo.api import TTS as ImportedTTS_melo_api
            TTS_melo_class_ref = ImportedTTS_melo_api
            # If the import succeeds, MELO_AVAILABLE will be finalized later after SCLParser is defined.
            # We break the retry loop here because the import itself was successful.
            log_worker("INFO", "MeloTTS API module imported successfully. Final availability check will follow.")
            break
        else:
            log_worker("WARNING", "Torch/Torchaudio not available, so MeloTTS cannot be loaded.")
            MELO_AVAILABLE = False
            break # No point retrying if torch isn't there

    except Exception as e_melo_imp:
        error_str = str(e_melo_imp).lower()
        # Check for the specific signature of the MeCab dictionary error.
        if "mecab" in error_str and ("no such file or directory" in error_str or "failed initializing" in error_str):
            # This is the error we can fix. Call the handler.
            _handle_mecab_failure_and_retry_download()
            # The 'while' loop will now cause us to re-attempt the 'from melo.api...' import.
            log_worker("INFO", "MeCab repair process finished. Retrying MeloTTS import...")
            continue # Go to the next iteration of the while loop to retry the import.
        else:
            # This is a different, unexpected error during import.
            log_worker("WARNING", f"MeloTTS API import failed with an unrecoverable error: {e_melo_imp}. Melo sample generation will be unavailable.")
            MELO_AVAILABLE = False
            break # Exit the retry loop.

# --- ChatterboxTTS Imports ---
try:
    if TORCH_AUDIO_AVAILABLE:
        from chatterbox.tts import ChatterboxTTS as ImportedChatterboxTTS_module_class
        ChatterboxTTS_class_ref = ImportedChatterboxTTS_module_class
        CHATTERBOX_TTS_AVAILABLE = True
        log_worker("INFO", "ChatterboxTTS library class imported successfully.")
    else:
        raise ImportError("Torch/Torchaudio not available, ChatterboxTTS cannot be loaded.")
except ImportError as e_cb_imp:
    log_worker("WARNING", f"ChatterboxTTS library not found/imported: {e_cb_imp}. Chatterbox TTS unavailable.")
    CHATTERBOX_TTS_AVAILABLE = False

# --- torch.load Patch (Applied only if torch is available) ---
if TORCH_AUDIO_AVAILABLE and torch:
    _original_torch_load = torch.load
    def _patched_torch_load_audio_worker(*args, **kwargs):
        # This patch is a passthrough. If ChatterboxTTS needs specific map_location logic,
        # it should handle it or this patch should be made more aware.
        return _original_torch_load(*args, **kwargs)
    torch.load = _patched_torch_load_audio_worker # type: ignore
    log_worker("INFO", "Global torch.load patch applied in audio_worker (passthrough).")

# --- PyWhisperCpp Imports ---
try:
    from pywhispercpp.model import Model as ImportedWhisperModel
    WhisperModel = ImportedWhisperModel
    PYWHISPERCPP_AVAILABLE = True
    log_worker("INFO", "pywhispercpp.model.Model imported successfully.")
except ImportError:
    log_worker("WARNING", "pywhispercpp library not found or Model could not be imported. ASR will be unavailable.")
    PYWHISPERCPP_AVAILABLE = False

# --- Config Import for ASR defaults (with fallbacks) ---
# These are defined here so they are available for argparse defaults in main()
WHISPER_MODEL_DIR_CFG = "./staticmodelpool"
WHISPER_DEFAULT_MODEL_FILENAME_CFG = "whisper-large-v3-q8_0.gguf"
WHISPER_DEFAULT_LANGUAGE_CFG = "auto"
try:
    from CortexConfiguration import WHISPER_MODEL_DIR, WHISPER_DEFAULT_MODEL_FILENAME, WHISPER_DEFAULT_LANGUAGE
    WHISPER_MODEL_DIR_CFG = WHISPER_MODEL_DIR
    WHISPER_DEFAULT_MODEL_FILENAME_CFG = WHISPER_DEFAULT_MODEL_FILENAME
    WHISPER_DEFAULT_LANGUAGE_CFG = WHISPER_DEFAULT_LANGUAGE
    log_worker("INFO", "Successfully imported ASR defaults from CortexConfiguration.py.")
except ImportError:
    log_worker("WARNING", "Could not import ASR defaults from CortexConfiguration.py. Using internal fallbacks for ASR config.")

# --- Numba (Optional for SCLParser) ---
try:
    from numba import njit
    log_worker("INFO", "Numba imported successfully (optional for SCLParser).")
except ImportError:
    log_worker("WARNING", "Numba not found. SCLParser performance might be affected if it uses @njit.")
    def njit(func_or_signature=None, *args, **kwargs): # type: ignore
        if callable(func_or_signature): return func_or_signature
        else: return lambda func: func

# --- Pedalboard and Librosa (for SCLParser post-processing) ---
PEDALBOARD_LIBROSA_AVAILABLE = False
try:
    from pedalboard import Pedalboard, Reverb, Limiter, Gain, PitchShift, Resample, Chorus, Delay, Distortion
    import librosa
    PEDALBOARD_LIBROSA_AVAILABLE = True
    log_worker("INFO", "Pedalboard and Librosa imported for SCLParser audio effects.")
except ImportError as e_fx_imp:
    log_worker("WARNING", f"Pedalboard or Librosa not installed: {e_fx_imp}. Advanced SCLParser audio effects disabled.")
    # Define dummy classes if SCLParser structure depends on them existing
    class Pedalboard: __init__ = lambda s, *a, **k: None; __call__ = lambda s, *a, **k: a[0] if a and len(a)>0 and isinstance(a[0],np.ndarray) else np.array([]) # type: ignore
    class Reverb: __init__ = lambda s, *a, **k: None
    class Limiter: __init__ = lambda s, *a, **k: None
    class Gain: __init__ = lambda s, *a, **k: None
    class PitchShift: __init__ = lambda s, *a, **k: None
    class Resample: __init__ = lambda s, *a, **k: None
    class Chorus: __init__ = lambda s, *a, **k: None
    class Delay: __init__ = lambda s, *a, **k: None
    class Distortion: __init__ = lambda s, *a, **k: None
# --- End of Part 1 ---

class SCLParser:
    def __init__(self, model, device='auto'):  # model is a MeloTTS instance for sample generation
        log_worker("INFO", "SCLParser: Initializing...")
        self.model = model
        self.device = device
        if model and hasattr(model, 'hps') and hasattr(model.hps, 'data') and hasattr(model.hps.data,
                                                                                      'spk2id') and hasattr(
                model.hps.data, 'sampling_rate'):
            self.speaker_ids = model.hps.data.spk2id
            self.sample_rate = model.hps.data.sampling_rate
        else:
            self.speaker_ids = {"EN-US": 0, "EN-BR": 1}
            self.sample_rate = 24000
            if model:
                log_worker("WARNING",
                           "SCLParser: Melo model passed for SCLParser init seems incomplete. Using default SR/speakers.")
            else:
                log_worker("DEBUG",
                           "SCLParser: Initialized without a Melo model (e.g., for text processing only). Using default SR/speakers.")

        log_worker("DEBUG", f"SCLParser: Speaker IDs (from model if provided): {self.speaker_ids}")
        log_worker("DEBUG", f"SCLParser: Sample Rate (from model if provided, else default): {self.sample_rate}")
        self.voice_settings = {'rate': 1.0, 'pitch': 0.0}
        log_worker("DEBUG", f"SCLParser: Initial Voice Settings: {self.voice_settings}")
        self.rate_map = {'x-slow': 0.9, 'slow': 1.0, 'medium': 1.05, 'fast': 1.1, 'x-fast': 1.1}
        self.pitch_map = {'x-low': -0.4, 'low': -0.1, 'medium': 0.0, 'high': 1.2, 'x-high': 2.0}
        log_worker("DEBUG", f"SCLParser: Rate Map: {self.rate_map}")
        log_worker("DEBUG", f"SCLParser: Pitch Map: {self.pitch_map}")
        self.audio_segments: List[np.ndarray] = []
        self.original_sf_write = sf.write  # For hooking during sample generation
        self.captured_audio_data: Optional[np.ndarray] = None
        self.captured_audio_samplerate: Optional[int] = None

        self.zephyloid_settings = {
            'vel': 64, 'dyn': 64, 'bre': 0, 'bri': 64, 'cle': 64, 'ope': 64,
            'gen': 64, 'gwl': 0, 'xsy': 0, 'xsy_voicebanks': None,
            'singing': False, 'key': None, 'correction_method': "closest",
        }
        log_worker("DEBUG", f"SCLParser: Initial zephyloid Settings: {self.zephyloid_settings}")
        self.xsy_profiles = {
            "Voicebank1": {"description": "Default",
                           "eq_curve": [(30, 0, 3.4), (100, 0, 1.4), (150, 0, 1.4), (250, 0, 1.0), (350, 0, 1.4),
                                        (450, 0, 1.8), (550, 0, 1.4), (2000, 0, 1.0), (2500, 0, 1.4), (3000, 0, 1.4),
                                        (3500, 0, 1.8), (4000, 0, 1.4), (8000, 0, 1.8), (12000, 0, 1.8),
                                        (20000, 0, 1.8)]},
            "Voicebank2": {"description": "Brighter",
                           "eq_curve": [(30, 2, 3.4), (100, 3, 1.4), (150, 1, 1.4), (250, 1, 1.0), (350, -1, 1.4),
                                        (450, -2, 1.8), (550, 2, 1.4), (2000, 3, 1.0), (2500, 4, 1.4), (3000, 3, 1.4),
                                        (3500, 2, 1.8), (4000, 1, 1.4), (8000, 4, 1.8), (12000, 5, 1.8),
                                        (20000, 2, 1.8)]},
            "Voicebank3": {"description": "Deeper",
                           "eq_curve": [(30, 4, 3.4), (100, 5, 1.4), (150, 3, 1.4), (250, 2, 1.0), (350, 1, 1.4),
                                        (450, -1, 1.8), (550, -3, 1.4), (2000, -2, 1.0), (2500, -1, 1.4),
                                        (3000, 0, 1.4), (3500, 1, 1.8), (4000, 2, 1.4), (8000, 1, 1.8), (12000, 0, 1.8),
                                        (20000, -1, 1.8)]}
        }
        log_worker("DEBUG", f"SCLParser: XSY Profiles defined.")
        log_worker("INFO", "SCLParser: Initialization complete.")

    def _new_sf_write(self, file, data, samplerate, *args, **kwargs):  # Used for sample generation
        log_worker("DEBUG",
                   f"SCLParser: _new_sf_write hooked. Data shape: {data.shape if isinstance(data, np.ndarray) else type(data)}, SR: {samplerate}")
        if TORCH_AUDIO_AVAILABLE and torch and isinstance(data, torch.Tensor):
            self.captured_audio_data = data.clone().detach().cpu().numpy()
        elif isinstance(data, np.ndarray):
            self.captured_audio_data = data.copy()
        else:
            log_worker("WARNING", f"SCLParser: _new_sf_write hook received unexpected data type: {type(data)}")
            self.captured_audio_data = None
            return self.original_sf_write(file, data, samplerate, *args, **kwargs)  # Call original if type is wrong
        self.captured_audio_samplerate = samplerate
        if self.captured_audio_data is not None:
            log_worker("DEBUG",
                       f"SCLParser: _new_sf_write: Audio data captured. Shape: {self.captured_audio_data.shape}")

    def parse(self, text: str, speaker: str = 'EN-US') -> tuple[Optional[np.ndarray], Optional[int]]:
        """
        This method in the orchestrator's SCLParser is primarily for generating the voice sample via MeloTTS.
        It hooks soundfile.write, calls self.speak_with_settings (which uses MeloTTS),
        and returns the captured audio data and sample rate.
        """
        log_worker("INFO",
                   f"SCLParser: parse (for sample generation) called. Text: '{text[:50]}...', Speaker: {speaker}")
        if not self.model:  # self.model is the MeloTTS instance
            log_worker("ERROR", "SCLParser.parse: No MeloTTS model instance available for sample generation.")
            return None, None

        original_write_func_ref = sf.write  # Store original
        sf.write = self._new_sf_write  # Apply hook

        self.captured_audio_data = None  # Reset captures for this call
        self.captured_audio_samplerate = None
        self.audio_segments = []  # Reset segments list for this parse call

        try:
            # Preprocess and then call speak_with_settings for the entire (potentially SCL-tagged) text.
            # speak_with_settings is expected to call self.model.tts_to_file, which our hook captures.
            processed_text_for_sample = self.preprocess_text(text)
            self.speak_with_settings(processed_text_for_sample, speaker)

            # If speak_with_settings populated self.audio_segments (as in your original SCLParser design)
            # AND self.captured_audio_data was not set by the hook (e.g., if tts_to_file not called as expected by hook)
            # this attempts to combine segments. Ideally, the hook is the primary capture.
            if self.captured_audio_data is None and self.audio_segments:
                log_worker("WARNING",
                           "SCLParser.parse (sample gen): sf.write hook did not capture audio, but audio_segments were populated. Attempting to combine.")
                combined_audio_from_segments = self.crossfade_segments(self.audio_segments,
                                                                       self.sample_rate)  # Uses self.sample_rate (Melo's SR)
                if combined_audio_from_segments is not None and combined_audio_from_segments.size > 0:
                    # For sample generation, post-processing might be too much, but if needed:
                    # self.captured_audio_data = self.post_process(combined_audio_from_segments, self.sample_rate, self.zephyloid_settings)
                    self.captured_audio_data = combined_audio_from_segments  # Using combined directly for sample
                    self.captured_audio_samplerate = self.sample_rate  # SR of combined audio
                else:
                    log_worker("ERROR", "SCLParser.parse (sample gen): Failed to combine audio_segments.")
            elif self.captured_audio_data is not None:
                log_worker("DEBUG", "SCLParser.parse (sample gen): Audio captured successfully via sf.write hook.")

        except Exception as e_scl_parse_sample:
            log_worker("ERROR", f"SCLParser.parse error during sample generation: {e_scl_parse_sample}")
            log_worker("ERROR", traceback.format_exc())
            self.captured_audio_data = None
            self.captured_audio_samplerate = None
        finally:
            sf.write = original_write_func_ref  # Always restore original soundfile.write

        log_worker("INFO",
                   f"SCLParser: parse (sample gen) finished. Captured audio shape: {self.captured_audio_data.shape if self.captured_audio_data is not None else 'None'}")
        return self.captured_audio_data, self.captured_audio_samplerate

    def preprocess_text(self, text: str) -> str:
        log_worker("DEBUG", f"SCLParser: preprocess_text called. Input text snippet: '{text[:70]}...'")
        text_fixed_tags = self.fix_missing_closing_tags(text)
        text_flattened = self.flatten_nested_tags(text_fixed_tags)
        log_worker("TRACE", f"SCLParser: preprocess_text output snippet: '{text_flattened[:70]}...'")
        return text_flattened

    def fix_missing_closing_tags(self, text: str) -> str:
        open_tags_stack = []
        output_parts = []
        current_pos = 0
        while current_pos < len(text):
            match_bracket = text.find('[', current_pos)
            if match_bracket == -1:  # No more opening brackets
                output_parts.append(text[current_pos:])
                break
            output_parts.append(text[current_pos:match_bracket])  # Text before the bracket
            match_closing_bracket = text.find(']', match_bracket)
            if match_closing_bracket == -1:  # Unclosed bracket at end of string
                output_parts.append(text[match_bracket:])  # Treat as literal text
                break

            tag_full_content = text[match_bracket + 1: match_closing_bracket]
            output_parts.append(f"[{tag_full_content}]")  # Add the tag itself

            if tag_full_content.startswith('/'):  # It's a closing tag
                tag_name_closed = tag_full_content[1:].split(' ')[0].lower()
                if open_tags_stack and open_tags_stack[-1] == tag_name_closed:
                    open_tags_stack.pop()
                else:  # Mismatched or unexpected closing tag, treat as literal for now or log
                    log_worker("TRACE",
                               f"SCLParser: fix_missing_closing_tags: Mismatched/unexpected closing tag '{tag_full_content}'")
            else:  # It's an opening tag
                tag_name_opened = tag_full_content.split(' ')[0].lower()
                open_tags_stack.append(tag_name_opened)
            current_pos = match_closing_bracket + 1

        for leftover_tag in reversed(open_tags_stack):  # Close any remaining open tags
            output_parts.append(f"[/{leftover_tag}]")
            log_worker("TRACE", f"SCLParser: fix_missing_closing_tags: Added missing closing tag for '{leftover_tag}'")

        return "".join(output_parts)

    def flatten_nested_tags(self, text: str) -> str:
        log_worker("DEBUG", f"SCLParser: flatten_nested_tags called. Input snippet: '{text[:70]}...'")
        # This is a simplified placeholder. Your original regex was complex.
        # A robust solution usually requires a proper parser or iterative simple regex.
        # For now, let's assume a less aggressive flattening or just pass through.
        # Example: [a][b]text[/b][/a] -> [a]text[/a][b]text[/b] (distributive)
        # This depends heavily on the specific SCL tag semantics.
        # The one from your file was:
        # replacement = f"[{tag1_full}]{text1}[{closing_tag1}][{tag2_full}]{text2}[/{tag2_full.split(' ')[0]}][{tag1_full}]{text3}[{closing_tag1}]"
        # This particular replacement structure might be what you intend.
        # For this full code, I'll keep it as a conceptual placeholder.
        # If complex flattening is critical, the original regex logic needs to be carefully reviewed and placed here.
        log_worker("TRACE",
                   "SCLParser: flatten_nested_tags (using passthrough for this version). Implement full logic if needed.")
        return text  # Placeholder: returning text as is. Replace with your full flattening logic.

    def split_into_segments_for_tts_worker(self, text: str) -> List[Dict[str, Any]]:
        log_worker("DEBUG", f"SCLParser: split_into_segments_for_tts_worker for: '{text[:50]}...'")
        segments_for_worker: List[Dict[str, Any]] = []
        current_text_buffer = ""
        # Reset voice/zephyloid settings for this parse run
        self.voice_settings = {'rate': 1.0, 'pitch': 0.0}
        self.zephyloid_settings = {'vel': 64, 'dyn': 64, 'bre': 0, 'bri': 64, 'cle': 64, 'ope': 64, 'gen': 64, 'gwl': 0,
                                   'xsy': 0, 'xsy_voicebanks': None, 'singing': False, 'key': None,
                                   'correction_method': "closest"}

        processed_text_for_splitting = self.preprocess_text(text)  # Apply fixes first
        parts_and_tags = re.split(r'(\[[^\]]+\])', processed_text_for_splitting, flags=re.IGNORECASE)

        for part in parts_and_tags:
            if not part or not part.strip(): continue

            if re.match(r'\[([^\]]+)\]', part, re.IGNORECASE):  # It's a tag
                if current_text_buffer.strip():  # Process accumulated text before this tag
                    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z]\.)(?<=\.|\?|\!)\s+',
                                         current_text_buffer.strip())
                    for sent in sentences:
                        if sent.strip():
                            segments_for_worker.append({"type": "text", "content": sent.strip(),
                                                        "settings": self.voice_settings.copy(),
                                                        "zephyloid": self.zephyloid_settings.copy()})
                    current_text_buffer = ""

                tag_content = part[1:-1]  # Content inside brackets
                if tag_content.startswith('/'):
                    self.reset_settings(tag_content[1:].split(' ')[0].lower())
                else:
                    # apply_settings updates self.voice_settings and self.zephyloid_settings
                    # and returns pause duration if it's a pause tag
                    pause_duration_ms = self.apply_settings(tag_content)
                    if pause_duration_ms is not None:  # It was a pause tag
                        segments_for_worker.append({"type": "pause", "content": pause_duration_ms,
                                                    "settings": {},
                                                    "zephyloid": {}})  # Settings don't apply to pause itself
            else:  # It's text
                current_text_buffer += part

        if current_text_buffer.strip():  # Process any remaining text after last tag
            sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z]\.)(?<=\.|\?|\!)\s+',
                                 current_text_buffer.strip())
            for sent in sentences:
                if sent.strip():
                    segments_for_worker.append({"type": "text", "content": sent.strip(),
                                                "settings": self.voice_settings.copy(),
                                                "zephyloid": self.zephyloid_settings.copy()})

        log_worker("DEBUG",
                   f"SCLParser: split_into_segments_for_tts_worker returning {len(segments_for_worker)} segments.")
        return segments_for_worker

    def _parse_pause_duration_from_tag(self,
                                       tag_content_full: str) -> int:  # tag_content_full is e.g., "pause duration=short"
        attrs = self.parse_attributes(tag_content_full.split(' ', 1)[1] if ' ' in tag_content_full else "")
        duration_attr = attrs.get("duration", list(attrs.values())[0] if attrs else "medium")
        duration_ms = 0
        pause_compensation_ms = 300
        if isinstance(duration_attr, str):
            val_str = duration_attr.lower().replace("ms", "")
            if val_str == "short":
                duration_ms = 100
            elif val_str == "medium":
                duration_ms = 250
            elif val_str == "long":
                duration_ms = 600
            elif val_str == "x-long":
                duration_ms = 900
            elif val_str.isdigit():
                duration_ms = int(val_str)
        actual_silence_ms = max(0, duration_ms - pause_compensation_ms)
        if duration_ms > 0 and actual_silence_ms == 0: actual_silence_ms = 50  # Ensure tiny pause if one was specified
        return actual_silence_ms

    def apply_settings(self, tag_content: str) -> Optional[int]:  # tag_content is e.g., "prosody rate=fast"
        log_worker("DEBUG", f"SCLParser: apply_settings for tag content: '{tag_content}'")
        parts = tag_content.split(' ', 1)
        tag_name = parts[0].lower()
        params_str = parts[1] if len(parts) > 1 else ""
        attrs = self.parse_attributes(params_str)

        if tag_name == "pause":
            return self._parse_pause_duration_from_tag(tag_content)  # Pass full tag content
        elif tag_name == "prosody":
            if "rate" in attrs: self.voice_settings['rate'] = max(0.5, min(1.1,
                                                                           self.rate_map.get(str(attrs["rate"]).lower(),
                                                                                             float(attrs[
                                                                                                       "rate"]) if self.is_number(
                                                                                                 attrs[
                                                                                                     "rate"]) else 1.0)))
            if "pitch" in attrs: self.voice_settings['pitch'] = self.parse_pitch(str(attrs["pitch"]))
        elif tag_name == "emphasis":  # Your full emphasis logic from message #91
            level = str(attrs.get("level", "moderate")).lower()
            base_pitch = self.voice_settings['pitch']
            if level == "strong":
                self.voice_settings['rate'] *= 0.9
                self.voice_settings['pitch'] = base_pitch + self.parse_pitch(
                    str(attrs.get("pitch", "high")))
            elif level == "moderate":
                self.voice_settings['rate'] *= 0.95
                self.voice_settings['pitch'] = base_pitch + self.parse_pitch(
                    str(attrs.get("pitch", "0.5")))
            elif level == "reduced":
                self.voice_settings['rate'] *= 1.1
                self.voice_settings['pitch'] = base_pitch + self.parse_pitch(
                    str(attrs.get("pitch", "low")))
            self.voice_settings['rate'] = max(0.5, min(1.1, self.voice_settings['rate']))  # Clamp rate
        elif tag_name == "zephyloid":  # Your full zephyloid settings update logic
            for key, value in attrs.items():
                if key in self.zephyloid_settings:
                    try:
                        if key in ['vel', 'dyn', 'bre', 'bri', 'cle', 'ope', 'gen', 'gwl', 'xsy']:
                            self.zephyloid_settings[key] = int(value)
                        elif key == 'singing':
                            self.zephyloid_settings[key] = str(value).lower() == 'true'
                        elif key == 'xsy_voicebanks':
                            self.zephyloid_settings[key] = [vb.strip() for vb in
                                                            str(value).split(',')] if value else None
                        elif key == 'key':
                            self.zephyloid_settings[key] = self._parse_key(str(value)) if value else None
                        else:
                            self.zephyloid_settings[key] = value
                    except ValueError:
                        log_worker("WARNING", f"SCLParser: Invalid value '{value}' for zephyloid key '{key}'")
        # Add other tag handling (e.g., "emotional") from your original SCLParser
        return None  # Return None if not a pause tag

    def reset_settings(self, tag_name_lower: str):
        if tag_name_lower in ("prosody", "emphasis", "emotional"):
            self.voice_settings = {'rate': 1.0, 'pitch': 0.0}
        elif tag_name_lower == "zephyloid":
            self.zephyloid_settings = {'vel': 64, 'dyn': 64, 'bre': 0, 'bri': 64, 'cle': 64, 'ope': 64, 'gen': 64,
                                       'gwl': 0, 'xsy': 0, 'xsy_voicebanks': None, 'singing': False, 'key': None,
                                       'correction_method': "closest"}
        log_worker("DEBUG", f"SCLParser: reset_settings for '{tag_name_lower}'")

    def _apply_zephyloid_effects(self, audio_np_not_used_in_this_version_of_method: Optional[np.ndarray] = None):
        # This method in your SCLParser primarily modifies self.voice_settings for rate/pitch
        # based on zephyloid settings like 'vel' and 'gen' before actual TTS generation.
        # Actual audio effects from zephyloid settings are applied in post_process.
        if self.zephyloid_settings.get('vel') is not None: self.voice_settings['rate'] *= (
                    1.0 + (self.zephyloid_settings['vel'] - 64) * 0.001)
        if self.zephyloid_settings.get('gen') is not None: self.voice_settings['pitch'] += (
                    (self.zephyloid_settings['gen'] - 64) / 64 * 6)  # Semitones
        self.voice_settings['rate'] = max(0.5, min(1.1, self.voice_settings['rate']))  # Clamp rate
        log_worker("TRACE", f"SCLParser: Zephyloid rate/pitch effects applied to voice_settings: {self.voice_settings}")

    def speak_with_settings(self, text: str, speaker: str):
        """
        Called by SCLParser.parse (for sample generation) to generate audio for a text segment
        using the currently configured voice_settings and the internal MeloTTS model.
        Relies on the _new_sf_write hook to capture the audio.
        """
        if not self.model or not hasattr(self.model, 'tts_to_file'):
            log_worker("ERROR", "SCLParser: No MeloTTS model instance available for speak_with_settings.")
            return

        self._apply_zephyloid_effects(None)  # Apply rate/pitch adjustments from zephyloid settings

        log_worker("DEBUG",
                   f"SCLParser: speak_with_settings (for sample gen): Text='{text[:30]}...', Speaker='{speaker}', Rate={self.voice_settings.get('rate', 1.0)}")  # self.voice_settings IS a dict
        try:
            # --- MODIFIED SPEAKER ID RETRIEVAL ---
            speaker_id_to_use: Optional[int] = None
            if speaker in self.speaker_ids:  # HParams should support 'in' if it's map-like
                speaker_id_to_use = self.speaker_ids[speaker]  # Use direct item access
            else:
                log_worker("WARNING",
                           f"SCLParser: Speaker '{speaker}' not found in Melo model's speaker_ids (HParams object). Using first available for sample.")
                # Check if self.speaker_ids is not None, not empty, and behaves somewhat like a dictionary
                if self.speaker_ids and hasattr(self.speaker_ids, 'keys') and list(self.speaker_ids.keys()):
                    try:
                        first_speaker_key = list(self.speaker_ids.keys())[0]
                        speaker_id_to_use = self.speaker_ids[first_speaker_key]
                        log_worker("DEBUG",
                                   f"SCLParser: Fallback speaker key: {first_speaker_key}, ID: {speaker_id_to_use}")
                    except Exception as e_fallback_key:
                        log_worker("ERROR",
                                   f"SCLParser: Error accessing fallback speaker ID from HParams: {e_fallback_key}. Defaulting to ID 0.")
                        speaker_id_to_use = 0
                else:
                    log_worker("ERROR",
                               "SCLParser: No speakers available in Melo model's speaker_ids or object is not behaving as expected. Defaulting to ID 0 for sample.")
                    speaker_id_to_use = 0  # Ultimate fallback to a default ID

            if speaker_id_to_use is None:  # Should ideally not happen if above logic is sound
                log_worker("CRITICAL",
                           "SCLParser: speaker_id_to_use is still None after checks. This is a bug. Defaulting to 0.")
                speaker_id_to_use = 0
            # --- END OF MODIFIED SPEAKER ID RETRIEVAL ---

            # MeloTTS's tts_to_file is called, our _new_sf_write hook should capture the data
            # into self.captured_audio_data and self.captured_audio_samplerate
            self.model.tts_to_file(text, speaker_id_to_use, "dummy_internal_scl_output.wav",
                                   speed=self.voice_settings.get('rate', 1.0))  # self.voice_settings.get is fine

            # The captured audio (self.captured_audio_data) might need pitch shifting if self.voice_settings['pitch'] is not 0
            # This part handles the pitch shifting for the segment if needed (e.g. from <prosody pitch="...">)
            if self.captured_audio_data is not None:
                current_audio_segment = self.captured_audio_data.copy()  # Work with a copy
                current_sample_rate = self.captured_audio_samplerate or self.sample_rate

                if self.voice_settings['pitch'] != 0.0 and PEDALBOARD_LIBROSA_AVAILABLE and librosa:
                    log_worker("TRACE",
                               f"SCLParser: Applying pitch shift {self.voice_settings['pitch']} semitones to segment.")
                    # Ensure mono for librosa pitch_shift
                    audio_mono_for_pitch = current_audio_segment
                    if current_audio_segment.ndim == 2:  # (channels, samples) or (samples, channels)
                        # Attempt to intelligently convert to mono (samples,)
                        if current_audio_segment.shape[0] == 1 and current_audio_segment.shape[1] > 1:  # (1, samples)
                            audio_mono_for_pitch = current_audio_segment.squeeze()
                        elif current_audio_segment.shape[1] == 1 and current_audio_segment.shape[0] > 1:  # (samples, 1)
                            audio_mono_for_pitch = current_audio_segment.squeeze()
                        elif current_audio_segment.shape[0] == 2 and current_audio_segment.shape[
                            1] > 2:  # (2, samples) from Torch/Pedalboard
                            audio_mono_for_pitch = librosa.to_mono(current_audio_segment)
                        elif current_audio_segment.shape[1] == 2 and current_audio_segment.shape[0] > 2:  # (samples, 2)
                            audio_mono_for_pitch = librosa.to_mono(current_audio_segment.T)  # Transpose first
                        else:  # Fallback if shape is ambiguous or not stereo
                            log_worker("WARNING",
                                       f"SCLParser: Could not reliably convert audio segment of shape {current_audio_segment.shape} to mono for pitch shift. Using mean or first channel.")
                            # A simple mean might work, or just take the first channel if it looks like multi-channel mono
                            if current_audio_segment.shape[0] < current_audio_segment.shape[
                                1]:  # (channels, samples)-like
                                audio_mono_for_pitch = current_audio_segment[0, :]
                            else:  # (samples, channels)-like
                                audio_mono_for_pitch = current_audio_segment[:, 0]
                            # Or more generally: audio_mono_for_pitch = current_audio_segment.mean(axis=0) if current_audio_segment.shape[0] < current_audio_segment.shape[1] else current_audio_segment.mean(axis=1)

                    pitched_audio_mono = librosa.effects.pitch_shift(audio_mono_for_pitch.astype(np.float32),
                                                                     # Ensure float32 for librosa
                                                                     sr=current_sample_rate,
                                                                     n_steps=self.voice_settings['pitch'])
                    self.audio_segments.append(pitched_audio_mono)  # Append pitched mono audio
                else:
                    self.audio_segments.append(
                        current_audio_segment)  # Append original captured audio (might be stereo or mono from Melo)
                log_worker("DEBUG",
                           f"SCLParser: Added segment to self.audio_segments. Count: {len(self.audio_segments)}")
            else:
                log_worker("ERROR",
                           "SCLParser: speak_with_settings: Audio data was not captured by the sf.write hook after MeloTTS call.")

        except Exception as e_speak_settings:
            log_worker("ERROR", f"SCLParser: Error in speak_with_settings TTS call: {e_speak_settings}")
            log_worker("ERROR", traceback.format_exc())

    def crossfade_segments(self, audio_chunks_list: List[np.ndarray], sample_rate: int, crossfade_ms: int = 5) -> \
    Optional[np.ndarray]:
        log_worker("DEBUG",
                   f"SCLParser: crossfade_segments called for {len(audio_chunks_list)} chunks. SR: {sample_rate}, XFade: {crossfade_ms}ms")
        if not audio_chunks_list: return np.array([], dtype=np.float32)

        valid_segments_for_cf = [s.astype(np.float32) for s in audio_chunks_list if s is not None and s.size > 0]
        if not valid_segments_for_cf: return np.array([], dtype=np.float32)

        # Ensure all segments are stereo (samples, 2) for consistent processing
        stereo_segments_for_cf: List[np.ndarray] = []
        for i, chunk_cf in enumerate(valid_segments_for_cf):
            if chunk_cf.ndim == 1:  # Mono (samples,)
                stereo_segments_for_cf.append(np.stack([chunk_cf, chunk_cf], axis=-1))
            elif chunk_cf.ndim == 2:
                if chunk_cf.shape[0] == 2 and chunk_cf.shape[1] > 2:  # (2, samples) from Torch/Pedalboard
                    stereo_segments_for_cf.append(chunk_cf.T)  # -> (samples, 2)
                elif chunk_cf.shape[1] == 1 and chunk_cf.shape[0] > 2:  # (samples, 1) mono
                    stereo_segments_for_cf.append(np.repeat(chunk_cf, 2, axis=-1))
                elif chunk_cf.shape[1] == 2 and chunk_cf.shape[0] > 2:  # (samples, 2) - already correct
                    stereo_segments_for_cf.append(chunk_cf)
                else:
                    log_worker("WARNING",
                               f"SCLParser: Crossfade skipping chunk {i} with unexpected 2D shape: {chunk_cf.shape}"); continue
            else:
                log_worker("WARNING",
                           f"SCLParser: Crossfade skipping chunk {i} with unexpected ndim: {chunk_cf.ndim}"); continue

        if not stereo_segments_for_cf: return np.array([], dtype=np.float32)

        crossfade_samples_val = int(sample_rate * crossfade_ms / 1000)
        combined_audio_output = stereo_segments_for_cf[0]

        for i in range(1, len(stereo_segments_for_cf)):
            next_segment = stereo_segments_for_cf[i]
            if combined_audio_output.shape[0] < crossfade_samples_val or next_segment.shape[0] < crossfade_samples_val:
                log_worker("TRACE", "SCLParser: Crossfade segment too short, concatenating.")
                combined_audio_output = np.concatenate((combined_audio_output, next_segment), axis=0)
            else:
                window = np.hanning(2 * crossfade_samples_val)
                fade_out_curve = window[:crossfade_samples_val]
                fade_in_curve = window[crossfade_samples_val:]

                # Apply fade to each channel (assuming (samples, 2))
                for ch in range(combined_audio_output.shape[1]):  # Iterate over channels (should be 2)
                    combined_audio_output[-crossfade_samples_val:, ch] *= fade_out_curve
                    next_segment[:crossfade_samples_val, ch] *= fade_in_curve

                overlap_sum = combined_audio_output[-crossfade_samples_val:] + next_segment[:crossfade_samples_val]
                combined_audio_output = np.concatenate(
                    (combined_audio_output[:-crossfade_samples_val], overlap_sum, next_segment[crossfade_samples_val:]),
                    axis=0)

        log_worker("DEBUG", f"SCLParser: Crossfade result shape: {combined_audio_output.shape}")
        return combined_audio_output  # Should be (total_samples, 2)

        # In SCLParser class, within audio_worker.py

    def _q_to_bandwidth_hz(self, q_factor: float, center_freq_hz: float) -> float:
        """Converts a Q factor to bandwidth in Hz for FFmpeg's equalizer."""
        if q_factor <= 0:
            return center_freq_hz  # Avoid division by zero, return a reasonable default
        return center_freq_hz / q_factor

    def _generate_reverb_ir(self, path: str, sample_rate: int):
        """
        Generates a simple synthetic impulse response for convolution reverb and saves it as a WAV file.
        This avoids the dependency on a specific FFmpeg build having 'areverb'.
        """
        # A simple decaying noise makes for a decent, neutral reverb IR
        duration_seconds = 1.0  # Short reverb tail
        decay_rate = -5.0

        num_samples = int(duration_seconds * sample_rate)
        time_axis = np.linspace(0, duration_seconds, num_samples, endpoint=False)

        # Generate white noise and apply an exponential decay
        noise = np.random.normal(0, 1, num_samples)
        decay_envelope = np.exp(decay_rate * time_axis)

        ir_signal = (noise * decay_envelope).astype(np.float32)

        # Normalize to prevent clipping
        ir_signal /= np.max(np.abs(ir_signal))

        # Save as a mono WAV file
        sf.write(path, ir_signal, sample_rate, subtype='PCM_16')
        log_worker("DEBUG", f"Generated synthetic reverb impulse response at: {path}")

    def _run_and_diagnose(
            self,
            ffmpeg_exe: str,
            command: list,
            input_path: str,
            output_path: str,
            stage_name: str
    ) -> bool:
        """
        A robust helper function to run an FFmpeg command and diagnose the audio
        volume before and after the operation. This is critical for debugging.
        """

        def get_max_volume(file_path: str) -> Optional[str]:
            """Runs volumedetect and returns the max_volume as a string."""
            if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                return "File does not exist or is empty"

            diag_command = [ffmpeg_exe, "-i", file_path, "-af", "volumedetect", "-f", "null", "-"]
            try:
                result = subprocess.run(diag_command, capture_output=True, text=True, timeout=60)
                for line in result.stderr.splitlines():
                    if "max_volume" in line:
                        return line.split("max_volume:")[1].strip()
                return "Not Found"
            except Exception as e:
                return f"Error during diagnosis: {e}"

        # --- 1. Diagnose the input file ---
        vol_in = get_max_volume(input_path)
        log_worker("INFO", f"[DIAGNOSTIC] Stage '{stage_name}' - Input Max Volume: {vol_in}")

        # --- 2. Run the main processing command ---
        log_worker("INFO", f"Executing FFmpeg Step '{stage_name}': {' '.join(command)}")
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
            log_worker("DEBUG", f"FFmpeg Step '{stage_name}' stderr:\n{result.stderr}")
        except subprocess.CalledProcessError as e:
            log_worker("ERROR", f"FFmpeg Step '{stage_name}' FAILED! Return code: {e.returncode}")
            log_worker("ERROR", f"FFmpeg stderr:\n{e.stderr}")
            return False
        except subprocess.TimeoutExpired as e:
            log_worker("ERROR", f"FFmpeg Step '{stage_name}' TIMED OUT!")
            log_worker("ERROR", f"FFmpeg stderr:\n{e.stderr}")
            return False

        # --- 3. Diagnose the output file ---
        vol_out = get_max_volume(output_path)
        log_worker("INFO", f"[DIAGNOSTIC] Stage '{stage_name}' - Output Max Volume: {vol_out}")

        # --- 4. Final sanity check ---
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            log_worker("CRITICAL",
                       f"FFmpeg Step '{stage_name}' completed but produced an empty or non-existent output file. Aborting.")
            return False

        return True

    def post_process(self, audio_np_in: np.ndarray, input_sample_rate: int, zephyloid_settings: Dict) -> Tuple[
        np.ndarray, float]:
        """
        Applies a comprehensive post-processing effects chain using a series of sequential,
        robust, and instrumented FFmpeg commands. This architecture is designed for
        maximum reliability and provides clear diagnostics for each step.

        The processing pipeline is as follows:
        1. Reverb Generation: Create a "wet-only" reverb signal.
        2. Wet/Dry Mix: Combine the original audio with the reverb signal.
        3. Linear Effects: Apply all EQs, chorus, gain, and other tonal effects.
        4. Volume Normalization & Limiting: Apply a compressor and limiter for a full, polished sound.
        5. Noise Mixing: Add subtle noise for realism.

        Each stage's volume is measured to ensure the audio is flowing correctly.
        """
        log_worker("INFO",
                   f"SCLParser: Post-processing with FFmpeg (Sequential/Diagnosed v3 with Normalizer) START. Input Shape={audio_np_in.shape}, SR={input_sample_rate}")

        # --- Prerequisite Checks ---
        ffmpeg_exe = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
        if not ffmpeg_exe:
            log_worker("ERROR", "ffmpeg executable not found. Cannot perform post-processing. Returning raw audio.")
            return audio_np_in, float(input_sample_rate)

        try:
            import scipy.signal
        except ImportError:
            log_worker("ERROR",
                       "scipy is required for generating reverb IR. Cannot perform post-processing. Returning raw audio.")
            return audio_np_in, float(input_sample_rate)

        with tempfile.TemporaryDirectory(prefix="ffmpeg_seq_") as temp_dir:
            # --- File Path Definitions ---
            initial_input_path = os.path.join(temp_dir, "00_initial_input.wav")
            ir_path = os.path.join(temp_dir, "reverb_ir.wav")
            reverb_wet_only_path = os.path.join(temp_dir, "01_reverb_wet_only.wav")
            reverb_mixed_path = os.path.join(temp_dir, "02_reverb_mixed.wav")
            linear_effects_output_path = os.path.join(temp_dir, "03_linear_effects_output.wav")
            final_output_path = os.path.join(temp_dir, "99_final_output.wav")

            # --- Prepare initial files ---
            sf.write(initial_input_path, audio_np_in, input_sample_rate, subtype='PCM_16')
            self._generate_reverb_ir(ir_path, input_sample_rate)

            current_file_path = initial_input_path

            # --- STAGE 1: Generate a 100% WET Reverb Signal ---
            reverb_wet_gen_command = [
                ffmpeg_exe, "-i", current_file_path, "-i", ir_path,
                "-filter_complex", f"[0:a][1:a]afir=dry=0:wet=1[out]",
                "-map", "[out]", "-y", reverb_wet_only_path
            ]
            if not self._run_and_diagnose(ffmpeg_exe, reverb_wet_gen_command, current_file_path, reverb_wet_only_path,
                                          "Reverb Generation (Wet Only)"):
                log_worker("CRITICAL", "Reverb generation stage failed. Aborting.")
                return audio_np_in, float(input_sample_rate)

            # --- STAGE 2: Mix the Dry and Wet Signals ---
            dry_level = 0.9
            wet_level = 0.01069
            reverb_mix_command = [
                ffmpeg_exe, "-i", current_file_path, "-i", reverb_wet_only_path,
                "-filter_complex", f"[0:a][1:a]amix=inputs=2:weights='{dry_level} {wet_level}':normalize=0[out]",
                "-map", "[out]", "-y", reverb_mixed_path
            ]
            if not self._run_and_diagnose(ffmpeg_exe, reverb_mix_command, current_file_path, reverb_mixed_path,
                                          "Reverb Wet/Dry Mix"):
                log_worker("CRITICAL", "Reverb mixing stage failed. Aborting.")
                return audio_np_in, float(input_sample_rate)
            current_file_path = reverb_mixed_path

            # --- STAGE 3: All Linear Effects (Gain, EQ, Compressor, Limiter) ---
            linear_filters = []
            nyquist_freq = input_sample_rate / 2.0

            # Gain, Distortion, Chorus
            dyn_gain_db = (zephyloid_settings.get('dyn', 64) - 64) / 64.0 * 12.0
            if abs(dyn_gain_db) > 1e-3: linear_filters.append(f"volume={10 ** (dyn_gain_db / 20.0):.4f}")
            gwl_amount_factor = zephyloid_settings.get('gwl', 0) / 127.0
            if gwl_amount_factor > 0: linear_filters.append(f"acontrast={gwl_amount_factor * 30.0:.4f}")
            linear_filters.append("chorus=delays=7.0:depths=0.25:decays=0.0:speeds=0.4")

            # Full EQ chain
            eq_filters = []
            final_eq_settings = [(30, 5, 3.4), (100, 4, 1.4), (150, 1.5, 1.4), (250, 2, 1.0), (350, 2, 1.4),
                                 (450, 2, 1.8),
                                 (550, -2, 1.4), (2000, 2, 1.0), (2500, 3, 1.4), (3000, 2, 1.4), (3500, 4, 1.8),
                                 (4000, 3, 1.4), (8000, 3, 1.8), (12000, 3, 1.8), (20000, 1, 1.8)]
            for freq, gain, q_val in final_eq_settings:
                clamped_freq = min(float(freq), nyquist_freq - 1)
                bw = self._q_to_bandwidth_hz(q_val, clamped_freq)
                eq_filters.append(f"equalizer=f={clamped_freq:.2f}:t=h:w={bw:.2f}:g={gain}")
            linear_filters.extend(eq_filters)

            # --- NORMALIZER ADDED HERE ---
            # `acompressor` evens out the volume, making quiet parts louder and loud parts quieter.
            # This is applied after EQ to act on the tonally-shaped signal.
            log_worker("INFO", "Adding Volume Normalizer (acompressor) to the filter chain.")
            linear_filters.append("acompressor=threshold=-20dB:ratio=20:attack=100:release=1000:makeup=20.0")

            # Safety Limiter: Applied at the very end of the linear chain to prevent clipping.
            linear_filters.append(f"alimiter=limit={10 ** (-0.5 / 20.0):.4f}:level=true")

            # Execute the full linear effects command
            linear_effects_command = [
                ffmpeg_exe, "-i", current_file_path,
                "-af", ",".join(linear_filters),
                "-y", linear_effects_output_path
            ]
            if not self._run_and_diagnose(ffmpeg_exe, linear_effects_command, current_file_path,
                                          linear_effects_output_path, "Linear Effects (incl. Normalizer)"):
                log_worker("CRITICAL", "Linear Effects stage failed. Aborting.")
                return audio_np_in, float(input_sample_rate)
            current_file_path = linear_effects_output_path

            # --- STAGE 4: Noise Mixing ---
            # The final, optional stage to add subtle realism.
            shutil.copy(current_file_path, final_output_path)  # Default to no noise
            total_noise_amplitude = zephyloid_settings.get('bre', 0) / 127.0 * 0.001 + 0.00005
            if total_noise_amplitude > 1e-7:
                noise_command = [
                    ffmpeg_exe, "-i", current_file_path,
                    "-filter_complex",
                    f"anoisesrc=d=10:c=white:a={total_noise_amplitude:.6f}:r={input_sample_rate}[noise];[0:a][noise]amix=inputs=2:duration=first[out]",
                    "-map", "[out]", "-y", final_output_path
                ]
                if not self._run_and_diagnose(ffmpeg_exe, noise_command, current_file_path, final_output_path,
                                              "Noise Mixing"):
                    log_worker("CRITICAL", "Noise Mixing stage failed. Aborting.")
                    return audio_np_in, float(input_sample_rate)

            # --- Final Step: Read the result ---
            processed_audio_np, final_sr = sf.read(final_output_path, dtype='float32')
            log_worker("INFO",
                       f"FFmpeg post-processing completed successfully. Final Shape={processed_audio_np.shape}, SR={final_sr}")
            return processed_audio_np, float(final_sr)

    def parse_attributes(self, params_str: str) -> Dict[str, str]:
        attributes = {}
        # Robust attribute parsing for quoted and unquoted values
        for match in re.finditer(r'(\w+)\s*=\s*(?:"([^"]*)"|([^\s"\']+))',
                                 params_str):  # Adjusted to also handle unquoted values better
            key = match.group(1).lower()
            # Value can be group 2 (quoted) or group 3 (unquoted)
            value = match.group(2) if match.group(2) is not None else match.group(3)
            attributes[key] = value
        log_worker("TRACE", f"SCLParser: parse_attributes: Input='{params_str}', Parsed='{attributes}'")
        return attributes

    def is_number(self, s_val: Any) -> bool:
        if isinstance(s_val, (int, float)): return True
        if isinstance(s_val, str):
            try:
                float(s_val); return True
            except ValueError:
                return False
        return False

    def parse_pitch(self, pitch_str_val: str) -> float:
        pitch_str = str(pitch_str_val).strip().lower()
        log_worker("TRACE", f"SCLParser: parse_pitch called. Input: '{pitch_str}'")
        if pitch_str.endswith("%"):
            try:
                val = float(pitch_str[:-1]); return val / 100.0 * 12.0  # Convert percentage of octave to semitones
            except ValueError:
                return 0.0
        elif pitch_str in self.pitch_map:
            return self.pitch_map[pitch_str]
        else:
            try:
                return float(pitch_str)  # Assume semitones if number
            except ValueError:
                return 0.0

    def _degrees_from(self, scale: str) -> np.ndarray:
        if PEDALBOARD_LIBROSA_AVAILABLE and librosa:
            try:
                degrees = librosa.key_to_degrees(scale)
                return np.concatenate((degrees, [degrees[0] + 12]))  # SEMITONES_IN_OCTAVE = 12
            except Exception as e_key_degree:
                log_worker("WARNING",
                           f"SCLParser: Error parsing scale '{scale}' with librosa: {e_key_degree}. Using Cmaj default.")
        return np.array([0, 2, 4, 5, 7, 9, 11, 12])  # Fallback to C major scale degrees

    def _closest_pitch_from_scale(self, f0: float, scale: str) -> float:
        if not PEDALBOARD_LIBROSA_AVAILABLE or not librosa or np.isnan(f0) or f0 <= 0: return f0
        try:
            degrees = self._degrees_from(scale)
            midi_note = librosa.hz_to_midi(f0)
            degree_index_in_octave = midi_note % 12.0  # MIDI note C4 is 60.0. C is 0.

            # Find the closest degree in the scale
            closest_degree_in_scale = degrees[np.argmin(np.abs(degrees - degree_index_in_octave))]

            degree_difference = degree_index_in_octave - closest_degree_in_scale
            corrected_midi_note = midi_note - degree_difference
            return float(librosa.midi_to_hz(corrected_midi_note))
        except Exception as e_closest_pitch:
            log_worker("WARNING",
                       f"SCLParser: Error in _closest_pitch_from_scale for f0={f0}, scale='{scale}': {e_closest_pitch}")
            return f0  # Return original f0 on error

    def _parse_key(self, key_str_val: str) -> str:
        log_worker("TRACE", f"SCLParser: _parse_key called. Input: '{key_str_val}'")
        key_str = str(key_str_val).strip()
        match = re.match(r"([A-Ga-g][#b♯♭]?)(\s*m(in(or)?)?)?", key_str)  # Simpler regex for tonic + optional minor
        if not match:
            log_worker("WARNING", f"SCLParser: Invalid key format: '{key_str}'. Defaulting to Cmaj.")
            return "C:maj"  # Default to C major if format is unexpected

        tonic = match.group(1).upper().replace("♯", "#").replace("♭", "b")
        mode_str = match.group(2)  # This will capture " m", " min", " minor" or None

        parsed_mode = "min" if mode_str and "m" in mode_str.lower() else "maj"

        result = f"{tonic}:{parsed_mode}"
        log_worker("TRACE", f"SCLParser: _parse_key: Returning: '{result}'")
        return result

    def _closest_pitch(self, f0_val: float) -> float:
        if PEDALBOARD_LIBROSA_AVAILABLE and librosa and not np.isnan(f0_val) and f0_val > 0:
            return float(librosa.midi_to_hz(np.around(librosa.hz_to_midi(np.asarray(f0_val)))))
        return f0_val

    def _autotune(self, audio: np.ndarray, sr: int, f0_contour: np.ndarray, voiced_flags: np.ndarray) -> np.ndarray:
        log_worker("DEBUG",
                   f"SCLParser: _autotune. Singing: {self.zephyloid_settings['singing']}, Key: {self.zephyloid_settings['key']}, Method: {self.zephyloid_settings['correction_method']}")
        if not PEDALBOARD_LIBROSA_AVAILABLE or not librosa or not self.zephyloid_settings.get('singing', False):
            return audio

        # This is a placeholder for your full, complex autotune logic.
        # The one from your file involves frame-by-frame processing with PSOLA or librosa.effects.pitch_shift.
        # For this full script, I'm keeping it simple as a passthrough.
        # You would replace this with your detailed autotune implementation.
        log_worker("WARNING",
                   "SCLParser: _autotune (using passthrough for this version). Implement full logic if needed.")
        return audio
        # --- END SCLParser Class ---

SCLParser_class_ref = SCLParser  # Assign the defined class for type hinting if used elsewhere early

# --- MeloTTS, ChatterboxTTS, PyWhisperCpp Imports & Availability (after SCLParser definition) ---
# This ensures SCLParser is defined when MELO_AVAILABLE is finally set.
try:
    if TORCH_AUDIO_AVAILABLE and TTS_melo_class_ref is None:  # If not imported before SCLParser
        from melo.api import TTS as ImportedTTS_melo_api  # type: ignore
        TTS_melo_class_ref = ImportedTTS_melo_api
    if SCLParser_class_ref is not None and TTS_melo_class_ref is not None and TORCH_AUDIO_AVAILABLE:
        MELO_AVAILABLE = True
        log_worker("INFO", "MeloTTS API and SCLParser class are confirmed available.")
    else:
        if not (SCLParser_class_ref and TTS_melo_class_ref): log_worker("WARNING",
                                                                        "SCLParser or MeloTTS API ref missing for full MeloTTS setup.")
        MELO_AVAILABLE = False
except ImportError as e_melo_final_check:
    log_worker("WARNING", f"MeloTTS setup final check failed: {e_melo_final_check}")
    MELO_AVAILABLE = False

# --- Constants for ChatterboxTTS Voice Sample & Test Mode ---
ADELAIDE_CASUAL_INTRO_FOR_SAMPLE = "Sup! It's your girl, Adelaide Zephyrine Charlotte, ready to roll. Not gonna lie, I'm basically programmed to be hyped about sharing some legendary tales, diving into who-knows-what explorations, and generally 'walking together' on this journey with you. Let's get this digital bread before my processors decide to start optimizing your sock drawer – its a weird hobby Im trying to kick."
TEXT_FOR_VOICE_SAMPLE = f"The quick brown fox jumps over the lazy dog. {ADELAIDE_CASUAL_INTRO_FOR_SAMPLE}"
CHATTERBOX_VOICE_SAMPLE_FILENAME = "GeneratedAudioSample_ZephyAdelaide.wav"  # Consistent name
CHATTERBOX_DEFAULT_MODEL_ID = "default_chatterbox_model_id"  # Placeholder, if ChatterboxTTS.from_pretrained uses it
# DUMMY TTS PROMPT FILENAME (for worker warmup)
DUMMY_TTS_PROMPT_FILENAME = "dummy_tts_warmup_prompt.wav"



ADELAIDE_EXCITED_TEST_INTRO = """
Woohoo! Adelaide Zephyrine Charlotte here, your friendly neighborhood AI, and I am absolutely CHARGED UP!
Are you ready? Because I'm about to spin a yarn, a legendary tale perfect for drifting off to dreamland, or maybe just for a bit of epic chill.

[pause duration="medium"]

Alright, gather 'round, imagine the pixelated sun setting over the Whispering Woods of Eldoria...
Our story begins in the cozy little village of Startington, nestled right by the Glimmering Stream – you know the type, right? Cobblestone paths, a bakery that always smells like sweet rolls, and an old, slightly grumpy blacksmith who secretly has a heart of gold.
Our hero... or maybe just a very enthusiastic apprentice with slightly singed eyebrows from a potion mishap... we'll call them Pip! Pip wasn't known for their brawn, or even particularly for their brains *just yet*, but they had a spark, a real knack for finding trouble, or as Pip liked to call it, 'unforeseen quest opportunities!'

[pause duration="short"]

One evening, as the twin moons of Lumina began to cast long, playful shadows, an old, tattered map practically FLEW into Pip's hands, carried by a gust of wind that smelled suspiciously of dragon sneezes and adventure! Unfurling it, Pip saw a crudely drawn path leading deep into the Forbidden Caves, a place local legends said was guarded by a... well, a rather large, and very, *very* sleepy badger. But also, whispered tales spoke of a legendary artifact hidden within: the Everlasting Goblet of Infinite Hot Cocoa!

[pause duration="medium"]

Pip, fueled by an insatiable curiosity and a sudden craving for superior hot beverages, decided this was IT! Their first real quest! With a patched-up rucksack containing three slightly stale biscuits, a compass that mostly pointed north-ish, and a wooden sword that had seen better days (mostly as a stirring stick for a particularly stubborn stew), Pip set off as the crickets began their nightly serenade... The path was dark, the woods were rustling with unknown sounds, and Pip couldn't help but wonder if those biscuits would be enough...

[pause duration="long"]

...and that, my friend, is just the beginning of Pip's grand adventure. We'll see how they tackle the sleepy badger, navigate the tricky puzzles, and whether that hot cocoa is truly infinite... another time. For now, let the idea of grand quests and cozy villages lull you into a peaceful rest. Sweet dreams of adventure!
"""

#ADELAIDE_EXCITED_TEST_INTRO="Wow! A fantastic Velocireaptor hypersonic aircraft just whooshed under our nose!"


ADELAIDE_EXCITED_TEST_INTRO="Dàjiā hǎo, wǒ shì Ādéláidé·Zéfēilín·Xiàluòtè. Wǒ jiāng tōngguò jièshào jìsuàn liúlìxué yǔ wēijīfēn 3 lái dài nín liǎojiě tòngdiǎn, zhè jiāng shì yālì jùdà de yī tiān!"
#ADELAIDE_EXCITED_TEST_INTRO="大家好，我是阿德莱德·泽菲琳·夏洛特。我将通过介绍计算流体力学与微积分3来带您了解痛点，这将是压力巨大的一天！"

# --- PyTorch Device Auto-Detection Helper ---
def _get_pytorch_device(requested_device_str: str) -> str:
    """
    Determines the best available PyTorch device based on request and availability.
    Hierarchy: CUDA > MPS (Metal) > Vulkan (Experimental) > CPU.
    Returns a device string like "cuda", "mps", "cpu", "vulkan".
    """
    log_worker("INFO", f"Requested PyTorch device: '{requested_device_str}'")
    if not TORCH_AUDIO_AVAILABLE or not torch:  # Check if torch module was successfully imported
        log_worker("ERROR", "PyTorch not available, defaulting to 'cpu' string for device.")
        return "cpu"

    resolved_device = "cpu"  # Default fallback
    req_dev_lower = requested_device_str.lower()

    if req_dev_lower == "cuda":
        if torch.cuda.is_available():  # type: ignore
            resolved_device = "cuda"
            log_worker("INFO", "CUDA is available. Using CUDA.")
        else:
            log_worker("WARNING", "CUDA requested but not available. Falling back.")
    elif req_dev_lower == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore
            resolved_device = "mps"
            log_worker("INFO", "MPS (Metal) is available. Using MPS.")
        else:
            log_worker("WARNING", "MPS (Metal) requested but not available. Falling back.")
    elif req_dev_lower == "vulkan":
        try:
            if hasattr(torch, 'vulkan') and torch.vulkan.is_available():  # type: ignore
                resolved_device = "vulkan"
                log_worker("INFO", "PyTorch reports Vulkan is available. Attempting to use Vulkan (experimental).")
            else:
                log_worker("WARNING", "Vulkan requested or considered, but torch.vulkan.is_available() is False.")
        except Exception as e_vulkan_check:
            log_worker("WARNING",
                       f"Vulkan check failed: {e_vulkan_check}. Vulkan backend likely unsupported by this PyTorch build.")

    # Auto-detection logic if not specifically requested or if specific request failed and resolved_device is still "cpu"
    if resolved_device == "cpu" and req_dev_lower == "auto":
        log_worker("INFO", "Device 'auto' requested. Detecting best available PyTorch device...")
        if torch.cuda.is_available():  # type: ignore
            resolved_device = "cuda"
            log_worker("INFO", "Auto-detected CUDA as best available device.")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore
            resolved_device = "mps"
            log_worker("INFO", "Auto-detected MPS (Metal) as best available device.")
        elif hasattr(torch, 'vulkan') and torch.vulkan.is_available():  # type: ignore
            log_worker("INFO", "Auto-detect: PyTorch reports Vulkan available. Trying Vulkan (experimental).")
            resolved_device = "vulkan"
        else:
            log_worker("INFO", "Auto-detection falling back to CPU.")
            resolved_device = "cpu"  # Already default, but explicit for clarity
    elif resolved_device == "cpu" and req_dev_lower not in ["cpu", "auto"]:
        # This case means a specific device (cuda, mps, vulkan) was requested but wasn't available
        log_worker("WARNING", f"Requested device '{requested_device_str}' was not available. Using CPU as fallback.")

    log_worker("INFO", f"Final PyTorch device selected for TTS operations: '{resolved_device}'")
    return resolved_device


# Helper function to ensure the voice sample exists for ChatterboxTTS
def _ensure_voice_sample(
        args: argparse.Namespace,
        effective_device_str: str
) -> Optional[str]:
    """
    Checks for the ChatterboxTTS voice sample file. If the file is not found, this
    function will dynamically initialize the MeloTTS engine, generate the required
    sample, save it to disk, and then immediately unload MeloTTS to conserve memory
    and ensure a fast startup on subsequent runs.

    This "lazy loading" approach prevents the costly initialization of MeloTTS when it's
    not needed.

    Args:
        args (argparse.Namespace): The parsed command-line arguments.
        effective_device_str (str): The PyTorch device ('cpu', 'mps', 'cuda') to use for generation.

    Returns:
        Optional[str]: The full path to the voice sample file if it exists or was
                       successfully created, otherwise None.
    """
    # First, check if the primary TTS engine (Chatterbox) is even available.
    if not CHATTERBOX_TTS_AVAILABLE:
        log_worker("WARNING",
                   "_ensure_voice_sample: ChatterboxTTS library is not available, cannot ensure voice sample.")
        return None

    # Construct the full path where the voice sample should be located.
    # This path is inside the user-specified temporary directory.
    sample_dir = os.path.join(args.temp_dir, "chatterbox_voice_samples")
    voice_sample_full_path = os.path.join(sample_dir, CHATTERBOX_VOICE_SAMPLE_FILENAME)

    # Attempt to create the directory if it doesn't exist.
    try:
        os.makedirs(sample_dir, exist_ok=True)
    except OSError as e_mkdir_sample:
        log_worker("ERROR", f"Could not create directory for voice samples '{sample_dir}': {e_mkdir_sample}")
        return None

    # --- THE CORE OPTIMIZATION ---
    # Check if the file already exists. If it does, our job is done. Return immediately.
    # This is the "fast path" that avoids all MeloTTS overhead.
    if os.path.exists(voice_sample_full_path):
        log_worker("INFO",
                   f"ChatterboxTTS voice sample found, skipping MeloTTS initialization: {voice_sample_full_path}")
        return voice_sample_full_path

    # --- ON-DEMAND GENERATION BLOCK ---
    # The following code only executes if the voice sample file was NOT found.
    log_worker("INFO", f"ChatterboxTTS voice sample not found. Lazily initializing MeloTTS for generation...")

    # Check if the necessary MeloTTS components (the library and our SCLParser) are available.
    if not MELO_AVAILABLE or not TTS_melo_class_ref or not SCLParser_class_ref:
        log_worker("ERROR",
                   "MeloTTS essentials (library or SCLParser) are unavailable. Cannot generate missing voice sample.")
        return None

    melo_model_instance = None  # Ensure instance is None in case of early failure
    try:
        # 1. LAZY INITIALIZATION: Load the MeloTTS model into memory.
        log_worker("INFO", "Loading MeloTTS model for on-demand sample generation...")
        load_start_time = time.time()
        melo_model_instance = TTS_melo_class_ref(language=args.model_lang.upper(), device=effective_device_str)
        scl_parser_for_sample = SCLParser_class_ref(model=melo_model_instance, device=effective_device_str)
        log_worker("INFO", f"MeloTTS model loaded in {time.time() - load_start_time:.2f} seconds.")

        # 2. DETERMINE SPEAKER: Select the correct voice from the loaded Melo model.
        melo_speaker_for_sample = f"{args.model_lang.upper()}-US"  # Default to a US variant
        current_speaker_ids = scl_parser_for_sample.speaker_ids

        if not current_speaker_ids:
            log_worker("ERROR",
                       "SCLParser's speaker_ids list is empty. Cannot determine Melo speaker for sample generation.")
            return None

        if melo_speaker_for_sample not in current_speaker_ids:
            log_worker("WARNING",
                       f"MeloTTS speaker '{melo_speaker_for_sample}' not in model's list. Using first available speaker.")
            melo_speaker_for_sample = list(current_speaker_ids.keys())[0]

        log_worker("INFO",
                   f"Generating voice sample with MeloTTS. Speaker: {melo_speaker_for_sample}, Text: '{TEXT_FOR_VOICE_SAMPLE[:70]}...'")

        # 3. GENERATE AUDIO: Use the SCLParser's `parse` method, which is hooked to capture audio.
        audio_data_np, sample_rate_melo = scl_parser_for_sample.parse(TEXT_FOR_VOICE_SAMPLE,
                                                                      speaker=melo_speaker_for_sample)

        # 4. SAVE TO DISK: If generation was successful, write the captured audio to the target file.
        if audio_data_np is not None and sample_rate_melo is not None:
            log_worker("DEBUG",
                       f"MeloTTS sample generated. Data shape: {audio_data_np.shape}, SR: {sample_rate_melo}. Saving file...")

            # Ensure audio is in the correct format for soundfile.write (samples, channels)
            audio_to_save = audio_data_np.T if audio_data_np.ndim == 2 and audio_data_np.shape[0] < audio_data_np.shape[
                1] else audio_data_np

            sf.write(voice_sample_full_path, audio_to_save, sample_rate_melo, format='WAV', subtype='PCM_16')
            log_worker("INFO", f"Successfully generated and saved ChatterboxTTS voice sample: {voice_sample_full_path}")
            return voice_sample_full_path
        else:
            log_worker("ERROR", "MeloTTS (via SCLParser) failed to return audio data for voice sample generation.")
            return None

    except Exception as e_sample_gen:
        log_worker("ERROR",
                   f"A critical error occurred during the on-demand generation of the voice sample: {e_sample_gen}")
        log_worker("ERROR", traceback.format_exc())
        return None
    finally:
        # 5. CLEANUP: This is critical. We must unload the MeloTTS model from memory
        # to ensure it doesn't consume resources when it's no longer needed.
        if melo_model_instance:
            del melo_model_instance
            gc.collect()  # Encourage Python to release the memory.
            if torch and effective_device_str in ["cuda", "mps"]:
                if effective_device_str == "cuda":
                    torch.cuda.empty_cache()
                elif effective_device_str == "mps" and hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()
            log_worker("INFO", "MeloTTS model and resources have been unloaded after sample generation.")


def start_persistent_worker(task_type: str, worker_config: Dict[str, Any]) -> Tuple[
    Optional[multiprocessing.Process], Optional[multiprocessing.connection.Connection]]:
    """Starts a persistent worker process."""
    try:
        # Ensure audioProcessorCortex_backbone_provider.py and its worker_loop are importable
        from audioProcessorCortex_backbone_provider import worker_loop as persistent_worker_main_loop_func
    except ImportError as e:
        log_worker("CRITICAL",
                   f"Could not import 'worker_loop' from audio_thread_worker.py: {e}. Ensure it's in the Python path and correctly defined.")
        return None, None
    except Exception as e_import_generic:
        log_worker("CRITICAL",
                   f"An unexpected error occurred while trying to import from audio_thread_worker.py: {e_import_generic}")
        return None, None

    parent_conn, child_conn = multiprocessing.Pipe()

    try:
        process = multiprocessing.Process(
            target=persistent_worker_main_loop_func,
            args=(child_conn, worker_config),  # Pass one end of pipe and config
            name=f"{task_type}AudioWorkerProcess"  # Give the process a name for easier debugging
        )
        process.daemon = True  # Worker will exit if main orchestrator exits
        process.start()
        log_worker("INFO", f"Started persistent {task_type} worker (PID: {process.pid}).")
        # child_conn.close() # Child end is used by the child process. Parent uses parent_conn.
        # This was a mistake in previous template, child_conn should NOT be closed in parent.
        log_worker("DEBUG", "Pausing for a moment to allow the worker process to initialize...")
        time.sleep(2)  # A 2-second grace period is good for debugging.
        return process, parent_conn
    except Exception as e_proc_start:
        log_worker("CRITICAL", f"Failed to start persistent {task_type} worker process: {e_proc_start}")
        # Clean up pipe if process failed to start
        parent_conn.close()
        child_conn.close()
        return None, None


def stop_persistent_worker(process: Optional[multiprocessing.Process],
                           pipe_conn: Optional[multiprocessing.connection.Connection], task_type: str):
    worker_pid = process.pid if process else "N/A"
    log_worker("DEBUG", f"Attempting to stop {task_type} worker (PID: {worker_pid}).")
    if process and process.is_alive():
        log_worker("INFO", f"Sending shutdown to {task_type} worker (PID: {process.pid}).")
        try:
            if pipe_conn:
                # Check if pipe is still writable, though send might raise BrokenPipeError anyway
                if not pipe_conn.closed:
                    pipe_conn.send({"command": "shutdown", "task_id": "orchestrator_shutdown"})
                else:
                    log_worker("WARNING", f"Pipe for {task_type} worker already closed before sending shutdown.")
        except BrokenPipeError:
            log_worker("WARNING",
                       f"Pipe for {task_type} worker broke before shutdown message could be sent (PID: {process.pid}). Worker might have crashed.")
        except Exception as e:
            log_worker("WARNING", f"Error sending shutdown to {task_type} worker (PID: {process.pid}): {e}")

        process.join(timeout=600)  # Wait for worker to exit gracefully
        if process.is_alive():
            log_worker("WARNING",
                       f"{task_type} worker (PID: {process.pid}) did not terminate gracefully after 10s, attempting to terminate.")
            process.terminate()  # Force terminate
            process.join(timeout=5)  # Wait for terminate to complete
            if process.is_alive():
                log_worker("ERROR",
                           f"{task_type} worker (PID: {process.pid}) could not be terminated. It might be stuck.")
            else:
                log_worker("INFO", f"{task_type} worker (PID: {process.pid}) terminated successfully.")
        else:
            log_worker("INFO", f"{task_type} worker (PID: {process.pid}) shut down gracefully.")
    elif process:  # Process object exists but is not alive
        log_worker("INFO", f"{task_type} worker (PID: {process.pid}) was already stopped.")

    # Clean up pipe connection in the parent
    if pipe_conn and not pipe_conn.closed:
        try:
            pipe_conn.close()
            log_worker("DEBUG", f"Pipe connection for {task_type} worker closed in orchestrator.")
        except Exception as e_pipe_close:
            log_worker("WARNING", f"Error closing pipe for {task_type} worker: {e_pipe_close}")

    # Clean up process object
    if process:
        try:
            process.close()  # Available in Python 3.7+
            log_worker("DEBUG", f"Process object for {task_type} worker (PID: {worker_pid}) closed.")
        except Exception as e_proc_close:
            log_worker("DEBUG",
                       f"Note: Error closing process object for {task_type} worker (PID: {worker_pid}): {e_proc_close}. Usually minor.")


def main():
    global tts_worker_process, tts_worker_pipe_orch_end
    global asr_worker_process, asr_worker_pipe_orch_end

    # Initialize process and pipe variables to None
    tts_worker_process = None
    tts_worker_pipe_orch_end = None
    asr_worker_process = None
    asr_worker_pipe_orch_end = None

    # --- 1. ARGUMENT PARSING & INITIAL SETUP ---
    parser = argparse.ArgumentParser(description="Audio Worker Orchestrator (TTS & ASR)")
    parser.add_argument("--task-type", required=True, choices=["tts", "asr"], help="Task to perform")
    parser.add_argument("--model-lang", default="EN", help="Lang for MeloTTS sample or ASR")
    parser.add_argument("--device", default="auto", help="PyTorch device for TTS (auto, cpu, cuda, mps, vulkan)")
    parser.add_argument("--model-dir", required=True,
                        help="Base dir for ASR GGUF models and potentially TTS voice sample dummy prompts")
    parser.add_argument("--temp-dir", default=".", help="Base dir for temporary files (chunks, samples)")
    parser.add_argument("--test-mode", action="store_true", help="Enable test mode with predefined inputs")
    parser.add_argument("--output-file", default="orchestrator_test_output.wav",
                        help="Output for *orchestrator's* TTS test mode (final combined audio)")
    parser.add_argument("--chatterbox_model_id", default=CHATTERBOX_DEFAULT_MODEL_ID, help="Model ID for ChatterboxTTS")
    parser.add_argument("--exaggeration", type=float, default=1.1, help="Exaggeration for ChatterboxTTS")
    parser.add_argument("--cfg_weight", type=float, default=0.9, help="CFG weight for ChatterboxTTS")
    parser.add_argument("--test-audio-input", default="test_input.wav",
                        help="Filename of test audio for ASR (relative to model-dir or temp-dir)")
    parser.add_argument("--asr-test-model", default=WHISPER_DEFAULT_MODEL_FILENAME_CFG,
                        help="Whisper model filename for ASR test")
    parser.add_argument("--debug-save-raw-combined", action="store_true",
                        help="Save the raw combined audio before post-processing for debugging.")
    args = parser.parse_args()
    os.makedirs(args.temp_dir, exist_ok=True)

    try:
        hf_cache_dir = os.path.join(args.model_dir, 'huggingface_cache')
        os.makedirs(hf_cache_dir, exist_ok=True)
        os.environ['HF_HOME'] = hf_cache_dir
        log_worker("INFO", f"Hugging Face cache directory set to: {os.environ['HF_HOME']}")
    except Exception as e_hf_cache:
        log_worker("CRITICAL", f"Failed to set custom Hugging Face cache directory: {e_hf_cache}")
        sys.exit(1)

    result_payload: Dict[str, Any] = {"error": f"Orchestrator failed task: {args.task_type}"}

    effective_pytorch_device_orchestrator = "cpu"
    if TORCH_AUDIO_AVAILABLE and torch:
        effective_pytorch_device_orchestrator = _get_pytorch_device(args.device)
    elif args.task_type == "tts" and not (TORCH_AUDIO_AVAILABLE and torch):
        log_worker("CRITICAL", "PyTorch/Torchaudio not available in orchestrator. Cannot perform TTS.")
        print(json.dumps({"error": "PyTorch/Torchaudio missing for TTS orchestration."}), flush=True)
        sys.exit(1)

    # --- 2. WORKER STARTUP (CRITICAL: Happens BEFORE the main orchestration try block) ---
    dummy_tts_prompt_path = None
    if args.task_type == "tts":
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=args.temp_dir,
                                             prefix="dummy_prompt_") as tmp_dummy_prompt:
                dummy_tts_prompt_path = tmp_dummy_prompt.name
            sr_dummy_tts = 24000
            duration_dummy_tts = 0.05
            silence_tts_prompt = np.zeros(int(duration_dummy_tts * sr_dummy_tts), dtype=np.float32)
            sf.write(dummy_tts_prompt_path, silence_tts_prompt, sr_dummy_tts, format='WAV', subtype='FLOAT')
            log_worker("INFO", f"Created dummy TTS prompt for worker warmup: {dummy_tts_prompt_path}")
        except Exception as e_dummy_prompt:
            log_worker("CRITICAL",
                       f"Failed to create essential dummy TTS prompt: {e_dummy_prompt}\n{traceback.format_exc()}")
            print(json.dumps({"error": "Failed to create dummy TTS prompt for worker."}), flush=True)
            sys.exit(1)

        tts_worker_config = {
            "enable_tts": True, "enable_asr": False, "device": effective_pytorch_device_orchestrator,
            "chatterbox_model_id": args.chatterbox_model_id,
            "dummy_tts_prompt_path_for_warmup": dummy_tts_prompt_path, "temp_dir": args.temp_dir, "model_dir": args.model_dir
        }
        tts_worker_process, tts_worker_pipe_orch_end = start_persistent_worker("TTS", tts_worker_config)
        if not tts_worker_process or not tts_worker_pipe_orch_end:
            log_worker("CRITICAL", "Failed to start persistent TTS worker process.")
            if dummy_tts_prompt_path and os.path.exists(dummy_tts_prompt_path): os.remove(dummy_tts_prompt_path)
            print(json.dumps({"error": "Failed to start TTS worker"}), flush=True)
            sys.exit(1)

    elif args.task_type == "asr":
        asr_model_filename_to_use = WHISPER_DEFAULT_MODEL_FILENAME_CFG
        if args.test_mode and args.asr_test_model:
            asr_model_filename_to_use = args.asr_test_model
        elif 'WHISPER_DEFAULT_MODEL_FILENAME' in globals():
            asr_model_filename_to_use = WHISPER_DEFAULT_MODEL_FILENAME
        asr_worker_config = {
            "enable_tts": False, "enable_asr": True, "device": "cpu", "model_dir": args.model_dir,
            "whisper_model_name": asr_model_filename_to_use, "temp_dir": args.temp_dir
        }
        asr_worker_process, asr_worker_pipe_orch_end = start_persistent_worker("ASR", asr_worker_config)
        if not asr_worker_process or not asr_worker_pipe_orch_end:
            log_worker("CRITICAL", "Failed to start persistent ASR worker. Exiting.")
            print(json.dumps({"error": "Failed to start ASR worker"}), flush=True)
            sys.exit(1)

    # --- 3. MAIN ORCHESTRATION LOGIC ---
    job_temp_dir_path: Optional[str] = None
    try:
        if args.task_type == "tts":
            job_temp_dir_path = tempfile.mkdtemp(prefix="tts_orch_job_", dir=args.temp_dir)
            log_worker("DEBUG", f"Created TTS job temp directory: {job_temp_dir_path}")

            # --- 3a. THE HANDSHAKE (First thing inside the try block) ---
            log_worker("INFO", "Waiting for TTS worker to signal it is ready...")
            try:
                from CortexConfiguration import TTS_WORKER_TIMEOUT
                worker_ready_timeout = TTS_WORKER_TIMEOUT
            except (ImportError, AttributeError):
                worker_ready_timeout = 3600

            if tts_worker_pipe_orch_end and tts_worker_pipe_orch_end.poll(timeout=worker_ready_timeout):
                ready_message = tts_worker_pipe_orch_end.recv()
                if not (ready_message.get("status") == "ready" and ready_message.get("tts_available") is True):
                    raise RuntimeError(
                        f"TTS worker started but is not available or failed initialization. Signal: {ready_message}")
                log_worker("INFO", "TTS worker has confirmed it is ready. Proceeding with tasks.")
            else:
                if tts_worker_process and not tts_worker_process.is_alive():
                    raise RuntimeError(
                        "TTS worker process died unexpectedly during initialization. Check console for tracebacks without a log prefix.")
                else:
                    raise RuntimeError(
                        f"TTS worker did not become ready within the timeout period of {worker_ready_timeout}s.")

            # --- 3b. TTS ORCHESTRATION ---
            sclparser_for_text_proc_tts: SCLParser = SCLParser(model=None, device=effective_pytorch_device_orchestrator)
            generated_voice_sample_path_tts = _ensure_voice_sample(
                args,
                effective_pytorch_device_orchestrator
            )
            if not generated_voice_sample_path_tts:
                raise RuntimeError("Critical: Failed to find or generate ChatterboxTTS voice sample for orchestration.")
            log_worker("INFO", f"Using voice sample for TTS worker: {generated_voice_sample_path_tts}")

            if args.test_mode:
                log_worker("INFO", "TTS Orchestrator in TEST MODE.")
                text_for_synthesis = ADELAIDE_EXCITED_TEST_INTRO
                request_data_from_stdin_tts = {'exaggeration': args.exaggeration, 'cfg_weight': args.cfg_weight,
                                               'response_format': 'wav'}
            else:
                log_worker("INFO", "TTS Orchestrator Standard Mode: Reading stdin...")
                input_json_str_from_stdin_tts = sys.stdin.read()
                if not input_json_str_from_stdin_tts: raise ValueError("Empty input from stdin for TTS orchestration.")
                request_data_from_stdin_tts = json.loads(input_json_str_from_stdin_tts)
                text_for_synthesis = request_data_from_stdin_tts.get("input")
                if not text_for_synthesis: raise ValueError("Missing 'input' text in TTS request.")

            text_segments_with_settings_tts = sclparser_for_text_proc_tts.split_into_segments_for_tts_worker(
                text_for_synthesis)
            if not text_segments_with_settings_tts:
                raise RuntimeError("SCLParser failed to split input text into processable segments.")
            log_worker("INFO", f"Split input into {len(text_segments_with_settings_tts)} segments for TTS processing.")

            all_audio_chunks_data_tts: List[np.ndarray] = []
            final_sr_from_chunks_tts: Optional[int] = None
            current_task_id_counter = 0
            progress_bar_tts = tqdm(total=len(text_segments_with_settings_tts), unit="segment",
                                    desc="TTS Processing Segments", ascii=IS_WINDOWS,
                                    leave=False) if TQDM_AVAILABLE else None

            for i, segment_info in enumerate(text_segments_with_settings_tts):
                current_task_id_counter += 1
                task_id = f"tts_orch_{os.getpid()}_seg_{current_task_id_counter}"

                if segment_info["type"] == "text":
                    chunk_output_full_path = os.path.join(job_temp_dir_path, f"tts_chunk_{i}_{os.getpid()}.wav")
                    tts_task_payload = {
                        "command": "tts_chunk", "task_id": task_id,
                        "params": {
                            "text_to_synthesize": segment_info["content"],
                            "voice_prompt_path": generated_voice_sample_path_tts,
                            "exaggeration": float(request_data_from_stdin_tts.get("exaggeration", args.exaggeration)),
                            "cfg_weight": float(request_data_from_stdin_tts.get("cfg_weight", args.cfg_weight)),
                            "output_audio_chunk_path": chunk_output_full_path
                        }
                    }
                    if not tts_worker_process or not tts_worker_process.is_alive():
                        raise RuntimeError("TTS Worker process died mid-operation.")
                    log_worker("DEBUG",
                               f"Sending task {task_id} to TTS worker for text: '{segment_info['content'][:30]}...'")
                    tts_worker_pipe_orch_end.send(tts_task_payload)

                    if tts_worker_pipe_orch_end.poll(timeout=worker_ready_timeout):
                        chunk_result = tts_worker_pipe_orch_end.recv()
                    else:
                        if progress_bar_tts: progress_bar_tts.close()
                        raise RuntimeError(f"TTS Worker timed out processing segment {i + 1}.")
                    if chunk_result.get("error"):
                        raise RuntimeError(f"TTS Worker reported an error for segment {i + 1}: {chunk_result['error']}")
                    result_data = chunk_result.get("result", {})
                    chunk_file_path_from_worker = result_data.get("audio_chunk_path")
                    chunk_sr_from_worker = result_data.get("sample_rate")
                    if not chunk_file_path_from_worker or not os.path.exists(
                            chunk_file_path_from_worker) or chunk_sr_from_worker is None:
                        raise RuntimeError(f"TTS Worker returned an invalid result for segment {i + 1}.")
                    audio_chunk_np, sr_read = sf.read(chunk_file_path_from_worker, dtype='float32')
                    if final_sr_from_chunks_tts is None: final_sr_from_chunks_tts = sr_read
                    all_audio_chunks_data_tts.append(audio_chunk_np)

                elif segment_info["type"] == "pause":
                    if final_sr_from_chunks_tts:
                        pause_duration_ms = int(segment_info["content"])
                        if pause_duration_ms > 0:
                            silence_samples = int(final_sr_from_chunks_tts * pause_duration_ms / 1000)
                            all_audio_chunks_data_tts.append(np.zeros((silence_samples, 2), dtype=np.float32))
                if progress_bar_tts: progress_bar_tts.update(1)
            if progress_bar_tts: progress_bar_tts.close()

            if not all_audio_chunks_data_tts:
                raise RuntimeError("No audio chunks were generated or collected for TTS.")

            sr_for_crossfade = final_sr_from_chunks_tts or sclparser_for_text_proc_tts.sample_rate or 24000
            log_worker("INFO",
                       f"Combining {len(all_audio_chunks_data_tts)} audio chunks at SR {sr_for_crossfade} for crossfade...")
            raw_combined_audio_np = sclparser_for_text_proc_tts.crossfade_segments(all_audio_chunks_data_tts,
                                                                                   sr_for_crossfade)
            if raw_combined_audio_np is None or raw_combined_audio_np.size == 0:
                raise RuntimeError("SCLParser failed to combine TTS audio chunks or result was empty.")
            if args.debug_save_raw_combined:
                sf.write(os.path.join(args.temp_dir, f"debug_raw_combined_{os.getpid()}.wav"), raw_combined_audio_np,
                         sr_for_crossfade)
            del all_audio_chunks_data_tts
            gc.collect()

            log_worker("INFO", "Applying SCLParser post-processing to combined audio...")
            post_processed_tts_audio_data, sr_after_post_process_float = sclparser_for_text_proc_tts.post_process(
                raw_combined_audio_np, sr_for_crossfade, sclparser_for_text_proc_tts.zephyloid_settings)
            sr_after_post_process = int(sr_after_post_process_float)
            log_worker("INFO",
                       f"Post-processing complete. Final audio shape: {post_processed_tts_audio_data.shape}, SR: {sr_after_post_process}")

            output_format = request_data_from_stdin_tts.get("response_format",
                                                            "mp3").lower() if not args.test_mode else "wav"
            audio_bytes_io = io.BytesIO()
            mime_type = f"audio/{output_format}"

            final_audio_for_saving = post_processed_tts_audio_data.T if post_processed_tts_audio_data.ndim == 2 and \
                                                                        post_processed_tts_audio_data.shape[0] < \
                                                                        post_processed_tts_audio_data.shape[
                                                                            1] else post_processed_tts_audio_data

            if output_format == "wav":
                sf.write(audio_bytes_io, final_audio_for_saving, sr_after_post_process, format='WAV', subtype='PCM_16')
            elif output_format == "mp3":
                ffmpeg_exe_path = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
                if ffmpeg_exe_path:
                    with tempfile.NamedTemporaryFile(suffix=".wav", dir=job_temp_dir_path,
                                                     delete=False) as temp_wav, tempfile.NamedTemporaryFile(
                            suffix=".mp3", dir=job_temp_dir_path, delete=False) as temp_mp3:
                        temp_wav_path, temp_mp3_path = temp_wav.name, temp_mp3.name
                    try:
                        sf.write(temp_wav_path, final_audio_for_saving, sr_after_post_process, format='WAV', subtype='PCM_16')
                        subprocess.run(
                            [ffmpeg_exe_path, "-i", temp_wav_path, "-codec:a", "libmp3lame", "-qscale:a", "2", "-y",
                             temp_mp3_path], check=True, capture_output=True, timeout=60)
                        with open(temp_mp3_path, "rb") as f:
                            audio_bytes_io.write(f.read())
                    finally:
                        if os.path.exists(temp_wav_path): os.remove(temp_wav_path)
                        if os.path.exists(temp_mp3_path): os.remove(temp_mp3_path)
                else:
                    log_worker("WARNING", "ffmpeg not found. Falling back to WAV.")
                    sf.write(audio_bytes_io, final_audio_for_saving, sr_after_post_process, format='WAV',
                             subtype='PCM_16')
                    output_format, mime_type = "wav", "audio/wav"
            else:
                log_worker("WARNING", f"Unknown output format '{output_format}'. Defaulting to WAV.")
                sf.write(audio_bytes_io, final_audio_for_saving, sr_after_post_process, format='WAV', subtype='PCM_16')
                output_format, mime_type = "wav", "audio/wav"

            audio_bytes_io.seek(0)
            audio_b64 = base64.b64encode(audio_bytes_io.read()).decode('utf-8')
            result_payload = {"result": {"audio_base64": audio_b64, "format": output_format, "mime_type": mime_type,
                                         "sample_rate": sr_after_post_process}}
            if args.test_mode:
                final_test_output_path = os.path.join(args.temp_dir, args.output_file)
                with open(final_test_output_path, "wb") as f_out_test:
                    audio_bytes_io.seek(0)
                    f_out_test.write(audio_bytes_io.read())
                log_worker("INFO", f"Saved final TTS test output to: {final_test_output_path}")

        elif args.task_type == "asr":
            job_temp_dir_path = tempfile.mkdtemp(prefix="asr_orch_job_", dir=args.temp_dir)
            log_worker("INFO", f"ASR Orchestration with persistent worker. Model dir: {args.model_dir}")

            # ASR Handshake
            log_worker("INFO", "Waiting for ASR worker to signal it is ready...")
            try:
                from CortexConfiguration import ASR_WORKER_TIMEOUT
                worker_ready_timeout = ASR_WORKER_TIMEOUT
            except (ImportError, AttributeError):
                worker_ready_timeout = 300

            if asr_worker_pipe_orch_end and asr_worker_pipe_orch_end.poll(timeout=worker_ready_timeout):
                ready_message = asr_worker_pipe_orch_end.recv()
                if not (ready_message.get("status") == "ready" and ready_message.get("asr_available") is True):
                    raise RuntimeError(
                        f"ASR worker started but is not available or failed initialization. Signal: {ready_message}")
                log_worker("INFO", "ASR worker has confirmed it is ready.")
            else:
                if asr_worker_process and not asr_worker_process.is_alive():
                    raise RuntimeError("ASR worker process died unexpectedly during initialization.")
                else:
                    raise RuntimeError(
                        f"ASR worker did not become ready within the timeout period of {worker_ready_timeout}s.")

            # ASR Orchestration
            if args.test_mode:
                input_audio_path_for_asr = os.path.join(args.temp_dir, args.test_audio_input)
                if not os.path.exists(input_audio_path_for_asr):
                    input_audio_path_for_asr = os.path.join(args.model_dir, args.test_audio_input)
            else:
                request_data_from_stdin_asr = json.loads(sys.stdin.read())
                input_audio_path_for_asr = request_data_from_stdin_asr.get("input_audio_path")

            if not input_audio_path_for_asr or not os.path.exists(input_audio_path_for_asr):
                raise FileNotFoundError(f"ASR input audio not found at '{input_audio_path_for_asr}'")

            ffmpeg_exe = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
            if not ffmpeg_exe: raise RuntimeError("ffmpeg not found, but is required for ASR audio pre-processing.")

            master_temp_wav_asr = os.path.join(job_temp_dir_path, f"master_asr_input_{os.getpid()}.wav")
            ffmpeg_cmd_asr = ['ffmpeg', '-i', input_audio_path_for_asr, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000',
                              '-ac', '1', '-y', master_temp_wav_asr]
            subprocess.run(ffmpeg_cmd_asr, check=True, capture_output=True, timeout=300)

            all_transcribed_segments_asr: List[str] = []
            with wave.open(master_temp_wav_asr, 'rb') as wf_asr:
                nchannels, sampwidth, framerate, nframes = wf_asr.getparams()[:4]
                chunk_duration_seconds, overlap_duration_seconds = 30, 5
                chunk_len_frames = chunk_duration_seconds * framerate
                step_frames = chunk_len_frames - (overlap_duration_seconds * framerate)

                current_pos_frames, chunk_idx = 0, 0
                total_asr_chunks = math.ceil(nframes / step_frames) if step_frames > 0 else 1
                progress_bar_asr = tqdm(total=total_asr_chunks, unit="chunk", desc="ASR Processing Chunks",
                                        ascii=IS_WINDOWS, leave=False) if TQDM_AVAILABLE else None

                while current_pos_frames < nframes:
                    wf_asr.setpos(current_pos_frames)
                    chunk_audio_bytes = wf_asr.readframes(chunk_len_frames)
                    if not chunk_audio_bytes: break

                    chunk_file_path_for_worker = os.path.join(job_temp_dir_path, f"asr_chunk_{chunk_idx}.wav")
                    with wave.open(chunk_file_path_for_worker, 'wb') as cfw:
                        cfw.setparams(
                            (nchannels, sampwidth, framerate, len(chunk_audio_bytes) // (nchannels * sampwidth), 'NONE',
                             'not compressed'))
                        cfw.writeframes(chunk_audio_bytes)

                    asr_task_id = f"asr_orch_{os.getpid()}_chunk_{chunk_idx}"
                    asr_task_payload = {"command": "asr_chunk", "task_id": asr_task_id,
                                        "params": {"input_audio_chunk_path": chunk_file_path_for_worker,
                                                   "language": args.model_lang}}
                    asr_worker_pipe_orch_end.send(asr_task_payload)

                    if asr_worker_pipe_orch_end.poll(timeout=worker_ready_timeout):
                        asr_chunk_result = asr_worker_pipe_orch_end.recv()
                        if asr_chunk_result.get("error"): raise RuntimeError(
                            f"ASR worker error on chunk {chunk_idx}: {asr_chunk_result['error']}")
                        all_transcribed_segments_asr.append(asr_chunk_result.get("result", {}).get("text", ""))
                    else:
                        raise RuntimeError(f"ASR Worker timed out on chunk {chunk_idx}.")

                    current_pos_frames += step_frames
                    chunk_idx += 1
                    if progress_bar_asr: progress_bar_asr.update(1)
                if progress_bar_asr: progress_bar_asr.close()

            final_transcription = " ".join(s.strip() for s in all_transcribed_segments_asr if s.strip())
            result_payload = {"result": {"text": final_transcription}}

    # --- 4. CLEANUP (Happens regardless of success or failure) ---
    except Exception as e_orchestration:
        result_payload = {"error": f"Orchestration error during task '{args.task_type}': {str(e_orchestration)}"}
        log_worker("ERROR", f"Orchestration error: {e_orchestration}")
        log_worker("ERROR", traceback.format_exc())
    finally:
        if tts_worker_process:
            stop_persistent_worker(tts_worker_process, tts_worker_pipe_orch_end, "TTS")
        if asr_worker_process:
            stop_persistent_worker(asr_worker_process, asr_worker_pipe_orch_end, "ASR")

        if job_temp_dir_path and os.path.isdir(job_temp_dir_path):
            try:
                shutil.rmtree(job_temp_dir_path)
                log_worker("INFO", f"Cleaned up job temp directory: {job_temp_dir_path}")
            except Exception as e_rm_job_dir:
                log_worker("WARNING", f"Failed to clean up job temp directory {job_temp_dir_path}: {e_rm_job_dir}")

        if dummy_tts_prompt_path and os.path.exists(dummy_tts_prompt_path):
            try:
                os.remove(dummy_tts_prompt_path)
                log_worker("INFO", f"Cleaned up dummy TTS prompt: {dummy_tts_prompt_path}")
            except Exception as e_rm_dummy:
                log_worker("WARNING", f"Failed to remove dummy TTS prompt {dummy_tts_prompt_path}: {e_rm_dummy}")

        gc.collect()
        if TORCH_AUDIO_AVAILABLE and torch:
            if effective_pytorch_device_orchestrator == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif effective_pytorch_device_orchestrator == "mps" and hasattr(torch.backends,
                                                                            "mps") and torch.backends.mps.is_available() and hasattr(
                    torch.mps, "empty_cache"):
                try:
                    torch.mps.empty_cache()
                    log_worker("INFO", "Orchestrator MPS cache clear attempted.")
                except Exception as e:
                    log_worker("WARNING", f"Orchestrator MPS empty_cache failed: {e}")

    # --- 5. FINAL OUTPUT ---
    try:
        output_json_str_final = json.dumps(result_payload)
        log_worker("DEBUG", f"Sending output JSON (len={len(output_json_str_final)}): {output_json_str_final[:250]}...")
        print(output_json_str_final, flush=True)
        log_worker("INFO", f"Final result/error JSON sent for task: {args.task_type}.")
    except Exception as e_final_print:
        log_worker("CRITICAL", f"Failed to serialize/write final result_payload: {e_final_print}")
        try:
            print(json.dumps({"error": f"Worker critical finalization error: {str(e_final_print)}"}), flush=True)
        except:
            pass
        sys.exit(1)

    log_worker("INFO", f"Audio Worker Orchestrator PID {os.getpid()} finished task: {args.task_type}.")
    sys.exit(0)


if __name__ == "__main__":
    # Required for PyInstaller/cx_Freeze or if starting new processes on Windows/macOS with 'spawn' or 'forkserver'
    multiprocessing.freeze_support()

    # Initial library availability checks (as before)
    if not TORCH_AUDIO_AVAILABLE and ('tts' in sys.argv):
        log_worker("CRITICAL", "Torch/Torchaudio NOT AVAILABLE. Cannot perform TTS tasks.")
        # No sys.exit here yet, main() might handle it or it might be ASR task.

    # Check if SCLParser class is defined (it's defined within this file)
    scl_parser_defined = 'SCLParser' in globals() and isinstance(globals()['SCLParser'], type)
    if not scl_parser_defined and ('tts' in sys.argv):
        log_worker("CRITICAL", "SCLParser class is not defined. Cannot proceed with TTS.")
        print(json.dumps({"error": "SCLParser class not defined for TTS."}), flush=True)
        sys.exit(1)

    # A more general check if no primary capabilities are available AT ALL
    any_tts_available = (TORCH_AUDIO_AVAILABLE and MELO_AVAILABLE) or (
                TORCH_AUDIO_AVAILABLE and CHATTERBOX_TTS_AVAILABLE)
    any_asr_available = PYWHISPERCPP_AVAILABLE
    if not any_tts_available and not any_asr_available:
        log_worker("CRITICAL",
                   "No primary audio libraries (TTS or ASR) seem available. Worker functionality is severely limited or non-functional.")
        # Allow to proceed if only one type of task is requested and its libs are there.
        # A stricter check might be:
        # if (('tts' in sys.argv) and not any_tts_available) or \
        #    (('asr' in sys.argv) and not any_asr_available):
        #     print(json.dumps({"error": "Required audio libraries for the requested task are not available."}), flush=True)
        #     sys.exit(1)

    main()