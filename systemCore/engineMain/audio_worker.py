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
from config import *
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
try:
    if TORCH_AUDIO_AVAILABLE:
        from melo.api import TTS as ImportedTTS_melo_api
        TTS_melo_class_ref = ImportedTTS_melo_api
        log_worker("INFO", "MeloTTS API class imported (for sample gen). SCLParser check will follow its definition.")
    else:
        # This path won't execute if TORCH_AUDIO_AVAILABLE is False, but as a guard:
        log_worker("WARNING", "Torch/Torchaudio not available, so MeloTTS cannot be loaded for sample generation.")
        MELO_AVAILABLE = False # Explicitly
except ImportError as e_melo_imp:
    log_worker("WARNING", f"MeloTTS API import failed: {e_melo_imp}. Melo sample gen will fail.")
    MELO_AVAILABLE = False

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
    from config import WHISPER_MODEL_DIR, WHISPER_DEFAULT_MODEL_FILENAME, WHISPER_DEFAULT_LANGUAGE
    WHISPER_MODEL_DIR_CFG = WHISPER_MODEL_DIR
    WHISPER_DEFAULT_MODEL_FILENAME_CFG = WHISPER_DEFAULT_MODEL_FILENAME
    WHISPER_DEFAULT_LANGUAGE_CFG = WHISPER_DEFAULT_LANGUAGE
    log_worker("INFO", "Successfully imported ASR defaults from config.py.")
except ImportError:
    log_worker("WARNING", "Could not import ASR defaults from config.py. Using internal fallbacks for ASR config.")

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
                   f"SCLParser: speak_with_settings (for sample gen): Text='{text[:30]}...', Speaker='{speaker}', Rate={self.voice_settings.get('rate', 1.0)}")
        try:
            speaker_id_to_use = self.speaker_ids.get(speaker)
            if speaker_id_to_use is None:
                log_worker("WARNING",
                           f"SCLParser: Speaker '{speaker}' not found in Melo model. Using first available for sample.")
                speaker_id_to_use = list(self.speaker_ids.values())[0] if self.speaker_ids else 0

            # MeloTTS's tts_to_file is called, our _new_sf_write hook should capture the data
            # into self.captured_audio_data and self.captured_audio_samplerate
            self.model.tts_to_file(text, speaker_id_to_use, "dummy_internal_scl_output.wav",
                                   speed=self.voice_settings.get('rate', 1.0))

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
                        if current_audio_segment.shape[0] < current_audio_segment.shape[1] and \
                                current_audio_segment.shape[0] <= 2:  # (ch, samps)
                            audio_mono_for_pitch = librosa.to_mono(current_audio_segment)
                        elif current_audio_segment.shape[1] < current_audio_segment.shape[0] and \
                                current_audio_segment.shape[1] <= 2:  # (samps, ch)
                            audio_mono_for_pitch = librosa.to_mono(current_audio_segment.T)
                        else:  # Fallback if shape is ambiguous
                            audio_mono_for_pitch = current_audio_segment.mean(axis=0) if current_audio_segment.shape[
                                                                                             0] < \
                                                                                         current_audio_segment.shape[
                                                                                             1] else current_audio_segment.mean(
                                axis=1)

                    pitched_audio_mono = librosa.effects.pitch_shift(audio_mono_for_pitch, sr=current_sample_rate,
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

    def post_process(self, audio_np_in: np.ndarray, input_sample_rate: int, zephyloid_settings: Dict) -> np.ndarray:
        log_worker("DEBUG",
                   f"SCLParser: post_process. Input shape {audio_np_in.shape}, SR: {input_sample_rate}, Zephyloid: {zephyloid_settings}")
        if not PEDALBOARD_LIBROSA_AVAILABLE or not TORCH_AUDIO_AVAILABLE or not torch or not F_torchaudio:
            log_worker("WARNING",
                       "Missing effects libraries (Pedalboard/Librosa or Torch/Torchaudio), skipping SCLParser post-processing.")
            return audio_np_in

        processed_audio_np = audio_np_in.astype(np.float32)

        # Ensure input is (channels, samples) for effects
        if processed_audio_np.ndim == 1:
            log_worker("TRACE", "SCLParser: post_process promoting mono to stereo (2, samples) for effects.")
            processed_audio_np = np.stack([processed_audio_np, processed_audio_np], axis=0)
        elif processed_audio_np.ndim == 2:
            if processed_audio_np.shape[1] == 2 and processed_audio_np.shape[0] > 2:  # (samples, 2)
                log_worker("TRACE", "SCLParser: post_process transposing (samples, 2) to (2, samples) for effects.")
                processed_audio_np = processed_audio_np.T
            elif processed_audio_np.shape[0] == 1 and processed_audio_np.shape[1] > 2:  # (1, samples) mono
                log_worker("TRACE",
                           "SCLParser: post_process repeating mono channel to stereo (2, samples) for effects.")
                processed_audio_np = np.repeat(processed_audio_np, 2, axis=0)

        if processed_audio_np.ndim != 2 or processed_audio_np.shape[0] != 2:
            log_worker("ERROR",
                       f"SCLParser Post-process: Expected 2 channels (2, samples), got shape {processed_audio_np.shape}. Skipping effects chain.")
            return audio_np_in  # Return original if shape is still wrong for subsequent effects

        current_sr_for_effects = float(input_sample_rate)
        # Convert to tensor early for DYN, BRE, and potentially noise addition later
        audio_tensor_intermediate = torch.from_numpy(processed_audio_np.copy()).to(self.device)

        # DYN gain
        dyn_gain_db_val = (zephyloid_settings.get('dyn', 64) - 64) / 64.0 * 12.0
        audio_tensor_intermediate = F_torchaudio.gain(audio_tensor_intermediate, dyn_gain_db_val)  # type: ignore
        log_worker("TRACE", f"SCLParser Post-process: Applied DYN gain: {dyn_gain_db_val:.2f}dB")

        # BRE noise
        bre_amount = zephyloid_settings.get('bre', 0) / 127.0 * 0.001
        if bre_amount > 0 and TORCH_AUDIO_AVAILABLE and torch:
            noise_bre_tensor = torch.randn_like(audio_tensor_intermediate) * bre_amount
            audio_tensor_intermediate = audio_tensor_intermediate + noise_bre_tensor
            log_worker("TRACE", f"SCLParser Post-process: Applied BRE noise. Amount factor: {bre_amount:.5f}")

        audio_for_pedalboard_np = audio_tensor_intermediate.cpu().numpy()  # (channels, samples) for pedalboard

        # Resample for Pedalboard if necessary
        if current_sr_for_effects != 44100.0 and PEDALBOARD_LIBROSA_AVAILABLE and librosa:
            log_worker("TRACE",
                       f"SCLParser Post-process: Resampling for Pedalboard from {current_sr_for_effects}Hz to 44100Hz.")
            resampled_channels_list = []
            for ch_idx_pb in range(audio_for_pedalboard_np.shape[0]):  # Iterate over channels
                resampled_channels_list.append(
                    librosa.resample(y=audio_for_pedalboard_np[ch_idx_pb], orig_sr=current_sr_for_effects,
                                     target_sr=44100.0, res_type='kaiser_fast'))
            audio_for_pedalboard_np = np.stack(resampled_channels_list)  # Stack back into (channels, samples)
            current_sr_for_effects = 44100.0  # Update sample rate after resampling
            log_worker("TRACE",
                       f"SCLParser Post-process: Resampled. New shape for Pedalboard: {audio_for_pedalboard_np.shape}")

        board_effects_list = [
            Reverb(room_size=1.0, damping=0.2, wet_level=0.5069, dry_level=0.4),
            Chorus(rate_hz=0.4, depth=0.25, centre_delay_ms=7.0, feedback=0.0, mix=0.02),
            Delay(delay_seconds=0.5, feedback=0.01, mix=0.0002),
        ]
        gwl_amount_factor = zephyloid_settings.get('gwl', 0) / 127.0
        if gwl_amount_factor > 0:
            drive_db_gwl = gwl_amount_factor * 20.0
            board_effects_list.insert(0, Distortion(drive_db=drive_db_gwl))
            log_worker("TRACE",
                       f"SCLParser Post-process: Added GWL (Distortion) to Pedalboard. Drive: {drive_db_gwl:.2f}dB")

        # This is where the original pedalboard processing happened.
        # We will process with pedalboard, then add noise, then convert to tensor for EQ.

        audio_after_pedalboard_effects_np = audio_for_pedalboard_np  # Initialize with pre-pedalboard audio

        if PEDALBOARD_LIBROSA_AVAILABLE:
            try:
                board = Pedalboard(board_effects_list)
                log_worker("TRACE", "SCLParser Post-process: Applying Pedalboard effects...")
                audio_after_pedalboard_effects_np = board(audio_for_pedalboard_np, sample_rate=current_sr_for_effects)
                log_worker("TRACE",
                           f"SCLParser Post-process: Pedalboard effects applied. Shape: {audio_after_pedalboard_effects_np.shape}")
            except Exception as e_pb_apply:
                log_worker("ERROR", f"SCLParser Post-process: Error applying Pedalboard effects: {e_pb_apply}")
                # audio_after_pedalboard_effects_np remains the audio *before* pedalboard attempt in case of error

        # --- ADD BROWNIAN-LIKE NOISE ---
        # This is added *after* the main pedalboard effects (reverb, chorus, delay, distortion)
        # but *before* the subsequent EQ chain.
        if TORCH_AUDIO_AVAILABLE and torch and F_torchaudio:
            log_worker("TRACE", "SCLParser Post-process: Attempting to add Brownian-like noise.")
            # Convert numpy array (output from pedalboard or pre-pedalboard if PB failed/skipped) to tensor
            audio_tensor_for_noise_addition = torch.from_numpy(audio_after_pedalboard_effects_np.copy()).to(self.device)

            # 1. Generate white noise
            white_noise_tensor = torch.randn_like(audio_tensor_for_noise_addition)

            # 2. Filter white noise to make it "brownish" using a strong low-pass filter.
            # A low cutoff frequency (e.g., 20-60 Hz) will give it a rumbly, low-frequency character.
            brownian_cutoff_freq = 40.0  # Hz; adjust as needed for desired "color"
            brownish_noise_tensor = F_torchaudio.lowpass_biquad(
                white_noise_tensor,
                sample_rate=current_sr_for_effects,  # Ensure this is the correct SR at this point
                cutoff_freq=brownian_cutoff_freq
            )

            # 3. Scale the noise. "0.1 on the wet" is interpreted as a gain factor for the noise.
            # torch.randn_like produces noise with std=1. A gain of 0.1 means noise std_dev will be 0.1.
            # This might be quite audible. For subtle realism, much lower values (0.001-0.01) are common.
            # We'll use 0.1 as requested.
            brownian_noise_gain = 0.1
            scaled_brownish_noise_tensor = brownish_noise_tensor * brownian_noise_gain

            # 4. Add the scaled noise to the audio
            audio_tensor_with_noise = audio_tensor_for_noise_addition + scaled_brownish_noise_tensor
            log_worker("TRACE",
                       f"SCLParser Post-process: Brownian-like noise added. Gain: {brownian_noise_gain}, Cutoff: {brownian_cutoff_freq}Hz.")

            # This tensor now goes to the EQ stage
            audio_tensor_for_eq = audio_tensor_with_noise
        else:
            log_worker("WARNING",
                       "SCLParser Post-process: Torch/Torchaudio not available for Brownian noise generation, skipping noise addition.")
            # If noise couldn't be added, proceed with the audio as it was after pedalboard
            audio_tensor_for_eq = torch.from_numpy(audio_after_pedalboard_effects_np.copy()).to(self.device)
        # --- END BROWNIAN-LIKE NOISE ---

        current_sr_float_for_eq = float(current_sr_for_effects)  # Ensure it's float for F_torchaudio

        # EQ (BRI, CLE, OPE, GEN, XSY, Standard final)
        bri_gain_val = (zephyloid_settings.get('bri', 64) - 64) / 64.0 * 6.0
        audio_tensor_for_eq = F_torchaudio.equalizer_biquad(audio_tensor_for_eq, current_sr_float_for_eq,
                                                            center_freq=8000.0, gain=bri_gain_val, Q=1.0)
        log_worker("TRACE", f"SCLParser Post-process: Applied BRI EQ. Gain: {bri_gain_val:.2f}dB at 8kHz")

        cle_gain_val = (zephyloid_settings.get('cle', 64) - 64) / 64.0 * 4.0
        audio_tensor_for_eq = F_torchaudio.equalizer_biquad(audio_tensor_for_eq, current_sr_float_for_eq,
                                                            center_freq=4000.0, gain=cle_gain_val, Q=1.2)
        log_worker("TRACE", f"SCLParser Post-process: Applied CLE EQ. Gain: {cle_gain_val:.2f}dB at 4kHz")

        ope_shift_val = (zephyloid_settings.get('ope', 64) - 64) / 64.0
        ope_center_freq = 1000.0 + ope_shift_val * 500.0;
        ope_gain = abs(ope_shift_val) * 3.0
        if ope_shift_val != 0:
            audio_tensor_for_eq = F_torchaudio.equalizer_biquad(audio_tensor_for_eq, current_sr_float_for_eq,
                                                                center_freq=ope_center_freq, gain=ope_gain, Q=1.5)
            log_worker("TRACE",
                       f"SCLParser Post-process: Applied OPE EQ. Freq: {ope_center_freq:.0f}Hz, Gain: {ope_gain:.2f}dB")

        gen_formant_shift_val = (zephyloid_settings.get('gen', 64) - 64) / 64.0 * 3.0
        gen_center_freq = 1500.0 + (zephyloid_settings.get('gen', 64) - 64) / 64.0 * 200.0
        if gen_formant_shift_val != 0:
            audio_tensor_for_eq = F_torchaudio.equalizer_biquad(audio_tensor_for_eq, current_sr_float_for_eq,
                                                                center_freq=gen_center_freq,
                                                                gain=abs(gen_formant_shift_val), Q=1.2)
            log_worker("TRACE",
                       f"SCLParser Post-process: Applied GEN (Formant) EQ. Freq: {gen_center_freq:.0f}Hz, Gain: {abs(gen_formant_shift_val):.2f}dB")

        xsy_voicebanks_val = zephyloid_settings.get('xsy_voicebanks')
        xsy_blend_val = zephyloid_settings.get('xsy', 0) / 127.0
        if isinstance(xsy_voicebanks_val, list) and len(xsy_voicebanks_val) == 2 and 0 < xsy_blend_val < 1:
            vb1_name, vb2_name = xsy_voicebanks_val
            profile1 = self.xsy_profiles.get(vb1_name);
            profile2 = self.xsy_profiles.get(vb2_name)
            if profile1 and profile2:
                audio_vb1 = audio_tensor_for_eq.clone()
                for freq, gain, q_val in profile1["eq_curve"]: audio_vb1 = F_torchaudio.equalizer_biquad(audio_vb1,
                                                                                                         current_sr_float_for_eq,
                                                                                                         freq, gain,
                                                                                                         q_val)
                audio_vb2 = audio_tensor_for_eq.clone()
                for freq, gain, q_val in profile2["eq_curve"]: audio_vb2 = F_torchaudio.equalizer_biquad(audio_vb2,
                                                                                                         current_sr_float_for_eq,
                                                                                                         freq, gain,
                                                                                                         q_val)
                audio_tensor_for_eq = (1 - xsy_blend_val) * audio_vb1 + xsy_blend_val * audio_vb2
                log_worker("TRACE", f"SCLParser Post-process: Applied XSY EQ. Blend: {xsy_blend_val:.2f}")

        final_eq_settings = [(30, 5, 3.4), (100, 4, 1.4), (150, 1.5, 1.4), (250, 2, 1.0), (350, 2, 1.4), (450, 2, 1.8),
                             (550, -2, 1.4), (2000, 2, 1.0), (2500, 3, 1.4), (3000, 2, 1.4), (3500, 4, 1.8),
                             (4000, 3, 1.4), (8000, 3, 1.8), (12000, 3, 1.8), (20000, 1, 1.8)]
        for freq, gain, q_val in final_eq_settings:
            audio_tensor_for_eq = F_torchaudio.equalizer_biquad(audio_tensor_for_eq, current_sr_float_for_eq, freq,
                                                                gain, q_val)
        log_worker("TRACE", "SCLParser Post-process: Standard final EQ applied.")

        if TORCH_AUDIO_AVAILABLE and torch:
            noise_final_tensor = torch.randn_like(audio_tensor_for_eq) * 0.0002
            audio_tensor_for_eq = audio_tensor_for_eq + noise_final_tensor
            mod_freq, mod_depth = 1.0, 0.03
            t_axis = torch.arange(audio_tensor_for_eq.shape[1],
                                  device=audio_tensor_for_eq.device) / current_sr_float_for_eq
            modulation_tensor = (1 + mod_depth * torch.sin(2 * torch.pi * mod_freq * t_axis)).float()  # type: ignore
            audio_tensor_for_eq = audio_tensor_for_eq * modulation_tensor.unsqueeze(0)
            log_worker("TRACE", "SCLParser Post-process: Final noise and amp modulation applied.")

        final_audio_numpy_for_limiter = audio_tensor_for_eq.cpu().numpy()
        if PEDALBOARD_LIBROSA_AVAILABLE:
            try:
                final_limiter_board = Pedalboard([Limiter(threshold_db=-1.0, release_ms=50.0)])
                final_audio_numpy_for_limiter = final_limiter_board(final_audio_numpy_for_limiter,
                                                                    sample_rate=current_sr_float_for_eq)
                log_worker("DEBUG", "SCLParser Post-process: Final Limiter applied.")
            except Exception as e_final_limiter_apply:
                log_worker("ERROR", f"SCLParser Post-process: Error applying final Limiter: {e_final_limiter_apply}")

        log_worker("DEBUG",
                   f"SCLParser: post_process: Final audio shape before returning: {final_audio_numpy_for_limiter.shape}")
        return final_audio_numpy_for_limiter

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
        melo_model_instance: Optional[Any],  # Type should be melo.api.TTS if fully typed
        scl_parser_instance: Optional[SCLParser],  # Using SCLParser class defined in this file
        effective_device_str: str  # The device string determined by _get_pytorch_device
) -> Optional[str]:
    """
    Checks for Chatterbox voice sample, generates it with MeloTTS (via SCLParser) if missing.
    Returns the path to the sample file, or None if generation fails or prerequisites missing.
    """
    # This function is called only if args.task_type == "tts"
    if not CHATTERBOX_TTS_AVAILABLE:  # From global flag set after ChatterboxTTS import attempt
        log_worker("WARNING", "_ensure_voice_sample: ChatterboxTTS library not available, cannot ensure voice sample.")
        return None

    sample_dir = os.path.join(args.temp_dir, "chatterbox_voice_samples")  # Store samples in a sub-directory
    try:
        os.makedirs(sample_dir, exist_ok=True)
    except OSError as e_mkdir_sample:
        log_worker("ERROR", f"Could not create directory for voice samples '{sample_dir}': {e_mkdir_sample}")
        return None

    voice_sample_full_path = os.path.join(sample_dir, CHATTERBOX_VOICE_SAMPLE_FILENAME)

    if os.path.exists(voice_sample_full_path):
        log_worker("INFO", f"ChatterboxTTS voice sample found: {voice_sample_full_path}")
        return voice_sample_full_path

    log_worker("INFO",
               f"ChatterboxTTS voice sample not found at '{voice_sample_full_path}'. Attempting generation with MeloTTS...")

    if not MELO_AVAILABLE or not melo_model_instance or not scl_parser_instance:
        log_worker("ERROR",
                   "MeloTTS essentials (Melo model or SCLParser instance) unavailable for voice sample generation. Cannot proceed if sample is missing.")
        return None  # Cannot generate sample if MeloTTS or its parser isn't ready

    try:
        # Determine speaker for MeloTTS sample generation
        melo_speaker_for_sample = f"{args.model_lang.upper()}-US"  # Default to US variant
        if hasattr(scl_parser_instance, 'speaker_ids') and isinstance(scl_parser_instance.speaker_ids, dict):
            if melo_speaker_for_sample not in scl_parser_instance.speaker_ids:
                log_worker("WARNING",
                           f"Melo speaker '{melo_speaker_for_sample}' not in SCLParser's list for sample. Using first available.")
                if scl_parser_instance.speaker_ids:
                    melo_speaker_for_sample = list(scl_parser_instance.speaker_ids.keys())[0]
                else:  # Should not happen if Melo model loaded correctly
                    log_worker("ERROR", "SCLParser has no Melo speakers loaded! Cannot generate sample.")
                    return None
        else:
            log_worker("ERROR", "SCLParser instance missing speaker_ids. Cannot determine Melo speaker for sample.")
            return None

        log_worker("INFO",
                   f"Generating voice sample with MeloTTS. Speaker: {melo_speaker_for_sample}, Text: '{TEXT_FOR_VOICE_SAMPLE[:70]}...'")

        # SCLParser's parse method (when used for sample generation) should use its Melo model
        # and its sf.write hook to capture the audio.
        audio_data_np, sample_rate_melo = scl_parser_instance.parse(TEXT_FOR_VOICE_SAMPLE,
                                                                    speaker=melo_speaker_for_sample)

        if audio_data_np is not None and sample_rate_melo is not None:
            log_worker("DEBUG",
                       f"MeloTTS sample generated by SCLParser. Data shape: {audio_data_np.shape}, SR: {sample_rate_melo}. Saving...")

            # soundfile.write expects (samples, channels) or (samples,) for mono
            audio_to_save_for_sf = audio_data_np
            if audio_data_np.ndim == 2:
                # If SCLParser's parse returns (channels, samples) from PyTorch/Melo internal format
                if audio_data_np.shape[0] < audio_data_np.shape[1] and audio_data_np.shape[0] <= 2:
                    audio_to_save_for_sf = audio_data_np.T  # Transpose to (samples, channels)
                elif audio_data_np.shape[1] > 2 and audio_data_np.shape[
                    0] == 2:  # Already (samples, 2) or similar if SCLParser adjusted it
                    pass  # Assume (samples, channels)
                elif audio_data_np.shape[0] == 1 and audio_data_np.shape[1] > 2:  # (1, samples) mono from torch
                    audio_to_save_for_sf = audio_data_np.squeeze()  # (samples,)

            sf.write(voice_sample_full_path, audio_to_save_for_sf, sample_rate_melo, format='WAV', subtype='PCM_16')
            log_worker("INFO", f"Successfully generated and saved ChatterboxTTS voice sample: {voice_sample_full_path}")
            return voice_sample_full_path
        else:
            log_worker("ERROR", "MeloTTS (via SCLParser) failed to return audio data for voice sample.")
            return None
    except Exception as e_sample_gen_final:
        log_worker("ERROR", f"Failed to generate/save voice sample with MeloTTS: {e_sample_gen_final}")
        log_worker("ERROR", traceback.format_exc())
        return None


def start_persistent_worker(task_type: str, worker_config: Dict[str, Any]) -> Tuple[
    Optional[multiprocessing.Process], Optional[multiprocessing.connection.Connection]]:
    """Starts a persistent worker process."""
    try:
        # Ensure audio_thread_worker.py and its worker_loop are importable
        from audio_thread_worker import worker_loop as persistent_worker_main_loop_func
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

        process.join(timeout=10)  # Wait for worker to exit gracefully
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

    tts_worker_process = None
    tts_worker_pipe_orch_end = None
    asr_worker_process = None
    asr_worker_pipe_orch_end = None

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
    parser.add_argument("--exaggeration", type=float, default=0.5, help="Exaggeration for ChatterboxTTS")
    parser.add_argument("--cfg_weight", type=float, default=0.8, help="CFG weight for ChatterboxTTS")
    parser.add_argument("--test-audio-input", default="test_input.wav",
                        help="Filename of test audio for ASR (relative to model-dir or temp-dir)")
    parser.add_argument("--asr-test-model", default=WHISPER_DEFAULT_MODEL_FILENAME_CFG,
                        help="Whisper model filename for ASR test")
    parser.add_argument("--debug-save-raw-combined", action="store_true",
                        help="Save the raw combined audio before post-processing for debugging.")

    args = parser.parse_args()

    result_payload: Dict[str, Any] = {"error": f"Orchestrator failed task: {args.task_type}"}

    # Determine PyTorch device for main orchestrator thread (e.g., for SCLParser, MeloTTS)
    # and as a suggestion for the worker.
    effective_pytorch_device_orchestrator = "cpu"
    if TORCH_AUDIO_AVAILABLE and torch:
        effective_pytorch_device_orchestrator = _get_pytorch_device(args.device)
    elif args.task_type == "tts" and not (TORCH_AUDIO_AVAILABLE and torch):
        log_worker("CRITICAL", "PyTorch/Torchaudio not available in orchestrator. Cannot perform TTS.")
        print(json.dumps({"error": "PyTorch/Torchaudio missing for TTS orchestration."}), flush=True);
        sys.exit(1)

    # --- Worker Startup ---
    # A dummy prompt path for TTS worker warmup. Create a tiny silent WAV.
    dummy_tts_prompt_path = None
    if args.task_type == "tts":  # Only create if TTS worker will be started
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=args.temp_dir,
                                             prefix="dummy_prompt_") as tmp_dummy_prompt:
                dummy_tts_prompt_path = tmp_dummy_prompt.name
            sr_dummy_tts = 24000  # Match a common TTS SR or Chatterbox's expected SR
            duration_dummy_tts = 0.05  # 50ms
            silence_tts_prompt = np.zeros(int(duration_dummy_tts * sr_dummy_tts), dtype=np.float32)
            sf.write(dummy_tts_prompt_path, silence_tts_prompt, sr_dummy_tts, format='WAV',
                     subtype='PCM_F32')  # Or PCM_16
            log_worker("INFO", f"Created dummy TTS prompt for worker warmup: {dummy_tts_prompt_path}")
        except Exception as e_dummy_prompt:
            log_worker("WARNING",
                       f"Failed to create dummy TTS prompt for worker warmup: {e_dummy_prompt}. Warmup might be skipped or fail in worker.")
            dummy_tts_prompt_path = None  # Ensure it's None if creation failed

    if args.task_type == "tts":
        tts_worker_config = {
            "enable_tts": True,
            "enable_asr": False,
            "device": effective_pytorch_device_orchestrator,  # Worker will use this device
            "chatterbox_model_id": args.chatterbox_model_id,
            "dummy_tts_prompt_path_for_warmup": dummy_tts_prompt_path,
            "temp_dir": args.temp_dir  # For ASR warmup file if worker also handles ASR
        }
        tts_worker_process, tts_worker_pipe_orch_end = start_persistent_worker("TTS", tts_worker_config)
        if not tts_worker_process or not tts_worker_pipe_orch_end:
            log_worker("CRITICAL", "Failed to start persistent TTS worker. Exiting.")
            if dummy_tts_prompt_path and os.path.exists(dummy_tts_prompt_path): os.remove(dummy_tts_prompt_path)
            print(json.dumps({"error": "Failed to start TTS worker"}), flush=True);
            sys.exit(1)

    elif args.task_type == "asr":
        # Determine ASR model filename from config or args
        asr_model_filename_to_use = WHISPER_DEFAULT_MODEL_FILENAME_CFG
        if args.test_mode and args.asr_test_model:
            asr_model_filename_to_use = args.asr_test_model
        elif 'WHISPER_DEFAULT_MODEL_FILENAME' in globals():  # from config.py
            asr_model_filename_to_use = WHISPER_DEFAULT_MODEL_FILENAME

        asr_worker_config = {
            "enable_tts": False,
            "enable_asr": True,
            "device": "cpu",  # ASR via pywhispercpp is typically CPU-bound unless ggml has GPU layers enabled
            "model_dir": args.model_dir,
            "whisper_model_name": asr_model_filename_to_use,
            "temp_dir": args.temp_dir  # For ASR warmup file
        }
        asr_worker_process, asr_worker_pipe_orch_end = start_persistent_worker("ASR", asr_worker_config)
        if not asr_worker_process or not asr_worker_pipe_orch_end:
            log_worker("CRITICAL", "Failed to start persistent ASR worker. Exiting.")
            print(json.dumps({"error": "Failed to start ASR worker"}), flush=True);
            sys.exit(1)

    # --- Main Orchestration Logic ---
    job_temp_dir_path: Optional[str] = None  # To store path for cleanup
    try:
        if args.task_type == "tts":
            log_worker("INFO",
                       f"TTS Orchestration with persistent worker. Orchestrator/Worker device: '{effective_pytorch_device_orchestrator}'")
            if not CHATTERBOX_TTS_AVAILABLE:  # Global flag for orchestrator's own checks
                raise RuntimeError(
                    "ChatterboxTTS library not available in orchestrator (needed for SCL checks or if used directly).")

            job_temp_dir_path = tempfile.mkdtemp(prefix="tts_orch_job_", dir=args.temp_dir)
            log_worker("DEBUG", f"Created TTS job temp directory: {job_temp_dir_path}")

            all_audio_chunks_data_tts: List[np.ndarray] = []
            final_sr_from_chunks_tts: Optional[int] = None

            # SCLParser instance in the orchestrator for text splitting and final post-processing
            # Model is None here as actual synthesis is by worker. Melo model is for sample gen below.
            sclparser_for_text_proc_tts: SCLParser = SCLParser(model=None, device=effective_pytorch_device_orchestrator)

            # 1. Ensure Voice Sample for ChatterboxTTS (uses MeloTTS via SCLParser locally in orchestrator)
            generated_voice_sample_path_tts: Optional[str] = None
            melo_for_sample_init_tts: Optional[Any] = None  # MeloTTS instance
            sclparser_for_sample_init_tts: Optional[SCLParser] = None  # SCLParser with Melo instance

            if MELO_AVAILABLE and TTS_melo_class_ref and SCLParser_class_ref:
                log_worker("INFO", "Initializing MeloTTS for voice sample generation...")
                melo_for_sample_init_tts = TTS_melo_class_ref(language=args.model_lang.upper(),
                                                              device=effective_pytorch_device_orchestrator)
                sclparser_for_sample_init_tts = SCLParser_class_ref(model=melo_for_sample_init_tts,
                                                                    device=effective_pytorch_device_orchestrator)
            else:
                log_worker("WARNING",
                           "MeloTTS or SCLParser class not available. Voice sample generation might fail if sample doesn't exist.")

            generated_voice_sample_path_tts = _ensure_voice_sample(args, melo_for_sample_init_tts,
                                                                   sclparser_for_sample_init_tts,
                                                                   effective_pytorch_device_orchestrator)
            if not generated_voice_sample_path_tts:
                raise RuntimeError("Critical: Failed to find or generate ChatterboxTTS voice sample for orchestration.")
            log_worker("INFO", f"Using voice sample for TTS worker: {generated_voice_sample_path_tts}")

            # 2. Get text to synthesize
            text_for_synthesis: Optional[str] = None
            request_data_from_stdin_tts: Dict[str, Any] = {}  # Store full request for params like exaggeration

            if args.test_mode:
                log_worker("INFO", "TTS Orchestrator in TEST MODE.")
                text_for_synthesis = ADELAIDE_EXCITED_TEST_INTRO
                request_data_from_stdin_tts['exaggeration'] = args.exaggeration
                request_data_from_stdin_tts['cfg_weight'] = args.cfg_weight
                request_data_from_stdin_tts['response_format'] = "wav"  # Test mode outputs WAV
            else:
                log_worker("INFO", "TTS Orchestrator Standard Mode: Reading stdin...")
                input_json_str_from_stdin_tts = sys.stdin.read()
                if not input_json_str_from_stdin_tts: raise ValueError("Empty input from stdin for TTS orchestration.")
                request_data_from_stdin_tts = json.loads(input_json_str_from_stdin_tts)
                text_for_synthesis = request_data_from_stdin_tts.get("input")
                if not text_for_synthesis: raise ValueError("Missing 'input' text in TTS request.")

            # 3. Split text into phrases/segments using SCLParser
            text_segments_with_settings_tts = sclparser_for_text_proc_tts.split_into_segments_for_tts_worker(
                text_for_synthesis)
            if not text_segments_with_settings_tts:
                raise RuntimeError("SCLParser failed to split input text into processable segments.")
            log_worker("INFO", f"Split input into {len(text_segments_with_settings_tts)} segments for TTS processing.")

            # 4. Process each segment
            current_task_id_counter = 0
            progress_bar_tts = None
            if TQDM_AVAILABLE:
                progress_bar_tts = tqdm(total=len(text_segments_with_settings_tts), unit="segment",
                                        desc="TTS Processing Segments", ascii=IS_WINDOWS, leave=False)

            for i, segment_info in enumerate(text_segments_with_settings_tts):
                current_task_id_counter += 1
                task_id = f"tts_orch_{os.getpid()}_seg_{current_task_id_counter}"

                if segment_info["type"] == "text":
                    text_chunk_content = segment_info["content"]
                    # Output path for the chunk WAV file within the job's temp directory
                    chunk_output_filename = f"tts_chunk_{i}_{os.getpid()}.wav"  # Unique enough within job dir
                    chunk_output_full_path = os.path.join(job_temp_dir_path, chunk_output_filename)

                    exaggeration_val = float(request_data_from_stdin_tts.get("exaggeration", args.exaggeration))
                    cfg_weight_val = float(request_data_from_stdin_tts.get("cfg_weight", args.cfg_weight))

                    tts_task_payload = {
                        "command": "tts_chunk",
                        "task_id": task_id,
                        "params": {
                            "text_to_synthesize": text_chunk_content,
                            "chatterbox_model_id": args.chatterbox_model_id,
                            # Worker might use this if its model can vary
                            "voice_prompt_path": generated_voice_sample_path_tts,
                            "exaggeration": exaggeration_val,
                            "cfg_weight": cfg_weight_val,
                            "output_audio_chunk_path": chunk_output_full_path
                        }
                    }

                    if not tts_worker_pipe_orch_end or not tts_worker_process or not tts_worker_process.is_alive():
                        if progress_bar_tts: progress_bar_tts.close()
                        raise RuntimeError("TTS Worker process is not available or not alive.")

                    log_worker("DEBUG",
                               f"Sending task {task_id} to TTS worker: synthesize '{text_chunk_content[:30]}...'")
                    tts_worker_pipe_orch_end.send(tts_task_payload)

                    # Wait for result with timeout
                    # Use TTS_WORKER_TIMEOUT from config.py if available
                    worker_timeout_seconds = getattr(__import__('config', fromlist=['TTS_WORKER_TIMEOUT']),
                                                     'TTS_WORKER_TIMEOUT', 120)

                    if tts_worker_pipe_orch_end.poll(timeout=worker_timeout_seconds):
                        chunk_result = tts_worker_pipe_orch_end.recv()
                    else:
                        if progress_bar_tts: progress_bar_tts.close()
                        log_worker("ERROR",
                                   f"TTS Worker timed out (after {worker_timeout_seconds}s) for task {task_id} (segment {i + 1}).")
                        # Attempt to stop the worker as it might be stuck
                        stop_persistent_worker(tts_worker_process, tts_worker_pipe_orch_end, "TTS")
                        tts_worker_process, tts_worker_pipe_orch_end = None, None  # Mark as dead
                        raise RuntimeError(f"TTS Worker timed out for segment {i + 1}.")

                    if chunk_result.get("error"):
                        if progress_bar_tts: progress_bar_tts.close()
                        raise RuntimeError(
                            f"TTS Worker for segment {i + 1} (task {task_id}) reported error: {chunk_result['error']}")

                    # Process successful chunk result
                    result_data = chunk_result.get("result", {})
                    chunk_file_path_from_worker = result_data.get("audio_chunk_path")
                    chunk_sr_from_worker = result_data.get("sample_rate")

                    if not chunk_file_path_from_worker or not os.path.exists(
                            chunk_file_path_from_worker) or chunk_sr_from_worker is None:
                        if progress_bar_tts: progress_bar_tts.close()
                        raise RuntimeError(
                            f"TTS Worker for segment {i + 1} returned invalid audio path/SR. Path: '{chunk_file_path_from_worker}'")

                    # Load the generated audio chunk
                    audio_chunk_np, sr_read = sf.read(chunk_file_path_from_worker, dtype='float32')
                    log_worker("DEBUG",
                               f"Successfully received and loaded TTS chunk {i + 1} (task {task_id}). Path: {chunk_file_path_from_worker}, SR: {sr_read}, Shape: {audio_chunk_np.shape}")

                    if final_sr_from_chunks_tts is None:
                        final_sr_from_chunks_tts = sr_read
                    elif final_sr_from_chunks_tts != sr_read:
                        # This is a critical issue if sample rates are inconsistent.
                        # Should ideally resample or ensure worker produces consistent SR.
                        log_worker("CRITICAL",
                                   f"Sample rate mismatch between TTS chunks! Expected {final_sr_from_chunks_tts}, got {sr_read} for chunk {i + 1}. This may cause issues.")
                        # For now, we'll proceed but this needs addressing if it occurs.

                    all_audio_chunks_data_tts.append(audio_chunk_np)

                elif segment_info["type"] == "pause":
                    if final_sr_from_chunks_tts is None:  # If first segment is a pause, we don't have SR yet
                        log_worker("WARNING",
                                   "Encountered a pause segment before any audio segments. Cannot determine sample rate for silence. Skipping pause.")
                    else:
                        pause_duration_ms = int(segment_info["content"])
                        if pause_duration_ms > 0:
                            silence_samples = int(final_sr_from_chunks_tts * pause_duration_ms / 1000)
                            # Assume stereo silence if other chunks are stereo, else mono
                            num_channels_for_silence = 2  # Default to stereo as Chatterbox usually outputs stereo
                            if all_audio_chunks_data_tts and all_audio_chunks_data_tts[0].ndim == 1:
                                num_channels_for_silence = 1

                            silence_shape = (silence_samples,
                                             num_channels_for_silence) if num_channels_for_silence == 2 else (
                                silence_samples,)
                            silence_chunk = np.zeros(silence_shape, dtype=np.float32)
                            all_audio_chunks_data_tts.append(silence_chunk)
                            log_worker("DEBUG", f"Added silence chunk of {pause_duration_ms}ms for task {task_id}.")

                if progress_bar_tts: progress_bar_tts.update(1)

            if progress_bar_tts: progress_bar_tts.close()

            # 5. Combine and Post-Process Chunks
            if not all_audio_chunks_data_tts:
                raise RuntimeError("No audio chunks were generated or collected for TTS.")

            # Ensure a valid sample rate is set for combining
            # If all segments were pauses, final_sr_from_chunks_tts would be None.
            # Fallback to SCLParser's default or a common TTS SR.
            final_sr_tts_combined = final_sr_from_chunks_tts or sclparser_for_text_proc_tts.sample_rate or 24000
            log_worker("INFO",
                       f"Combining {len(all_audio_chunks_data_tts)} audio chunks at SR {final_sr_tts_combined}...")

            raw_combined_audio_np = sclparser_for_text_proc_tts.crossfade_segments(all_audio_chunks_data_tts,
                                                                                   final_sr_tts_combined)
            if raw_combined_audio_np is None or raw_combined_audio_np.size == 0:
                raise RuntimeError("SCLParser failed to combine TTS audio chunks or result was empty.")
            log_worker("INFO", f"Raw combined audio shape: {raw_combined_audio_np.shape}")

            # Optional: Save raw combined audio for debugging
            if args.debug_save_raw_combined:
                raw_debug_filename = f"debug_raw_combined_{os.getpid()}.wav"
                raw_debug_filepath = os.path.join(args.temp_dir, raw_debug_filename)  # Save in main temp, not job temp
                try:
                    # sf.write expects (samples, channels) or (samples,)
                    audio_to_save_raw = raw_combined_audio_np
                    if raw_combined_audio_np.ndim == 2 and raw_combined_audio_np.shape[0] < raw_combined_audio_np.shape[
                        1]:  # (channels, samples)
                        audio_to_save_raw = raw_combined_audio_np.T
                    sf.write(raw_debug_filepath, audio_to_save_raw, final_sr_tts_combined, format='WAV',
                             subtype='PCM_F32')  # Save as float for full quality
                    log_worker("DEBUG", f"Saved raw combined audio for debugging: {raw_debug_filepath}")
                except Exception as e_raw_save:
                    log_worker("WARNING", f"Failed to save raw combined audio for debugging: {e_raw_save}")

            # Free memory from individual chunks list
            del all_audio_chunks_data_tts
            gc.collect()

            # Apply SCLParser post-processing to the *entire combined* raw audio
            log_worker("INFO", "Applying SCLParser post-processing to combined audio...")
            # SCLParser's post_process expects zephyloid_settings from the *last state* of the SCL text.
            # This is appropriate if the effects are meant to be global to the utterance.
            # If SCL had per-segment post-processing, this would be much more complex.
            post_processed_tts_audio = sclparser_for_text_proc_tts.post_process(
                raw_combined_audio_np,
                final_sr_tts_combined,
                sclparser_for_text_proc_tts.zephyloid_settings  # Uses the settings accumulated by SCLParser
            )
            log_worker("INFO", f"Post-processing complete. Final audio shape: {post_processed_tts_audio.shape}")

            # 6. Convert to Target Format and Base64 Encode (largely same as your original)
            output_format = request_data_from_stdin_tts.get("response_format",
                                                            "mp3").lower() if not args.test_mode else "wav"
            audio_bytes_io = io.BytesIO()
            mime_type = f"audio/{output_format}"

            # Prepare for saving (post_processed_tts_audio from SCLParser is (channels, samples) numpy)
            # Soundfile expects (samples, channels) for stereo or (samples,) for mono
            audio_to_save_sf = post_processed_tts_audio  # Default if already (samples, channels) or (samples,)
            if post_processed_tts_audio.ndim == 2 and post_processed_tts_audio.shape[0] < \
                    post_processed_tts_audio.shape[1]:  # (channels, samples)
                # This is common for PyTorch tensors, so SCLParser might return this
                audio_to_save_sf = post_processed_tts_audio.T  # Transpose to (samples, channels)

            # Torchaudio save expects (channels, samples)
            audio_to_save_torch = torch.from_numpy(post_processed_tts_audio.astype(np.float32))
            if audio_to_save_torch.ndim == 1:  # (samples,) -> (1, samples) then (C, samples)
                audio_to_save_torch = audio_to_save_torch.unsqueeze(0)
            if audio_to_save_torch.shape[0] == 1:  # Ensure stereo if it became (1,samples)
                audio_to_save_torch = audio_to_save_torch.repeat(2, 1)

            if output_format == "wav":
                sf.write(audio_bytes_io, audio_to_save_sf, final_sr_tts_combined, format='WAV', subtype='PCM_16')
            elif output_format == "mp3":
                # Your existing MP3 conversion logic using torchaudio or ffmpeg fallback
                ffmpeg_exe_path = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
                can_torch_save_mp3_flag = TORCH_AUDIO_AVAILABLE and torchaudio and hasattr(torchaudio,
                                                                                           'save') and torchaudio.get_audio_backend() == "sox_io"

                if can_torch_save_mp3_flag:
                    try:
                        torchaudio.save(audio_bytes_io, audio_to_save_torch.cpu(), final_sr_tts_combined, format="mp3")
                        log_worker("INFO", "Saved final audio as MP3 using torchaudio.")
                    except Exception as e_mp3_torch_save:
                        log_worker("WARNING",
                                   f"Torchaudio MP3 save failed: {e_mp3_torch_save}. Falling back to ffmpeg if available.")
                        can_torch_save_mp3_flag = False  # Force fallback

                if not can_torch_save_mp3_flag:  # Fallback to ffmpeg or WAV
                    if ffmpeg_exe_path:
                        log_worker("INFO", "Attempting MP3 conversion using ffmpeg.")
                        with tempfile.NamedTemporaryFile(suffix=".wav", dir=args.temp_dir,
                                                         delete=False) as temp_wav_for_mp3, \
                                tempfile.NamedTemporaryFile(suffix=".mp3", dir=args.temp_dir,
                                                            delete=False) as temp_mp3_out:
                            temp_wav_path = temp_wav_for_mp3.name
                            temp_mp3_path = temp_mp3_out.name
                        try:
                            sf.write(temp_wav_path, audio_to_save_sf, final_sr_tts_combined, format='WAV',
                                     subtype='PCM_16')
                            ffmpeg_cmd = [
                                ffmpeg_exe_path, "-i", temp_wav_path,
                                "-codec:a", "libmp3lame", "-qscale:a", "2",  # VBR quality 2
                                "-y", temp_mp3_path
                            ]
                            subprocess.run(ffmpeg_cmd, check=True, capture_output=True, timeout=60)
                            with open(temp_mp3_path, "rb") as f_mp3:
                                audio_bytes_io.write(f_mp3.read())
                            log_worker("INFO", "Successfully converted to MP3 using ffmpeg.")
                        except Exception as e_ffmpeg_mp3:
                            log_worker("ERROR",
                                       f"ffmpeg MP3 conversion failed: {e_ffmpeg_mp3}. Falling back to WAV format.")
                            audio_bytes_io = io.BytesIO()  # Reset BytesIO
                            sf.write(audio_bytes_io, audio_to_save_sf, final_sr_tts_combined, format='WAV',
                                     subtype='PCM_16')
                            output_format = "wav"
                            mime_type = "audio/wav"
                        finally:
                            if os.path.exists(temp_wav_path): os.remove(temp_wav_path)
                            if os.path.exists(temp_mp3_path): os.remove(temp_mp3_path)
                    else:  # No ffmpeg, save as WAV
                        log_worker("WARNING", "ffmpeg not found for MP3 conversion. Falling back to WAV format.")
                        audio_bytes_io = io.BytesIO()  # Reset
                        sf.write(audio_bytes_io, audio_to_save_sf, final_sr_tts_combined, format='WAV',
                                 subtype='PCM_16')
                        output_format = "wav"
                        mime_type = "audio/wav"
            else:  # Default to WAV if format is unknown
                log_worker("WARNING", f"Unknown output format '{output_format}'. Defaulting to WAV.")
                sf.write(audio_bytes_io, audio_to_save_sf, final_sr_tts_combined, format='WAV', subtype='PCM_16')
                output_format = "wav"
                mime_type = "audio/wav"

            audio_bytes_io.seek(0)
            audio_b64 = base64.b64encode(audio_bytes_io.read()).decode('utf-8')
            result_payload = {
                "result": {
                    "audio_base64": audio_b64,
                    "format": output_format,
                    "mime_type": mime_type,
                    "sample_rate": final_sr_tts_combined
                }
            }

            if args.test_mode:
                # Save the final processed audio for test mode
                final_test_output_path = os.path.join(args.temp_dir, args.output_file)  # Use main temp dir
                with open(final_test_output_path, "wb") as f_out_test:
                    audio_bytes_io.seek(0)
                    f_out_test.write(audio_bytes_io.read())
                log_worker("INFO", f"Saved final TTS test output to: {final_test_output_path}")
                if "result" in result_payload:  # Should always be true here
                    result_payload["result"]["file_path_test_output"] = final_test_output_path


        elif args.task_type == "asr":
            log_worker("INFO", f"ASR Orchestration with persistent worker. Model dir: {args.model_dir}")
            job_temp_dir_path = tempfile.mkdtemp(prefix="asr_orch_job_", dir=args.temp_dir)  # ASR job temp dir

            master_temp_wav_asr: Optional[str] = None
            all_transcribed_segments_asr: List[str] = []

            input_audio_path_for_asr: Optional[str] = None
            # Determine ASR model and language from args or config (as in your original)
            asr_model_to_use_for_task = WHISPER_DEFAULT_MODEL_FILENAME_CFG  # Fallback from config.py
            if args.test_mode and args.asr_test_model:
                asr_model_to_use_for_task = args.asr_test_model
            elif 'WHISPER_DEFAULT_MODEL_FILENAME' in globals():  # Check if loaded from config
                asr_model_to_use_for_task = WHISPER_DEFAULT_MODEL_FILENAME

            asr_lang_to_use_for_task = args.model_lang if args.model_lang else WHISPER_DEFAULT_LANGUAGE_CFG

            if args.test_mode:
                log_worker("INFO", "ASR Orchestrator in TEST MODE.")
                # Construct path for test audio input
                input_audio_path_for_asr = os.path.join(args.temp_dir, args.test_audio_input)
                if not os.path.exists(input_audio_path_for_asr):
                    input_audio_path_for_asr = os.path.join(args.model_dir, args.test_audio_input)  # Try model_dir
                if not os.path.exists(input_audio_path_for_asr):
                    raise FileNotFoundError(
                        f"ASR test audio '{args.test_audio_input}' not found in '{args.temp_dir}' or '{args.model_dir}'.")
                # ASR model and lang for test mode are already set by worker config or overridden by args.
            else:  # Standard Mode from stdin
                log_worker("INFO", "ASR Orchestrator Standard Mode: Reading stdin.")
                input_json_str_from_stdin_asr = sys.stdin.read()
                if not input_json_str_from_stdin_asr: raise ValueError("Empty input from stdin for ASR orchestration.")
                request_data_from_stdin_asr = json.loads(input_json_str_from_stdin_asr)

                input_audio_path_for_asr = request_data_from_stdin_asr.get("input_audio_path")
                if not input_audio_path_for_asr or not os.path.exists(input_audio_path_for_asr):
                    raise FileNotFoundError(
                        f"ASR input_audio_path '{input_audio_path_for_asr}' missing or does not exist.")

                # Override model and lang if provided in stdin request, else use defaults/args
                asr_model_to_use_for_task = request_data_from_stdin_asr.get("whisper_model_name",
                                                                            asr_model_to_use_for_task)
                asr_lang_to_use_for_task = request_data_from_stdin_asr.get("language", asr_lang_to_use_for_task)

            log_worker("INFO",
                       f"ASR Task: Input Audio='{input_audio_path_for_asr}', Model='{asr_model_to_use_for_task}', Lang='{asr_lang_to_use_for_task}'")

            # Pre-process with FFmpeg to create master 16kHz mono WAV (same as your original logic)
            ffmpeg_exe = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
            if not str(input_audio_path_for_asr).lower().endswith(".wav"):
                if not ffmpeg_exe:
                    raise RuntimeError("ffmpeg is required for non-WAV input to ASR, and ffmpeg was not found.")
                master_temp_wav_asr = os.path.join(job_temp_dir_path, f"master_asr_input_{os.getpid()}.wav")
                ffmpeg_cmd_asr = [
                    ffmpeg_exe, '-i', input_audio_path_for_asr,
                    '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
                    '-y', master_temp_wav_asr
                ]
                log_worker("DEBUG", f"Running ffmpeg for ASR pre-processing: {' '.join(ffmpeg_cmd_asr)}")
                proc_ffmpeg_asr = subprocess.run(ffmpeg_cmd_asr, capture_output=True, text=True, check=False,
                                                 timeout=300)
                if proc_ffmpeg_asr.returncode != 0:
                    raise RuntimeError(f"ffmpeg ASR pre-processing failed. Stderr: {proc_ffmpeg_asr.stderr.strip()}")
                log_worker("INFO", f"FFmpeg ASR pre-processing successful: {master_temp_wav_asr}")
            else:  # Input is already WAV, but we still need to ensure 16kHz mono for Whisper
                # Option 1: Assume it's correct if it's WAV (simpler, but risky)
                # master_temp_wav_asr = input_audio_path_for_asr
                # Option 2: Always process with ffmpeg to ensure format (safer)
                if ffmpeg_exe:
                    master_temp_wav_asr = os.path.join(job_temp_dir_path,
                                                       f"master_asr_input_standardized_{os.getpid()}.wav")
                    ffmpeg_cmd_asr_std = [
                        ffmpeg_exe, '-i', input_audio_path_for_asr,
                        '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
                        '-y', master_temp_wav_asr
                    ]
                    log_worker("DEBUG", f"Running ffmpeg for ASR WAV standardization: {' '.join(ffmpeg_cmd_asr_std)}")
                    proc_ffmpeg_asr_std = subprocess.run(ffmpeg_cmd_asr_std, capture_output=True, text=True,
                                                         check=False, timeout=300)
                    if proc_ffmpeg_asr_std.returncode != 0:
                        log_worker("WARNING",
                                   f"ffmpeg ASR WAV standardization failed. Stderr: {proc_ffmpeg_asr_std.stderr.strip()}. Using original WAV.")
                        master_temp_wav_asr = input_audio_path_for_asr  # Fallback to original if standardization fails
                    else:
                        log_worker("INFO", f"FFmpeg ASR WAV standardization successful: {master_temp_wav_asr}")
                else:  # No ffmpeg, use WAV as is and hope for the best
                    log_worker("WARNING", "Input is WAV but ffmpeg not found to ensure 16kHz/mono. Using as-is.")
                    master_temp_wav_asr = input_audio_path_for_asr

            # Chunk the master WAV and send to ASR worker (same logic as your original for chunking)
            with wave.open(master_temp_wav_asr, 'rb') as wf_asr:
                nchannels, sampwidth, framerate, nframes, comptype, compname = wf_asr.getparams()
                if framerate != 16000 or nchannels != 1 or sampwidth != 2:  # 2 bytes for pcm_s16le
                    raise ValueError(
                        f"Master ASR WAV file '{master_temp_wav_asr}' is not 16kHz mono 16-bit PCM as required. Params: {wf_asr.getparams()}")

                chunk_duration_seconds = 30  # Standard for Whisper
                overlap_duration_seconds = 5  # Overlap to help with context

                chunk_len_frames = chunk_duration_seconds * framerate
                overlap_frames = overlap_duration_seconds * framerate
                step_frames = chunk_len_frames - overlap_frames
                if step_frames <= 0: step_frames = chunk_len_frames // 2  # Ensure progress

                current_pos_frames = 0
                chunk_idx = 0
                total_asr_chunks_to_process = math.ceil(nframes / step_frames) if step_frames > 0 else 1

                progress_bar_asr = None
                if TQDM_AVAILABLE:
                    progress_bar_asr = tqdm(total=total_asr_chunks_to_process, unit="chunk",
                                            desc="ASR Processing Chunks", ascii=IS_WINDOWS, leave=False)

                while current_pos_frames < nframes:
                    wf_asr.setpos(current_pos_frames)
                    frames_to_read = min(chunk_len_frames, nframes - current_pos_frames)
                    chunk_audio_bytes = wf_asr.readframes(frames_to_read)

                    if not chunk_audio_bytes: break  # End of file

                    chunk_file_path_for_worker = os.path.join(job_temp_dir_path,
                                                              f"asr_chunk_{chunk_idx}_{os.getpid()}.wav")
                    with wave.open(chunk_file_path_for_worker, 'wb') as cfw:
                        cfw.setnchannels(nchannels)
                        cfw.setsampwidth(sampwidth)
                        cfw.setframerate(framerate)
                        cfw.writeframes(chunk_audio_bytes)

                    asr_task_id = f"asr_orch_{os.getpid()}_chunk_{chunk_idx}"
                    asr_task_payload = {
                        "command": "asr_chunk",
                        "task_id": asr_task_id,
                        "params": {
                            "input_audio_chunk_path": chunk_file_path_for_worker,
                            "language": asr_lang_to_use_for_task
                            # The ASR worker already knows its model_dir and model_name from its init config
                        }
                    }

                    if not asr_worker_pipe_orch_end or not asr_worker_process or not asr_worker_process.is_alive():
                        if progress_bar_asr: progress_bar_asr.close()
                        raise RuntimeError("ASR Worker process is not available or not alive.")

                    log_worker("DEBUG",
                               f"Sending task {asr_task_id} to ASR worker for chunk: {chunk_file_path_for_worker}")
                    asr_worker_pipe_orch_end.send(asr_task_payload)

                    asr_worker_timeout_seconds = getattr(__import__('config', fromlist=['ASR_WORKER_TIMEOUT']),
                                                         'ASR_WORKER_TIMEOUT', 300)

                    if asr_worker_pipe_orch_end.poll(timeout=asr_worker_timeout_seconds):
                        asr_chunk_result = asr_worker_pipe_orch_end.recv()
                    else:
                        if progress_bar_asr: progress_bar_asr.close()
                        log_worker("ERROR",
                                   f"ASR Worker timed out (after {asr_worker_timeout_seconds}s) for task {asr_task_id} (chunk {chunk_idx}).")
                        stop_persistent_worker(asr_worker_process, asr_worker_pipe_orch_end, "ASR")
                        asr_worker_process, asr_worker_pipe_orch_end = None, None
                        raise RuntimeError(f"ASR Worker timed out for chunk {chunk_idx}.")

                    if asr_chunk_result.get("error"):
                        if progress_bar_asr: progress_bar_asr.close()
                        raise RuntimeError(
                            f"ASR Worker for chunk {chunk_idx} (task {asr_task_id}) reported error: {asr_chunk_result['error']}")

                    transcribed_text_from_chunk = asr_chunk_result.get("result", {}).get("text", "")
                    all_transcribed_segments_asr.append(transcribed_text_from_chunk)
                    log_worker("DEBUG",
                               f"Received ASR result for chunk {chunk_idx}: '{transcribed_text_from_chunk[:30]}...'")

                    # Clean up individual ASR chunk file unless told to keep
                    if not os.getenv("KEEP_ASR_CHUNKS", "false").lower() == "true":
                        try:
                            os.remove(chunk_file_path_for_worker)
                        except Exception as e_rm_asr_c:
                            log_worker("WARNING",
                                       f"Failed to remove ASR chunk file {chunk_file_path_for_worker}: {e_rm_asr_c}")

                    if current_pos_frames + frames_to_read >= nframes and frames_to_read < chunk_len_frames:  # Last chunk was smaller
                        pass  # allow progress bar to complete
                    elif current_pos_frames + step_frames >= nframes and step_frames < chunk_len_frames:  # About to step past end
                        current_pos_frames = nframes  # ensure loop terminates
                    else:
                        current_pos_frames += step_frames

                    chunk_idx += 1
                    if progress_bar_asr: progress_bar_asr.update(1)

                if progress_bar_asr: progress_bar_asr.close()

            final_transcription = " ".join(s.strip() for s in all_transcribed_segments_asr if s.strip()).strip()
            log_worker("INFO", f"Final combined ASR transcription: '{final_transcription[:100]}...'")
            result_payload = {"result": {"text": final_transcription}}

    except Exception as e_orchestration:
        result_payload = {"error": f"Orchestration error during task '{args.task_type}': {str(e_orchestration)}"}
        log_worker("ERROR", f"Orchestration error: {e_orchestration}")
        log_worker("ERROR", traceback.format_exc())
    finally:
        # --- Worker Shutdown ---
        if tts_worker_process:
            stop_persistent_worker(tts_worker_process, tts_worker_pipe_orch_end, "TTS")
        if asr_worker_process:
            stop_persistent_worker(asr_worker_process, asr_worker_pipe_orch_end, "ASR")

        # --- General Cleanup ---
        # Remove job-specific temp directory
        if job_temp_dir_path and os.path.isdir(job_temp_dir_path):
            try:
                shutil.rmtree(job_temp_dir_path)
                log_worker("INFO", f"Cleaned up job temp directory: {job_temp_dir_path}")
            except Exception as e_rm_job_dir:
                log_worker("WARNING", f"Failed to clean up job temp directory {job_temp_dir_path}: {e_rm_job_dir}")

        # Remove dummy TTS prompt if created
        if dummy_tts_prompt_path and os.path.exists(dummy_tts_prompt_path):
            try:
                os.remove(dummy_tts_prompt_path)
                log_worker("INFO", f"Cleaned up dummy TTS prompt: {dummy_tts_prompt_path}")
            except Exception as e_rm_dummy:
                log_worker("WARNING", f"Failed to remove dummy TTS prompt {dummy_tts_prompt_path}: {e_rm_dummy}")

        # MeloTTS instance cleanup (if created for sample gen)
        if 'melo_for_sample_init_tts' in locals() and melo_for_sample_init_tts:
            del melo_for_sample_init_tts
        if 'sclparser_for_sample_init_tts' in locals() and sclparser_for_sample_init_tts:
            del sclparser_for_sample_init_tts

        gc.collect()
        if TORCH_AUDIO_AVAILABLE and torch and effective_pytorch_device_orchestrator == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif TORCH_AUDIO_AVAILABLE and torch and effective_pytorch_device_orchestrator == "mps" and hasattr(
                torch.backends, "mps") and torch.backends.mps.is_available() and hasattr(torch.mps, "empty_cache"):
            try:
                torch.mps.empty_cache(); log_worker("INFO", "Orchestrator MPS cache clear attempted.")  # type: ignore
            except Exception as mps_e_final:
                log_worker("WARNING", f"Orchestrator MPS empty_cache failed: {mps_e_final}")

    # --- Print final JSON to stdout ---
    try:
        output_json_str_final = json.dumps(result_payload)
        log_worker("DEBUG", f"Sending output JSON (len={len(output_json_str_final)}): {output_json_str_final[:250]}...")
        print(output_json_str_final, flush=True)
        log_worker("INFO", f"Final result/error JSON sent for task: {args.task_type}.")
    except Exception as e_final_print:
        log_worker("CRITICAL", f"Failed to serialize/write final result_payload: {e_final_print}")
        try:
            # Try to send a minimal error if the main payload fails
            print(json.dumps({"error": f"Worker critical finalization error: {str(e_final_print)}"}), flush=True)
        except:
            pass  # If even this fails, not much more can be done via stdout
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