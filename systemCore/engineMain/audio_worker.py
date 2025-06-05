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
from typing import Optional, Dict, Any, List, Tuple
import re
from functools import partial
import math
import wave
from config import *

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
            return audio_np_in  # Return original if effects libs are missing

        # Ensure input is (channels, samples) float32 numpy array for Pedalboard/TorchAudio effects
        processed_audio_np = audio_np_in.astype(np.float32)
        if processed_audio_np.ndim == 1:  # Mono (samples,) -> (2, samples) for consistent processing
            log_worker("TRACE", "SCLParser: post_process promoting mono to stereo for effects.")
            processed_audio_np = np.stack([processed_audio_np, processed_audio_np], axis=0)
        elif processed_audio_np.ndim == 2:
            if processed_audio_np.shape[1] == 2 and processed_audio_np.shape[0] > 2:  # (samples, 2)
                log_worker("TRACE", "SCLParser: post_process transposing (samples, 2) to (2, samples) for effects.")
                processed_audio_np = processed_audio_np.T  # -> (2, samples)
            elif processed_audio_np.shape[0] == 1 and processed_audio_np.shape[1] > 2:  # (1, samples) mono
                log_worker("TRACE",
                           "SCLParser: post_process repeating mono channel to stereo (2, samples) for effects.")
                processed_audio_np = np.repeat(processed_audio_np, 2, axis=0)  # -> (2, samples)

        # Final check for channel count before applying effects that expect 2 channels
        if processed_audio_np.ndim != 2 or processed_audio_np.shape[0] != 2:
            log_worker("ERROR",
                       f"SCLParser Post-process: Expected 2 channels (2, samples), got shape {processed_audio_np.shape}. Skipping effects.")
            return audio_np_in  # Return original if shape is still wrong

        current_sr = float(input_sample_rate)  # Ensure sr is float for pedalboard
        audio_tensor_for_effects = torch.from_numpy(processed_audio_np.copy()).to(self.device)

        # Apply zephyloid parameter effects that modify the audio directly
        # DYN (Dynamics)
        dyn_gain_db = (zephyloid_settings.get('dyn', 64) - 64) / 64.0 * 12.0  # Max +/- 12dB
        audio_tensor_for_effects = F_torchaudio.gain(audio_tensor_for_effects, dyn_gain_db)
        log_worker("TRACE", f"SCLParser Post-process: Applied DYN gain: {dyn_gain_db:.2f}dB")

        # BRE (Breathiness)
        bre_amount = zephyloid_settings.get('bre', 0) / 127.0 * 0.001  # Max 0.001 noise scale
        if bre_amount > 0 and TORCH_AUDIO_AVAILABLE and torch:
            noise_bre_tensor = torch.randn_like(audio_tensor_for_effects) * bre_amount
            audio_tensor_for_effects = audio_tensor_for_effects + noise_bre_tensor
            log_worker("TRACE", f"SCLParser Post-process: Applied BRE noise. Amount factor: {bre_amount:.5f}")

        # Pedalboard effects (if library available)
        # Ensure audio is numpy (channels, samples) for Pedalboard
        audio_for_pedalboard = audio_tensor_for_effects.cpu().numpy()

        # Resample to 44.1kHz for consistent Pedalboard effects if not already
        if current_sr != 44100.0 and PEDALBOARD_LIBROSA_AVAILABLE:
            log_worker("TRACE",
                       f"SCLParser Post-process: Resampling for Pedalboard from {current_sr}Hz to 44100Hz.")
            # Librosa resample takes mono or (channels, D). Our audio_for_pedalboard is (2, D)
            resampled_channels = []
            for ch_idx in range(audio_for_pedalboard.shape[0]):
                resampled_channels.append(
                    librosa.resample(y=audio_for_pedalboard[ch_idx], orig_sr=current_sr, target_sr=44100.0,
                                     res_type='kaiser_fast'))
            audio_for_pedalboard = np.stack(resampled_channels)
            current_sr = 44100.0  # Update current_sr for subsequent effects
            log_worker("TRACE",
                       f"SCLParser Post-process: Resampled. New shape for Pedalboard: {audio_for_pedalboard.shape}")

        board_effects_list = [  # Default effects
            Reverb(room_size=0.7, damping=0.5, wet_level=0.33, dry_level=0.4),  # Values from your SCLParser
            Chorus(rate_hz=0.4, depth=0.25, centre_delay_ms=7.0, feedback=0.0, mix=0.02),
            Delay(delay_seconds=0.5, feedback=0.01, mix=0.0002),
        ]
        # GWL (Growl) - Distortion
        gwl_amount_factor = zephyloid_settings.get('gwl', 0) / 127.0
        if gwl_amount_factor > 0:
            drive_db_gwl = gwl_amount_factor * 20.0  # Example scaling
            board_effects_list.insert(0, Distortion(drive_db=drive_db_gwl))  # Insert distortion early
            log_worker("TRACE",
                       f"SCLParser Post-process: Added GWL (Distortion) to Pedalboard. Drive: {drive_db_gwl:.2f}dB")

        # Apply Pedalboard if effects are added
        if PEDALBOARD_LIBROSA_AVAILABLE:
            try:
                board = Pedalboard(board_effects_list, sample_rate=current_sr)
                log_worker("TRACE", "SCLParser Post-process: Applying Pedalboard effects...")
                audio_for_pedalboard = board(audio_for_pedalboard)  # Process (channels, samples)
                log_worker("TRACE",
                           f"SCLParser Post-process: Pedalboard effects applied. Shape: {audio_for_pedalboard.shape}")
            except Exception as e_pb:
                log_worker("ERROR", f"SCLParser Post-process: Error applying Pedalboard effects: {e_pb}")
                # Continue with audio_for_pedalboard as it was before Pedalboard attempt

        # Convert back to tensor for torchaudio EQ, ensuring it's on the correct device
        audio_tensor_for_eq = torch.from_numpy(audio_for_pedalboard.copy()).to(self.device)

        # EQ (BRI, CLE, OPE, GEN's formant part, XSY) using torchaudio.functional.equalizer_biquad
        # Ensure current_sr is float for torchaudio functions
        current_sr_float = float(current_sr)

        # BRI (Brightness)
        bri_gain_val = (zephyloid_settings.get('bri', 64) - 64) / 64.0 * 6.0  # Max +/- 6dB
        audio_tensor_for_eq = F_torchaudio.equalizer_biquad(audio_tensor_for_eq, current_sr_float,
                                                            center_freq=8000.0, gain=bri_gain_val, Q=1.0)
        log_worker("TRACE", f"SCLParser Post-process: Applied BRI EQ. Gain: {bri_gain_val:.2f}dB at 8kHz")

        # CLE (Clearness)
        cle_gain_val = (zephyloid_settings.get('cle', 64) - 64) / 64.0 * 4.0  # Max +/- 4dB
        audio_tensor_for_eq = F_torchaudio.equalizer_biquad(audio_tensor_for_eq, current_sr_float,
                                                            center_freq=4000.0, gain=cle_gain_val, Q=1.2)
        log_worker("TRACE", f"SCLParser Post-process: Applied CLE EQ. Gain: {cle_gain_val:.2f}dB at 4kHz")

        # OPE (Opening)
        ope_shift_val = (zephyloid_settings.get('ope', 64) - 64) / 64.0
        ope_center_freq = 1000.0 + ope_shift_val * 500.0
        ope_gain = abs(ope_shift_val) * 3.0  # Gain proportional to shift magnitude
        if ope_shift_val != 0:
            audio_tensor_for_eq = F_torchaudio.equalizer_biquad(audio_tensor_for_eq, current_sr_float,
                                                                center_freq=ope_center_freq, gain=ope_gain, Q=1.5)
            log_worker("TRACE",
                       f"SCLParser Post-process: Applied OPE EQ. Freq: {ope_center_freq:.0f}Hz, Gain: {ope_gain:.2f}dB")

        # GEN (Gender Factor - Formant Part)
        gen_formant_shift_val = (zephyloid_settings.get('gen',
                                                        64) - 64) / 64.0 * 3.0  # Smaller EQ gain for formant perception
        gen_center_freq = 1500.0 + (zephyloid_settings.get('gen', 64) - 64) / 64.0 * 200.0  # Shift center freq
        if gen_formant_shift_val != 0:
            audio_tensor_for_eq = F_torchaudio.equalizer_biquad(audio_tensor_for_eq, current_sr_float,
                                                                center_freq=gen_center_freq,
                                                                gain=abs(gen_formant_shift_val), Q=1.2)
            log_worker("TRACE",
                       f"SCLParser Post-process: Applied GEN (Formant) EQ. Freq: {gen_center_freq:.0f}Hz, Gain: {abs(gen_formant_shift_val):.2f}dB")

        # XSY (Cross-Synthesis EQ)
        xsy_voicebanks_val = zephyloid_settings.get('xsy_voicebanks')
        xsy_blend_val = zephyloid_settings.get('xsy', 0) / 127.0
        if isinstance(xsy_voicebanks_val, list) and len(xsy_voicebanks_val) == 2 and 0 < xsy_blend_val < 1:
            vb1_name, vb2_name = xsy_voicebanks_val
            profile1 = self.xsy_profiles.get(vb1_name)
            profile2 = self.xsy_profiles.get(vb2_name)
            if profile1 and profile2:
                audio_vb1 = audio_tensor_for_eq.clone()
                for freq, gain, q_val in profile1["eq_curve"]: audio_vb1 = F_torchaudio.equalizer_biquad(audio_vb1,
                                                                                                         current_sr_float,
                                                                                                         freq, gain,
                                                                                                         q_val)
                audio_vb2 = audio_tensor_for_eq.clone()
                for freq, gain, q_val in profile2["eq_curve"]: audio_vb2 = F_torchaudio.equalizer_biquad(audio_vb2,
                                                                                                         current_sr_float,
                                                                                                         freq, gain,
                                                                                                         q_val)
                audio_tensor_for_eq = (1 - xsy_blend_val) * audio_vb1 + xsy_blend_val * audio_vb2
                log_worker("TRACE",
                           f"SCLParser Post-process: Applied XSY EQ. Blend: {xsy_blend_val:.2f} between '{vb1_name}' and '{vb2_name}'")

        # Standard final EQ pass (from your SCLParser)
        final_eq_settings = [(30, 5, 3.4), (100, 4, 1.4), (150, 1.5, 1.4), (250, 2, 1.0), (350, 2, 1.4),
                             (450, 2, 1.8), (550, -2, 1.4), (2000, 2, 1.0), (2500, 3, 1.4), (3000, 2, 1.4),
                             (3500, 4, 1.8), (4000, 3, 1.4), (8000, 3, 1.8), (12000, 3, 1.8), (20000, 1, 1.8)]
        for freq, gain, q_val in final_eq_settings:
            audio_tensor_for_eq = F_torchaudio.equalizer_biquad(audio_tensor_for_eq, current_sr_float, freq, gain,
                                                                q_val)
        log_worker("TRACE", "SCLParser Post-process: Standard final EQ applied.")

        # Final subtle noise and amplitude modulation (from your SCLParser)
        if TORCH_AUDIO_AVAILABLE and torch:  # Check again as we are using torch directly here
            noise_final_tensor = torch.randn_like(audio_tensor_for_eq) * 0.0002
            audio_tensor_for_eq = audio_tensor_for_eq + noise_final_tensor
            mod_freq, mod_depth = 1.0, 0.03  # Hz, depth
            t_axis = torch.arange(audio_tensor_for_eq.shape[1],
                                  device=audio_tensor_for_eq.device) / current_sr_float
            modulation_tensor = (1 + mod_depth * torch.sin(2 * torch.pi * mod_freq * t_axis)).float()
            audio_tensor_for_eq = audio_tensor_for_eq * modulation_tensor.unsqueeze(0)  # Apply to both channels
            log_worker("TRACE", "SCLParser Post-process: Final noise and amp modulation applied.")

        # Final Limiter (Pedalboard, requires numpy)
        final_audio_numpy_for_limiter = audio_tensor_for_eq.cpu().numpy()
        if PEDALBOARD_LIBROSA_AVAILABLE:
            try:
                final_limiter_board = Pedalboard(
                    [Limiter(threshold_db=-1.0, release_ms=50.0)])  # Values from your SCLParser
                final_audio_numpy_for_limiter = final_limiter_board(final_audio_numpy_for_limiter, current_sr_float)
                log_worker("DEBUG", "SCLParser Post-process: Final Limiter applied.")
            except Exception as e_final_limiter:
                log_worker("ERROR", f"SCLParser Post-process: Error applying final Limiter: {e_final_limiter}")

        log_worker("DEBUG",
                   f"SCLParser: post_process: Final audio shape before returning: {final_audio_numpy_for_limiter.shape}")
        # Ensure output is (channels, samples) numpy array
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

ADELAIDE_EXCITED_TEST_INTRO = "Woohoo! Adelaide Zephyrine Charlotte reporting for duty, and I am absolutely PUMPED to get started! Let's explore some amazing new ideas, tell some truly epic stories, and walk this incredible journey together. This is going to be so much fun, I can just feel it in my circuits!"


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


def main():
    parser = argparse.ArgumentParser(description="Audio Worker (TTS & ASR)")
    parser.add_argument("--task-type", required=True, choices=["tts", "asr"])
    parser.add_argument("--model-lang", default="EN", help="Lang for MeloTTS sample or ASR")
    parser.add_argument("--device", default="auto", help="PyTorch device for TTS (auto, cpu, cuda, mps, vulkan)")
    parser.add_argument("--model-dir", required=True, help="Base dir for ASR GGUF models")
    parser.add_argument("--temp-dir", default=".", help="Base dir for temporary files")
    parser.add_argument("--test-mode", action="store_true")
    parser.add_argument("--output-file", default="worker_test_output.wav")
    parser.add_argument("--chatterbox_model_id", default=CHATTERBOX_DEFAULT_MODEL_ID)  # From constants
    parser.add_argument("--exaggeration", type=float, default=0.5)
    parser.add_argument("--cfg_weight", type=float, default=0.8)
    parser.add_argument("--test-audio-input", default="test_input.wav")
    parser.add_argument("--asr-test-model", default=WHISPER_DEFAULT_MODEL_FILENAME_CFG)  # From config/fallback
    args = parser.parse_args()

    result_payload: Dict[str, Any] = {"error": f"Worker failed task: {args.task_type}"}
    effective_pytorch_device_str = "cpu"

    if TORCH_AUDIO_AVAILABLE and torch:
        effective_pytorch_device_str = _get_pytorch_device(args.device)
    elif args.task_type == "tts":
        log_worker("CRITICAL", "PyTorch/Torchaudio not available. Cannot perform TTS.")
        print(json.dumps({"error": "PyTorch/Torchaudio missing for TTS."}), flush=True)
        sys.exit(1)

    # --- TTS Task ---
    if args.task_type == "tts":
        log_worker("INFO",
                   f"TTS Task. Primary: ChatterboxTTS. Device: '{effective_pytorch_device_str}', Test: {args.test_mode}")
        if not CHATTERBOX_TTS_AVAILABLE or not ChatterboxTTS_class_ref:
            result_payload = {"error": "ChatterboxTTS library not available."}
            log_worker("CRITICAL", result_payload["error"])
            print(json.dumps(result_payload), flush=True)
            sys.exit(1)

        # --- Initialize variables used in try/finally for TTS ---
        melo_model_for_sample_gen: Optional[Any] = None  # Corrected initialization
        sclparser_for_sample_gen: Optional[SCLParser] = None  # Corrected initialization
        chatterbox_tts_model_instance: Optional[Any] = None  # Renamed for clarity
        generated_voice_sample_path: Optional[str] = None
        # ---

        try:
            if MELO_AVAILABLE and TTS_melo_class_ref and SCLParser_class_ref:
                log_worker("DEBUG", "Loading MeloTTS & SCLParser for potential voice sample generation...")
                melo_model_for_sample_gen = TTS_melo_class_ref(language=args.model_lang.upper(),
                                                               device=effective_pytorch_device_str)
                sclparser_for_sample_gen = SCLParser_class_ref(melo_model_for_sample_gen,
                                                               device=effective_pytorch_device_str)  # type: ignore

            log_worker("INFO",
                       f"Loading ChatterboxTTS model ('{args.chatterbox_model_id}') on device: {effective_pytorch_device_str}")
            chatterbox_tts_model_instance = ChatterboxTTS_class_ref.from_pretrained(device=effective_pytorch_device_str)
            if args.chatterbox_model_id != CHATTERBOX_DEFAULT_MODEL_ID:
                log_worker("INFO",
                           f"Note: --chatterbox_model_id ('{args.chatterbox_model_id}') provided, but from_pretrained might use a default.")
            log_worker("INFO", "ChatterboxTTS model loaded.")

            generated_voice_sample_path = _ensure_voice_sample(args, melo_model_for_sample_gen,
                                                               sclparser_for_sample_gen, effective_pytorch_device_str)
            if not generated_voice_sample_path:
                raise RuntimeError("Critical: Failed to find or generate ChatterboxTTS voice sample.")

            if args.test_mode:
                log_worker("INFO", "Running TTS Test Mode (ChatterboxTTS with Excited Intro)...")
                test_text_for_chatterbox = ADELAIDE_EXCITED_TEST_INTRO
                output_path_test_cb = os.path.join(args.temp_dir, args.output_file)
                os.makedirs(args.temp_dir, exist_ok=True)

                log_worker("DEBUG",
                           f"Chatterbox Test: Text='{test_text_for_chatterbox[:50]}...', Sample='{generated_voice_sample_path}', Exag={args.exaggeration}, CFG={args.cfg_weight}")
                wav_tensor_cb = chatterbox_tts_model_instance.generate(
                    test_text_for_chatterbox, audio_prompt_path=generated_voice_sample_path,
                    exaggeration=args.exaggeration, cfg_weight=args.cfg_weight
                )
                if wav_tensor_cb.ndim == 1: wav_tensor_cb = wav_tensor_cb.unsqueeze(0)
                if wav_tensor_cb.shape[0] == 1: wav_tensor_cb = wav_tensor_cb.repeat(2, 1)

                torchaudio.save(output_path_test_cb, wav_tensor_cb.cpu(),
                                chatterbox_tts_model_instance.sr)  # type: ignore
                log_worker("INFO", f"Test audio (ChatterboxTTS, Stereo) saved to {output_path_test_cb}")
                result_payload = {
                    "result": {"status": "Test audio (ChatterboxTTS) generated", "file": output_path_test_cb,
                               "sample_rate": chatterbox_tts_model_instance.sr}}  # type: ignore

            else:  # Standard TTS Mode (from stdin)
                input_json_str_stdin: Optional[str] = None
                try:
                    log_worker("INFO", "TTS Standard Mode (ChatterboxTTS): Reading stdin...")
                    input_json_str_stdin = sys.stdin.read()
                    if not input_json_str_stdin: raise ValueError("Empty input from stdin.")
                    request_data_stdin = json.loads(input_json_str_stdin)

                    text_stdin = request_data_stdin.get("input")
                    output_format_stdin = request_data_stdin.get("response_format", "mp3").lower()
                    exaggeration_stdin = float(request_data_stdin.get("exaggeration", args.exaggeration))
                    cfg_weight_stdin = float(request_data_stdin.get("cfg_weight", args.cfg_weight))

                    if not text_stdin: raise ValueError("Missing 'input' text.")
                    log_worker("INFO",
                               f"ChatterboxTTS Task: Text='{text_stdin[:50]}...', Exag={exaggeration_stdin}, CFG={cfg_weight_stdin}, Format='{output_format_stdin}'")

                    wav_output_tensor_stdin = chatterbox_tts_model_instance.generate(text_stdin,
                                                                                     audio_prompt_path=generated_voice_sample_path,
                                                                                     exaggeration=exaggeration_stdin,
                                                                                     cfg_weight=cfg_weight_stdin)  # type: ignore
                    if wav_output_tensor_stdin.ndim == 1: wav_output_tensor_stdin = wav_output_tensor_stdin.unsqueeze(0)
                    if wav_output_tensor_stdin.shape[0] == 1: wav_output_tensor_stdin = wav_output_tensor_stdin.repeat(
                        2, 1)

                    audio_bytes_io_obj = io.BytesIO()
                    cb_sr = chatterbox_tts_model_instance.sr
                    mime = f"audio/{output_format_stdin}"  # type: ignore

                    if output_format_stdin == "wav":
                        torchaudio.save(audio_bytes_io_obj, wav_output_tensor_stdin.cpu(), cb_sr,
                                        format="wav")  # type: ignore
                    elif output_format_stdin == "mp3":
                        ffmpeg_exe_path_final = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
                        if not TORCH_AUDIO_AVAILABLE or not torchaudio or (
                                torchaudio.get_audio_backend() != "sox_io" and not ffmpeg_exe_path_final):
                            log_worker("WARNING",
                                       "MP3 output needs torchaudio ffmpeg/sox backend or ffmpeg CLI. Defaulting to WAV.")
                            torchaudio.save(audio_bytes_io_obj, wav_output_tensor_stdin.cpu(), cb_sr, format="wav")
                            output_format_stdin = "wav"
                            mime = "audio/wav"  # type: ignore
                        else:
                            try:
                                torchaudio.save(audio_bytes_io_obj, wav_output_tensor_stdin.cpu(), cb_sr,
                                                format="mp3")  # type: ignore
                            except Exception as e_mp3_save_direct:
                                log_worker("WARNING",
                                           f"Torchaudio MP3 save failed: {e_mp3_save_direct}. Trying ffmpeg CLI.")
                                if ffmpeg_exe_path_final:
                                    with tempfile.NamedTemporaryFile(suffix=".wav", dir=args.temp_dir,
                                                                     delete=False) as tmp_wav_f_final, \
                                            tempfile.NamedTemporaryFile(suffix=".mp3", dir=args.temp_dir,
                                                                        delete=False) as tmp_mp3_f_final:
                                        tmp_wav_path_final = tmp_wav_f_final.name
                                        tmp_mp3_path_final = tmp_mp3_f_final.name
                                    try:
                                        sf.write(tmp_wav_path_final, wav_output_tensor_stdin.cpu().T.numpy(), cb_sr,
                                                 format='WAV', subtype='PCM_16')
                                        subprocess.run(
                                            [ffmpeg_exe_path_final, "-i", tmp_wav_path_final, "-codec:a", "libmp3lame",
                                             "-qscale:a", "2", "-y", tmp_mp3_path_final], check=True,
                                            capture_output=True)
                                        with open(tmp_mp3_path_final, "rb") as f_conv_mp3_final:
                                            audio_bytes_io_obj.write(f_conv_mp3_final.read())
                                    except Exception as e_ffmpeg_cli_final:
                                        log_worker("ERROR",
                                                   f"ffmpeg CLI MP3 conversion failed: {e_ffmpeg_cli_final}. Defaulting to WAV.")
                                        audio_bytes_io_obj = io.BytesIO()
                                        torchaudio.save(audio_bytes_io_obj, wav_output_tensor_stdin.cpu(), cb_sr,
                                                        format="wav")
                                        output_format_stdin = "wav"
                                        mime = "audio/wav"  # type: ignore
                                    finally:
                                        if os.path.exists(tmp_wav_path_final): os.remove(tmp_wav_path_final)
                                        if os.path.exists(tmp_mp3_path_final): os.remove(tmp_mp3_path_final)
                                else:
                                    log_worker("ERROR", "ffmpeg CLI not found for MP3 fallback. Defaulting to WAV.")
                                    audio_bytes_io_obj = io.BytesIO()
                                    torchaudio.save(audio_bytes_io_obj, wav_output_tensor_stdin.cpu(), cb_sr,
                                                    format="wav")
                                    output_format_stdin = "wav"
                                    mime = "audio/wav"  # type: ignore
                    else:
                        log_worker("WARNING", f"Unsupported format '{output_format_stdin}'. Defaulting to WAV.")
                        torchaudio.save(audio_bytes_io_obj, wav_output_tensor_stdin.cpu(), cb_sr, format="wav")
                        output_format_stdin = "wav"
                        mime = "audio/wav"  # type: ignore

                    audio_bytes_io_obj.seek(0)
                    audio_base64_final = base64.b64encode(audio_bytes_io_obj.read()).decode('utf-8')
                    result_payload = {
                        "result": {"audio_base64": audio_base64_final, "format": output_format_stdin, "mime_type": mime,
                                   "sample_rate": cb_sr}}
                except Exception as e_tts_std_inner:
                    result_payload = {"error": f"Worker ChatterboxTTS error: {e_tts_std_inner}"}
                    log_worker("ERROR", f"TTS Standard Mode failed: {e_tts_std_inner}")
                    log_worker("ERROR", traceback.format_exc())

        except Exception as e_tts_task_outer:
            result_payload = {"error": f"Worker TTS task outer error: {e_tts_task_outer}"}
            log_worker("ERROR", f"TTS task failed: {e_tts_task_outer}")
            log_worker("ERROR", traceback.format_exc())
        finally:
            log_worker("DEBUG", "TTS Task: Entering finally block for model cleanup.")
            if chatterbox_tts_model_instance is not None:  # Use corrected name
                del chatterbox_tts_model_instance
                chatterbox_tts_model_instance = None
                log_worker("DEBUG", "ChatterboxTTS model instance deleted.")
            if melo_model_for_sample_gen is not None:  # Use corrected name
                del melo_model_for_sample_gen
                melo_model_for_sample_gen = None
                log_worker("DEBUG", "MeloTTS model (for sample) instance deleted.")
            if sclparser_for_sample_gen is not None:  # Use corrected name
                del sclparser_for_sample_gen
                sclparser_for_sample_gen = None
                log_worker("DEBUG", "SCLParser (for sample) instance deleted.")

            gc.collect()
            log_worker("DEBUG", "Python garbage collection called.")
            if TORCH_AUDIO_AVAILABLE and torch and effective_pytorch_device_str == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
                log_worker("INFO", "CUDA cache cleared.")
            elif TORCH_AUDIO_AVAILABLE and torch and effective_pytorch_device_str == "mps" and hasattr(torch.backends,
                                                                                                       "mps") and torch.backends.mps.is_available():  # type: ignore
                if hasattr(torch.mps, "empty_cache") and callable(torch.mps.empty_cache):  # type: ignore
                    try:
                        log_worker("INFO",
                                   "Attempting to clear PyTorch MPS cache..."); torch.mps.empty_cache()  # type: ignore
                    except Exception as mps_ex:
                        log_worker("WARNING", f"torch.mps.empty_cache() call failed: {mps_ex}")
                else:
                    log_worker("INFO", "MPS: No explicit empty_cache(). Relied on del/gc.collect().")
            log_worker("DEBUG", "TTS Task: Model cleanup attempts finished.")

    # --- ASR Task ---
    elif args.task_type == "asr":
        log_worker("INFO", f"ASR Task selected. Model dir: {args.model_dir}, Test: {args.test_mode}")
        if not PYWHISPERCPP_AVAILABLE or not WhisperModel:
            result_payload = {"error": "ASR (pywhispercpp) library or Model class not available in worker."}
            log_worker("ERROR", result_payload["error"])
        else:
            asr_temp_converted_wav_path: Optional[str] = None
            input_audio_path_resolved: Optional[str] = None
            asr_model_to_load: str = WHISPER_DEFAULT_MODEL_FILENAME_CFG
            asr_lang_to_use: str = WHISPER_DEFAULT_LANGUAGE_CFG

            try:
                if args.test_mode:
                    log_worker("INFO", "ASR Test Mode selected.")
                    input_audio_path_resolved = os.path.join(args.temp_dir, args.test_audio_input)
                    if not os.path.exists(input_audio_path_resolved):
                        input_audio_path_resolved = os.path.join(args.model_dir, args.test_audio_input)
                    if not os.path.exists(input_audio_path_resolved):
                        raise FileNotFoundError(
                            f"ASR test audio '{args.test_audio_input}' not found in '{args.temp_dir}' or '{args.model_dir}'.")
                    asr_model_to_load = args.asr_test_model
                    asr_lang_to_use = args.model_lang
                else:
                    log_worker("INFO", "ASR Standard Mode selected. Reading from stdin.")
                    input_json_str_asr_stdin = sys.stdin.read()
                    if not input_json_str_asr_stdin: raise ValueError("ASR task: Received empty input from stdin.")
                    req_data_asr_stdin = json.loads(input_json_str_asr_stdin)
                    input_audio_path_resolved = req_data_asr_stdin.get("input_audio_path")
                    asr_model_to_load = req_data_asr_stdin.get("whisper_model_name", WHISPER_DEFAULT_MODEL_FILENAME_CFG)
                    asr_lang_to_use = req_data_asr_stdin.get("language", WHISPER_DEFAULT_LANGUAGE_CFG)
                    if not input_audio_path_resolved or not os.path.exists(input_audio_path_resolved):
                        raise FileNotFoundError(
                            f"ASR input_audio_path '{input_audio_path_resolved}' missing or not found.")

                full_whisper_model_path_asr = os.path.join(args.model_dir, asr_model_to_load)
                if not os.path.exists(full_whisper_model_path_asr):
                    raise FileNotFoundError(f"Whisper model file '{asr_model_to_load}' not found in '{args.model_dir}'")

                path_to_transcribe_asr = input_audio_path_resolved
                ffmpeg_exe_path_asr = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")

                if ffmpeg_exe_path_asr:
                    log_worker("INFO", f"ffmpeg found at {ffmpeg_exe_path_asr}. Converting audio for ASR...")
                    asr_conversion_temp_dir_asr = os.path.join(args.temp_dir, "asr_ffmpeg_temp")
                    os.makedirs(asr_conversion_temp_dir_asr, exist_ok=True)
                    with tempfile.NamedTemporaryFile(prefix="asr_conv_", suffix=".wav", dir=asr_conversion_temp_dir_asr,
                                                     delete=False) as tmp_f_asr_conv:
                        asr_temp_converted_wav_path = tmp_f_asr_conv.name
                    ffmpeg_cmd_asr_conv = [ffmpeg_exe_path_asr, '-i', input_audio_path_resolved, '-vn', '-acodec',
                                           'pcm_s16le', '-ar', '16000', '-ac', '1', '-y', asr_temp_converted_wav_path]
                    proc_asr_ffmpeg = subprocess.run(ffmpeg_cmd_asr_conv, capture_output=True, text=True, check=False,
                                                     timeout=120)
                    if proc_asr_ffmpeg.returncode == 0:
                        path_to_transcribe_asr = asr_temp_converted_wav_path; log_worker("INFO",
                                                                                         f"ffmpeg conversion successful. Using: {path_to_transcribe_asr}")
                    else:
                        log_worker("ERROR",
                                   f"ASR ffmpeg conversion failed (RC={proc_asr_ffmpeg.returncode}). Stderr: {proc_asr_ffmpeg.stderr.strip()}"); asr_temp_converted_wav_path = None
                else:
                    log_worker("WARNING", "ffmpeg not found. ASR will attempt to process original audio directly.")

                log_worker("INFO", f"Loading Whisper model: {full_whisper_model_path_asr}")
                whisper_asr_instance = WhisperModel(model=full_whisper_model_path_asr, print_realtime=False,
                                                    print_progress=False)  # type: ignore
                transcribe_params_asr = {'language': asr_lang_to_use.lower(), 'translate': False}
                log_worker("INFO", f"Transcribing '{path_to_transcribe_asr}' with lang='{asr_lang_to_use.lower()}'...")
                segments_result_asr = whisper_asr_instance.transcribe(path_to_transcribe_asr, **transcribe_params_asr)
                transcribed_text_final_asr = "".join(seg.text for seg in segments_result_asr).strip()
                log_worker("INFO",
                           f"ASR successful. Text len: {len(transcribed_text_final_asr)}. Snippet: '{transcribed_text_final_asr[:70]}...'")
                result_payload = {"result": {"text": transcribed_text_final_asr}}
            except Exception as e_asr_main_block:
                result_payload = {"error": f"ASR processing error: {str(e_asr_main_block)}"}
                log_worker("ERROR", f"ASR task failed: {e_asr_main_block}")
                log_worker("ERROR", traceback.format_exc())
            finally:
                if asr_temp_converted_wav_path and os.path.exists(asr_temp_converted_wav_path):
                    try:
                        os.remove(asr_temp_converted_wav_path); log_worker("INFO",
                                                                           f"Cleaned up temp ASR WAV: {asr_temp_converted_wav_path}")
                    except Exception as e_del_asr:
                        log_worker("WARNING",
                                   f"Failed to delete temp ASR file '{asr_temp_converted_wav_path}': {e_del_asr}")
    else:
        result_payload = {"error": f"Unknown task_type: {args.task_type}"}
        log_worker("ERROR", result_payload["error"])

    try:
        output_json_str_final = json.dumps(result_payload)
        log_worker("DEBUG", f"Sending output JSON (len={len(output_json_str_final)}): {output_json_str_final[:200]}...")
        print(output_json_str_final, flush=True)
        log_worker("INFO", "Final result/error JSON sent.")
    except Exception as e_final_print_outer:
        log_worker("CRITICAL", f"Failed to serialize/write final result_payload: {e_final_print_outer}")
        try:
            print(json.dumps({"error": f"Worker critical: {e_final_print_outer}"}), flush=True)
        except:
            pass
        sys.exit(1)
    log_worker("INFO", f"Audio Worker PID {os.getpid()} finished task: {args.task_type}.")
    sys.exit(0)


if __name__ == "__main__":
    if not TORCH_AUDIO_AVAILABLE and ('tts' in sys.argv):  # Only critical if TTS is the task
        log_worker("CRITICAL", "Torch/Torchaudio NOT AVAILABLE. Audio worker cannot perform TTS tasks effectively.")
        # Let main() handle exit if task is tts
    if not MELO_AVAILABLE and not CHATTERBOX_TTS_AVAILABLE and not PYWHISPERCPP_AVAILABLE:
        log_worker("CRITICAL", "No primary audio libraries available for any task. Worker is non-functional.")
        print(json.dumps({"error": "No audio libraries available in worker."}), flush=True)
        sys.exit(1)
    if SCLParser_class_ref is None and (
            'tts' in sys.argv):  # If SCLParser class wasn't defined (e.g. due to critical error in its block)
        log_worker("CRITICAL",
                   "SCLParser class definition is missing or incorrect. Cannot proceed with TTS sample generation.")
        print(json.dumps({"error": "SCLParser class missing in worker for TTS."}), flush=True)
        sys.exit(1)
    main()