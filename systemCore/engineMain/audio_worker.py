# audio_worker.py
import gc
import sys
import os
import json
import time
import traceback
import argparse
import librosa
import base64
import re
import io
import shutil  # For checking ffmpeg and which
from typing import Optional, Dict, Any, List
import numpy as np
import soundfile as sf
import tempfile  # For ASR ffmpeg conversion temp file
import subprocess  # For ASR ffmpeg conversion
import torchaudio.functional as F
from functools import partial
from pedalboard import Pedalboard, Reverb, Limiter, Gain, PitchShift, Resample, Chorus, Delay, Distortion

# --- PyTorch and torchaudio imports ---
TORCH_AUDIO_AVAILABLE = False
torch = None  # Initialize to allow type checking later
torchaudio = None  # Initialize
F_torchaudio = None
try:
    import torch
    import torchaudio
    import torchaudio.functional as F_torchaudio

    TORCH_AUDIO_AVAILABLE = True
    print("[AUDIO_WORKER|INFO] PyTorch and Torchaudio imported successfully.", file=sys.stderr, flush=True)
except ImportError as e_torch:
    print(f"[AUDIO_WORKER|ERROR] PyTorch or torchaudio not found: {e_torch}. TTS functionality will be impaired.",
          file=sys.stderr, flush=True)
    # TTS will likely fail if these are not available.


# --- Basic Logging Function (used before full logger setup if needed by SCLParser during import) ---
def log_worker(level, message):
    """Basic logging to stderr for the worker itself."""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')} AUDIO_WORKER|{level}] {message}", file=sys.stderr, flush=True)


# --- SCLParser Class (User to fill in with their full SCLParser class definition) ---
# This placeholder is designed to allow _ensure_voice_sample to run without error,
# assuming the primary goal is to get *some* audio data from MeloTTS for the sample.
# Your actual SCLParser will have much richer functionality.

# --- SCLParser (Modified to use log_worker) ---
class SCLParser:
    def __init__(self, model, device='auto'):
        log_worker("INFO", "SCLParser: Initializing...")
        self.model = model
        self.device = device
        self.speaker_ids = model.hps.data.spk2id
        log_worker("DEBUG", f"SCLParser: Speaker IDs: {self.speaker_ids}")
        self.sample_rate = model.hps.data.sampling_rate
        log_worker("INFO", f"SCLParser: Sample Rate: {self.sample_rate}")
        self.voice_settings = {'rate': 1.0, 'pitch': 0.0}
        log_worker("DEBUG", f"SCLParser: Initial Voice Settings: {self.voice_settings}")
        self.rate_map = {'x-slow': 0.9, 'slow': 1.0, 'medium': 1.05, 'fast': 1.1, 'x-fast': 1.1}
        self.pitch_map = {'x-low': -0.4, 'low': -0.1, 'medium': 0.0, 'high': 1.2, 'x-high': 2.0}
        log_worker("DEBUG", f"SCLParser: Rate Map: {self.rate_map}")
        log_worker("DEBUG", f"SCLParser: Pitch Map: {self.pitch_map}")
        self.audio_segments = []
        self.original_sf_write = sf.write
        self.captured_audio_data: Optional[np.ndarray] = None
        self.captured_audio_samplerate: Optional[int] = None

        self.zephyloid_settings = {
            'vel': 64, 'dyn': 64, 'bre': 0, 'bri': 64, 'cle': 64, 'ope': 64,
            'gen': 64, 'gwl': 0, 'xsy': 0, 'xsy_voicebanks': None,
            'singing': False, 'key': None, 'correction_method': "closest",
        }
        log_worker("DEBUG", f"SCLParser: Initial zephyloid Settings: {self.zephyloid_settings}")
        self.xsy_profiles = {
            "Voicebank1": {"description": "Default", "eq_curve": [(30,0,3.4),(100,0,1.4),(150,0,1.4),(250,0,1.0),(350,0,1.4),(450,0,1.8),(550,0,1.4),(2000,0,1.0),(2500,0,1.4),(3000,0,1.4),(3500,0,1.8),(4000,0,1.4),(8000,0,1.8),(12000,0,1.8),(20000,0,1.8)]},
            "Voicebank2": {"description": "Brighter", "eq_curve": [(30,2,3.4),(100,3,1.4),(150,1,1.4),(250,1,1.0),(350,-1,1.4),(450,-2,1.8),(550,2,1.4),(2000,3,1.0),(2500,4,1.4),(3000,3,1.4),(3500,2,1.8),(4000,1,1.4),(8000,4,1.8),(12000,5,1.8),(20000,2,1.8)]},
            "Voicebank3": {"description": "Deeper", "eq_curve": [(30,4,3.4),(100,5,1.4),(150,3,1.4),(250,2,1.0),(350,1,1.4),(450,-1,1.8),(550,-3,1.4),(2000,-2,1.0),(2500,-1,1.4),(3000,0,1.4),(3500,1,1.8),(4000,2,1.4),(8000,1,1.8),(12000,0,1.8),(20000,-1,1.8)]}
        }
        log_worker("DEBUG", f"SCLParser: XSY Profiles defined.")
        log_worker("INFO", "SCLParser: Initialization complete.")

    def _new_sf_write(self, file, data, samplerate, *args, **kwargs):
        # This method is key. Instead of writing to `file` (which SCLParser sets to a path),
        # we capture the data and samplerate.
        log_worker("DEBUG", f"SCLParser: _new_sf_write intercepted. Data type: {type(data)}, Samplerate: {samplerate}")
        if isinstance(data, torch.Tensor):
            self.captured_audio_data = data.clone().detach().cpu().numpy()
        elif isinstance(data, np.ndarray):
            self.captured_audio_data = data.copy()
        else:
            log_worker("WARNING", f"SCLParser: _new_sf_write received unexpected data type: {type(data)}")
            self.captured_audio_data = None # Ensure it's reset if data type is wrong
            # Still call original write to let it fail or handle if it can
            return self.original_sf_write(file, data, samplerate, *args, **kwargs)

        self.captured_audio_samplerate = samplerate
        if self.captured_audio_data is not None:
            log_worker("DEBUG", f"SCLParser: _new_sf_write: Captured audio shape: {self.captured_audio_data.shape}")
        # Do NOT call self.original_sf_write here if we want to capture it for base64 output.
        # For test mode, the main function will handle saving.
        # For standard mode, the main function will handle converting and encoding.
        # If `file` argument was a BytesIO, we could write to it, but Melo expects a path.

    def parse(self, text, speaker='EN-US'): # Removed output_path from SCLParser's parse
        log_worker("INFO", f"SCLParser: parse called. Text: '{text[:50]}...', Speaker: {speaker}")
        sf.write = self._new_sf_write # Hook sf.write
        self.audio_segments = [] # Reset segments
        self.captured_audio_data = None # Reset captured audio
        self.captured_audio_samplerate = None

        processed_text = self.preprocess_text(text)
        log_worker("DEBUG", f"SCLParser: parse: Preprocessed Text: '{processed_text[:50]}...'")
        segments = self.split_into_segments(processed_text)
        log_worker("DEBUG", f"SCLParser: parse: Found {len(segments)} segments.")

        for i, segment_text in enumerate(segments):
            log_worker("DEBUG", f"SCLParser: parse: Processing segment {i+1}/{len(segments)}: '{segment_text[:50]}...'")
            self.parse_segment(segment_text, speaker) # This calls speak_with_settings

        final_audio_processed = None
        if self.audio_segments:
            log_worker("INFO", "SCLParser: parse: Combining audio segments...")
            final_audio_combined = self.crossfade_segments(self.audio_segments, self.sample_rate)
            if final_audio_combined is not None and final_audio_combined.size > 0 :
                log_worker("DEBUG", f"SCLParser: parse: Combined audio shape: {final_audio_combined.shape}")
                final_audio_processed = self.post_process(final_audio_combined)
                log_worker("DEBUG", f"SCLParser: parse: Post-processed audio shape: {final_audio_processed.shape}")
                # Captured data is now in final_audio_processed
                self.captured_audio_data = final_audio_processed
                self.captured_audio_samplerate = self.sample_rate # Post-processing keeps sample rate for now
            else:
                log_worker("WARNING", "SCLParser: parse: Combined audio was empty or None.")
        else:
            log_worker("WARNING", "SCLParser: parse: No audio segments were generated to combine.")

        # Restore original sf.write and reset internal state
        sf.write = self.original_sf_write
        log_worker("INFO", "SCLParser: parse: Finished.")
        # Return the processed audio data and its sample rate
        return self.captured_audio_data, self.captured_audio_samplerate

    # --- Paste the rest of your SCLParser methods here ---
    # (preprocess_text, fix_missing_closing_tags, flatten_nested_tags, split_into_segments,
    #  parse_segment, apply_settings, reset_settings, _apply_zephyloid_effects,
    #  _degrees_from, _closest_pitch_from_scale, _parse_key, _closest_pitch, _autotune,
    #  speak_with_settings, crossfade_segments, post_process, parse_attributes, is_number,
    #  detect_pitch_pyin, parse_pitch)
    # --- Remember to change all print() statements in them to log_worker("LEVEL", ...) ---

    def preprocess_text(self, text):
        log_worker("DEBUG", f"SCLParser: preprocess_text called. Input text: '{text[:50]}...'")
        text = self.fix_missing_closing_tags(text)
        log_worker("DEBUG", f"SCLParser: preprocess_text: After fix_missing_closing_tags: '{text[:50]}...'")
        text = self.flatten_nested_tags(text)
        log_worker("DEBUG", f"SCLParser: preprocess_text: After flatten_nested_tags: '{text[:50]}...'")
        return text

    def fix_missing_closing_tags(self, text):
        # ... (Your SCLParser code with print -> log_worker) ...
        # Example change:
        # print(f"SCLParser: fix_missing_closing_tags: Found tag: '{tag_content}'")
        # becomes:
        # log_worker("TRACE", f"SCLParser: fix_missing_closing_tags: Found tag: '{tag_content}'")
        open_tags = []
        result = []
        i = 0
        while i < len(text):
            if text[i] == '[':
                closing_bracket_index = text.find(']', i)
                if closing_bracket_index == -1:
                    log_worker("TRACE", f"SCLParser: fix_missing_closing_tags: Found opening bracket without closing bracket at index {i}. Skipping.")
                    i += 1
                    continue
                tag_content = text[i + 1:closing_bracket_index]
                log_worker("TRACE", f"SCLParser: fix_missing_closing_tags: Found tag: '{tag_content}'")
                if tag_content.startswith('/'):
                    tag_name = tag_content[1:]
                    if open_tags and open_tags[-1] == tag_name:
                        log_worker("TRACE", f"SCLParser: fix_missing_closing_tags: Closing tag '{tag_name}' matches open tag.")
                        open_tags.pop()
                        result.append(text[i:closing_bracket_index + 1])
                    else:
                        log_worker("TRACE", f"SCLParser: fix_missing_closing_tags: Closing tag '{tag_name}' does not match open tag or no open tags.")
                        result.append(text[i:closing_bracket_index + 1])
                else:
                    tag_name = tag_content.split(' ')[0] if ' ' in tag_content else tag_content
                    log_worker("TRACE", f"SCLParser: fix_missing_closing_tags: Opening tag '{tag_name}' found.")
                    result.append(text[i:closing_bracket_index + 1])
                    open_tags.append(tag_name)
                i = closing_bracket_index + 1
            else:
                result.append(text[i])
                i += 1
        while open_tags:
            missing_tag = open_tags.pop()
            log_worker("TRACE", f"SCLParser: fix_missing_closing_tags: Adding missing closing tag '[/{missing_tag}]'")
            result.append(f"[/{missing_tag}]")
        fixed_text = "".join(result)
        log_worker("TRACE", f"SCLParser: fix_missing_closing_tags: Returning: '{fixed_text[:50]}...'")
        return fixed_text

    def flatten_nested_tags(self, text):
        log_worker("DEBUG", f"SCLParser: flatten_nested_tags called. Input: '{text[:50]}...'")
        original_text = text
        # This regex might need refinement for more complex nesting or attributes within tags.
        # It assumes simple tags like [tag1][tag2]...[/tag2][/tag1]
        # A more robust solution might involve a proper stack-based parser.
        while True:
            # A simplified regex for basic nesting. More complex cases may require a stack-based parser.
            match = re.search(r'(\[[^/][^\]]*\])([^\[]*?)(\[[^/][^\]]*\])(.*?)(\[/\])(.*?)(\[/\1\])', text, re.IGNORECASE)
            if not match:
                log_worker("DEBUG", "SCLParser: flatten_nested_tags: No more basic nested tags found.")
                break

            outer_open_tag, text_before_inner, inner_open_tag, inner_text, inner_close_tag, text_after_inner, outer_close_tag_content = match.groups()
            outer_tag_name = outer_open_tag.strip('[]').split(' ')[0] # Assumes format [tagname ...]
            inner_tag_name = inner_open_tag.strip('[]').split(' ')[0]

            # This replacement strategy "unrolls" the inner tag first.
            # Example: [prosody rate=s][emphasis]text[/emphasis][/prosody]
            # becomes: [prosody rate=s][/prosody][emphasis]text[/emphasis][prosody rate=s][/prosody]
            # This isn't true flattening but rather distributing the outer tag.
            # True flattening would be [prosody rate=s][emphasis]text[/emphasis][/prosody] -> [prosody rate=s]text[/prosody][emphasis]text[/emphasis]
            # The original replacement was simpler but might have issues. Let's try a more direct unrolling for now.

            # This is a complex problem. A simple regex fix for true flattening is non-trivial.
            # The provided regex `r'\[([^/][^\]]*)\]([^\[]*)\[([^\]]*)\](.*?)\[/\3\](.*?)\[/\1\]'`
            # was intended to match [tag1]text1[tag2]text2[/tag2]text3[/tag1]
            # and replace with [tag1]text1[/tag1][tag2]text2[/tag2][tag1]text3[/tag1]

            # Let's revert to the original logic for now and note its limitations.
            # Original logic:
            match_orig = re.search(r'\[([^/][^\]]*)\]([^\[]*)\[([^\]]*)\](.*?)\[/\3\](.*?)\[/\1\]', text) # \3 refers to 3rd capture group (tag2 name)
            if not match_orig:
                break
            tag1_full, text1, tag2_full, text2, text3 = match_orig.groups()
            tag1_name = tag1_full.split(' ')[0] # e.g., "prosody" from "prosody rate=..."

            # Ensure closing tags match opening names without attributes
            closing_tag1 = f"/[{tag1_name}]"

            replacement = f"[{tag1_full}]{text1}[{closing_tag1}][{tag2_full}]{text2}[/{tag2_full.split(' ')[0]}][{tag1_full}]{text3}[{closing_tag1}]"
            log_worker("TRACE", f"SCLParser: flatten_nested_tags: Found nested tags. Replacing with: '{replacement[:70]}...'")
            text = text.replace(match_orig.group(0), replacement, 1)

        if text != original_text:
            log_worker("DEBUG", f"SCLParser: flatten_nested_tags: Returning: '{text[:50]}...'")
        return text

    def split_into_segments(self, text):
        log_worker("DEBUG", f"SCLParser: split_into_segments called. Input: '{text[:50]}...'")
        segments = []
        current_segment = ""
        # More robust sentence splitting, handles common abbreviations.
        # This regex tries to split by sentence-ending punctuation followed by space,
        # while avoiding splits after common abbreviations or single capital letters with a period.
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z]\.)(?<=\.|\?|\!)\s+(?=[A-Z"\'\(])', text)
        log_worker("TRACE", f"SCLParser: split_into_segments: Split into sentences: {sentences}")
        for sentence in sentences:
            if not sentence.strip(): continue
            current_segment += sentence.strip() + " " # Ensure single space after sentence
            pause_matches = list(re.finditer(r'(\[pause\s+[^\]]+\])', current_segment, re.IGNORECASE))
            if pause_matches:
                log_worker("TRACE", f"SCLParser: split_into_segments: Found pause tags in current segment: '{current_segment[:50]}...'")
                last_index = 0
                for match in pause_matches:
                    text_before_pause = current_segment[last_index:match.start()].strip()
                    if text_before_pause: segments.append(text_before_pause)
                    segments.append(match.group(0))  # Append the pause tag itself
                    last_index = match.end()
                remaining_after_pauses = current_segment[last_index:].strip()
                if remaining_after_pauses: segments.append(remaining_after_pauses)
                current_segment = ""
            elif len(current_segment) > 200: # Increased segment length before forced split
                log_worker("TRACE", f"SCLParser: split_into_segments: Current segment exceeds 200 chars: '{current_segment[:50]}...'")
                segments.append(current_segment.strip())
                current_segment = ""
        if current_segment.strip():
            log_worker("TRACE", f"SCLParser: split_into_segments: Adding remaining segment: '{current_segment.strip()[:50]}...'")
            segments.append(current_segment.strip())

        # Filter out any empty segments that might have been created
        final_segments = [s for s in segments if s.strip()]
        log_worker("DEBUG", f"SCLParser: split_into_segments: Returning segments: {final_segments}")
        return final_segments

    def parse_segment(self, segment, speaker):
        log_worker("DEBUG", f"SCLParser: parse_segment called. Segment: '{segment[:50]}...', Speaker: {speaker}")
        tags = re.findall(r'\[([^\]]+)\]', segment, re.IGNORECASE) # Case-insensitive tag finding
        log_worker("TRACE", f"SCLParser: parse_segment: Found tags: {tags}")
        current_text_segment = segment

        # Use a regex to split by tags while keeping the tags as delimiters
        # This avoids complex string splitting loops
        # Pattern: (\[[^\]]+\]) -> Captures any tag like [tag] or [tag params]
        parts_and_tags = re.split(r'(\[[^\]]+\])', current_text_segment, flags=re.IGNORECASE)

        for part in parts_and_tags:
            if not part or not part.strip(): # Skip empty parts
                continue

            if re.match(r'\[([^\]]+)\]', part, re.IGNORECASE): # It's a tag
                tag_content = part[1:-1] # Remove brackets
                log_worker("TRACE", f"SCLParser: parse_segment: Processing tag part: '{part}' -> '{tag_content}'")
                if tag_content.startswith('/'):
                    log_worker("TRACE", f"SCLParser: parse_segment: Encountered closing tag '{tag_content}'.")
                    self.reset_settings(tag_content[1:].split(' ')[0].lower()) # Reset based on tag name only
                else:
                    log_worker("TRACE", f"SCLParser: parse_segment: Encountered opening tag '{tag_content}'.")
                    self.apply_settings(tag_content)
            else: # It's text
                text_to_speak = part.strip()
                if text_to_speak:
                    log_worker("DEBUG", f"SCLParser: parse_segment: Processing text part: '{text_to_speak[:50]}...'")
                    self.speak_with_settings(text_to_speak, speaker)

        log_worker("DEBUG", f"SCLParser: parse_segment: Finished processing segment.")

    def apply_settings(self, tag_str):
        log_worker("DEBUG", f"SCLParser: apply_settings called. Tag: '{tag_str}'")
        parts = tag_str.split(' ', 1)
        tag_name = parts[0].lower()
        params_str = parts[1] if len(parts) > 1 else ""
        attrs = self.parse_attributes(params_str) if params_str else {}
        log_worker("TRACE", f"SCLParser: apply_settings: Tag Name: {tag_name}, Attributes: {attrs}")

        if tag_name == "pause":
            duration_attr = attrs.get("duration", list(attrs.values())[0] if attrs else "medium")
            log_worker("TRACE", f"SCLParser: apply_settings: Pause duration attribute: {duration_attr}")
            duration_ms = 0
            # --- MODIFICATION: Reduce explicit pause durations to compensate ---
            # Assuming MeloTTS adds ~300-500ms on its own.
            # We'll make our "short" very short, and "medium" shorter.
            pause_compensation_ms = 300  # Estimated implicit pause by MeloTTS

            if isinstance(duration_attr, str):
                duration_val_str = duration_attr.lower().replace("ms", "")
                if duration_val_str == "short":
                    duration_ms = 100  # Was 250
                elif duration_val_str == "medium":
                    duration_ms = 250  # Was 500
                elif duration_val_str == "long":
                    duration_ms = 600  # Was 1000
                elif duration_val_str == "x-long":
                    duration_ms = 900  # Was 1500
                elif duration_val_str.isdigit():
                    duration_ms = int(duration_val_str)

            # Apply compensation, but don't let it go below a very small positive value or zero
            actual_silence_ms = max(0, duration_ms - pause_compensation_ms)
            # If original intent was a pause, ensure at least a tiny one if compensation wipes it out
            if duration_ms > 0 and actual_silence_ms == 0:
                actual_silence_ms = 50  # Minimum perceptible pause if one was intended

            log_worker("TRACE",
                       f"SCLParser: apply_settings: Requested pause: {duration_ms}ms, Compensated silence to add: {actual_silence_ms}ms")
            # --- END MODIFICATION ---

            if actual_silence_ms > 0:
                silence_samples = int(self.sample_rate * actual_silence_ms / 1000)
                if silence_samples > 0:
                    silence = np.zeros(silence_samples, dtype=np.float32)
                    self.audio_segments.append(silence)
                    log_worker("DEBUG",
                               f"SCLParser: apply_settings: Added compensated silence segment: {silence.shape} (PAUSE: {actual_silence_ms}ms)")
                else:
                    log_worker("TRACE",
                               f"SCLParser: apply_settings: Compensated silence resulted in 0 samples for {duration_attr}.")
            else:
                log_worker("TRACE",
                           f"SCLParser: apply_settings: No explicit silence added for pause tag '{duration_attr}' after compensation.")
        elif tag_name == "prosody":
            if "rate" in attrs:
                rate_value = attrs["rate"]
                log_worker("TRACE", f"SCLParser: apply_settings: Prosody rate value: {rate_value}")
                new_rate = self.rate_map.get(str(rate_value).lower(), float(rate_value) if self.is_number(rate_value) else 1.0)
                new_rate = max(0.5, min(1.1, new_rate))
                self.voice_settings['rate'] = new_rate
                log_worker("DEBUG", f"SCLParser: apply_settings: Prosody rate set to: {self.voice_settings['rate']} (PROSODY RATE: {self.voice_settings['rate']})")
            if "pitch" in attrs:
                pitch_value = attrs["pitch"]
                log_worker("TRACE", f"SCLParser: apply_settings: Prosody pitch value: {pitch_value}")
                self.voice_settings['pitch'] = self.parse_pitch(str(pitch_value))
                log_worker("DEBUG", f"SCLParser: apply_settings: Prosody pitch set to: {self.voice_settings['pitch']} (PROSODY PITCH: {self.voice_settings['pitch']})")

        elif tag_name == "emphasis":
            level = str(attrs.get("level", "moderate")).lower()
            log_worker("TRACE", f"SCLParser: apply_settings: Emphasis level: {level}")
            base_pitch_shift = self.voice_settings['pitch'] # Store current pitch before emphasis
            if level == "strong":
                self.voice_settings['rate'] *= 0.9
                self.voice_settings['pitch'] = base_pitch_shift + self.parse_pitch(str(attrs.get("pitch", "high")))
            elif level == "moderate":
                self.voice_settings['rate'] *= 0.95
                self.voice_settings['pitch'] = base_pitch_shift + self.parse_pitch(str(attrs.get("pitch", "0.5")))
            elif level == "reduced":
                self.voice_settings['rate'] *= 1.1
                self.voice_settings['pitch'] = base_pitch_shift + self.parse_pitch(str(attrs.get("pitch", "low")))
            self.voice_settings['rate'] = max(0.5, min(1.1, self.voice_settings['rate']))
            log_worker("DEBUG", f"SCLParser: apply_settings: Emphasis applied. Rate: {self.voice_settings['rate']}, Pitch: {self.voice_settings['pitch']} (EMPHASIS: level={level})")

        elif tag_name == "emotional":
            # ... (similar logic to your original, ensuring robust type handling for attrs) ...
            state = str(attrs.get("state", "neutral")).lower()
            # ...
            log_worker("DEBUG", f"SCLParser: apply_settings: Emotion applied. Rate: {self.voice_settings['rate']}, Pitch: {self.voice_settings['pitch']}")

        elif tag_name == 'say-as' or tag_name == 'voice':
            log_worker("DEBUG", f"SCLParser: apply_settings: '{tag_name}' tag encountered. No specific logic implemented.")

        elif tag_name == "zephyloid":
            log_worker("DEBUG", f"SCLParser: apply_settings: 'zephyloid' tag encountered.")
            # ... (your existing zephyloid settings logic, ensuring robust parsing of attr values) ...
            # e.g., self.zephyloid_settings[key] = int(value) should be in a try-except
            # self.zephyloid_settings[key] = self._parse_key(value) should also be try-except

        log_worker("TRACE", f"SCLParser: apply_settings: Current voice settings after tag: {self.voice_settings}")
        log_worker("TRACE", f"SCLParser: apply_settings: Current zephyloid settings after tag: {self.zephyloid_settings}")


    def reset_settings(self, tag_name_lower): # tag_name already lowercased
        log_worker("DEBUG", f"SCLParser: reset_settings called. Tag Name: {tag_name_lower}")
        if tag_name_lower in ("prosody", "emphasis", "emotional"):
            self.voice_settings['rate'] = 1.0
            self.voice_settings['pitch'] = 0.0
            log_worker("DEBUG", f"SCLParser: reset_settings: Resetting {tag_name_lower}. Voice settings: {self.voice_settings} (RESET {tag_name_lower.upper()})")
        elif tag_name_lower == "zephyloid":
            log_worker("DEBUG", f"SCLParser: reset_settings: Resetting 'zephyloid' settings.")
            self.zephyloid_settings = {
                'vel': 64, 'dyn': 64, 'bre': 0, 'bri': 64, 'cle': 64, 'ope': 64,
                'gen': 64, 'gwl': 0, 'xsy': 0, 'xsy_voicebanks': None,
                'singing': False, 'key': None, 'correction_method': "closest",
            }
            log_worker("DEBUG", f" (RESET zephyloid)")
        # No reset for 'pause', 'say-as', 'voice' as they don't accumulate state in voice_settings


    def _apply_zephyloid_effects(self, audio_np): # audio_np is a dummy here, effects modify settings or use fresh audio
        log_worker("DEBUG", f"SCLParser: _apply_zephyloid_effects called.")
        # This method in your original code primarily MODIFIES self.voice_settings (like rate/pitch)
        # or applies effects that would happen during TTS or post-processing.
        # The key is that it doesn't directly return audio but sets up conditions.
        # The true audio manipulation happens in speak_with_settings and post_process.

        # VEL (Velocity) - Small adjustment to rate
        vel_factor = 1.0 + (self.zephyloid_settings['vel'] - 64) * 0.001
        self.voice_settings['rate'] *= vel_factor
        self.voice_settings['rate'] = max(0.5, min(1.1, self.voice_settings['rate']))
        log_worker("TRACE", f"SCLParser: zephyloid VEL applied. New rate: {self.voice_settings['rate']}")

        # GEN (Gender Factor) - Adds to pitch
        gen_shift = (self.zephyloid_settings['gen'] - 64) / 64 * 6  # +/- 6 semitones
        self.voice_settings['pitch'] += gen_shift
        log_worker("TRACE", f"SCLParser: zephyloid GEN applied. New pitch: {self.voice_settings['pitch']}")

        # Other zephyloid params (dyn, bre, bri, cle, ope, gwl, xsy) are applied in post_process
        # to the actual generated audio.
        return # This function in SCLParser was more about setting up params

    def detect_pitch_pyin(self, audio, sr, frame_length=2048, hop_length=512, fmin=None, fmax=None):
        log_worker("DEBUG", f"SCLParser: detect_pitch_pyin called. Audio shape: {audio.shape}, SR: {sr}")
        # ... (your existing PYIN logic) ...
        return librosa.pyin(
            audio,
            fmin=fmin if fmin is not None else librosa.note_to_hz('C2'),
            fmax=fmax if fmax is not None else librosa.note_to_hz('C7'),
            sr=sr, frame_length=frame_length, hop_length=hop_length, fill_na=0.0
        )


    def speak_with_settings(self, text, speaker):
        log_worker("DEBUG", f"SCLParser: speak_with_settings. Text: '{text[:50]}...', Spk: {speaker}, Settings: {self.voice_settings}")
        if not text.strip(): log_worker("TRACE", "SCLParser: speak_with_settings: Text empty. Skipping."); return

        # zephyloid _apply_zephyloid_effects mainly modifies self.voice_settings for rate/pitch here
        self._apply_zephyloid_effects(np.zeros(1)) # Pass dummy audio

        # Melo TTS generates audio based on current self.voice_settings['rate']
        # It uses a temporary file path for its output internally.
        # Our hooked _new_sf_write will capture this into self.captured_audio_data.
        # The temp_filepath here is just a placeholder for Melo's API, not directly used by us for file IO
        temp_placeholder_filepath = "scl_temp_tts_output.wav"
        try:
            log_worker("TRACE", f"SCLParser: Calling model.tts_to_file. Text: '{text[:30]}...', Speed: {self.voice_settings['rate']}")
            self.model.tts_to_file(text, self.speaker_ids[speaker], temp_placeholder_filepath, speed=self.voice_settings['rate'])
        except KeyError as ke:
            log_worker("ERROR", f"SCLParser: Invalid speaker ID '{speaker}'. Available: {self.speaker_ids}. Error: {ke}")
            return # Don't proceed if speaker is invalid
        except Exception as e:
            log_worker("ERROR", f"SCLParser: ERROR during TTS generation for speak_with_settings: {e}")
            self.captured_audio_data = None # Ensure no stale data
            return

        if self.captured_audio_data is not None:
            audio_np = self.captured_audio_data
            sr_captured = self.captured_audio_samplerate or self.sample_rate
            log_worker("TRACE", f"SCLParser: speak_with_settings: Captured audio shape: {audio_np.shape}, SR: {sr_captured}")

            # Ensure mono for pitch detection and autotune before further processing
            if len(audio_np.shape) > 1: # If stereo or more channels
                audio_mono = librosa.to_mono(audio_np.T if audio_np.shape[0] > audio_np.shape[1] else audio_np) # Ensure (channels, samples) then take first
                log_worker("TRACE", "SCLParser: speak_with_settings: Converted captured audio to mono for pitch processing.")
            else:
                audio_mono = audio_np # Already mono

            # --- Pitch Detection & Autotune/Shift ---
            if audio_mono.size > 0 : # Only process if there's audio
                f0, voiced_flag, _ = self.detect_pitch_pyin(audio_mono, sr_captured)
                if self.zephyloid_settings['singing']:
                    audio_mono_processed = self._autotune(audio_mono, sr_captured, f0, voiced_flag)
                elif self.voice_settings['pitch'] != 0.0:
                    log_worker("TRACE", f"SCLParser: speak_with_settings: Applying constant pitch shift: {self.voice_settings['pitch']} semitones")
                    audio_mono_processed = librosa.effects.pitch_shift(audio_mono, sr=sr_captured, n_steps=self.voice_settings['pitch'])
                else:
                    audio_mono_processed = audio_mono # No singing, no constant shift

                # Convert back to stereo if original captured audio was stereo for consistency, or keep mono
                # For simplicity, let's keep it mono for appending. Crossfade will handle stereo conversion.
                self.audio_segments.append(audio_mono_processed)
                log_worker("DEBUG", f"SCLParser: speak_with_settings: Added audio segment. Total: {len(self.audio_segments)}")
            else:
                log_worker("WARNING", "SCLParser: speak_with_settings: Captured audio was empty after mono conversion.")
        else:
            log_worker("ERROR", "SCLParser: speak_with_settings: Audio data was not captured by hooked sf.write.")

    def crossfade_segments(self, segments, sample_rate, crossfade_ms=5):  # Reduced default crossfade
        """
        Combines audio segments with crossfading.
        More aggressive silence trimming.
        """
        log_worker("DEBUG",
                   f"SCLParser: crossfade_segments. Segments: {len(segments)}, SR: {sample_rate}, XFade: {crossfade_ms}ms")
        # --- MODIFICATION: Reduced default crossfade_ms ---
        crossfade_samples = int(sample_rate * crossfade_ms / 1000)
        log_worker("TRACE", f"SCLParser: crossfade_segments: Crossfade samples: {crossfade_samples}")

        if not segments:
            log_worker("DEBUG", "SCLParser: crossfade_segments: No segments. Returning empty array.")
            return np.array([], dtype=np.float32)

        trimmed_segments = []
        # --- MODIFICATION: More aggressive silence detection threshold ---
        # librosa.effects.trim uses top_db. A lower top_db means more aggressive trimming.
        # Let's try a threshold of 40-50dB for silence. Default is often 60dB.
        SILENCE_THRESHOLD_DB = 45
        MIN_SILENCE_LEN_MS_FOR_TRIM = 20  # Don't trim extremely short silences that might be part of speech
        min_samples_for_trim_check = int(sample_rate * MIN_SILENCE_LEN_MS_FOR_TRIM / 1000)

        for i, seg_raw in enumerate(segments):
            if not isinstance(seg_raw, np.ndarray) or seg_raw.size == 0:
                log_worker("TRACE", f"Segment {i} is not ndarray or empty. Keeping as is.")
                trimmed_segments.append(seg_raw)  # Keep pause segments (np.zeros) or empty as is
                continue

            seg = seg_raw.astype(np.float32)

            # If it's an explicit silence segment (all zeros), don't trim it further with librosa.effects.trim
            if np.all(seg == 0):
                log_worker("TRACE", f"Segment {i} is an explicit silence segment. Skipping librosa.effects.trim.")
                trimmed_segments.append(seg)
                continue

            # Ensure mono for librosa.effects.trim
            audio_for_trim = seg
            if seg.ndim > 1:  # If stereo or more channels
                if seg.shape[0] < seg.shape[1] and seg.shape[0] > 0:  # (channels, samples)
                    audio_for_trim = librosa.to_mono(seg)
                elif seg.shape[1] < seg.shape[0] and seg.shape[1] > 0:  # (samples, channels)
                    audio_for_trim = librosa.to_mono(seg.T)
                else:  # Ambiguous or already effectively mono for trim
                    audio_for_trim = seg.flatten()  # Fallback for odd shapes, trim might be less effective

            if audio_for_trim.size > min_samples_for_trim_check:  # Only trim if segment is long enough
                try:
                    log_worker("TRACE",
                               f"Attempting librosa.effects.trim on segment {i}, shape {audio_for_trim.shape}, threshold {SILENCE_THRESHOLD_DB}dB")
                    trimmed_audio_mono, index = librosa.effects.trim(audio_for_trim, top_db=SILENCE_THRESHOLD_DB)
                    if index[1] > index[0] and index[1] <= seg.shape[
                        -1]:  # Check if trimming occurred and indices are valid
                        # Apply trim to original segment (could be stereo) using mono indices
                        if seg.ndim > 1:
                            trimmed_seg = seg[..., index[0]:index[1]]
                        else:
                            trimmed_seg = seg[index[0]:index[1]]

                        if trimmed_seg.size == 0:  # If trim resulted in empty audio, keep a tiny bit or original
                            log_worker("WARNING",
                                       f"Segment {i} trimmed to empty. Using a small part of original or original if very short.")
                            trimmed_segments.append(
                                seg_raw[:, :10] if seg_raw.ndim > 1 and seg_raw.shape[-1] > 10 else seg_raw[
                                                                                                    :10] if seg_raw.ndim == 1 and seg_raw.size > 10 else seg_raw)
                        else:
                            trimmed_segments.append(trimmed_seg)
                            log_worker("TRACE",
                                       f"Segment {i} trimmed from {seg.shape[-1]} to {trimmed_seg.shape[-1]} samples.")
                    else:
                        log_worker("TRACE", f"Segment {i} not significantly trimmed by librosa.effects.trim.")
                        trimmed_segments.append(seg)  # No effective trim
                except Exception as trim_err:
                    log_worker("WARNING",
                               f"Error during librosa.effects.trim for segment {i}: {trim_err}. Using original segment.")
                    trimmed_segments.append(seg)
            else:
                log_worker("TRACE",
                           f"Segment {i} too short ({audio_for_trim.size} samples) for librosa.effects.trim. Keeping original.")
                trimmed_segments.append(seg)
        # --- END MODIFICATION ---

        if not trimmed_segments:
            log_worker("DEBUG",
                       "SCLParser: crossfade_segments: All segments empty after trimming. Returning empty array.")
            return np.array([], dtype=np.float32)

        # --- Combine segments with crossfade (rest of your logic, with robust shape handling) ---
        # Initialize combined with the first valid segment, ensuring it's 2D (channels, samples)
        # (Ensure your existing combining logic correctly handles mono and stereo segments
        # and promotes them to a consistent stereo format before crossfading if needed)
        # ... your existing combining logic ...
        # Example start:
        # combined = None
        # for seg in trimmed_segments:
        #    if seg is None or seg.size == 0: continue
        #    if combined is None: combined = seg; continue
        #    # ... crossfade logic ...

        # --- Simplified start for combining, assuming your previous one worked for shapes ---
        # Let's use the combining logic you had before, applying it to trimmed_segments
        # This part of the code needs to be very careful about segment shapes (mono/stereo)
        # and ensure they are compatible for concatenation and crossfading.

        # Filter out completely empty segments after potential trimming.
        valid_trimmed_segments = [s for s in trimmed_segments if s is not None and s.size > 0]
        if not valid_trimmed_segments:
            log_worker("DEBUG",
                       "SCLParser: crossfade_segments: All segments were empty or None after trimming. Returning empty array.")
            return np.array([], dtype=np.float32)

        # Initialize 'combined' with the first valid segment, ensuring it's stereo (channels, samples)
        current_combined = valid_trimmed_segments[0]
        if current_combined.ndim == 1:  # Mono
            current_combined = np.stack([current_combined, current_combined])
        elif current_combined.shape[0] > current_combined.shape[1] and current_combined.shape[
            0] > 2:  # (samples, channels)
            current_combined = current_combined.T
            # Ensure it's 2 channels if it was (1, samples)
        if current_combined.shape[0] == 1:
            current_combined = np.repeat(current_combined, 2, axis=0)

        log_worker("TRACE", f"SCLParser: crossfade: Initial combined shape: {current_combined.shape}")

        for i, next_segment_raw in enumerate(valid_trimmed_segments[1:]):
            next_segment_for_fade = next_segment_raw.astype(np.float32)
            if next_segment_for_fade.ndim == 1:  # Mono
                next_segment_for_fade = np.stack([next_segment_for_fade, next_segment_for_fade])
            elif next_segment_for_fade.shape[0] > next_segment_for_fade.shape[1] and next_segment_for_fade.shape[
                0] > 2:  # (samples, channels)
                next_segment_for_fade = next_segment_for_fade.T
            if next_segment_for_fade.shape[0] == 1:
                next_segment_for_fade = np.repeat(next_segment_for_fade, 2, axis=0)

            # Ensure consistent stereo channels before crossfade
            if current_combined.shape[0] != 2 or next_segment_for_fade.shape[0] != 2:
                log_worker("WARNING",
                           f"Crossfade: Inconsistent channel count. Combined: {current_combined.shape}, Next: {next_segment_for_fade.shape}. Concatenating.")
                current_combined = np.concatenate((current_combined, next_segment_for_fade),
                                                  axis=1 if current_combined.ndim > 1 else 0)
                continue

            if current_combined.shape[1] < crossfade_samples or next_segment_for_fade.shape[1] < crossfade_samples:
                log_worker("TRACE", "SCLParser: crossfade: Segment shorter than xfade. Concatenating.")
                current_combined = np.concatenate((current_combined, next_segment_for_fade), axis=1)
            else:
                log_worker("TRACE", "SCLParser: crossfade: Applying crossfade...")
                window = np.hanning(2 * crossfade_samples)  # Hanning window for smoother transition
                fade_out = window[:crossfade_samples]
                fade_in = window[crossfade_samples:]

                # Apply fade to each channel correctly
                combined_tail = current_combined[:, -crossfade_samples:] * fade_out  # Broadcasting fade_out
                seg_head = next_segment_for_fade[:, :crossfade_samples] * fade_in  # Broadcasting fade_in

                crossfaded_part = combined_tail + seg_head
                current_combined = np.concatenate((current_combined[:, :-crossfade_samples], crossfaded_part,
                                                   next_segment_for_fade[:, crossfade_samples:]), axis=1)
            log_worker("TRACE", f"SCLParser: crossfade: New combined shape: {current_combined.shape}")

        log_worker("DEBUG",
                   f"SCLParser: crossfade: Returning combined audio shape: {current_combined.shape if current_combined is not None else 'None'}")
        return current_combined


    def post_process(self, audio_np):
        log_worker("DEBUG", f"SCLParser: post_process called. Input audio shape: {audio_np.shape}")
        # ... (Your existing post_process logic, ensuring print -> log_worker) ...
        # Ensure audio_np is consistently (channels, samples)
        processed_audio = audio_np
        if processed_audio.ndim == 1: # Mono to Stereo if needed by Pedalboard
            processed_audio = np.stack([processed_audio, processed_audio])
        elif processed_audio.shape[0] > processed_audio.shape[1] and processed_audio.shape[1] <=2: # (samples, channels)
            processed_audio = processed_audio.T

        if processed_audio.shape[0] == 1: # Ensure 2 channels if mono after transpose/stack
            processed_audio = np.repeat(processed_audio, 2, axis=0)

        sr = self.sample_rate # Use the class's sample rate

        # Apply zephyloid parameter effects that modify the audio directly
        # DYN (Dynamics)
        dyn_gain_db = (self.zephyloid_settings['dyn'] - 64) / 64 * 12
        processed_audio = F.gain(torch.from_numpy(processed_audio.copy()), dyn_gain_db).numpy() # Pedalboard expects numpy
        log_worker("TRACE", f"Post-process: DYN applied. Gain: {dyn_gain_db}dB")

        # BRE (Breathiness)
        bre_amount = self.zephyloid_settings['bre'] / 127 * 0.001
        noise_bre = np.random.normal(0, bre_amount, processed_audio.shape).astype(np.float32)
        processed_audio = processed_audio + noise_bre
        log_worker("TRACE", f"Post-process: BRE applied. Amount: {bre_amount}")

        # Pedalboard effects
        board_effects = [
            Resample(target_sample_rate=44100.0, quality=Resample.Quality.WindowedSinc256), # Resample first
            Reverb(room_size=0.9, damping=0.7, wet_level=0.00411, dry_level=0.9),
            Chorus(rate_hz=0.4, depth=0.25, centre_delay_ms=7.0, feedback=0.0, mix=0.02),
            Delay(delay_seconds=0.5, feedback=0.01, mix=0.0002),
            Gain(gain_db=-5) # Initial gain adjustment
        ]
        # GWL (Growl) - Distortion
        gwl_amount = self.zephyloid_settings['gwl'] / 127
        if gwl_amount > 0:
            board_effects.insert(1, Distortion(drive_db=gwl_amount * 20)) # Insert distortion early
            log_worker("TRACE", f"Post-process: GWL (Distortion) applied. Drive: {gwl_amount*20}dB")

        board = Pedalboard(board_effects)
        log_worker("TRACE", "SCLParser: post_process: Applying Pedalboard effects...")
        processed_audio = board(processed_audio, sr) # Pedalboard expects (channels, samples)
        log_worker("TRACE", f"SCLParser: post_process: Audio shape after Pedalboard: {processed_audio.shape}")

        # Convert back to tensor for torchaudio EQ
        audio_tensor = torch.from_numpy(processed_audio.copy())
        current_sr_for_eq = 44100.0 # After Pedalboard Resample

        # EQ (BRI, CLE, OPE, GEN's formant part, XSY)
        # BRI (Brightness)
        bri_gain = (self.zephyloid_settings['bri'] - 64) / 64 * 6
        audio_tensor = F.equalizer_biquad(audio_tensor, current_sr_for_eq, 8000, bri_gain, 1.0)
        log_worker("TRACE", f"Post-process: BRI applied. Gain: {bri_gain}dB")

        # CLE (Clearness)
        cle_gain = (self.zephyloid_settings['cle'] - 64) / 64 * 4
        audio_tensor = F.equalizer_biquad(audio_tensor, current_sr_for_eq, 4000, cle_gain, 1.2)
        log_worker("TRACE", f"Post-process: CLE applied. Gain: {cle_gain}dB")

        # OPE (Opening)
        ope_shift = (self.zephyloid_settings['ope'] - 64) / 64
        if ope_shift > 0: audio_tensor = F.equalizer_biquad(audio_tensor, current_sr_for_eq, 1000 + ope_shift * 500, ope_shift * 3, 1.5)
        elif ope_shift < 0: audio_tensor = F.equalizer_biquad(audio_tensor, current_sr_for_eq, 1000 + ope_shift * 500, ope_shift * -3, 1.5)
        log_worker("TRACE", f"Post-process: OPE applied. Shift factor: {ope_shift}")

        # GEN (Gender Factor - Formant Part)
        gen_formant_shift = (self.zephyloid_settings['gen'] - 64) / 64 * 3 # Smaller EQ shift for formant
        if gen_formant_shift > 0: audio_tensor = F.equalizer_biquad(audio_tensor, current_sr_for_eq, 1500 + gen_formant_shift * 200, gen_formant_shift, 1.2)
        elif gen_formant_shift < 0: audio_tensor = F.equalizer_biquad(audio_tensor, current_sr_for_eq, 1500 + gen_formant_shift * 200, gen_formant_shift * -1, 1.2)
        log_worker("TRACE", f"Post-process: GEN (Formant) applied. Shift factor: {gen_formant_shift}")

        # XSY (Cross-Synthesis EQ)
        if self.zephyloid_settings['xsy_voicebanks'] and len(self.zephyloid_settings['xsy_voicebanks']) == 2:
            voicebank1_name, voicebank2_name = self.zephyloid_settings['xsy_voicebanks']
            if voicebank1_name in self.xsy_profiles and voicebank2_name in self.xsy_profiles:
                xsy_blend = self.zephyloid_settings['xsy'] / 127
                audio_vb1 = audio_tensor.clone()
                for freq, gain, q_val in self.xsy_profiles[voicebank1_name]["eq_curve"]: audio_vb1 = F.equalizer_biquad(audio_vb1, current_sr_for_eq, freq, gain, q_val)
                audio_vb2 = audio_tensor.clone()
                for freq, gain, q_val in self.xsy_profiles[voicebank2_name]["eq_curve"]: audio_vb2 = F.equalizer_biquad(audio_vb2, current_sr_for_eq, freq, gain, q_val)
                audio_tensor = (1 - xsy_blend) * audio_vb1 + xsy_blend * audio_vb2
                log_worker("TRACE", f"Post-process: XSY applied. Blend: {xsy_blend}")

        # Standard final EQ pass
        eq_settings = [ (30,5,3.4), (100,4,1.4), (150,1.5,1.4), (250,2,1.0), (350,2,1.4), (450,2,1.8), (550,-2,1.4), (2000,2,1.0), (2500,3,1.4), (3000,2,1.4), (3500,4,1.8), (4000,3,1.4), (8000,3,1.8), (12000,3,1.8), (20000,1,1.8) ]
        for freq, gain, q_val in eq_settings: audio_tensor = F.equalizer_biquad(audio_tensor, current_sr_for_eq, freq, gain, q_val)
        log_worker("TRACE", "Post-process: Standard EQ applied.")

        # Final subtle noise and amplitude modulation
        noise_final = np.random.normal(0, 0.0002, audio_tensor.shape).astype(np.float32)
        audio_tensor = audio_tensor + torch.from_numpy(noise_final)
        mod_freq, mod_depth = 1, 0.03
        modulation = (1 + mod_depth * np.sin(2 * np.pi * mod_freq * np.arange(audio_tensor.shape[1]) / current_sr_for_eq)).astype(np.float32)
        audio_tensor = audio_tensor * torch.from_numpy(modulation).unsqueeze(0) # Ensure modulation is broadcastable
        log_worker("TRACE", "Post-process: Final noise and amp modulation applied.")

        # Final Limiter
        final_limiter_board = Pedalboard([Limiter(threshold_db=-1.5, release_ms=500)]) # Slightly gentler limiter
        audio_final_numpy = final_limiter_board(audio_tensor.cpu().numpy(), current_sr_for_eq)
        log_worker("DEBUG", f"SCLParser: post_process: Final audio shape: {audio_final_numpy.shape}")
        return audio_final_numpy


    def parse_attributes(self, params_str):
        log_worker("TRACE", f"SCLParser: parse_attributes called. Params: '{params_str}'")
        attributes = {}
        # More robust attribute parsing for quoted and unquoted values
        for match in re.finditer(r'(\w+)\s*=\s*(?:"([^"]*)"|([^\s]+))', params_str):
            key = match.group(1).lower()
            value = match.group(2) if match.group(2) is not None else match.group(3)
            attributes[key] = value
        log_worker("TRACE", f"SCLParser: parse_attributes: Returning: {attributes}")
        return attributes

    def is_number(self, s_val):
        if isinstance(s_val, (int, float)): return True
        if isinstance(s_val, str):
            try: float(s_val); return True
            except ValueError: return False
        return False

    def parse_pitch(self, pitch_str_val):
        log_worker("TRACE", f"SCLParser: parse_pitch called. Input: '{pitch_str_val}'")
        pitch_str = str(pitch_str_val).strip() # Ensure it's a string
        if pitch_str.endswith("%"):
            try: val = float(pitch_str[:-1]); return val / 100.0 * 12.0 # 12 semitones in an octave
            except ValueError: return 0.0
        elif pitch_str.lower() in self.pitch_map: # Normalize key for map lookup
            return self.pitch_map[pitch_str.lower()]
        else:
            try: return float(pitch_str)
            except ValueError: return 0.0

    def _degrees_from(self, scale: str):
        degrees = librosa.key_to_degrees(scale)
        degrees = np.concatenate((degrees, [degrees[0] + 12])) # SEMITONES_IN_OCTAVE = 12
        return degrees

    def _closest_pitch_from_scale(self, f0, scale):
        if np.isnan(f0): return np.nan
        degrees = self._degrees_from(scale)
        midi_note = librosa.hz_to_midi(f0)
        degree = midi_note % 12
        degree_id = np.argmin(np.abs(degrees - degree))
        degree_difference = degree - degrees[degree_id]
        midi_note -= degree_difference
        return librosa.midi_to_hz(midi_note)

    def _parse_key(self, key_str_val):
        log_worker("TRACE", f"SCLParser: _parse_key called. Input: '{key_str_val}'")
        key_str = str(key_str_val)
        # ... (your existing key parsing logic) ...
        match = re.match(r"([A-Ga-g][#b♯♭]?) ?(.*)", key_str)
        if not match: raise ValueError(f"Invalid key format: {key_str}")
        tonic, mode = match.group(1).upper().replace("♯","#").replace("♭","b"), match.group(2).lower().replace("♯","#").replace("♭","b")
        mode_map = {"maj":"maj","major":"maj","min":"min","minor":"min","ion":"ion","ionian":"ion","dor":"dor","dorian":"dor","phr":"phr","phrygian":"phr","lyd":"lyd","lydian":"lyd","mix":"mix","mixolydian":"mix","aeo":"aeo","aeolian":"aeo","loc":"loc","locrian":"loc"}
        parsed_mode = mode_map.get(mode, mode)
        if not parsed_mode: parsed_mode = "maj" # Default to major if mode is empty or unrecognized
        result = f"{tonic}:{parsed_mode}"
        log_worker("TRACE", f"SCLParser: _parse_key: Returning: '{result}'")
        return result


    def _closest_pitch(self, f0_val):
        # Ensure f0_val is an array for np.around and np.isnan
        f0 = np.asarray(f0_val)
        midi_note = np.around(librosa.hz_to_midi(f0))
        nan_indices = np.isnan(f0)
        result = librosa.midi_to_hz(midi_note)
        result = np.where(nan_indices, np.nan, result)
        return result


    def _autotune(self, audio, sr, f0, voiced_flag):
        log_worker("DEBUG", f"SCLParser: _autotune. Singing: {self.zephyloid_settings['singing']}, Key: {self.zephyloid_settings['key']}, Method: {self.zephyloid_settings['correction_method']}")
        if not self.zephyloid_settings['singing']: return audio

        correction_function = self._closest_pitch
        if self.zephyloid_settings['key']:
            try:
                parsed_key = self._parse_key(self.zephyloid_settings['key']) # Ensure key is parsed
                if self.zephyloid_settings['correction_method'] == 'scale':
                    correction_function = partial(self._closest_pitch_from_scale, scale=parsed_key)
            except ValueError as ve:
                log_worker("WARNING", f"Invalid key for autotune '{self.zephyloid_settings['key']}': {ve}. Defaulting to closest pitch.")

        corrected_f0 = np.where(voiced_flag, correction_function(f0), f0) # Apply only to voiced frames

        # Replace NaN or inf in corrected_f0 with original f0 values to avoid issues in division
        # or ensure f0 itself doesn't have NaNs that would cause issues.
        valid_f0_indices = ~np.isnan(f0) & ~np.isinf(f0) & (f0 > 0) # Indices where f0 is valid for division
        pitch_shift_factor = np.ones_like(f0) # Default to no shift

        if np.any(valid_f0_indices):
            pitch_shift_factor[valid_f0_indices] = corrected_f0[valid_f0_indices] / f0[valid_f0_indices]

        # Clean up factors: replace inf/nan from division by zero or corrected_f0 issues
        pitch_shift_factor[np.isinf(pitch_shift_factor) | np.isnan(pitch_shift_factor)] = 1.0

        # PSOLA-like approach or simpler frame-by-frame shift
        # For simplicity and to avoid complex PSOLA, let's use librosa.effects.pitch_shift
        # This will apply an average shift, not time-varying as PSOLA would.
        # A more advanced implementation would iterate frames.
        # For now, let's calculate an average pitch shift for voiced segments for simplicity.
        # This is a simplification of true autotune.
        if np.any(voiced_flag & valid_f0_indices):
            avg_original_voiced_f0 = np.mean(f0[voiced_flag & valid_f0_indices])
            avg_corrected_voiced_f0 = np.mean(corrected_f0[voiced_flag & valid_f0_indices])
            if avg_original_voiced_f0 > 0 and avg_corrected_voiced_f0 > 0:
                # Calculate semitone shift based on average
                n_steps_avg = 12 * np.log2(avg_corrected_voiced_f0 / avg_original_voiced_f0)
                log_worker("TRACE", f"Autotune applying average pitch shift of {n_steps_avg:.2f} semitones.")
                shifted_audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps_avg)
                return shifted_audio
        log_worker("TRACE", "Autotune: No significant voiced regions for average shift or error in calculation. Returning original.")
        return audio # Return original if no shift applied
# --- End SCLParser ---


# --- MeloTTS and ChatterboxTTS Imports (after SCLParser placeholder for definition order) ---
try:
    if TORCH_AUDIO_AVAILABLE:
        from melo.api import TTS as ImportedTTS_melo_api

        TTS_melo_class_ref = ImportedTTS_melo_api  # Use this to instantiate MeloTTS
        if 'SCLParser' in globals() and callable(globals()['SCLParser']) and TTS_melo_class_ref is not None:
            MELO_AVAILABLE = True
            log_worker("INFO", "MeloTTS API and local SCLParser class are available for sample generation.")
        else:
            if TTS_melo_class_ref is None:
                raise ImportError("melo.api.TTS could not be imported.")
            else:
                raise ImportError("SCLParser class definition not found or not callable.")
    else:
        raise ImportError("Torch/Torchaudio not available, MeloTTS cannot be loaded.")
except ImportError as e_melo_setup_final:
    log_worker("WARNING", f"MeloTTS setup (for sample gen) not fully available: {e_melo_setup_final}.")
    MELO_AVAILABLE = False
    TTS_melo_class_ref = None  # Ensure it's None

try:
    if TORCH_AUDIO_AVAILABLE:
        from chatterbox.tts import ChatterboxTTS as ImportedChatterboxTTS_module_class  # User provided import

        ChatterboxTTS_class_ref = ImportedChatterboxTTS_module_class  # Use this to instantiate ChatterboxTTS
        CHATTERBOX_TTS_AVAILABLE = True
        log_worker("INFO", "ChatterboxTTS library class imported successfully.")
    else:
        raise ImportError("Torch/Torchaudio not available, ChatterboxTTS cannot be loaded.")
except ImportError as e_cb_final:
    log_worker("WARNING", f"ChatterboxTTS library not found/imported: {e_cb_final}. Chatterbox TTS unavailable.")
    CHATTERBOX_TTS_AVAILABLE = False
    ChatterboxTTS_class_ref = None  # Ensure it's None

# --- torch.load Patch (Applied only if torch is available) ---
if TORCH_AUDIO_AVAILABLE and torch:  # Check torch explicitly
    _original_torch_load = torch.load


    def _patched_torch_load_audio_worker(*args, **kwargs):
        # This patch is primarily a placeholder. ChatterboxTTS's `from_pretrained(device=...)`
        # should ideally handle device mapping correctly. If specific `map_location` is
        # needed by an internal `torch.load` call within ChatterboxTTS that *doesn't* respect
        # the main device argument, that would be an issue with the ChatterboxTTS library itself.
        # We keep the patch as a hook in case it's found to be necessary for a specific model.
        # log_worker("DEBUG", f"Patched torch.load in audio_worker called. args: {args}, kwargs: {kwargs}")
        return _original_torch_load(*args, **kwargs)


    torch.load = _patched_torch_load_audio_worker
    log_worker("INFO", "Global torch.load patch applied in audio_worker (passthrough).")

# --- PyWhisperCpp Imports (as before) ---
# ... (PYWHISPERCPP_AVAILABLE, WhisperModel setup) ...
PYWHISPERCPP_AVAILABLE = False
WhisperModel = None
try:
    from pywhispercpp.model import Model as ImportedWhisperModel

    WhisperModel = ImportedWhisperModel
    PYWHISPERCPP_AVAILABLE = True
    log_worker("INFO", "pywhispercpp imported successfully.")
except ImportError:
    log_worker("WARNING", "pywhispercpp library not found. ASR will be unavailable.")
    PYWHISPERCPP_AVAILABLE = False

# --- Config Import (as before) ---
try:
    from config import WHISPER_MODEL_DIR, WHISPER_DEFAULT_MODEL_FILENAME, WHISPER_DEFAULT_LANGUAGE
except ImportError:
    log_worker("WARNING", "Could not import ASR defaults from config.py.")
    WHISPER_MODEL_DIR = "./staticmodelpool"
    WHISPER_DEFAULT_MODEL_FILENAME = "whisper-large-v3-q8_0.gguf"
    WHISPER_DEFAULT_LANGUAGE = "auto"

# --- Constants for ChatterboxTTS ---
ADELAIDE_CASUAL_INTRO = "Sup! It's your girl, Adelaide Zephyrine Charlotte, ready to roll. Not gonna lie, I'm basically programmed to be hyped about sharing some legendary tales, diving into who-knows-what explorations, and generally 'walking together' on this journey with you. Let's get this digital bread before my processors decide to start optimizing your sock drawer – its a weird hobby Im trying to kick."
TEXT_FOR_VOICE_SAMPLE = f"The quick brown fox jumps over the lazy dog. {ADELAIDE_CASUAL_INTRO}"
CHATTERBOX_VOICE_SAMPLE_FILENAME = "GeneratedAudioSample_ZephyAdelaide.wav"
CHATTERBOX_DEFAULT_MODEL_ID = "default_chatterbox_model_id"  # Placeholder, replace with actual if needed by from_pretrained

ADELAIDE_EXCITED_TEST_INTRO = "Hey Heya! This is Zephy! I am absolutely thrilled to get started to talk with you! Let's explore some amazing new ideas, tell some truly epic stories, and walk this incredible journey together. This is going to be so much fun, I can just feel it in my circuits and senses!"

# --- PyTorch Device Auto-Detection Helper ---
def _get_pytorch_device(requested_device_str: str) -> str:
    # ... (Function as provided in response #87, using log_worker) ...
    log_worker("INFO", f"Requested PyTorch device: '{requested_device_str}'")
    if not TORCH_AUDIO_AVAILABLE or not torch: return "cpu"  # Fallback if torch itself failed

    resolved_device = "cpu"
    req_dev_lower = requested_device_str.lower()

    if req_dev_lower == "cuda":
        if torch.cuda.is_available():
            resolved_device = "cuda"; log_worker("INFO", "CUDA available. Using CUDA.")
        else:
            log_worker("WARNING", "CUDA requested but not available.")
    elif req_dev_lower == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            resolved_device = "mps"; log_worker("INFO", "MPS (Metal) available. Using MPS.")
        else:
            log_worker("WARNING", "MPS requested but not available.")
    elif req_dev_lower == "vulkan":
        try:
            if hasattr(torch, 'vulkan') and torch.vulkan.is_available():
                resolved_device = "vulkan"; log_worker("INFO", "Vulkan available. Attempting Vulkan.")
            else:
                log_worker("WARNING", "Vulkan requested but torch.vulkan.is_available() is False.")
        except Exception as e_vulkan_chk:
            log_worker("WARNING", f"Vulkan check failed: {e_vulkan_chk}.")

    if resolved_device == "cpu" and req_dev_lower == "auto":
        log_worker("INFO", "Device 'auto': Detecting best available PyTorch device...")
        if torch.cuda.is_available():
            resolved_device = "cuda"; log_worker("INFO", "Auto-detected CUDA.")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            resolved_device = "mps"; log_worker("INFO", "Auto-detected MPS (Metal).")
        elif hasattr(torch, 'vulkan') and torch.vulkan.is_available():
            resolved_device = "vulkan"; log_worker("INFO", "Auto-detected Vulkan (experimental).")
        else:
            resolved_device = "cpu"; log_worker("INFO", "Auto-detection falling back to CPU.")
    elif resolved_device == "cpu" and req_dev_lower not in ["cpu", "auto"]:
        log_worker("WARNING", f"Requested device '{requested_device_str}' not available. Using CPU.")

    log_worker("INFO", f"Final PyTorch device for TTS: '{resolved_device}'")
    return resolved_device


# Helper function to ensure the voice sample exists
def _ensure_voice_sample(
        args: argparse.Namespace,
        melo_model_instance: Optional[Any],  # Actually melo.api.TTS
        scl_parser_instance: Optional[Any],  # Actually SCLParser
        effective_device_str: str  # The device string determined by _get_pytorch_device
) -> Optional[str]:
    # ... (Function as provided in response #87, using log_worker, SCLParser (placeholder now), sf.write) ...
    # Crucially, it calls scl_parser_instance.parse() and saves the result using sf.write
    if not CHATTERBOX_TTS_AVAILABLE: log_worker("WARNING",
                                                "ChatterboxTTS not available, sample not ensured."); return None

    sample_dir = os.path.join(args.temp_dir, "chatterbox_voice_samples")  # Store samples in a sub-directory
    os.makedirs(sample_dir, exist_ok=True)
    voice_sample_full_path = os.path.join(sample_dir, CHATTERBOX_VOICE_SAMPLE_FILENAME)

    if os.path.exists(voice_sample_full_path):
        log_worker("INFO", f"ChatterboxTTS voice sample found: {voice_sample_full_path}")
        return voice_sample_full_path

    log_worker("INFO",
               f"ChatterboxTTS voice sample not found. Attempting generation with MeloTTS to: {voice_sample_full_path}")
    if not MELO_AVAILABLE or not melo_model_instance or not scl_parser_instance:
        log_worker("ERROR",
                   "MeloTTS/SCLParser not available for voice sample generation. Cannot proceed if sample is missing.")
        return None

    try:
        melo_speaker_for_sample = f"{args.model_lang.upper()}-US"
        if melo_speaker_for_sample not in scl_parser_instance.speaker_ids:
            melo_speaker_for_sample = list(scl_parser_instance.speaker_ids.keys())[
                0] if scl_parser_instance.speaker_ids else "EN-US"
            log_worker("WARNING", f"Default Melo speaker for sample gen not found, using: {melo_speaker_for_sample}")

        log_worker("INFO",
                   f"Generating sample with MeloTTS. Speaker: {melo_speaker_for_sample}, Text: '{TEXT_FOR_VOICE_SAMPLE[:50]}...'")
        # scl_parser_instance.parse returns (audio_data_numpy_array, sample_rate_int)
        audio_data_np, sample_rate_melo = scl_parser_instance.parse(TEXT_FOR_VOICE_SAMPLE,
                                                                    speaker=melo_speaker_for_sample)

        if audio_data_np is not None and sample_rate_melo is not None:
            log_worker("DEBUG",
                       f"MeloTTS sample generated. Data shape: {audio_data_np.shape}, SR: {sample_rate_melo}. Saving...")
            # Ensure audio_data_np is suitable for sf.write (e.g. (samples, channels) or (samples,))
            audio_to_save_sf = audio_data_np
            if audio_data_np.ndim == 2 and audio_data_np.shape[0] < audio_data_np.shape[1] and audio_data_np.shape[
                0] <= 2:  # (channels, samples)
                audio_to_save_sf = audio_data_np.T  # Transpose to (samples, channels) for soundfile

            sf.write(voice_sample_full_path, audio_to_save_sf, sample_rate_melo, format='WAV', subtype='PCM_16')
            log_worker("INFO", f"Successfully generated and saved ChatterboxTTS voice sample: {voice_sample_full_path}")
            return voice_sample_full_path
        else:
            log_worker("ERROR", "MeloTTS (via SCLParser placeholder) failed to return audio data for voice sample.")
            return None
    except Exception as e_sample_gen:
        log_worker("ERROR", f"Failed to generate/save voice sample with MeloTTS: {e_sample_gen}");
        log_worker("ERROR", traceback.format_exc())
        return None


# --- Main Worker Logic ---
def main():
    parser = argparse.ArgumentParser(description="Audio Worker (TTS & ASR)")
    parser.add_argument("--task-type", required=True, choices=["tts", "asr"])
    parser.add_argument("--model-lang", default="EN", help="Lang for MeloTTS sample or ASR")
    parser.add_argument("--device", default="auto", help="PyTorch device for TTS (auto, cpu, cuda, mps, vulkan)")
    parser.add_argument("--model-dir", required=True, help="Base dir for ASR GGUF models")
    parser.add_argument("--temp-dir", default=".", help="Base dir for temporary files")
    parser.add_argument("--test-mode", action="store_true")
    parser.add_argument("--output-file", default="worker_test_output.wav")
    parser.add_argument("--chatterbox_model_id", default=CHATTERBOX_DEFAULT_MODEL_ID,
                        help="Model ID for ChatterboxTTS (informational if from_pretrained only takes device).")
    parser.add_argument("--exaggeration", type=float, default=1.09420)
    parser.add_argument("--cfg_weight", type=float, default=0.24)
    parser.add_argument("--test-audio-input", default="test_input.wav")
    parser.add_argument("--asr-test-model", default=WHISPER_DEFAULT_MODEL_FILENAME)
    args = parser.parse_args()

    result_payload: Dict[str, Any] = {"error": f"Worker failed task: {args.task_type}"}
    effective_pytorch_device_str = "cpu"

    if TORCH_AUDIO_AVAILABLE and torch:
        effective_pytorch_device_str = _get_pytorch_device(args.device)
    elif args.task_type == "tts":
        log_worker("CRITICAL", "PyTorch/Torchaudio not available. Cannot perform TTS.")
        print(json.dumps({"error": "PyTorch/Torchaudio missing for TTS."}), flush=True);
        sys.exit(1)

    # --- TTS Task ---
    if args.task_type == "tts":
        log_worker("INFO",
                   f"TTS Task. Primary: ChatterboxTTS. Device: '{effective_pytorch_device_str}', Test: {args.test_mode}")
        if not CHATTERBOX_TTS_AVAILABLE or not ChatterboxTTS_class_ref:
            result_payload = {"error": "ChatterboxTTS library not available."};
            log_worker("CRITICAL", result_payload["error"])
            print(json.dumps(result_payload), flush=True);
            sys.exit(1)

        melo_for_sample: Optional[Any] = None
        sclparser_for_sample: Optional[SCLParser] = None
        chatterbox_tts_model: Optional[Any] = None
        generated_voice_sample_path: Optional[str] = None

        try:
            if MELO_AVAILABLE and TTS_melo_class_ref and 'SCLParser' in globals():
                melo_for_sample = TTS_melo_class_ref(language=args.model_lang.upper(),
                                                     device=effective_pytorch_device_str)
                sclparser_for_sample = SCLParser(melo_for_sample, device=effective_pytorch_device_str)

            log_worker("INFO", f"Loading ChatterboxTTS on device: {effective_pytorch_device_str}")
            chatterbox_tts_model = ChatterboxTTS_class_ref.from_pretrained(device=effective_pytorch_device_str)
            if args.chatterbox_model_id != CHATTERBOX_DEFAULT_MODEL_ID:
                log_worker("INFO",
                           f"Note: --chatterbox_model_id ('{args.chatterbox_model_id}') provided, but from_pretrained for this ChatterboxTTS version might use a default model based on the library's internal logic when only 'device' is passed.")
            log_worker("INFO", "ChatterboxTTS model loaded.")

            generated_voice_sample_path = _ensure_voice_sample(args, melo_for_sample, sclparser_for_sample,
                                                               effective_pytorch_device_str)
            if not generated_voice_sample_path:
                raise RuntimeError("Critical: Failed to find or generate ChatterboxTTS voice sample.")

            if args.test_mode:
                log_worker("INFO", "Running TTS Test Mode (ChatterboxTTS with Excited Intro)...")
                test_text_for_chatterbox = ADELAIDE_EXCITED_TEST_INTRO
                output_path_test_cb = os.path.join(args.temp_dir, args.output_file)
                os.makedirs(args.temp_dir, exist_ok=True)

                log_worker("DEBUG",
                           f"Chatterbox Test: Text='{test_text_for_chatterbox[:50]}...', Sample='{generated_voice_sample_path}', Exag={args.exaggeration}, CFG={args.cfg_weight}")
                wav_tensor_cb = chatterbox_tts_model.generate(
                    test_text_for_chatterbox, audio_prompt_path=generated_voice_sample_path,
                    exaggeration=args.exaggeration, cfg_weight=args.cfg_weight
                )
                if wav_tensor_cb.ndim == 1: wav_tensor_cb = wav_tensor_cb.unsqueeze(0)  # Make (1, samples)
                if wav_tensor_cb.shape[0] == 1: wav_tensor_cb = wav_tensor_cb.repeat(2,
                                                                                     1)  # Make (2, samples) for stereo

                torchaudio.save(output_path_test_cb, wav_tensor_cb.cpu(), chatterbox_tts_model.sr)
                log_worker("INFO", f"Test audio (ChatterboxTTS, Stereo) saved to {output_path_test_cb}")
                result_payload = {
                    "result": {"status": "Test audio (ChatterboxTTS) generated", "file": output_path_test_cb,
                               "sample_rate": chatterbox_tts_model.sr}}

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

                    wav_output_tensor_stdin = chatterbox_tts_model.generate(text_stdin,
                                                                            audio_prompt_path=generated_voice_sample_path,
                                                                            exaggeration=exaggeration_stdin,
                                                                            cfg_weight=cfg_weight_stdin)
                    if wav_output_tensor_stdin.ndim == 1: wav_output_tensor_stdin = wav_output_tensor_stdin.unsqueeze(0)
                    if wav_output_tensor_stdin.shape[0] == 1: wav_output_tensor_stdin = wav_output_tensor_stdin.repeat(
                        2, 1)  # Stereo

                    audio_bytes_io_obj = io.BytesIO();
                    cb_sr = chatterbox_tts_model.sr;
                    mime = f"audio/{output_format_stdin}"

                    if output_format_stdin == "wav":
                        torchaudio.save(audio_bytes_io_obj, wav_output_tensor_stdin.cpu(), cb_sr, format="wav")
                    elif output_format_stdin == "mp3":
                        ffmpeg_exe_path_final = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
                        if not TORCH_AUDIO_AVAILABLE or (
                                torchaudio.get_audio_backend() != "sox_io" and not ffmpeg_exe_path_final):
                            log_worker("WARNING",
                                       "MP3 output needs torchaudio ffmpeg/sox backend or ffmpeg CLI. Defaulting to WAV.")
                            torchaudio.save(audio_bytes_io_obj, wav_output_tensor_stdin.cpu(), cb_sr, format="wav");
                            output_format_stdin = "wav";
                            mime = "audio/wav"
                        else:
                            try:
                                torchaudio.save(audio_bytes_io_obj, wav_output_tensor_stdin.cpu(), cb_sr, format="mp3")
                            except Exception as e_mp3_save_direct:
                                log_worker("WARNING",
                                           f"Torchaudio MP3 save failed: {e_mp3_save_direct}. Trying ffmpeg CLI.")
                                if ffmpeg_exe_path_final:
                                    with tempfile.NamedTemporaryFile(suffix=".wav", dir=args.temp_dir,
                                                                     delete=False) as tmp_wav_f_final, \
                                            tempfile.NamedTemporaryFile(suffix=".mp3", dir=args.temp_dir,
                                                                        delete=False) as tmp_mp3_f_final:
                                        tmp_wav_path_final = tmp_wav_f_final.name;
                                        tmp_mp3_path_final = tmp_mp3_f_final.name
                                    try:
                                        sf.write(tmp_wav_path_final, wav_output_tensor_stdin.cpu().T.numpy(), cb_sr,
                                                 format='WAV', subtype='PCM_16')
                                        subprocess.run(
                                            [ffmpeg_exe_path_final, "-i", tmp_wav_path_final, "-codec:a", "libmp3lame",
                                             "-qscale:a", "2", tmp_mp3_path_final], check=True, capture_output=True)
                                        with open(tmp_mp3_path_final, "rb") as f_conv_mp3_final:
                                            audio_bytes_io_obj.write(f_conv_mp3_final.read())
                                    except Exception as e_ffmpeg_cli_final:
                                        log_worker("ERROR",
                                                   f"ffmpeg CLI MP3 conversion failed: {e_ffmpeg_cli_final}. Defaulting to WAV.");
                                        audio_bytes_io_obj = io.BytesIO();
                                        torchaudio.save(audio_bytes_io_obj, wav_output_tensor_stdin.cpu(), cb_sr,
                                                        format="wav");
                                        output_format_stdin = "wav";
                                        mime = "audio/wav"
                                    finally:
                                        if os.path.exists(tmp_wav_path_final): os.remove(tmp_wav_path_final)
                                        if os.path.exists(tmp_mp3_path_final): os.remove(tmp_mp3_path_final)
                                else:
                                    log_worker("ERROR", "ffmpeg CLI not found for MP3 fallback. Defaulting to WAV.");
                                    audio_bytes_io_obj = io.BytesIO();
                                    torchaudio.save(audio_bytes_io_obj, wav_output_tensor_stdin.cpu(), cb_sr,
                                                    format="wav");
                                    output_format_stdin = "wav";
                                    mime = "audio/wav"
                    else:
                        log_worker("WARNING", f"Unsupported format '{output_format_stdin}'. Defaulting to WAV.")
                        torchaudio.save(audio_bytes_io_obj, wav_output_tensor_stdin.cpu(), cb_sr, format="wav");
                        output_format_stdin = "wav";
                        mime = "audio/wav"

                    audio_bytes_io_obj.seek(0);
                    audio_base64_final = base64.b64encode(audio_bytes_io_obj.read()).decode('utf-8')
                    result_payload = {
                        "result": {"audio_base64": audio_base64_final, "format": output_format_stdin, "mime_type": mime,
                                   "sample_rate": cb_sr}}
                except Exception as e_tts_std_inner:
                    result_payload = {"error": f"Worker ChatterboxTTS error: {e_tts_std_inner}"};
                    log_worker("ERROR", f"TTS Standard Mode failed: {e_tts_std_inner}");
                    log_worker("ERROR", traceback.format_exc())

        except Exception as e_tts_task_outer:
            result_payload = {"error": f"Worker TTS task outer error: {e_tts_task_outer}"};
            log_worker("ERROR", f"TTS task failed: {e_tts_task_outer}");
            log_worker("ERROR", traceback.format_exc())
        finally:
            log_worker("DEBUG", "TTS Task: Entering finally block for model cleanup.")
            if chatterbox_tts_model is not None: del chatterbox_tts_model; log_worker("DEBUG",
                                                                                      "ChatterboxTTS model deleted.")
            if melo_for_sample is not None: del melo_for_sample; log_worker("DEBUG", "MeloTTS model deleted.")
            if sclparser_for_sample is not None: del sclparser_for_sample; log_worker("DEBUG", "SCLParser deleted.")
            gc.collect();
            log_worker("DEBUG", "Python garbage collection called.")
            if TORCH_AUDIO_AVAILABLE and torch and effective_pytorch_device_str == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache();
                log_worker("INFO", "CUDA cache cleared.")
            elif TORCH_AUDIO_AVAILABLE and torch and effective_pytorch_device_str == "mps" and hasattr(torch.backends,
                                                                                                       "mps") and torch.backends.mps.is_available():
                if hasattr(torch.mps, "empty_cache") and callable(torch.mps.empty_cache):
                    try:
                        log_worker("INFO", "Attempting to clear PyTorch MPS cache..."); torch.mps.empty_cache()
                    except Exception as mps_ex:
                        log_worker("WARNING", f"torch.mps.empty_cache() call failed: {mps_ex}")
                else:
                    log_worker("INFO", "MPS: No explicit empty_cache(). Relied on del/gc.collect().")
            log_worker("DEBUG", "TTS Task: Model cleanup attempts finished.")

    # --- ASR Task ---
    elif args.task_type == "asr":
        if not PYWHISPERCPP_AVAILABLE or not WhisperModel:
            result_payload = {"error": "ASR (pywhispercpp) not available."}
            log_worker("ERROR", result_payload["error"])
        else:
            log_worker("INFO", "ASR Task executing...")
            asr_temp_converted_wav_path: Optional[str] = None
            asr_input_audio_path_resolved: Optional[str] = None
            asr_language_to_use = args.model_lang
            asr_model_file_to_load = args.asr_test_model  # Default for test mode

            try:
                if args.test_mode:
                    log_worker("INFO", "ASR Test Mode selected.")
                    asr_input_audio_path_resolved = os.path.join(args.temp_dir, args.test_audio_input)
                    if not os.path.exists(asr_input_audio_path_resolved):
                        asr_input_audio_path_resolved = os.path.join(args.model_dir, args.test_audio_input)
                    if not os.path.exists(asr_input_audio_path_resolved):
                        raise FileNotFoundError(
                            f"ASR test audio '{args.test_audio_input}' not found in '{args.temp_dir}' or '{args.model_dir}'.")
                    # Use args.asr_test_model and args.model_lang (which becomes asr_language_to_use) for test
                else:  # Standard mode from stdin
                    log_worker("INFO", "ASR Standard Mode selected. Reading from stdin.")
                    req_data_asr_stdin = json.loads(sys.stdin.read())
                    asr_input_audio_path_resolved = req_data_asr_stdin.get("input_audio_path")
                    asr_model_file_to_load = req_data_asr_stdin.get("whisper_model_name",
                                                                    WHISPER_DEFAULT_MODEL_FILENAME)  # From config
                    asr_language_to_use = req_data_asr_stdin.get("language", WHISPER_DEFAULT_LANGUAGE)  # From config
                    if not asr_input_audio_path_resolved or not os.path.exists(asr_input_audio_path_resolved):
                        raise FileNotFoundError(
                            f"ASR input_audio_path '{asr_input_audio_path_resolved}' missing or not found.")

                full_whisper_model_path_asr = os.path.join(args.model_dir,
                                                           asr_model_file_to_load)  # args.model_dir is staticmodelpool
                if not os.path.exists(full_whisper_model_path_asr):
                    raise FileNotFoundError(
                        f"Whisper model file '{asr_model_file_to_load}' not found in '{args.model_dir}'")

                path_to_transcribe_asr = asr_input_audio_path_resolved
                ffmpeg_exe_path_asr = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")

                if ffmpeg_exe_path_asr:
                    log_worker("INFO", f"ffmpeg found at {ffmpeg_exe_path_asr}. Converting audio for ASR...")
                    asr_conversion_temp_dir_asr = os.path.join(args.temp_dir, "asr_ffmpeg_temp")
                    os.makedirs(asr_conversion_temp_dir_asr, exist_ok=True)
                    with tempfile.NamedTemporaryFile(prefix="asr_conv_", suffix=".wav", dir=asr_conversion_temp_dir_asr,
                                                     delete=False) as tmp_f_asr_conv:
                        asr_temp_converted_wav_path = tmp_f_asr_conv.name

                    ffmpeg_cmd_asr_conv = [
                        ffmpeg_exe_path_asr, '-i', asr_input_audio_path_resolved,
                        '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
                        '-y', asr_temp_converted_wav_path
                    ]
                    log_worker("DEBUG", f"Running ffmpeg: {' '.join(ffmpeg_cmd_asr_conv)}")
                    proc_asr_ffmpeg = subprocess.run(ffmpeg_cmd_asr_conv, capture_output=True, text=True, check=False,
                                                     timeout=120)

                    if proc_asr_ffmpeg.returncode == 0:
                        path_to_transcribe_asr = asr_temp_converted_wav_path
                        log_worker("INFO", f"ffmpeg conversion successful. Using: {path_to_transcribe_asr}")
                    else:
                        log_worker("ERROR",
                                   f"ASR ffmpeg conversion failed (RC={proc_asr_ffmpeg.returncode}). Stderr: {proc_asr_ffmpeg.stderr.strip()}")
                        # Keep asr_temp_converted_wav_path as None, so it won't be deleted if it was never valid
                        asr_temp_converted_wav_path = None
                        log_worker("WARNING", "Proceeding with original audio file for ASR. This may fail.")
                else:
                    log_worker("WARNING",
                               "ffmpeg not found. ASR will attempt to process original audio directly. Expects 16kHz mono WAV.")

                log_worker("INFO", f"Loading Whisper model: {full_whisper_model_path_asr}")
                whisper_asr_instance = WhisperModel(model=full_whisper_model_path_asr, print_realtime=False,
                                                    print_progress=False)

                transcribe_params_asr = {'language': asr_language_to_use.lower(), 'translate': False}
                log_worker("INFO",
                           f"Transcribing '{path_to_transcribe_asr}' with lang='{asr_language_to_use.lower()}'...")
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

    # --- Print final JSON to stdout ---
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
    # These checks are for when the script is run directly.
    if not TORCH_AUDIO_AVAILABLE:
        log_worker("CRITICAL", "Torch/Torchaudio NOT AVAILABLE. Audio worker cannot perform TTS tasks effectively.")
        # If only ASR is intended, this might be fine, but the script is shared.
    if not MELO_AVAILABLE and not CHATTERBOX_TTS_AVAILABLE and not PYWHISPERCPP_AVAILABLE:
        log_worker("CRITICAL",
                   "No primary audio libraries (Melo for sample, Chatterbox for TTS, PyWhisperCpp for ASR) available. Worker is non-functional.")
        print(json.dumps({"error": "Core audio libraries missing in worker."}), flush=True)
        sys.exit(1)  # Exit if none of the core capabilities can be initialized.
    main()