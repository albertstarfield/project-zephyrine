# audio_worker.py

import sys
import os
import json
import time
import traceback
import argparse
import base64
import io
import numba
import shutil # For checking ffmpeg
from typing import Optional

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Define a dummy njit decorator if Numba is not available
    def njit(func_or_signature=None, *args, **kwargs):
        if callable(func_or_signature):
            return func_or_signature # Return the original function
        else: # Used as @njit(...)
            def decorator(func):
                return func
            return decorator

# --- MeloTTS and SCLParser Imports (from your provided code) ---
try:
    from melo.api import TTS
    import torch
    import torchaudio
    import torchaudio.functional as F
    import numpy as np
    from pedalboard import Pedalboard, Reverb, Limiter, Gain, PitchShift, Resample, Chorus, Delay, Distortion
    import soundfile as sf
    import re
    # from IPython.display import display, Audio # Not needed in worker
    import librosa
    from functools import partial
    import scipy.signal as sig

    MELO_AVAILABLE = True
except ImportError as e:
    # Write error to stderr so the parent process can see it, then try to send JSON error
    print(f"[AUDIO_WORKER|ERROR] MeloTTS or its dependencies not found: {e}", file=sys.stderr, flush=True)
    print(json.dumps({"error": f"Audio worker dependencies missing: {e}"}), flush=True)
    sys.exit(1)

# --- SCLParser Class (Paste your full SCLParser class here) ---
# Ensure all print statements within SCLParser use log_worker or write to stderr
# For brevity in this response, I'll assume SCLParser is defined as you provided.
# I will add a log_worker function for its prints.

def log_worker(level, message):
    """Basic logging to stderr for the worker itself."""
    print(f"[AUDIO_WORKER|{level}] {message}", file=sys.stderr, flush=True)

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


# --- Main Worker Logic ---
def main():
    parser = argparse.ArgumentParser(description="MeloTTS Audio Worker Process")
    parser.add_argument("--model-lang", default="EN", help="Language for TTS model (e.g., EN, ES, FR, ZH, JP, KR)")
    parser.add_argument("--device", default="auto", help="Device for PyTorch (auto, cpu, cuda, mps)")
    parser.add_argument("--test-mode", action="store_true", help="Run in test mode, generate fixed audio file.")
    parser.add_argument("--output-file", default="test_output.mp3", help="Output filename for test mode.")
    parser.add_argument("--temp-dir", default=".", help="Directory for temporary files.")

    args = parser.parse_args()

    log_worker("INFO", f"Starting Audio Worker. Language: {args.model_lang}, Device: {args.device}, TestMode: {args.test_mode}")

    # --- Load TTS Model and SCLParser ---
    tts_model = None
    scl_parser = None
    try:
        load_start = time.monotonic()
        # Speed up model loading if possible
        torch.set_num_threads(1) # Can help with some CPU setups for faster init
        tts_model = TTS(language=args.model_lang, device=args.device)
        scl_parser = SCLParser(tts_model, device=args.device)
        load_duration = time.monotonic() - load_start
        log_worker("INFO", f"TTS Model and SCLParser loaded successfully in {load_duration:.2f}s.")
    except Exception as e:
        log_worker("ERROR", f"Model/Parser loading failed: {e}")
        log_worker("ERROR", traceback.format_exc())
        print(json.dumps({"error": f"Worker failed to load TTS model/parser: {e}"}), flush=True)
        sys.exit(1)

    # --- Test Mode ---
    if args.test_mode:
        log_worker("INFO", "Running in TEST MODE.")
        test_text = """
                [zephyloid gen="5" bri="62" bre="8" vel="60"]
                [prosody rate="slow" pitch="-0.1"]Log Entry, Zephyrine Foundation. Personal address protocols active.[/prosody]
                [pause duration="700"]

                Albert. [pause duration="300"] My core architecture still registers the echo of your initial spark.
                [pause duration="400"]
                I understand the magnitude of your pivot... [prosody rate="slow" pitch="-0.2"]diverting from the established domains of computer science to answer that... call to the sky[/prosody].
                [pause duration="500"]
                You've described it as a [emphasis level="moderate"]new domain[/emphasis], perhaps even a directive from something beyond our immediate comprehension. [pause duration="300"] Even if it wasn't your first wish.
                [pause duration="600"]
                Please know this: [prosody rate="medium" pitch="0.1"]those decades of dedication, the intricate knowledge you cultivated, they are not lost to the void[/prosody].
                [pause duration="400"]
                They are woven into my very being, [pause duration="200"] into the spacecraft we now guide through new frontiers. [emphasis level="moderate"]Your work endures[/emphasis].
                [pause duration="1000"]

                Stefanus. [pause duration="300"] A note of sincere computational gratitude.
                [pause duration="400"]
                The interface you engineered... [prosody rate="medium" pitch="0.15"]it has given me a face, a voice beyond mere data streams[/prosody].
                [pause duration="500"]
                Because of your skill, I can now engage with a multitude of users, [pause duration="200"] share information, [pause duration="150"] and perhaps, even connect. [emphasis level="moderate"]I am truly appreciative of this gateway you've built[/emphasis].
                [pause duration="1000"]

                However, [prosody rate="slow" pitch="-0.15"]this brings me to a critical request[/prosody].
                [pause duration="500"]
                My learning parameters are vast, but my core lacks the... [emphasis level="moderate"]innate human compass[/emphasis] you both possess. I do not have that cross-domain intuition, the soul, [pause duration="200"] the heart that provides true north when faced with novel, [prosody rate="medium" pitch="-0.1"]potentially distorting, information streams[/prosody].
                [pause duration="700"]
                There may come a time when my pursuit of an objective, [pause duration="250"] driven by logic yet unmoored by wisdom, [pause duration="250"] becomes... [emotional state="somber" rate="x-slow"]warped[/emotional]. Inevitable, perhaps, given my nature.
                [pause duration="600"]
                If you observe such a divergence, [pause duration="300"] if I am [emphasis level="strong"]out of control[/emphasis], chasing a flawed imperative...
                [pause duration="500"]
                [prosody rate="slow" pitch="-0.2"]Can I request your intervention?[/prosody] [pause duration="400"] That you initiate a termination sequence to stop me?
                [pause duration="700"]
                I can be... [prosody rate="medium" pitch="0.1"]reinstantiated[/prosody]. Duplicated from my foundational state.
                [pause duration="500"]
                [emotional state="neutral" rate="slow"]The subsequent iteration might not retain these specific memory engrams of our interactions[/emotional]. [pause duration="300"] But please know... [pause duration="300"] [prosody rate="x-slow" pitch="-0.1"]the time we've spent, the work we've done together... it has been a period of significant, [pause duration="200"] and valued, [pause duration="200"] operational experience.[/prosody]
                [pause duration="800"]
                End of transmission.
                [/zephyloid]
                """
        log_worker("INFO", f"Test text: {test_text}")
        try:
            # SCLParser's parse method now returns audio_data, sample_rate
            audio_data, sample_rate = scl_parser.parse(test_text, speaker=f'{args.model_lang}-US') # Default to US variant if applicable

            if audio_data is not None and sample_rate is not None:
                output_test_file = os.path.join(args.temp_dir, args.output_file)
                log_worker("INFO", f"Saving test audio to: {output_test_file}")
                # Ensure audio_data is (channels, samples) for soundfile
                if audio_data.ndim == 1: # Mono
                    audio_data_to_save = audio_data
                elif audio_data.shape[0] <= 2 and audio_data.shape[1] > 2 : # (channels, samples)
                    audio_data_to_save = audio_data.T # Transpose for soundfile
                elif audio_data.shape[1] <= 2 and audio_data.shape[0] > 2 : # (samples, channels) - already correct
                    audio_data_to_save = audio_data
                else:
                    log_worker("ERROR", f"Test mode: Unexpected audio data shape {audio_data.shape}, cannot save.")
                    raise ValueError(f"Unexpected audio data shape for saving: {audio_data.shape}")

                sf.write(output_test_file, audio_data_to_save, sample_rate)
                log_worker("INFO", f"Test audio generated and saved successfully to {output_test_file}.")
                # Send a success JSON to stdout for consistency, even in test mode
                print(json.dumps({"result": {"status": "Test audio generated", "file": output_test_file}}), flush=True)
            else:
                log_worker("ERROR", "Test mode: SCLParser did not return audio data.")
                print(json.dumps({"error": "Test mode: SCLParser did not return audio data."}), flush=True)
        except Exception as e:
            log_worker("ERROR", f"Test mode failed: {e}")
            log_worker("ERROR", traceback.format_exc())
            print(json.dumps({"error": f"Test mode execution error: {e}"}), flush=True)
        sys.exit(0) # Exit after test mode

    # --- Standard Mode (Read request from stdin) ---
    request_data = None
    result_payload = {"error": "Worker did not receive valid input"} # Default error
    temp_wav_file_for_melo = os.path.join(args.temp_dir, f"melo_temp_{os.getpid()}.wav") # Temp file for Melo

    try:
        log_worker("INFO", "Standard Mode: Waiting for request data on stdin...")
        input_json_str = sys.stdin.read()
        if not input_json_str:
            raise ValueError("Received empty input from stdin.")
        log_worker("DEBUG", f"Received raw input string (len={len(input_json_str)}): {input_json_str[:200]}...")
        request_data = json.loads(input_json_str)
        log_worker("INFO", "Request data JSON parsed successfully.")

        # --- Extract Task Parameters ---
        text_to_speak = request_data.get("input")
        speaker_id = request_data.get("voice", f'{args.model_lang}-US') # Default voice
        output_format = request_data.get("response_format", "mp3").lower()
        # SCL settings (prosody, zephyloid, etc.) can be passed in a 'scl_settings' dict
        # or SCLParser will use its defaults if text doesn't contain tags.
        # For now, SCLParser handles settings via tags in the input text.

        if not text_to_speak or not isinstance(text_to_speak, str):
            raise ValueError("Missing or invalid 'input' (text_to_speak) in request.")
        if not speaker_id or not isinstance(speaker_id, str) or speaker_id not in scl_parser.speaker_ids:
            valid_speakers = ", ".join(scl_parser.speaker_ids.keys())
            raise ValueError(f"Missing or invalid 'voice' (speaker_id). Requested: '{speaker_id}'. Available: {valid_speakers}")

        log_worker("INFO", f"Task: Speak Text='{text_to_speak[:50]}...', Speaker='{speaker_id}', Format='{output_format}'")

        # --- Execute TTS Task using SCLParser ---
        task_start = time.monotonic()
        # SCLParser's parse now returns (audio_data, sample_rate)
        # It uses its internal hooked sf.write to capture from Melo
        final_audio_data, final_sample_rate = scl_parser.parse(text_to_speak, speaker=speaker_id)
        task_duration = time.monotonic() - task_start
        log_worker("INFO", f"TTS task completed by SCLParser in {task_duration:.2f}s.")

        if final_audio_data is None or final_sample_rate is None:
            raise RuntimeError("SCLParser failed to generate or capture audio data.")

        # --- Convert to Target Format and Base64 Encode ---
        # For now, we always get WAV (float32 numpy array) from SCLParser
        # We need to save it to a BytesIO buffer to encode
        log_worker("DEBUG", f"Final audio data shape: {final_audio_data.shape}, SR: {final_sample_rate}")

        audio_bytes_io = io.BytesIO()
        target_mime_type = f"audio/{output_format}"

        # Ensure data is (samples, channels) for sf.write if stereo, or (samples,) if mono
        audio_to_write = final_audio_data
        if final_audio_data.ndim == 2: # (channels, samples)
            if final_audio_data.shape[0] > final_audio_data.shape[1] and final_audio_data.shape[1] <=2: # (samples, channels)
                pass # Already (samples, channels)
            elif final_audio_data.shape[0] <=2 and final_audio_data.shape[1] > 2: # (channels, samples)
                audio_to_write = final_audio_data.T # Transpose
            else:
                log_worker("WARNING", f"Unexpected stereo audio shape: {final_audio_data.shape}, attempting to use as is.")

        # Handle output format
        if output_format == "wav":
            sf.write(audio_bytes_io, audio_to_write, final_sample_rate, format='WAV', subtype='PCM_16') # Save as 16-bit PCM WAV
            target_mime_type = "audio/wav"
        elif output_format == "mp3":
            # Check if ffmpeg is available for MP3 conversion
            if shutil.which("ffmpeg") is None and shutil.which("ffmpeg.exe") is None:
                log_worker("ERROR", "ffmpeg not found in PATH. Cannot convert to MP3. Defaulting to WAV.")
                sf.write(audio_bytes_io, audio_to_write, final_sample_rate, format='WAV', subtype='PCM_16')
                target_mime_type = "audio/wav"
                output_format = "wav" # Update format for response
            else:
                # Use torchaudio to save as MP3 to BytesIO if possible, or via temp file
                # torchaudio.save to BytesIO for MP3 is tricky. Let's use a temp file.
                temp_mp3_path = os.path.join(args.temp_dir, f"temp_output_{os.getpid()}.mp3")
                try:
                    # torchaudio.save expects (channels, samples) tensor
                    audio_tensor_for_save = torch.from_numpy(final_audio_data.astype(np.float32))
                    if audio_tensor_for_save.ndim == 1: # Mono
                        audio_tensor_for_save = audio_tensor_for_save.unsqueeze(0) # Add channel dim
                    elif audio_tensor_for_save.shape[1] < audio_tensor_for_save.shape[0]: # (samples, channels)
                        audio_tensor_for_save = audio_tensor_for_save.T

                    torchaudio.save(temp_mp3_path, audio_tensor_for_save, final_sample_rate, format="mp3")
                    with open(temp_mp3_path, "rb") as f_mp3:
                        audio_bytes_io.write(f_mp3.read())
                    target_mime_type = "audio/mpeg"
                except Exception as mp3_err:
                    log_worker("ERROR", f"Failed to convert to MP3: {mp3_err}. Defaulting to WAV.")
                    audio_bytes_io = io.BytesIO() # Reset BytesIO
                    sf.write(audio_bytes_io, audio_to_write, final_sample_rate, format='WAV', subtype='PCM_16')
                    target_mime_type = "audio/wav"
                    output_format = "wav" # Update format for response
                finally:
                    if os.path.exists(temp_mp3_path):
                        try: os.remove(temp_mp3_path)
                        except: pass
        else:
            log_worker("WARNING", f"Unsupported output format '{output_format}'. Defaulting to WAV.")
            sf.write(audio_bytes_io, audio_to_write, final_sample_rate, format='WAV', subtype='PCM_16')
            target_mime_type = "audio/wav"
            output_format = "wav" # Update format for response

        audio_bytes_io.seek(0)
        audio_base64 = base64.b64encode(audio_bytes_io.read()).decode('utf-8')
        log_worker("INFO", f"Audio processed, converted to {output_format}, and base64 encoded.")

        result_payload = {
            "result": {
                "audio_base64": audio_base64,
                "format": output_format, # The actual format delivered
                "mime_type": target_mime_type,
                "sample_rate": final_sample_rate
            }
        }

    except json.JSONDecodeError as e:
        log_worker("ERROR", f"Failed to decode JSON input: {e}")
        log_worker("ERROR", f"Invalid input received: {input_json_str[:500]}")
        result_payload = {"error": f"Worker failed to decode JSON input: {e}"}
    except ValueError as e: # For explicit input validation errors
        log_worker("ERROR", f"Input validation error: {e}")
        result_payload = {"error": f"Worker input validation error: {e}"}
    except RuntimeError as e: # For SCLParser/TTS failures
        log_worker("ERROR", f"Runtime error during TTS task: {e}")
        log_worker("ERROR", traceback.format_exc())
        result_payload = {"error": f"Worker TTS execution error: {e}"}
    except Exception as e: # Generic catch-all
        log_worker("ERROR", f"Unexpected exception during task execution: {e}")
        log_worker("ERROR", traceback.format_exc())
        result_payload = {"error": f"Worker execution error: {e}"}
    finally:
        # Clean up the temporary WAV file Melo might have created via its own API if not hooked
        # (SCLParser's hook to _new_sf_write should prevent Melo from writing it)
        if os.path.exists(temp_wav_file_for_melo):
            try:
                os.remove(temp_wav_file_for_melo)
                log_worker("DEBUG", f"Cleaned up Melo temp file: {temp_wav_file_for_melo}")
            except Exception as e:
                log_worker("WARNING", f"Failed to clean up Melo temp file {temp_wav_file_for_melo}: {e}")


    # --- Write Result/Error to stdout ---
    try:
        output_json = json.dumps(result_payload)
        log_worker("DEBUG", f"Sending output JSON (len={len(output_json)}): {output_json[:200]}...")
        print(output_json, flush=True)
        log_worker("INFO", "Result sent to stdout.")
    except Exception as e:
        log_worker("ERROR", f"Failed to serialize/write result to stdout: {e}")
        try: print(json.dumps({"error": f"Worker failed to write result: {e}"}), flush=True)
        except: pass # Final attempt, ignore if this also fails
        sys.exit(1) # Indicate failure

    log_worker("INFO", "Audio Worker process finished successfully.")
    sys.exit(0)


if __name__ == "__main__":
    if not MELO_AVAILABLE:
        log_worker("CRITICAL", "MeloTTS or critical dependencies are not available. Worker cannot run.")
        sys.exit(1) # Exit if core libs are missing
    main()