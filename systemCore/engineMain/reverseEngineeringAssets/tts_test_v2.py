from melo.api import TTS
import torch
import torchaudio
import torchaudio.functional as F
import numpy as np
from pedalboard import Pedalboard, Reverb, Limiter, Gain, PitchShift, Resample, Chorus, Delay, Distortion
import soundfile as sf
import re
from IPython.display import display, Audio
import sys
import librosa
from functools import partial
import scipy.signal as sig

OUTPUT_FILE = "en-test-zeph-profile-v2-vocaloid.mp3"
SEMITONES_IN_OCTAVE = 12

class SCLParser:
    def __init__(self, model, device='auto'):
        print("SCLParser: Initializing...")
        self.model = model
        self.device = device
        self.speaker_ids = model.hps.data.spk2id
        print(f"SCLParser: Speaker IDs: {self.speaker_ids}")
        self.sample_rate = model.hps.data.sampling_rate
        print(f"SCLParser: Sample Rate: {self.sample_rate}")
        self.voice_settings = {
            'rate': 1.0,
            'pitch': 0.0,  # in semitones (constant offset)
        }
        print(f"SCLParser: Initial Voice Settings: {self.voice_settings}")
        # Rate map, with adjustments for max/min:
        self.rate_map = {
            'x-slow': 0.5,  # Keep x-slow at 0.5
            'slow': 1.0,   # Slow is now the default (1.0)
            'medium': 1.05,  # Slightly faster than normal
            'fast': 1.1,    # Max speed
            'x-fast': 1.1,   # Also max speed
        }
        self.pitch_map = {
            'x-low': -3.0,
            'low': -1.5,
            'medium': 0.0,
            'high': 1.5,
            'x-high': 3.0,
        }
        print(f"SCLParser: Rate Map: {self.rate_map}")
        print(f"SCLParser: Pitch Map: {self.pitch_map}")
        self.audio_segments = []
        self.original_sf_write = sf.write
        self.captured_audio = None

        # Vocaloid settings
        self.vocaloid_settings = {
            'vel': 64,  # Default values (middle of 0-127 range)
            'dyn': 64,
            'bre': 0,
            'bri': 64,
            'cle': 64,
            'ope': 64,
            'gen': 64,
            'gwl': 0,
            'xsy': 0,  # 0 = Voicebank1, 127 = Voicebank2
            'xsy_voicebanks': None, # No cross-synthesis by default
            'singing': False,   # Whether to apply autotune
            'key': None,      # Key for autotune (e.g., "C:maj")
            'correction_method': "closest",  # "closest" or "scale"
        }
        print(f"SCLParser: Initial Vocaloid Settings: {self.vocaloid_settings}")
        self.xsy_profiles = {
            "Voicebank1": {
                "description": "Default voice profile.",
                "eq_curve": [
                    (30, 0, 3.4), (100, 0, 1.4), (150, 0, 1.4), (250, 0, 1.0),
                    (350, 0, 1.4), (450, 0, 1.8), (550, 0, 1.4), (2000, 0, 1.0),
                    (2500, 0, 1.4), (3000, 0, 1.4), (3500, 0, 1.8), (4000, 0, 1.4),
                    (8000, 0, 1.8), (12000, 0, 1.8), (20000, 0, 1.8)
                ]
            },
            "Voicebank2": {
                "description": "Brighter, more airy voice profile for cross-synthesis.",
                "eq_curve": [
                    (30, 2, 3.4), (100, 3, 1.4), (150, 1, 1.4), (250, 1, 1.0),
                    (350, -1, 1.4), (450, -2, 1.8), (550, 2, 1.4), (2000, 3, 1.0),
                    (2500, 4, 1.4), (3000, 3, 1.4), (3500, 2, 1.8), (4000, 1, 1.4),
                    (8000, 4, 1.8), (12000, 5, 1.8), (20000, 2, 1.8)
                ]
            },
            "Voicebank3": {
                "description": "Deeper, more resonant voice profile for cross-synthesis.",
                "eq_curve": [
                    (30, 4, 3.4), (100, 5, 1.4), (150, 3, 1.4), (250, 2, 1.0),
                    (350, 1, 1.4), (450, -1, 1.8), (550, -3, 1.4), (2000, -2, 1.0),
                    (2500, -1, 1.4), (3000, 0, 1.4), (3500, 1, 1.8), (4000, 2, 1.4),
                    (8000, 1, 1.8), (12000, 0, 1.8), (20000, -1, 1.8)
                ]
            }
        }
        print(f"SCLParser: XSY Profiles: {self.xsy_profiles}")

        print("SCLParser: Initialization complete.")

    def _new_sf_write(self, file, data, samplerate, *args, **kwargs):
        print(f"SCLParser: _new_sf_write called. File: {file}, Samplerate: {samplerate}")
        if isinstance(data, torch.Tensor):
            self.captured_audio = data.clone().detach().cpu().numpy()
            print(f"SCLParser: _new_sf_write: Captured audio (Tensor): {self.captured_audio.shape}")

        else:
            self.captured_audio = data.copy()
            print(f"SCLParser: _new_sf_write: Captured audio (ndarray): {self.captured_audio.shape}")
        return self.original_sf_write(file, data, samplerate, *args, **kwargs)

    def detect_pitch_pyin(self, audio, sr, frame_length=2048, hop_length=512, fmin=None, fmax=None):
        """
        Detects pitch using the PYIN algorithm.
        """
        print(f"SCLParser: detect_pitch_pyin called. Audio shape: {audio.shape}, Sample Rate: {sr}")

        # Ensure mono for PYIN
        if len(audio.shape) > 1 and audio.shape[0] > 1:  # Check for stereo/multi-channel
            audio = librosa.to_mono(audio)
            print("SCLParser: detect_pitch_pyin: Converted audio to mono.")
        elif len(audio.shape) > 1: #1d array
            audio = audio[0]

        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=fmin if fmin is not None else librosa.note_to_hz('C2'),
            fmax=fmax if fmax is not None else librosa.note_to_hz('C7'),
            sr=sr,
            frame_length=frame_length,
            hop_length=hop_length,
            fill_na=0.0  # Replace unvoiced frames with 0 Hz
        )
        print(f"SCLParser: detect_pitch_pyin: Pitch detection complete. f0 shape: {f0.shape}")
        return f0, voiced_flag, voiced_probs

    def parse(self, text, output_path='output.mp3', speaker='EN-US'):
        print(f"SCLParser: parse called. Text: '{text}', Output Path: {output_path}, Speaker: {speaker}")
        sf.write = self._new_sf_write
        processed_text = self.preprocess_text(text)
        print(f"SCLParser: parse: Preprocessed Text: '{processed_text}'")
        segments = self.split_into_segments(processed_text)
        print(f"SCLParser: parse: Segments: {segments}")

        for i, segment in enumerate(segments):
            print(f"SCLParser: parse: Processing segment {i+1}/{len(segments)}: '{segment}'")
            self.parse_segment(segment, speaker)

        if self.audio_segments:
            print("SCLParser: parse: Combining audio segments...")
            final_audio = self.crossfade_segments(self.audio_segments, self.sample_rate)  # Use the updated crossfade_segments
            print(f"SCLParser: parse: Combined audio shape: {final_audio.shape}")
            final_audio_processed = self.post_process(final_audio)
            print(f"SCLParser: parse: Post-processed audio shape: {final_audio_processed.shape}")
            torchaudio.save(output_path, torch.tensor(final_audio_processed), self.sample_rate)

            if 'IPython.display' in sys.modules:
                print("SCLParser: parse: Displaying audio in IPython...")
                display(Audio(data=final_audio_processed, rate=self.sample_rate, autoplay=True))
        else:
            print("SCLParser: parse: No audio segments to combine.")

        self.audio_segments = []
        sf.write = self.original_sf_write
        self.captured_audio = None
        print("SCLParser: parse: Resetting internal state.")
        print("SCLParser: parse: Finished.")

    def preprocess_text(self, text):
        print(f"SCLParser: preprocess_text called. Input text: '{text}'")
        text = self.fix_missing_closing_tags(text)
        print(f"SCLParser: preprocess_text: After fix_missing_closing_tags: '{text}'")
        text = self.flatten_nested_tags(text)
        print(f"SCLParser: preprocess_text: After flatten_nested_tags: '{text}'")
        return text

    def fix_missing_closing_tags(self, text):
        print(f"SCLParser: fix_missing_closing_tags called. Input: '{text}'")
        open_tags = []
        result = []
        i = 0
        while i < len(text):
            if text[i] == '[':
                closing_bracket_index = text.find(']', i)
                if closing_bracket_index == -1:
                    print(f"SCLParser: fix_missing_closing_tags: Found opening bracket without closing bracket at index {i}. Skipping.")
                    i += 1
                    continue
                tag_content = text[i + 1:closing_bracket_index]
                print(f"SCLParser: fix_missing_closing_tags: Found tag: '{tag_content}'")
                if tag_content.startswith('/'):
                    tag_name = tag_content[1:]
                    if open_tags and open_tags[-1] == tag_name:
                        print(f"SCLParser: fix_missing_closing_tags: Closing tag '{tag_name}' matches open tag.")
                        open_tags.pop()
                        result.append(text[i:closing_bracket_index + 1])
                    else:
                        print(f"SCLParser: fix_missing_closing_tags: Closing tag '{tag_name}' does not match open tag or no open tags.")
                        result.append(text[i:closing_bracket_index + 1]) # Still append even if unmatched

                else:
                    tag_name = tag_content.split(' ')[0] if ' ' in tag_content else tag_content
                    print(f"SCLParser: fix_missing_closing_tags: Opening tag '{tag_name}' found.")
                    result.append(text[i:closing_bracket_index + 1])
                    open_tags.append(tag_name)
                i = closing_bracket_index + 1
            else:
                result.append(text[i])
                i += 1
        while open_tags:
            missing_tag = open_tags.pop()
            print(f"SCLParser: fix_missing_closing_tags: Adding missing closing tag '[/{missing_tag}]'")
            result.append(f"[/{missing_tag}]")
        fixed_text = "".join(result)
        print(f"SCLParser: fix_missing_closing_tags: Returning: '{fixed_text}'")
        return fixed_text

    def flatten_nested_tags(self, text):
        print(f"SCLParser: flatten_nested_tags called. Input: '{text}'")
        original_text = text
        while True:
            match = re.search(r'\[([^/][^\]]*)\]([^\[]*)\[([^\]]*)\](.*?)\[/\3\](.*?)\[/\1\]', text)
            if not match:
                print("SCLParser: flatten_nested_tags: No nested tags found.")
                break
            tag1, text1, tag2, text2, text3 = match.groups()
            print(f"SCLParser: flatten_nested_tags: Found nested tags: tag1='{tag1}', text1='{text1}', tag2='{tag2}', text2='{text2}', text3='{text3}'")
            replacement = f"[{tag1}]{text1}[/{tag1}][{tag2}]{text2}[/{tag2}][{tag1}]{text3}[/{tag1}]"
            print(f"SCLParser: flatten_nested_tags: Replacing with: '{replacement}'")
            text = text.replace(match.group(0), replacement, 1)
        if text != original_text:
          print(f"SCLParser: flatten_nested_tags: Returning: '{text}'")
        return text
    def split_into_segments(self, text):
        print(f"SCLParser: split_into_segments called. Input: '{text}'")
        segments = []
        current_segment = ""
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
        print(f"SCLParser: split_into_segments: Split into sentences: {sentences}")
        for sentence in sentences:
            current_segment += sentence + " "
            pause_matches = list(re.finditer(r'\[pause\s+([^\]]+)\]', current_segment))
            if pause_matches:
                print(f"SCLParser: split_into_segments: Found pause tags in current segment: '{current_segment}'")
                last_index = 0
                for match in pause_matches:
                    segments.append(current_segment[last_index:match.start()])
                    segments.append(match.group(0))  # Append the pause tag itself
                    last_index = match.end()
                segments.append(current_segment[last_index:])
                current_segment = ""
            elif len(current_segment) > 150:
                print(f"SCLParser: split_into_segments: Current segment exceeds 150 characters: '{current_segment}'")
                segments.append(current_segment)
                current_segment = ""
        if current_segment.strip():
            print(f"SCLParser: split_into_segments: Adding remaining segment: '{current_segment.strip()}'")
            segments.append(current_segment.strip())
        print(f"SCLParser: split_into_segments: Returning segments: {segments}")
        return segments

    def parse_segment(self, segment, speaker):
        print(f"SCLParser: parse_segment called. Segment: '{segment}', Speaker: {speaker}")
        tags = re.findall(r'\[([^\]]+)\]', segment)
        print(f"SCLParser: parse_segment: Found tags: {tags}")
        current_text = segment
        for tag in tags:
            parts = current_text.split(f"[{tag}]", 1)
            before_text = parts[0]
            print(f"SCLParser: parse_segment: Processing tag '{tag}'. Text before: '{before_text}'")
            if before_text.strip():
                self.speak_with_settings(before_text, speaker)
            if tag.startswith('/'):
                print(f"SCLParser: parse_segment: Encountered closing tag '{tag}'.")
                self.reset_settings(tag[1:])
            else:
                print(f"SCLParser: parse_segment: Encountered opening tag '{tag}'.")
                self.apply_settings(tag)
            current_text = parts[1] if len(parts) > 1 else ""
            print(f"SCLParser: parse_segment: Remaining text after tag: '{current_text}'")
        if current_text.strip():
            print(f"SCLParser: parse_segment: Processing remaining text: '{current_text}'")
            self.speak_with_settings(current_text, speaker)
        print(f"SCLParser: parse_segment: Finished processing segment.")

    def apply_settings(self, tag_str):
        print(f"SCLParser: apply_settings called. Tag: '{tag_str}'")
        parts = tag_str.split(' ', 1)
        tag_name = parts[0].lower()
        params = parts[1] if len(parts) > 1 else ""
        attrs = self.parse_attributes(params) if params else {}
        print(f"SCLParser: apply_settings: Tag Name: {tag_name}, Attributes: {attrs}")

        if tag_name == "pause":
            duration = attrs.get("duration", list(attrs.values())[0] if attrs else "medium")
            print(f"SCLParser: apply_settings: Pause duration: {duration}")
            if duration in ("short", "medium", "long", "x-long"):
                duration_ms = {"short": 250, "medium": 500, "long": 1000, "x-long": 1500}[duration]
            elif duration.isdigit():
                duration_ms = int(duration)
            else:
                duration_ms = 0
            print(f"SCLParser: apply_settings: Pause duration (ms): {duration_ms}")
            silence = np.zeros(int(self.sample_rate * duration_ms / 1000), dtype=np.float32)
            self.audio_segments.append(silence)
            print(f"SCLParser: apply_settings: Added silence segment: {silence.shape}")
            print(f"  (PAUSE: {duration_ms}ms)")

        elif tag_name == "prosody":
            if "rate" in attrs:
                rate_value = attrs["rate"]
                print(f"SCLParser: apply_settings: Prosody rate value: {rate_value}")
                # Get rate, but clamp to min 0.5 and max 1.1:
                new_rate = self.rate_map.get(rate_value, float(rate_value) if self.is_number(rate_value) else 1.0)
                new_rate = max(0.5, min(1.1, new_rate))  # Enforce limits
                self.voice_settings['rate'] = new_rate
                print(f"SCLParser: apply_settings: Prosody rate set to: {self.voice_settings['rate']}")
                print(f"  (PROSODY RATE: {self.voice_settings['rate']})")
            if "pitch" in attrs:
                pitch_value = attrs["pitch"]
                print(f"SCLParser: apply_settings: Prosody pitch value: {pitch_value}")
                self.voice_settings['pitch'] = self.parse_pitch(pitch_value)
                print(f"SCLParser: apply_settings: Prosody pitch set to: {self.voice_settings['pitch']}")
                print(f"  (PROSODY PITCH: {self.voice_settings['pitch']})")

        elif tag_name == "emphasis":
            level = attrs.get("level", "moderate")
            print(f"SCLParser: apply_settings: Emphasis level: {level}")
            if level == "strong":
                self.voice_settings['rate'] *= 0.9
                self.voice_settings['pitch'] = self.parse_pitch(attrs.get("pitch", "high"))
            elif level == "moderate":
                self.voice_settings['rate'] *= 0.95
                self.voice_settings['pitch'] = self.parse_pitch(attrs.get("pitch", "0.5"))
            elif level == "reduced":
                self.voice_settings['rate'] *= 1.1
                self.voice_settings['pitch'] = self.parse_pitch(attrs.get("pitch", "low"))
            # Clamp rate after emphasis adjustment:
            self.voice_settings['rate'] = max(0.5, min(1.1, self.voice_settings['rate']))
            print(f"SCLParser: apply_settings: Emphasis applied. Rate: {self.voice_settings['rate']}, Pitch: {self.voice_settings['pitch']}")
            print(f"  (EMPHASIS: level={level}, rate={self.voice_settings['rate']}, pitch={self.voice_settings['pitch']})")

        elif tag_name == "emotional":
            state = attrs.get("state", "neutral").lower()
            print(f"SCLParser: apply_settings: Emotional state: {state}")
            state_pitch_map = {"excited": 2.0, "somber": -2.0, "neutral": 0.0}
            state_rate_map = {"excited": 1.1, "somber": 0.9, "neutral": 1.0}
            rate_value = attrs.get("rate", None)
            if rate_value:
                print(f"SCLParser: apply_settings: Emotional rate value provided: {rate_value}")
                rate = self.rate_map.get(rate_value, float(rate_value) if self.is_number(rate_value) else 1.0)
            else:
                print(f"SCLParser: apply_settings: Using state-based rate: {state_rate_map.get(state, 1.0)}")
                rate = state_rate_map.get(state, 1.0)
            pitch_value = attrs.get("pitch", None)
            if pitch_value:
                print(f"SCLParser: apply_settings: Emotional pitch value provided: {pitch_value}")
                pitch = self.parse_pitch(pitch_value)
            else:
                print(f"SCLParser: apply_settings: Using state-based pitch: {state_pitch_map.get(state, 0.0)}")
                pitch = state_pitch_map.get(state, 0.0)
            self.voice_settings['rate'] *= rate
            self.voice_settings['pitch'] += pitch

            # Clamp rate after emotional adjustment:
            self.voice_settings['rate'] = max(0.5, min(1.1, self.voice_settings['rate']))
            print(f"SCLParser: apply_settings: Emotion applied. Rate: {self.voice_settings['rate']}, Pitch: {self.voice_settings['pitch']}")
            print(f"  (EMOTIONAL: state={state}, rate factor={rate}, pitch shift={pitch})")

        elif tag_name == 'say-as':
            print("SCLParser: apply_settings: 'say-as' tag encountered. No specific logic implemented.")
            pass
        elif tag_name == 'voice':
            print("SCLParser: apply_settings: 'voice' tag encountered. No specific logic implemented.")
            pass

        elif tag_name == "vocaloid":
            print(f"SCLParser: apply_settings: 'vocaloid' tag encountered.")
            for key, value in attrs.items():
                if key in self.vocaloid_settings:
                    if key == 'xsy_voicebanks':
                         # Split voicebanks and check if they are valid profiles
                        voicebanks = value.split(",")
                        if all(vb.strip() in self.xsy_profiles for vb in voicebanks):
                            self.vocaloid_settings[key] = [vb.strip() for vb in voicebanks]
                            print(f"SCLParser: apply_settings: Vocaloid XSY voicebanks set to: {self.vocaloid_settings[key]}")
                        else:
                            print(f"SCLParser: apply_settings: Invalid XSY voicebanks: {value}. Using default.")
                            self.vocaloid_settings[key] = None # Reset to default

                    elif key == 'xsy':
                        try:
                            xsy_val = int(value)
                            if 0 <= xsy_val <= 127:
                                self.vocaloid_settings[key] = xsy_val
                                print(f"SCLParser: apply_settings: Vocaloid XSY set to: {self.vocaloid_settings[key]}")
                            else:
                                print(f"SCLParser: apply_settings: Invalid XSY value: {value}.  Must be 0-127. Using default.")
                        except ValueError:
                            print(f"SCLParser: apply_settings: Invalid XSY value (not an integer): {value}. Using default.")
                    elif key == "singing":
                        self.vocaloid_settings[key] = value.lower() == "true"
                        print(f"SCLParser: apply_settings: Vocaloid singing set to: {self.vocaloid_settings[key]}")
                    elif key == "key":
                        # Validate and format the key string
                        try:
                            # Attempt to parse the key to validate it
                            librosa.key_to_degrees(self._parse_key(value))
                            self.vocaloid_settings[key] = self._parse_key(value)
                            print(f"SCLParser: apply_settings: Vocaloid key set to: {self.vocaloid_settings[key]}")
                        except Exception as e:
                            print(f"SCLParser: apply_settings: Invalid key: {value}. Error: {e}. Using default.")
                            self.vocaloid_settings[key] = None

                    elif key == "correction_method":
                        if value.lower() in ("closest", "scale"):
                            self.vocaloid_settings[key] = value.lower()
                            print(f"SCLParser: apply_settings: Vocaloid correction_method set to: {self.vocaloid_settings[key]}")
                        else:
                            print(f"SCLParser: apply_settings: Invalid correction_method: {value}. Using default.")

                    else:
                        try:
                            # All other vocaloid settings are 0-127 integers
                            val = int(value)
                            if 0 <= val <= 127:
                                self.vocaloid_settings[key] = val
                                print(f"SCLParser: apply_settings: Vocaloid {key} set to: {self.vocaloid_settings[key]}")
                            else:
                                print(f"SCLParser: apply_settings: Invalid {key} value: {value}. Must be 0-127. Using default.")
                        except ValueError:
                            print(f"SCLParser: apply_settings: Invalid {key} value (not an integer): {value}. Using default.")

        print(f"SCLParser: apply_settings: Current voice settings after applying tag: {self.voice_settings}")
        print(f"SCLParser: apply_settings: Current vocaloid settings after applying tag: {self.vocaloid_settings}")

    def reset_settings(self, tag_name):
        print(f"SCLParser: reset_settings called. Tag Name: {tag_name}")
        if tag_name in ("prosody", "emphasis", "emotional"):
            self.voice_settings['rate'] = 1.0  # Reset to default (which is now "slow")
            self.voice_settings['pitch'] = 0.0
            print(f"SCLParser: reset_settings: Resetting {tag_name}. Voice settings: {self.voice_settings}")
            print(f"  (RESET {tag_name.upper()})")
        elif tag_name == "vocaloid":
            print(f"SCLParser: reset_settings: Resetting 'vocaloid' settings.")
            self.vocaloid_settings = {
                'vel': 64,
                'dyn': 64,
                'bre': 0,
                'bri': 64,
                'cle': 64,
                'ope': 64,
                'gen': 64,
                'gwl': 0,
                'xsy': 0,
                'xsy_voicebanks': None,
                'singing': False,
                'key': None,
                'correction_method': "closest",
            }
            print(f"  (RESET VOCALOID)")

    def _apply_vocaloid_effects(self, audio_np):
        """Applies Vocaloid effects to the audio."""
        print(f"SCLParser: _apply_vocaloid_effects called. Input audio shape: {audio_np.shape}")

        # Ensure mono for processing
        if len(audio_np.shape) > 1 and audio_np.shape[0] > 1:
            audio_mono = librosa.to_mono(audio_np)
            print("SCLParser: _apply_vocaloid_effects: Converted to mono for processing.")
        elif len(audio_np.shape) > 1:
             audio_mono = audio_np[0]
        else:
            audio_mono = audio_np

        sr = self.sample_rate
        audio_tensor = torch.tensor(audio_mono).unsqueeze(0) # Convert to tensor for torchaudio

        # --- VEL (Velocity) --- Simulate by slightly adjusting speed
        vel_factor = 1.0 + (self.vocaloid_settings['vel'] - 64) * 0.001  # Small adjustment
        print(f"SCLParser: _apply_vocaloid_effects: Velocity factor: {vel_factor}")
        self.voice_settings['rate'] *= vel_factor  # Apply to the overall rate
        self.voice_settings['rate'] = max(0.5, min(1.1, self.voice_settings['rate'])) #clamp

        # --- DYN (Dynamics) --- Control output volume
        dyn_gain_db = (self.vocaloid_settings['dyn'] - 64) / 64 * 12  # +/- 12 dB range
        print(f"SCLParser: _apply_vocaloid_effects: Dynamics gain (dB): {dyn_gain_db}")
        audio_tensor = F.gain(audio_tensor, dyn_gain_db)

        # --- BRE (Breathiness) --- Add a small amount of noise
        bre_amount = self.vocaloid_settings['bre'] / 127 * 0.001  # Very subtle noise
        print(f"SCLParser: _apply_vocaloid_effects: Breathiness amount: {bre_amount}")
        noise = np.random.normal(0, bre_amount, audio_tensor.shape).astype(np.float32)
        audio_tensor = audio_tensor + torch.from_numpy(noise)

        # --- BRI (Brightness) --- Use EQ to boost higher frequencies
        bri_gain = (self.vocaloid_settings['bri'] - 64) / 64 * 6  # +/- 6 dB boost
        print(f"SCLParser: _apply_vocaloid_effects: Brightness gain (dB): {bri_gain}")
        audio_tensor = F.equalizer_biquad(audio_tensor, sr, 8000, bri_gain, 1.0)

        # --- CLE (Clearness) --- Use EQ and slight de-essing
        cle_gain = (self.vocaloid_settings['cle'] - 64) / 64 * 4  # +/- 4 dB
        print(f"SCLParser: _apply_vocaloid_effects: Clearness gain (dB): {cle_gain}")
        audio_tensor = F.equalizer_biquad(audio_tensor, sr, 4000, cle_gain, 1.2)
        # (De-essing would require a more sophisticated approach, possibly a separate plugin)

        # --- OPE (Opening) --- Simulate formant shifts with EQ
        ope_shift = (self.vocaloid_settings['ope'] - 64) / 64  # -1 to +1 range
        print(f"SCLParser: _apply_vocaloid_effects: Opening shift: {ope_shift}")
        # Example formant shift (this is a simplification)
        if ope_shift > 0:
            audio_tensor = F.equalizer_biquad(audio_tensor, sr, 1000, ope_shift * 3, 1.5)
        elif ope_shift < 0:
            audio_tensor = F.equalizer_biquad(audio_tensor, sr, 500, ope_shift * -3, 1.5)

        # --- GEN (Gender Factor) --- Pitch and formant shift
        gen_shift = (self.vocaloid_settings['gen'] - 64) / 64 * 6  # +/- 6 semitones
        print(f"SCLParser: _apply_vocaloid_effects: Gender shift (semitones): {gen_shift}")
        # We'll handle the pitch shift in PSOLA.  Formant shift here:
        if gen_shift > 0:
            audio_tensor = F.equalizer_biquad(audio_tensor, sr, 2000, gen_shift, 1.2)
        elif gen_shift < 0:
            audio_tensor = F.equalizer_biquad(audio_tensor, sr, 800, gen_shift * -1, 1.2)
        self.voice_settings['pitch'] += gen_shift


        # --- GWL (Growl) --- Distortion, pitch modulation, and EQ
        gwl_amount = self.vocaloid_settings['gwl'] / 127
        print(f"SCLParser: _apply_vocaloid_effects: Growl amount: {gwl_amount}")
        if gwl_amount > 0:
            # 1. Distortion (Pedalboard)
            board = Pedalboard([Distortion(drive_db=gwl_amount * 20)])  # Up to 20 dB drive
            audio_np_distorted = board(audio_tensor.cpu().numpy(), sr)

            # 2. Pitch modulation (very subtle)
            mod_freq = 5 + gwl_amount * 10  # 5-15 Hz modulation
            mod_depth = 0.01 + gwl_amount * 0.02  # 1-3% modulation depth
            modulation = (1 + mod_depth * np.sin(2 * np.pi * mod_freq * np.arange(len(audio_np_distorted[0])) / sr)).astype(np.float32)
            audio_np_distorted = audio_np_distorted * modulation

            # 3. EQ (emphasize lower frequencies)
            audio_tensor = torch.tensor(audio_np_distorted)
            audio_tensor = F.equalizer_biquad(audio_tensor, sr, 200, gwl_amount * 5, 1.8)
            audio_tensor = F.equalizer_biquad(audio_tensor, sr, 500, gwl_amount * -3, 1.4)


        # --- XSY (Cross-Synthesis) ---
        if self.vocaloid_settings['xsy_voicebanks']:
            print(f"SCLParser: _apply_vocaloid_effects: Applying XSY between {self.vocaloid_settings['xsy_voicebanks']}")
            voicebank1, voicebank2 = self.vocaloid_settings['xsy_voicebanks']
            xsy_blend = self.vocaloid_settings['xsy'] / 127  # 0.0 to 1.0

            # Apply EQ for Voicebank 1
            audio_tensor_vb1 = audio_tensor.clone()  # Start with a copy
            for freq, gain, q in self.xsy_profiles[voicebank1]["eq_curve"]:
                audio_tensor_vb1 = F.equalizer_biquad(audio_tensor_vb1, sr, freq, gain, q)

            # Apply EQ for Voicebank 2
            audio_tensor_vb2 = audio_tensor.clone() # Start with a copy
            for freq, gain, q in self.xsy_profiles[voicebank2]["eq_curve"]:
                audio_tensor_vb2 = F.equalizer_biquad(audio_tensor_vb2, sr, freq, gain, q)

            # Blend the two EQ'd signals
            audio_tensor = (1 - xsy_blend) * audio_tensor_vb1 + xsy_blend * audio_tensor_vb2

        return audio_tensor.cpu().numpy().flatten()

    def _degrees_from(self, scale: str):
        """Return the pitch classes (degrees) that correspond to the given scale"""
        degrees = librosa.key_to_degrees(scale)
        # To properly perform pitch rounding to the nearest degree from the scale, we need to repeat
        # the first degree raised by an octave. Otherwise, pitches slightly lower than the base degree
        # would be incorrectly assigned.
        degrees = np.concatenate((degrees, [degrees[0] + SEMITONES_IN_OCTAVE]))
        return degrees

    

    def _closest_pitch_from_scale(self, f0, scale):
        """Return the pitch closest to f0 that belongs to the given scale"""
        # Preserve nan.
        if np.isnan(f0):
            return np.nan
        degrees = self._degrees_from(scale)
        midi_note = librosa.hz_to_midi(f0)
        # Subtract the multiplicities of 12 so that we have the real-valued pitch class of the
        # input pitch.
        degree = midi_note % SEMITONES_IN_OCTAVE
        # Find the closest pitch class from the scale.
        degree_id = np.argmin(np.abs(degrees - degree))
        # Calculate the difference between the input pitch class and the desired pitch class.
        degree_difference = degree - degrees[degree_id]
        # Shift the input MIDI note number by the calculated difference.
        midi_note -= degree_difference
        # Convert to Hz.
        return librosa.midi_to_hz(midi_note)

    def _parse_key(self, key_str):
        """Parses the key string into the format required by librosa."""
        print(f"SCLParser: _parse_key called. Input: '{key_str}'")
        match = re.match(r"([A-Ga-g][#b♯♭]?) ?(.*)", key_str)
        if not match:
            raise ValueError(f"Invalid key format: {key_str}")

        tonic = match.group(1).upper()  # Ensure tonic is uppercase
        mode = match.group(2).lower()  # Ensure mode is lowercase

        # Replace unicode sharp/flat with ASCII equivalents
        tonic = tonic.replace("♯", "#").replace("♭", "b")
        mode = mode.replace("♯", "#").replace("♭", "b")

        # Mode abbreviations and full names
        mode_map = {
            "maj": "maj", "major": "maj",
            "min": "min", "minor": "min",
            "ion": "ion", "ionian": "ion",
            "dor": "dor", "dorian": "dor",
            "phr": "phr", "phrygian": "phr",
            "lyd": "lyd", "lydian": "lyd",
            "mix": "mix", "mixolydian": "mix",
            "aeo": "aeo", "aeolian": "aeo",
            "loc": "loc", "locrian": "loc",
        }
        mode = mode_map.get(mode, mode) # Get correct mode abbreviation

        result = f"{tonic}:{mode}"
        print(f"SCLParser: _parse_key: Returning: '{result}'")
        return result

    def _closest_pitch(self, f0):
        """Round the given pitch values to the nearest MIDI note numbers"""
        midi_note = np.around(librosa.hz_to_midi(f0))
        # To preserve the nan values.
        nan_indices = np.isnan(f0)
        # midi_note[nan_indices] = np.nan  # Incorrect: Cannot assign to a float array
        # Convert back to Hz.
        result = librosa.midi_to_hz(midi_note)  # Correct: Modify the result, not the intermediate midi_note
        result = np.where(nan_indices, np.nan, result) #  And assign nan to the result Hz values
        return result


    def _autotune(self, audio, sr, f0, voiced_flag):
        """Applies autotune to the audio based on detected pitch."""
        print(f"SCLParser: _autotune called. Singing: {self.vocaloid_settings['singing']}, Key: {self.vocaloid_settings['key']}, Method: {self.vocaloid_settings['correction_method']}")

        if not self.vocaloid_settings['singing']:
            print("SCLParser: _autotune: Singing is disabled. Returning original audio.")
            return audio

        if self.vocaloid_settings['key'] is None:
            print("SCLParser: _autotune: Key is not specified.  Using 'closest' pitch correction.")
            correction_function = self._closest_pitch
        else:
            print(f"SCLParser: _autotune: Using key: {self.vocaloid_settings['key']}")
            if self.vocaloid_settings['correction_method'] == 'scale':
                correction_function = partial(self._closest_pitch_from_scale, scale=self.vocaloid_settings['key'])
            else:  # Default to 'closest' even if an invalid method is specified
                correction_function = self._closest_pitch

        # Apply the chosen adjustment strategy to the pitch, but only where voiced.
        corrected_f0 = np.copy(f0)  # Work on a copy to avoid modifying the original f0
        for i in range(len(f0)):
            if voiced_flag[i]:
                corrected_f0[i] = correction_function(f0[i])

        # Create a pitch-shifted audio signal using the corrected pitch.  This replaces PSOLA.
        print("SCLParser: _autotune: Applying pitch shift based on corrected f0.")
        pitch_shift_factor = corrected_f0 / f0
        # Replace inf and nan with 1.0 (no shift) to avoid errors
        pitch_shift_factor[np.isinf(pitch_shift_factor)] = 1.0
        pitch_shift_factor[np.isnan(pitch_shift_factor)] = 1.0

        # Use librosa.effects.pitch_shift, applying it per frame.
        hop_length = 512  # Must match hop_length used in detect_pitch_pyin
        shifted_audio = np.zeros_like(audio)
        for i in range(len(pitch_shift_factor)):
            start_index = i * hop_length
            end_index = min((i + 1) * hop_length, len(audio))  # Handle last frame
            frame = audio[start_index:end_index]
            # Apply pitch shift to this frame.
            shifted_frame = librosa.effects.pitch_shift(frame, sr=sr, n_steps=12 * np.log2(pitch_shift_factor[i]))
            # Place the shifted frame into the output audio, handling length differences
            shifted_len = len(shifted_frame)
            if shifted_len <= (end_index-start_index):
                shifted_audio[start_index:start_index + shifted_len] = shifted_frame
            else: #rare case
                shifted_audio[start_index:end_index] = shifted_frame[:end_index-start_index]


        print("SCLParser: _autotune: Pitch shifting complete.")
        return shifted_audio


    def speak_with_settings(self, text, speaker):
        print(f"SCLParser: speak_with_settings called. Text: '{text}', Speaker: {speaker}, Settings: {self.voice_settings}")
        if not text.strip():
            print("SCLParser: speak_with_settings: Text is empty. Skipping.")
            return

        print(f"Speaking: '{text}' with settings: {self.voice_settings}")
        temp_filepath = "temp_audio.wav"
        
        # Apply Vocaloid effects *before* TTS
        modified_audio = self._apply_vocaloid_effects(np.zeros((1,self.sample_rate))) # Dummy audio, because we modify settings not audio

        try:
            self.model.tts_to_file(text, self.speaker_ids[speaker], temp_filepath, speed=self.voice_settings['rate'])
        except Exception as e:
            print(f"SCLParser: speak_with_settings: ERROR during TTS: {e}")
            return

        if self.captured_audio is not None:
            audio_np = self.captured_audio
            print(f"SCLParser: speak_with_settings: Captured audio shape: {audio_np.shape}")

            if len(audio_np.shape) == 1:
                audio_np = np.expand_dims(audio_np, axis=0)
                print("SCLParser: speak_with_settings: Expanded audio to shape: {audio_np.shape}")

            # Ensure mono for pitch detection and autotune
            if len(audio_np.shape) > 1 and audio_np.shape[0] > 1:
                audio_np = librosa.to_mono(audio_np)
                print("SCLParser: speak_with_settings: Converted to mono.")
            elif len(audio_np.shape) > 1:
                audio_np = audio_np[0]

            # --- Pitch Detection (PYIN) ---
            hop_length = 512
            frame_length = 2048
            f0, voiced_flag, _ = self.detect_pitch_pyin(audio_np, self.sample_rate, hop_length=hop_length, frame_length=frame_length)

            # --- Autotune (replaces PSOLA for pitch shifting) ---
            if self.vocaloid_settings['singing']:
                audio_np = self._autotune(audio_np, self.sample_rate, f0, voiced_flag)
            # --- Constant Pitch Shift (if not singing) ---
            elif self.voice_settings['pitch'] != 0.0:
                print(f"SCLParser: speak_with_settings: Applying constant pitch shift: {self.voice_settings['pitch']} semitones")
                audio_np = librosa.effects.pitch_shift(audio_np, sr=self.sample_rate, n_steps=self.voice_settings['pitch'])

            # --- End Autotune/Pitch Shift ---

            self.audio_segments.append(audio_np)
            print(f"SCLParser: speak_with_settings: Added audio segment to list. Total segments: {len(self.audio_segments)}")
        else:
            print("SCLParser: speak_with_settings: Error: Audio data was not captured.")
            return

    def crossfade_segments(self, segments, sample_rate, crossfade_ms=10):
        """
        Combines audio segments with crossfading, handling edge cases and silences.

        This improved version addresses the "weird silence" issue by:
        1.  Reducing the default crossfade duration to 10ms (adjustable).
        2.  Checking for and removing leading/trailing silence *before* crossfading.
        3.  Using a shorter, more precise silence detection threshold.

        Args:
            segments (list): List of NumPy arrays (audio segments).
            sample_rate (int):  The sample rate of the audio.
            crossfade_ms (int): Crossfade duration in milliseconds.
        """

        print(f"SCLParser: crossfade_segments called. Number of segments: {len(segments)}, Sample Rate: {sample_rate}, Crossfade (ms): {crossfade_ms}")
        crossfade_samples = int(sample_rate * crossfade_ms / 1000)
        print(f"SCLParser: crossfade_segments: Crossfade samples: {crossfade_samples}")

        if not segments:
            print("SCLParser: crossfade_segments: No segments. Returning empty array.")
            return np.array([], dtype=np.float32)

        # Trim leading/trailing silence from *each* segment *before* combining
        trimmed_segments = []
        for seg in segments:
            if len(seg.shape) > 1: #stereo
                mono_seg = librosa.to_mono(seg)  # Temporary mono for silence detection
            else: #already mono
                mono_seg = seg

            # Find first and last non-silent samples (more precise threshold)
            non_silent_indices = np.where(np.abs(mono_seg) > 1e-5)[0]

            if len(non_silent_indices) > 0:
                start = non_silent_indices[0]
                end = non_silent_indices[-1] + 1  # Include the last non-silent sample
                trimmed_seg = seg[:, start:end] if len(seg.shape) > 1 else seg[start:end]
                trimmed_segments.append(trimmed_seg)
            else:
                # Segment is entirely silent, keep it as is (might be intentional pause)
                trimmed_segments.append(seg)


        if not trimmed_segments:
            print("SCLParser: crossfade_segments: All segments were completely silent after trimming. Returning empty array.")
            return np.array([], dtype=np.float32)


        combined = trimmed_segments[0]
        print(f"SCLParser: crossfade_segments: Initial combined shape: {combined.shape}")

        for i, seg in enumerate(trimmed_segments[1:]):
            print(f"SCLParser: crossfade_segments: Processing segment {i+1}/{len(trimmed_segments[1:])}")

            if len(combined.shape) == 1:
                combined = np.expand_dims(combined, axis=0)
            if len(seg.shape) == 1:
                seg = np.expand_dims(seg, axis=0)

            if combined.shape[0] == 1 and seg.shape[0] == 1:
                combined = np.repeat(combined, 2, axis=0)
                seg = np.repeat(seg, 2, axis=0)
            elif combined.shape[0] == 2 and seg.shape[0] == 1:
                seg = np.repeat(seg, 2, axis=0)
            elif combined.shape[0] == 1 and seg.shape[0] == 2:
                combined = np.repeat(combined, 2, axis=0)

            if combined.shape[1] < crossfade_samples or seg.shape[1] < crossfade_samples:
                print("SCLParser: crossfade_segments: Segment or combined is shorter than crossfade length. Concatenating directly.")
                combined = np.concatenate((combined, seg), axis=1)
            else:
                print("SCLParser: crossfade_segments: Applying crossfade...")
                window = np.hanning(2 * crossfade_samples)
                fade_out = window[:crossfade_samples]
                fade_in = window[crossfade_samples:]

                combined_tail = combined[:, -crossfade_samples:] * fade_out
                seg_head = seg[:, :crossfade_samples] * fade_in
                crossfaded = combined_tail + seg_head
                combined = np.concatenate((combined[:, :-crossfade_samples], crossfaded, seg[:, crossfade_samples:]), axis=1)

            print(f"SCLParser: crossfade_segments: New combined shape: {combined.shape}")

        print(f"SCLParser: crossfade_segments: Returning combined audio: {combined.shape}")
        return combined

    def post_process(self, audio_np):
        print(f"SCLParser: post_process called. Input audio shape: {audio_np.shape}")

        # Ensure stereo for consistent processing
        if len(audio_np.shape) == 1:
            audio_stereo = np.stack([audio_np, audio_np], axis=0)
            print("SCLParser: post_process: Input was mono. Converted to stereo.")
        elif audio_np.shape[0] == 1:
            audio_stereo = np.repeat(audio_np, 2, axis=0)
            print("SCLParser: post_process: Input was single-channel. Converted to stereo.")
        else:
             audio_stereo = audio_np
        print(f"SCLParser: post_process: Processing audio shape: {audio_stereo.shape}")

        sample_rate = self.sample_rate

        board = Pedalboard([
            Resample(target_sample_rate=44100.0, quality=Resample.Quality.WindowedSinc256),
            Reverb(room_size=0.9, damping=0.7, wet_level=0.00411, dry_level=0.9),
            Limiter(threshold_db=-2, release_ms=1000),
            Chorus(rate_hz=0.4, depth=0.25, centre_delay_ms=7.0, feedback=0.0, mix=0.02),
            Delay(delay_seconds=0.5, feedback=0.01, mix=0.0002),
            Gain(gain_db=-5)
        ])
        print("SCLParser: post_process: Applying Pedalboard effects...")
        audio_resonance = board(audio_stereo, sample_rate)
        print(f"SCLParser: post_process: Audio shape after Pedalboard: {audio_resonance.shape}")

        audio_tensor = torch.tensor(audio_resonance)

        print("SCLParser: post_process: Applying EQ...")
        audio_tensor = F.equalizer_biquad(audio_tensor, sample_rate, center_freq=30, gain=5, Q=3.4)
        audio_tensor = F.equalizer_biquad(audio_tensor, sample_rate, center_freq=100, gain=4, Q=1.4)
        audio_tensor = F.equalizer_biquad(audio_tensor, sample_rate, center_freq=150, gain=1.5, Q=1.4)
        audio_tensor = F.equalizer_biquad(audio_tensor, sample_rate, center_freq=250, gain=2, Q=1.0)
        audio_tensor = F.equalizer_biquad(audio_tensor, sample_rate, center_freq=350, gain=2, Q=1.4)
        audio_tensor = F.equalizer_biquad(audio_tensor, sample_rate, center_freq=450, gain=2, Q=1.8)
        audio_tensor = F.equalizer_biquad(audio_tensor, sample_rate, center_freq=550, gain=-2, Q=1.4)
        audio_tensor = F.equalizer_biquad(audio_tensor, sample_rate, center_freq=2000, gain=2, Q=1.0)
        audio_tensor = F.equalizer_biquad(audio_tensor, sample_rate, center_freq=2500, gain=3, Q=1.4)
        audio_tensor = F.equalizer_biquad(audio_tensor, sample_rate, center_freq=3000, gain=2, Q=1.4)
        audio_tensor = F.equalizer_biquad(audio_tensor, sample_rate, center_freq=3500, gain=4, Q=1.8)
        audio_tensor = F.equalizer_biquad(audio_tensor, sample_rate, center_freq=4000, gain=3, Q=1.4)
        audio_tensor = F.equalizer_biquad(audio_tensor, sample_rate, center_freq=8000, gain=3, Q=1.8)
        audio_tensor = F.equalizer_biquad(audio_tensor, sample_rate, center_freq=12000, gain=3, Q=1.8)
        audio_tensor = F.equalizer_biquad(audio_tensor, sample_rate, center_freq=20000, gain=1, Q=1.8)
        print(f"SCLParser: post_process: Audio shape after EQ: {audio_tensor.shape}")

        # Add noise *after* EQ
        noise = np.random.normal(0, 0.0002, audio_tensor.shape).astype(np.float32)  # Adjust 0.0002 for noise level
        print(f"SCLParser: post_process: Adding noise. Noise shape: {noise.shape}")
        audio_tensor = audio_tensor + torch.from_numpy(noise)


        mod_freq = 1
        mod_depth = 0.03
        print(f"SCLParser: post_process: Applying amplitude modulation. Freq: {mod_freq}, Depth: {mod_depth}")
        modulation = (1 + mod_depth * np.sin(2 * np.pi * mod_freq * np.arange(audio_tensor.shape[1]) / sample_rate)).astype(np.float32)
        audio_tensor = audio_tensor * torch.from_numpy(modulation).unsqueeze(0)
        print(f"SCLParser: post_process: Audio shape after amplitude modulation: {audio_tensor.shape}")

        final_limiter = Pedalboard([
            Limiter(threshold_db=-2, release_ms=1000),
            Resample(target_sample_rate=44100.0, quality=Resample.Quality.WindowedSinc256)
        ])
        print("SCLParser: post_process: Applying final limiter and resampling...")
        audio_processed = final_limiter(audio_tensor.cpu().numpy(), sample_rate)
        print(f"SCLParser: post_process: Final audio shape after limiter and resampling: {audio_processed.shape}")

        return audio_processed

    def parse_attributes(self, params):
        print(f"SCLParser: parse_attributes called. Params: '{params}'")
        attributes = dict(re.findall(r'(\w+)\s*=\s*"([^"]+)"', params))
        print(f"SCLParser: parse_attributes: Returning: {attributes}")
        return attributes

    def is_number(self, s):
        #print(f"SCLParser: is_number called. Input: '{s}'")  # Avoid excessive printing
        try:
            float(s)
            return True
        except ValueError:
            return False

    def parse_pitch(self, pitch_str):
        print(f"SCLParser: parse_pitch called. Input: '{pitch_str}'")
        pitch_str = pitch_str.strip()
        if pitch_str.endswith("%"):
            try:
                val = float(pitch_str[:-1])
                result = val / 100.0 * 12
                print(f"SCLParser: parse_pitch: Percentage pitch. Returning: {result}")
                return result
            except ValueError:
                print("SCLParser: parse_pitch: Invalid percentage value. Returning 0.0")
                return 0.0
        elif pitch_str in self.pitch_map:
            result = self.pitch_map[pitch_str]
            print(f"SCLParser: parse_pitch: Mapped pitch value. Returning: {result}")
            return result
        else:
            try:
                result = float(pitch_str)
                print(f"SCLParser: parse_pitch: Numeric pitch value. Returning: {result}")
                return result
            except ValueError:
                print("SCLParser: parse_pitch: Invalid pitch value. Returning 0.0")
                return 0.0


if __name__ == '__main__':
    device = 'auto'
    print(f"Main: Setting device to: {device}")
    model = TTS(language='EN', device=device)
    print("Main: TTS model initialized.")
    parser = SCLParser(model, device=device)
    print("Main: SCLParser initialized.")
    text = """
[prosody rate="medium"]This is a demonstration of the SCL parser's capabilities.[/prosody]
[pause duration="500"]

[prosody rate="slow" pitch="-3"]Here's an example of slow, low-pitched speech. Notice the difference in both speed and intonation.[/prosody]
[pause duration="short"]

[prosody rate="fast" pitch="2.5"]And now, some fast, high-pitched speech!  It's almost like a chipmunk![/prosody]
[pause duration="long"]

[emphasis level="strong"]This word is strongly emphasized![/emphasis]
[pause duration="200"]
[emphasis level="moderate"]This phrase has moderate emphasis.[/emphasis]
[pause duration="x-long"]

[emotional state="excited"]Wow, this is incredibly exciting! I can't believe how well this works![/emotional]
[pause duration="short"]
[emotional state="somber" rate="0.8" pitch="-1"]This is a very somber and sad statement.  The voice should reflect the emotion.[/emotional]
[pause duration="medium"]

[vocaloid vel="75" dyn="85" bre="10" bri="68" cle="60" ope="70" gen="-5" gwl="20" xsy="50" xsy_voicebanks="Voicebank1,Voicebank2"]This section uses Vocaloid parameters to create a slightly robotic, stylized voice. We're blending Voicebank1 and Voicebank2.[/vocaloid]
[pause duration="1000"]

[vocaloid vel="60" dyn="70" bre="0" bri="50" cle="80" ope="40" gen="10" gwl="0" xsy="100" xsy_voicebanks="Voicebank2,Voicebank3"]Here's another Vocaloid example, with different settings and using Voicebank2 and Voicebank3 for a brighter, clearer sound with a higher gender factor.[/vocaloid]

[pause duration="short"]
Let's try some singing!
[pause duration="short"]

[vocaloid vel=80 dyn=100 gen=-5]
[prosody rate="0.6" pitch="0"]Do [/prosody][pause duration=100][prosody rate="0.6" pitch="2"]Re [/prosody][pause duration=100]
[prosody rate="0.6" pitch="4"]Mi [/prosody][pause duration=100][prosody rate="0.6" pitch="5"]Fa [/prosody][pause duration=100]
[prosody rate="0.6" pitch="7"]So [/prosody][pause duration=100][prosody rate="0.6" pitch="9"]La [/prosody][pause duration=100]
[prosody rate="0.6" pitch="11"]Ti [/prosody][pause duration=100][prosody rate="0.6" pitch="12"]Do![/prosody]
[/vocaloid]

[pause duration="medium"]

[vocaloid vel=90 dyn=95 gen=5]
[prosody rate="0.7" pitch="5"]G [/prosody][pause duration=150] [prosody rate=0.7 pitch="7"]A [/prosody] [pause duration=150]
[prosody rate="0.7" pitch="9"]B [/prosody][pause duration=150] [prosody rate=0.7 pitch="10"]C# [/prosody][pause duration=150]
[prosody rate="0.7" pitch="12"]D [/prosody][pause duration=150] [prosody rate="0.7" pitch="14"]E [/prosody] [pause duration=150]
[prosody rate="0.7" pitch="16"]F# [/prosody][pause duration=150] [prosody rate="0.7" pitch="17"]G# [/prosody]
[/vocaloid]

[pause duration="medium"]

[vocaloid vel=85 dyn=105 gen=-2]
[prosody rate=0.8 pitch="3"]Bb [/prosody][pause duration=200]
[prosody rate=0.8 pitch="5"]C [/prosody][pause duration=200]
[prosody rate=0.8 pitch="7"]D [/prosody][pause duration=200]
[prosody rate=0.8 pitch="8"]Eb [/prosody][pause duration=200]
[prosody rate=0.8 pitch="10"]F [/prosody][pause duration=200]
[prosody rate=0.8 pitch="12"]G [/prosody][pause duration=200]
[prosody rate=0.8 pitch="14"]A [/prosody][pause duration=200]
[prosody rate=0.8 pitch="15"]Bb[/prosody]
[/vocaloid]
[pause duration="medium"]
This concludes the demonstration.
"""

    print(f"Main: Input text: \n{text}")
    parser.parse(text, output_path=OUTPUT_FILE)
    print("Main: Parsing complete.")








"""
    A parser for a custom Speech Control Language (SCL) that allows fine-grained
    control over text-to-speech (TTS) output using the melo library.  SCL
    provides tags for controlling prosody (rate and pitch), inserting pauses,
    adding emphasis, simulating emotional tones, and applying Vocaloid-style
    voice effects.

    Args:
        model (TTS): The melo TTS model instance.
        device (str, optional): The device to use for TTS ('auto', 'cpu', or 'cuda').
            Defaults to 'auto'.

    Attributes:
        model (TTS): The melo TTS model.
        device (str): The device used for TTS.
        speaker_ids (dict): A dictionary mapping speaker names to IDs.
        sample_rate (int): The sample rate of the audio generated by the TTS model.
        voice_settings (dict):  Current voice settings (rate and pitch).
        rate_map (dict):  Mapping of rate keywords (e.g., "slow") to numeric values.
        pitch_map (dict): Mapping of pitch keywords (e.g., "high") to semitone values.
        audio_segments (list): A list to store generated audio segments.
        original_sf_write (function):  The original soundfile.write function.
        captured_audio (np.ndarray): Stores the captured audio data from the TTS engine.
        vocaloid_settings (dict): Dictionary of Vocaloid parameter settings.
        xsy_profiles (dict): Dictionary of EQ profiles for Vocaloid cross-synthesis.

    SCL Syntax:

    The SCL uses square bracket tags to control the speech output.  Tags can be
    nested (though nesting is automatically flattened), and closing tags reset
    the settings applied by the corresponding opening tag.  All tags are
    case-insensitive (e.g., `[Prosody]` is the same as `[prosody]`).

    1. Prosody Tag: `[prosody rate="..." pitch="..."]`

        - Controls the speaking rate and pitch.
        - Attributes:
            - `rate` (optional):  Sets the speaking rate.
                - Keywords: "x-slow", "slow", "medium", "fast", "x-fast"
                - Numeric values:  Floating-point numbers (e.g., 0.8, 1.2).  Values are clamped between 0.5 and 1.1.
                - Default: "medium" (which is internally mapped to a rate slightly faster than the model's default)
            - `pitch` (optional): Sets the pitch.
                - Keywords: "x-low", "low", "medium", "high", "x-high"
                - Numeric values:  Floating-point numbers representing semitone shifts (e.g., -2.0, 1.5).
                - Percentage values: Strings ending with "%" (e.g., "50%", "-20%").  These are converted to semitone shifts (100% = 12 semitones).
                - Default: "medium" (0.0 semitones)
        - Closing Tag: `[/prosody]` Resets rate and pitch to their default values (rate=1.0, pitch=0.0).

        Example:
        ```
        [prosody rate="slow" pitch="-2"]This is slow and low-pitched speech.[/prosody]
        [prosody rate="1.2" pitch="50%"]This is faster and higher-pitched.[/prosody]
        ```

    2. Pause Tag: `[pause duration="..."]`

        - Inserts a pause in the speech.
        - Attributes:
            - `duration` (required): Specifies the pause duration.
                - Keywords: "short" (250ms), "medium" (500ms), "long" (1000ms), "x-long" (1500ms)
                - Numeric values:  Integers representing milliseconds (e.g., 300, 750).  You can also include "ms" (e.g., "200ms").
        - This tag does not have a closing tag.

        Example:
        ```
        [pause duration="short"]  // Short pause (250ms)
        [pause duration="1000"] // Long pause (1000ms)
        [pause duration="750ms"] // Pause of 750ms
        ```

    3. Emphasis Tag: `[emphasis level="..." pitch="..."]`

        - Adds emphasis to a word or phrase.  Emphasis is achieved by modifying both rate and pitch.
        - Attributes:
            - `level` (optional): The level of emphasis.
                - Keywords: "strong", "moderate", "reduced"
                - Default: "moderate"
            - `pitch` (optional): Overrides the default pitch adjustment for the given emphasis level.  Uses the same values as the `pitch` attribute in the `prosody` tag.
        - Closing Tag: `[/emphasis]` Resets rate and pitch to their default values.

        Example:
        ```
        [emphasis level="strong"]This is emphasized![/emphasis]
        [emphasis level="reduced" pitch="low"]This is less emphasized.[/emphasis]
        ```

    4. Emotional Tag: `[emotional state="..." rate="..." pitch="..."]`

        - Adjusts the voice to convey emotion.
        - Attributes:
            - `state` (optional): The emotional state.
                - Keywords: "excited", "somber", "neutral"
                - Default: "neutral"
            - `rate` (optional): Overrides the default rate adjustment for the given emotional state.  Uses the same values as the `rate` attribute in the `prosody` tag.
            - `pitch` (optional): Overrides the default pitch adjustment for the given emotional state. Uses the same values as the `pitch` attribute in the `prosody` tag.
        - Closing Tag: `[/emotional]` Resets rate and pitch to their default values.
        - If `rate` or `pitch` are not provided, default values based on the `state` are used.

        Example:
        ```
        [emotional state="excited" rate="1.2"]This is exciting![/emotional]
        [emotional state="somber"]This is somber.[/emotional]
        ```

    5. Vocaloid Tag: `[vocaloid vel="..." dyn="..." bre="..." bri="..." cle="..." ope="..." gen="..." gwl="..." xsy="..." xsy_voicebanks="..."]`

        - Applies Vocaloid-style voice effects.  These effects are applied *before* the TTS engine generates the audio, allowing for a wide range of vocal modifications.
        - Attributes:
            - `vel` (Velocity):  Simulates vocal velocity (0-127, default 64).  Higher values can make the voice sound more forceful.
            - `dyn` (Dynamics): Controls the output volume (0-127, default 64).
            - `bre` (Breathiness): Adds a subtle breathy quality (0-127, default 0).
            - `bri` (Brightness):  Increases the high-frequency content (0-127, default 64).
            - `cle` (Clearness):  Adjusts the clarity of the voice (0-127, default 64).
            - `ope` (Opening): Simulates vocal tract opening (0-127, default 64).
            - `gen` (Gender):  Shifts the perceived gender of the voice (-127 to 127, default 64, negative values for more masculine, positive for more feminine).
            - `gwl` (Growl):  Adds a growl effect (0-127, default 0).
            - `xsy` (Cross-Synthesis): Blends between two voicebanks (0-127, default 0, 0=Voicebank1, 127=Voicebank2).
            - `xsy_voicebanks`:  Specifies the two voicebanks to use for cross-synthesis (comma-separated, e.g., "Voicebank1, Voicebank2").  Available voicebanks are "Voicebank1", "Voicebank2", and "Voicebank3".
        - Closing Tag: `[/vocaloid]` Resets all Vocaloid parameters to their default values.

        Example:
        ```
        [vocaloid vel="80" dyn="90" bre="20" bri="74" cle="54" ope="74" gen="-20" gwl="40" xsy="64" xsy_voicebanks="Voicebank1,Voicebank2"]This has Vocaloid effects.[/vocaloid]
        [vocaloid xsy="127" xsy_voicebanks="Voicebank2,Voicebank3"]This uses cross-synthesis.[/vocaloid]
        ```

    6. `say-as` and `voice` Tags: `[say-as ...]` and `[voice ...]`

        - These tags are recognized but do not have any implemented functionality.
        - They are placeholders for future extensions.

    Combining Tags:

    You can combine multiple tags to create complex effects. For example, you can use `[prosody]` to set the overall rate and pitch, and then use `[emphasis]` to emphasize specific words within that prosodic context.  Tags are processed sequentially, so later tags override earlier ones within the same segment.

    Example:
    ```
    [prosody rate="slow" pitch="-1"]This is slow speech. [emphasis level="strong"]This word is emphasized![/emphasis] And this is back to slow.[/prosody]
    ```

    Pitch Shifting:

    Pitch shifting is implemented using the `rubberband` library, which provides high-quality pitch shifting. If you intend to use pitch shifting, make sure to install `pyrubberband`:

    ```bash
    pip install pyrubberband
    ```

    If `pyrubberband` is not installed, the parser will still function, but pitch-shifting tags will not have any effect, and a warning message will be printed.

    Example combining various features (for a singing-like effect):
    ```
     [vocaloid vel="85" dyn="95" bre="5" bri="80" cle="55" ope="75" gen="-2" gwl="0" xsy="20" xsy_voicebanks="Voicebank1,Voicebank3"]
      [prosody rate="medium" pitch="x-high"]Never gonna make you cry[/prosody]
     [/vocaloid]
     [pause duration="short"]
    ```
"""