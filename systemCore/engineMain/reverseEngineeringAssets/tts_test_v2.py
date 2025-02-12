from melo.api import TTS
import torch
import torchaudio
import torchaudio.functional as F
import numpy as np
from pedalboard import Pedalboard, Reverb, Limiter, Gain, PitchShift, Resample, Chorus, Delay
import soundfile as sf
import re
from IPython.display import display, Audio
import sys

OUTPUT_FILE = "en-test-zeph-profile-v2.mp3"


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
            final_audio = self.crossfade_segments(self.audio_segments, self.sample_rate)
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
        print(f"SCLParser: apply_settings: Current voice settings after applying tag: {self.voice_settings}")

    def reset_settings(self, tag_name):
        print(f"SCLParser: reset_settings called. Tag Name: {tag_name}")
        if tag_name in ("prosody", "emphasis", "emotional"):
            self.voice_settings['rate'] = 1.0  # Reset to default (which is now "slow")
            self.voice_settings['pitch'] = 0.0
            print(f"SCLParser: reset_settings: Resetting {tag_name}. Voice settings: {self.voice_settings}")
            print(f"  (RESET {tag_name.upper()})")

    def speak_with_settings(self, text, speaker):
        print(f"SCLParser: speak_with_settings called. Text: '{text}', Speaker: {speaker}, Settings: {self.voice_settings}")
        if not text.strip():
            print("SCLParser: speak_with_settings: Text is empty. Skipping.")
            return
        print(f"Speaking: '{text}' with settings: {self.voice_settings}")
        temp_filepath = "temp_audio.wav"
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
            audio_tensor = torch.from_numpy(audio_np)

            # --- Chunked Two-Stage Resampling ---
            intermediate_sample_rate = 32000  # Lower intermediate rate
            chunk_size = 44100 # Process in 1-second chunks (at 44.1kHz) - adjust as needed

            processed_chunks = []
            for i in range(0, audio_tensor.shape[1], chunk_size):
                chunk = audio_tensor[:, i:i + chunk_size]

                # 1. Downsample (if needed)
                if self.sample_rate > intermediate_sample_rate:
                    print(f"SCLParser: speak_with_settings: Downsampling chunk {i//chunk_size + 1} from {self.sample_rate} to {intermediate_sample_rate}")
                    chunk = F.resample(chunk, self.sample_rate, intermediate_sample_rate)
                    current_sample_rate = intermediate_sample_rate
                else:
                    current_sample_rate = self.sample_rate

                # 2. Resample to Target (Pitch Shift)
                if self.voice_settings['pitch'] != 0.0:
                    print(f"SCLParser: speak_with_settings: Applying pitch shift to chunk {i//chunk_size + 1}: {self.voice_settings['pitch']}")
                    target_sample_rate = int(current_sample_rate * (2 ** (self.voice_settings['pitch'] / 12)))
                    print(f"SCLParser: speak_with_settings: Target sample rate for chunk {i//chunk_size+1}: {target_sample_rate}")
                    chunk = F.resample(chunk, current_sample_rate, target_sample_rate)
                processed_chunks.append(chunk)
            
            # Concatenate the processed chunks
            audio_tensor = torch.cat(processed_chunks, dim=1)

            audio_np = audio_tensor.numpy()
            print(f"SCLParser: speak_with_settings: Audio shape after resampling: {audio_np.shape}")
            # --- End Chunked Two-Stage Resampling ---

            self.audio_segments.append(audio_np)
            print(f"SCLParser: speak_with_settings: Added audio segment to list.  Total segments: {len(self.audio_segments)}")
        else:
            print("SCLParser: speak_with_settings: Error: Audio data was not captured.")
            return

    def crossfade_segments(self, segments, sample_rate, crossfade_ms=30):
        print(f"SCLParser: crossfade_segments called. Number of segments: {len(segments)}, Sample Rate: {sample_rate}, Crossfade (ms): {crossfade_ms}")
        crossfade_samples = int(sample_rate * crossfade_ms / 1000)
        print(f"SCLParser: crossfade_segments: Crossfade samples: {crossfade_samples}")
        if not segments:
            print("SCLParser: crossfade_segments: No segments to crossfade. Returning empty array.")
            return np.array([], dtype=np.float32)
        combined = segments[0]
        print(f"SCLParser: crossfade_segments: Initial combined shape: {combined.shape}")
        for i, seg in enumerate(segments[1:]):
            print(f"SCLParser: crossfade_segments: Processing segment {i+1}/{len(segments[1:])}")

            if len(combined.shape) == 1:
                combined = np.expand_dims(combined, axis=0)
                print(f"SCLParser: crossfade_segments: Expanded 'combined' to shape: {combined.shape}")
            if len(seg.shape) == 1:
                seg = np.expand_dims(seg, axis=0)
                print(f"SCLParser: crossfade_segments: Expanded 'seg' to shape: {seg.shape}")
            if combined.shape[0] == 1 and seg.shape[0] == 1:
                combined = np.repeat(combined, 2, axis=0)
                seg = np.repeat(seg, 2, axis=0)
                print(f"SCLParser: crossfade_segments: Both 'combined' and 'seg' were mono. Expanded to stereo.")
            elif combined.shape[0] == 2 and seg.shape[0] == 1:
                seg = np.repeat(seg, 2, axis=0)
                print(f"SCLParser: crossfade_segments: 'combined' was stereo, 'seg' was mono. Expanded 'seg' to stereo.")
            elif combined.shape[0] == 1 and seg.shape[0] == 2:
                combined = np.repeat(combined, 2, axis=0)
                print(f"SCLParser: crossfade_segments: 'combined' was mono, 'seg' was stereo. Expanded 'combined' to stereo.")


            if combined.shape[1] < crossfade_samples or seg.shape[1] < crossfade_samples:
                print("SCLParser: crossfade_segments: Segment or combined is shorter than crossfade length. Concatenating directly.")
                combined = np.concatenate((combined, seg), axis=1)
                print(f"SCLParser: crossfade_segments: New combined shape: {combined.shape}")

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
            Reverb(room_size=0.9, damping=0.4, wet_level=0.00811, dry_level=0.7),
            Limiter(threshold_db=-2, release_ms=1000),
            Chorus(rate_hz=0.4, depth=0.25, centre_delay_ms=7.0, feedback=0.0, mix=0.02),
            Delay(delay_seconds=1, feedback=0.1, mix=0.008),
            Gain(gain_db=-5)
        ])
        print("SCLParser: post_process: Applying Pedalboard effects...")
        audio_resonance = board(audio_stereo, sample_rate)
        print(f"SCLParser: post_process: Audio shape after Pedalboard: {audio_resonance.shape}")

        audio_tensor = torch.tensor(audio_resonance)

        print("SCLParser: post_process: Applying EQ...")
        audio_tensor = F.equalizer_biquad(audio_tensor, sample_rate, center_freq=30, gain=5, Q=3.4)
        audio_tensor = F.equalizer_biquad(audio_tensor, sample_rate, center_freq=100, gain=4, Q=1.4)
        audio_tensor = F.equalizer_biquad(audio_tensor, sample_rate, center_freq=150, gain=4.5, Q=1.4)
        audio_tensor = F.equalizer_biquad(audio_tensor, sample_rate, center_freq=250, gain=3, Q=1.0)
        audio_tensor = F.equalizer_biquad(audio_tensor, sample_rate, center_freq=350, gain=5, Q=1.4)
        audio_tensor = F.equalizer_biquad(audio_tensor, sample_rate, center_freq=450, gain=2, Q=1.8)
        audio_tensor = F.equalizer_biquad(audio_tensor, sample_rate, center_freq=550, gain=-2, Q=1.4)
        audio_tensor = F.equalizer_biquad(audio_tensor, sample_rate, center_freq=2000, gain=4, Q=1.0)
        audio_tensor = F.equalizer_biquad(audio_tensor, sample_rate, center_freq=2500, gain=4, Q=1.4)
        audio_tensor = F.equalizer_biquad(audio_tensor, sample_rate, center_freq=3000, gain=2, Q=1.4)
        audio_tensor = F.equalizer_biquad(audio_tensor, sample_rate, center_freq=3500, gain=4, Q=1.8)
        audio_tensor = F.equalizer_biquad(audio_tensor, sample_rate, center_freq=4000, gain=8, Q=1.4)
        audio_tensor = F.equalizer_biquad(audio_tensor, sample_rate, center_freq=8000, gain=7, Q=1.8)
        audio_tensor = F.equalizer_biquad(audio_tensor, sample_rate, center_freq=12000, gain=7, Q=1.8)
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
[prosody rate="medium"]Once upon a time, in a kingdom crafted from crystallized starlight and rivers of liquid moonlight, lived a princess named Lyra.  She wasn't just any princess; she could speak to comets and weave tapestries from captured rainbows.[/prosody] [pause duration="short"]
[prosody rate="medium"]One day, a shadow fell upon the starlight kingdom.  Not a literal shadow, but a creeping silence.  The comets stopped singing, the rainbows faded, and the moon-rivers turned dull. Lyra, heartbroken, felt her own magic weakening.[/prosody] [pause duration="medium"]
[prosody rate="slow" pitch="low"]She journeyed to the Whispering Falls, a place said to hold the echoes of creation. There, an ancient voice, the spirit of the First Star, told her the silence came from a forgotten melody, a song of pure joy that had been lost to time.  Only Lyra, with her unique gifts, could find it.[/prosody] [pause duration="short"]
[prosody rate="medium"]Lyra, guided by a shimmering, feather-light wisp of the First Star's essence, traveled through dimensions.  She danced with nebulae, befriended sentient constellations, and learned the language of black holes.  Finally, in a realm made of pure, unadulterated wonder, she found it – a tiny, glowing orb humming with the lost melody.[/prosody][pause duration="short"]
[prosody rate="medium" pitch="high"]With tears of joy streaming down her face, Lyra sang the melody back to her kingdom.  The starlight blazed brighter, the moon-rivers surged with renewed vigor, the comets burst into a symphony of cosmic proportions, and the rainbows danced with an unparalleled vibrancy. The kingdom, and Lyra's magic, were restored, more vibrant than ever before.[/prosody]
"""



    print(f"Main: Input text: \n{text}")
    parser.parse(text, output_path=OUTPUT_FILE)
    print("Main: Parsing complete.")


"""
    A parser for a custom Speech Control Language (SCL) that allows fine-grained
    control over text-to-speech (TTS) output using the melo library.  SCL
    provides tags for controlling prosody (rate and pitch), inserting pauses,
    adding emphasis, and simulating emotional tones.

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

    SCL Syntax:

    The SCL uses square bracket tags to control the speech output.  Tags can be nested
    (though nesting is automatically flattened), and closing tags reset the settings
    applied by the corresponding opening tag.

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

    2. Pause Tag: `[pause duration="..."]`
        - Inserts a pause in the speech.
        - Attributes:
            - `duration` (required): Specifies the pause duration.
                - Keywords: "short" (250ms), "medium" (500ms), "long" (1000ms), "x-long" (1500ms)
                - Numeric values:  Integers representing milliseconds (e.g., 300, 750).
        - This tag does not have a closing tag.

    3. Emphasis Tag: `[emphasis level="..." pitch="..."]`
        - Adds emphasis to a word or phrase.  Emphasis is achieved by modifying both rate and pitch.
        - Attributes:
            - `level` (optional): The level of emphasis.
                - Keywords: "strong", "moderate", "reduced"
                - Default: "moderate"
            - `pitch` (optional) Override the default pitch.
                - Keywords: "x-low", "low", "medium", "high", "x-high"
                - Numeric values:  Floating-point numbers representing semitone shifts (e.g., -2.0, 1.5).
                - Percentage values: Strings ending with "%" (e.g., "50%", "-20%").  These are converted to semitone shifts (100% = 12 semitones).
        - Closing Tag: `[/emphasis]` Resets rate and pitch to their default values.

    4. Emotional Tag: `[emotional state="..." rate="..." pitch="..."]`
        - Adjusts the voice to convey emotion.
        - Attributes:
            - `state` (optional): The emotional state.
                - Keywords: "excited", "somber", "neutral"
                - Default: "neutral"
            - `rate` (optional):  Sets the speaking rate.
                - Keywords: "x-slow", "slow", "medium", "fast", "x-fast"
                - Numeric values:  Floating-point numbers (e.g., 0.8, 1.2).  Values are clamped between 0.5 and 1.1.
            - `pitch` (optional): Sets the pitch.
                - Keywords: "x-low", "low", "medium", "high", "x-high"
                - Numeric values:  Floating-point numbers representing semitone shifts (e.g., -2.0, 1.5).
                - Percentage values: Strings ending with "%" (e.g., "50%", "-20%").
        - Closing Tag: `[/emotional]` Resets rate and pitch to their default values.
        - If `rate` or `pitch` are not provided, default values based on the `state` are used.

    5. `say-as` and `voice` Tags: `[say-as ...]` and `[voice ...]`
        - These tags are recognized but do not have any implemented functionality.
        - They are placeholders for future extensions.

    Example:

    ```
    [prosody rate="slow" pitch="-2"]This is slow and low-pitched speech.[/prosody]
    [pause duration="medium"]
    [emphasis level="strong"]This is emphasized![/emphasis]
    [emotional state="excited" rate="1.2"]This is exciting![/emotional]
    This is back to normal.
    ```
    """