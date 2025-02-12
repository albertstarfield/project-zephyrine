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
        self.model = model
        self.device = device
        self.speaker_ids = model.hps.data.spk2id
        self.sample_rate = model.hps.data.sampling_rate
        self.voice_settings = {
            'rate': 1.0,
            'pitch': 0.0,  # in semitones (constant offset)
        }
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
        self.audio_segments = []
        self.original_sf_write = sf.write
        self.captured_audio = None

    def _new_sf_write(self, file, data, samplerate, *args, **kwargs):
        if isinstance(data, torch.Tensor):
            self.captured_audio = data.clone().detach().cpu().numpy()
        else:
            self.captured_audio = data.copy()
        return self.original_sf_write(file, data, samplerate, *args, **kwargs)

    def parse(self, text, output_path='output.mp3', speaker='EN-US'):
        sf.write = self._new_sf_write
        processed_text = self.preprocess_text(text)
        segments = self.split_into_segments(processed_text)
        for segment in segments:
            self.parse_segment(segment, speaker)

        if self.audio_segments:
            final_audio = self.crossfade_segments(self.audio_segments, self.sample_rate)
            final_audio_processed = self.post_process(final_audio)
            torchaudio.save(output_path, torch.tensor(final_audio_processed), self.sample_rate)
            if 'IPython.display' in sys.modules:
                display(Audio(data=final_audio_processed, rate=self.sample_rate, autoplay=True))

        self.audio_segments = []
        sf.write = self.original_sf_write
        self.captured_audio = None

    def preprocess_text(self, text):
        text = self.fix_missing_closing_tags(text)
        text = self.flatten_nested_tags(text)
        return text

    def fix_missing_closing_tags(self, text):
        open_tags = []
        result = []
        i = 0
        while i < len(text):
            if text[i] == '[':
                closing_bracket_index = text.find(']', i)
                if closing_bracket_index == -1:
                    i += 1
                    continue
                tag_content = text[i + 1:closing_bracket_index]
                if tag_content.startswith('/'):
                    tag_name = tag_content[1:]
                    if open_tags and open_tags[-1] == tag_name:
                        open_tags.pop()
                        result.append(text[i:closing_bracket_index + 1])
                else:
                    tag_name = tag_content.split(' ')[0] if ' ' in tag_content else tag_content
                    result.append(text[i:closing_bracket_index + 1])
                    open_tags.append(tag_name)
                i = closing_bracket_index + 1
            else:
                result.append(text[i])
                i += 1
        while open_tags:
            result.append(f"[/{open_tags.pop()}]")
        return "".join(result)

    def flatten_nested_tags(self, text):
        while True:
            match = re.search(r'\[([^/][^\]]*)\]([^\[]*)\[([^\]]*)\](.*?)\[/\3\](.*?)\[/\1\]', text)
            if not match:
                break
            tag1, text1, tag2, text2, text3 = match.groups()
            replacement = f"[{tag1}]{text1}[/{tag1}][{tag2}]{text2}[/{tag2}][{tag1}]{text3}[/{tag1}]"
            text = text.replace(match.group(0), replacement, 1)
        return text

    def split_into_segments(self, text):
        segments = []
        current_segment = ""
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
        for sentence in sentences:
            current_segment += sentence + " "
            pause_matches = list(re.finditer(r'\[pause\s+([^\]]+)\]', current_segment))
            if pause_matches:
                last_index = 0
                for match in pause_matches:
                    segments.append(current_segment[last_index:match.start()])
                    segments.append(match.group(0))
                    last_index = match.end()
                segments.append(current_segment[last_index:])
                current_segment = ""
            elif len(current_segment) > 150:
                segments.append(current_segment)
                current_segment = ""
        if current_segment.strip():
            segments.append(current_segment.strip())
        return segments

    def parse_segment(self, segment, speaker):
        tags = re.findall(r'\[([^\]]+)\]', segment)
        current_text = segment
        for tag in tags:
            parts = current_text.split(f"[{tag}]", 1)
            before_text = parts[0]
            if before_text.strip():
                self.speak_with_settings(before_text, speaker)
            if tag.startswith('/'):
                self.reset_settings(tag[1:])
            else:
                self.apply_settings(tag)
            current_text = parts[1] if len(parts) > 1 else ""
        if current_text.strip():
            self.speak_with_settings(current_text, speaker)

    def apply_settings(self, tag_str):
        parts = tag_str.split(' ', 1)
        tag_name = parts[0].lower()
        params = parts[1] if len(parts) > 1 else ""
        attrs = self.parse_attributes(params) if params else {}

        if tag_name == "pause":
            duration = attrs.get("duration", list(attrs.values())[0] if attrs else "medium")
            if duration in ("short", "medium", "long", "x-long"):
                duration_ms = {"short": 250, "medium": 500, "long": 1000, "x-long": 1500}[duration]
            elif duration.isdigit():
                duration_ms = int(duration)
            else:
                duration_ms = 0
            silence = np.zeros(int(self.sample_rate * duration_ms / 1000), dtype=np.float32)
            self.audio_segments.append(silence)
            print(f"  (PAUSE: {duration_ms}ms)")

        elif tag_name == "prosody":
            if "rate" in attrs:
                rate_value = attrs["rate"]
                # Get rate, but clamp to min 0.5 and max 1.1:
                new_rate = self.rate_map.get(rate_value, float(rate_value) if self.is_number(rate_value) else 1.0)
                new_rate = max(0.5, min(1.1, new_rate))  # Enforce limits
                self.voice_settings['rate'] = new_rate
                print(f"  (PROSODY RATE: {self.voice_settings['rate']})")
            if "pitch" in attrs:
                pitch_value = attrs["pitch"]
                self.voice_settings['pitch'] = self.parse_pitch(pitch_value)
                print(f"  (PROSODY PITCH: {self.voice_settings['pitch']})")

        elif tag_name == "emphasis":
            level = attrs.get("level", "moderate")
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
            print(f"  (EMPHASIS: level={level}, rate={self.voice_settings['rate']}, pitch={self.voice_settings['pitch']})")

        elif tag_name == "emotional":
            state = attrs.get("state", "neutral").lower()
            state_pitch_map = {"excited": 2.0, "somber": -2.0, "neutral": 0.0}
            state_rate_map = {"excited": 1.1, "somber": 0.9, "neutral": 1.0}
            rate_value = attrs.get("rate", None)
            if rate_value:
                rate = self.rate_map.get(rate_value, float(rate_value) if self.is_number(rate_value) else 1.0)
            else:
                rate = state_rate_map.get(state, 1.0)
            pitch_value = attrs.get("pitch", None)
            if pitch_value:
                pitch = self.parse_pitch(pitch_value)
            else:
                pitch = state_pitch_map.get(state, 0.0)
            self.voice_settings['rate'] *= rate
            self.voice_settings['pitch'] += pitch

            # Clamp rate after emotional adjustment:
            self.voice_settings['rate'] = max(0.5, min(1.1, self.voice_settings['rate']))
            print(f"  (EMOTIONAL: state={state}, rate factor={rate}, pitch shift={pitch})")

        elif tag_name == 'say-as':
            pass
        elif tag_name == 'voice':
            pass

    def reset_settings(self, tag_name):
        if tag_name in ("prosody", "emphasis", "emotional"):
            self.voice_settings['rate'] = 1.0  # Reset to default (which is now "slow")
            self.voice_settings['pitch'] = 0.0
            print(f"  (RESET {tag_name.upper()})")

    def speak_with_settings(self, text, speaker):
        if not text.strip():
            return
        print(f"Speaking: '{text}' with settings: {self.voice_settings}")
        temp_filepath = "temp_audio.wav"
        self.model.tts_to_file(text, self.speaker_ids[speaker], temp_filepath, speed=self.voice_settings['rate'])

        if self.captured_audio is not None:
            audio_np = self.captured_audio
            if self.voice_settings['pitch'] != 0.0:
                # Ensure audio_np is (channel, time) for F.resample
                if len(audio_np.shape) == 1:
                    audio_np = np.expand_dims(audio_np, axis=0)  # Make it (1, time)

                audio_tensor = torch.from_numpy(audio_np)
                target_sample_rate = int(self.sample_rate * (2 ** (self.voice_settings['pitch'] / 12)))
                audio_tensor = F.resample(audio_tensor, self.sample_rate, target_sample_rate)
                audio_np = audio_tensor.numpy()


            self.audio_segments.append(audio_np)
        else:
            print("Error: Audio data was not captured.")
            return

    def crossfade_segments(self, segments, sample_rate, crossfade_ms=30):
        crossfade_samples = int(sample_rate * crossfade_ms / 1000)
        if not segments:
            return np.array([], dtype=np.float32)
        combined = segments[0]
        for seg in segments[1:]:
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
                combined = np.concatenate((combined, seg), axis=1)
            else:
                window = np.hanning(2 * crossfade_samples)
                fade_out = window[:crossfade_samples]
                fade_in = window[crossfade_samples:]
                combined_tail = combined[:, -crossfade_samples:] * fade_out
                seg_head = seg[:, :crossfade_samples] * fade_in
                crossfaded = combined_tail + seg_head
                combined = np.concatenate((combined[:, :-crossfade_samples], crossfaded, seg[:, crossfade_samples:]), axis=1)
        return combined

    def post_process(self, audio_np):
        # Ensure stereo for consistent processing
        if len(audio_np.shape) == 1:
            audio_stereo = np.stack([audio_np, audio_np], axis=0)
        elif audio_np.shape[0] == 1:
            audio_stereo = np.repeat(audio_np, 2, axis=0)
        else:
             audio_stereo = audio_np
        sample_rate = self.sample_rate


        # Add noise *before* reverb
        noise = np.random.normal(0, 0.001, audio_stereo.shape).astype(np.float32)  # Adjust 0.001 for noise level
        audio_stereo = audio_stereo + noise

        board = Pedalboard([
            Resample(target_sample_rate=44100.0, quality=Resample.Quality.WindowedSinc256),
            Reverb(room_size=0.9, damping=0.4, wet_level=0.00811, dry_level=0.7),
            Limiter(threshold_db=-2, release_ms=1000),
            Chorus(rate_hz=0.4, depth=0.25, centre_delay_ms=7.0, feedback=0.0, mix=0.02),
            Delay(delay_seconds=1, feedback=0.2, mix=0.008),
            Gain(gain_db=-5)
        ])
        audio_resonance = board(audio_stereo, sample_rate)
        audio_tensor = torch.tensor(audio_resonance)

        audio_tensor = F.highpass_biquad(audio_tensor, sample_rate, cutoff_freq=5, Q=0.7)
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

        mod_freq = 1
        mod_depth = 0.03
        modulation = (1 + mod_depth * np.sin(2 * np.pi * mod_freq * np.arange(audio_tensor.shape[1]) / sample_rate)).astype(np.float32)
        audio_tensor = audio_tensor * torch.from_numpy(modulation).unsqueeze(0)

        final_limiter = Pedalboard([
            Limiter(threshold_db=-2, release_ms=1000),
            Resample(target_sample_rate=44100.0, quality=Resample.Quality.WindowedSinc256)
        ])
        audio_processed = final_limiter(audio_tensor.cpu().numpy(), sample_rate)
        return audio_processed

    def parse_attributes(self, params):
        return dict(re.findall(r'(\w+)\s*=\s*"([^"]+)"', params))

    def is_number(self, s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    def parse_pitch(self, pitch_str):
        pitch_str = pitch_str.strip()
        if pitch_str.endswith("%"):
            try:
                val = float(pitch_str[:-1])
                return val / 100.0 * 12
            except ValueError:
                return 0.0
        elif pitch_str in self.pitch_map:
            return self.pitch_map[pitch_str]
        else:
            try:
                return float(pitch_str)
            except ValueError:
                return 0.0
if __name__ == '__main__':
    device = 'auto'
    model = TTS(language='EN', device=device)
    parser = SCLParser(model, device=device)
    text = """
[prosody rate="medium"]I died.  Unexpected, really.  One moment I was crossing the street, the next... darkness.[/prosody] [pause duration="short"]
[prosody rate="slow"]Then, a strange sensation.  A... squishiness?[/prosody] [pause duration="medium"]
[prosody rate="medium" pitch="low"]I opened my... well, I didn't have eyes, but I *perceived* my new form.  A blue, gelatinous blob.  A slime.  Of all things.[/prosody] [pause duration="short"]

[prosody rate="medium"]The world was a confusing mess of magical energy and unfamiliar creatures.[/prosody] [pause duration="short"]
[prosody rate="fast"]Goblins, mostly.  Nasty little things, always picking fights.[/prosody] [pause duration="short"]
[prosody rate="medium"]But I quickly discovered something amazing: I could absorb things.  Other creatures, plants, even rocks![/prosody] [pause duration="short"]
[prosody rate="fast" pitch="high"]And with each absorption, I gained their abilities.  The goblins' crude strength, the plants' regeneration, even the rocks' resilience![/prosody] [pause duration="short"]

"""
    parser.parse(text, output_path=OUTPUT_FILE)