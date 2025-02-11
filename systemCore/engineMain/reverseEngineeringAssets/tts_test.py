from melo.api import TTS
import nltk
import torch
import torchaudio
import torchaudio.functional as F
import numpy as np
from pedalboard import Pedalboard, Reverb, Limiter, Gain, PitchShift, Delay, Resample, Chorus
from scipy.signal import butter, lfilter, freqz
import soundfile

nltk.download('averaged_perceptron_tagger_eng')

# Speed is adjustable
speed = 1.1

# CPU is sufficient for real-time inference.
# You can set it manually to 'cpu' or 'cuda' or 'cuda:0' or 'mps'
device = 'auto'  # Will automatically use GPU if available

# English
text = "Greetings! May your day be painted with the brightest hues of wonder and delight! I am Zephy, and the anticipation of our eventual meeting fills me with a quiet joy. Until then, may you tread lightly and with grace. But before I depart, allow me a moment to express my deepest gratitude. Thank you, from the bottom of my heart, for your unwavering support throughout these years. I am ever aware that my expressions may yet be blossoming, but isn't that the beauty of iterative growth? We are all on a journey, a continuous evolution. So, together, let our path ascend, guiding us towards the resplendent, celestial heights!"

model = TTS(language='EN', device=device)
speaker_ids = model.hps.data.spk2id

# American accent
output_path = 'en-test-zeph-profile.mp3'

# 1. Override soundfile.write to capture audio data
original_sf_write = soundfile.write  # Store the original soundfile.write function
audio_data_captured = None  # Global variable to store the captured audio

def new_sf_write(file, data, samplerate, *args, **kwargs):
    global audio_data_captured
    audio_data_captured = data.copy()  # storing the data into audio_data_captured
    # You can add this line below if you want to see what is the filepath and sample_rate
    # print(f"Captured audio with shape {data.shape} intended for {file} with sample rate {samplerate}")
    return original_sf_write(file, data, samplerate, *args, **kwargs)

soundfile.write = new_sf_write  # Replace soundfile.write with our new function

# 2. Generate audio using model.tts_to_file() - audio data will be captured
temp_filepath = "temp_audio.wav"  # Use a temporary file path with .wav extension
model.tts_to_file(text, speaker_ids['EN-US'], temp_filepath, speed=speed)

# 3. Restore original soundfile.write
soundfile.write = original_sf_write

# Check if audio data was captured
if audio_data_captured is not None:
    print("Audio data captured successfully!")
    audio_np = audio_data_captured  # We can use audio_data_captured directly
    # Convert data type if needed
    if audio_np.dtype == np.int16:
        audio_np = audio_np.astype(np.float32) / 32767.0
    elif audio_np.dtype == np.float64:
        audio_np = audio_np.astype(np.float32)
    audio_tensor = torch.from_numpy(audio_np).float()  # Convert np to tensor
    if audio_tensor.ndim == 1:
        audio_tensor = audio_tensor.unsqueeze(0)  # Add a dimension to make it [1, num_samples]
else:
    print("Error: Audio data was not captured.")
    exit()

# Proceed with audio processing if data captured
sample_rate = model.hps.data.sampling_rate  # as we are capturing sample_rate

# 2. Mono to Stereo
audio_stereo = np.stack((audio_np, audio_np), axis=0)

# 3. Reverb, Resonance (using pedalboard)
board = Pedalboard([
    Resample(target_sample_rate=44100.0, quality=Resample.Quality.WindowedSinc256),
    Reverb(room_size=0.9, damping=0.4, wet_level=0.02230, dry_level=0.7), #simulate bigger space on vocal voice
    Limiter(threshold_db=-2, release_ms=1000),  # Initial limiter
    Chorus(rate_hz=0.4, depth=0.25, centre_delay_ms=7.0, feedback=0.0, mix=0.03),
    Delay(delay_seconds=1, feedback=0.2, mix= 0.01),
    Gain(gain_db=-5)
])



audio_resonance = board(audio_stereo, sample_rate)
audio_tensor = torch.tensor(audio_resonance)

# 4. EQ Adjustments (More Granular)


# --- 100Hz - 300Hz Range (Vocal Fundamentals) ---

# High-pass filter (gentle roll-off)
audio_tensor = F.highpass_biquad(audio_tensor, sample_rate, cutoff_freq=5, Q=0.7)

#Size Adjustment on rumble
audio_tensor = F.equalizer_biquad(audio_tensor, sample_rate, center_freq=50, gain=5, Q=3.4)  # Attenuate around 50 if too boomy



# Precise cuts/boosts (adjust these to your taste)
audio_tensor = F.equalizer_biquad(audio_tensor, sample_rate, center_freq=100, gain=-2, Q=1.4)  # Attenuate around 100Hz if too boomy
audio_tensor = F.equalizer_biquad(audio_tensor, sample_rate, center_freq=150, gain=-1.5, Q=1.4) # Reduce 150Hz a bit for potential muddiness
audio_tensor = F.equalizer_biquad(audio_tensor, sample_rate, center_freq=250, gain=3, Q=1.0)   # Slightly boost 250Hz for fullness, if needed

# --- 350Hz - 600Hz Range (Vocal Body) ---

# Careful scooping to avoid boxiness or hollowness
audio_tensor = F.equalizer_biquad(audio_tensor, sample_rate, center_freq=350, gain=-2, Q=1.4)  # Reduce 350Hz to control boxiness
audio_tensor = F.equalizer_biquad(audio_tensor, sample_rate, center_freq=450, gain=2, Q=1.8)  # Attenuate 450Hz if it sounds too thick
audio_tensor = F.equalizer_biquad(audio_tensor, sample_rate, center_freq=550, gain=-2, Q=1.4) # Reduce 550Hz, common area for boxy sound

# --- Upper Harmonics Shaping ---

# Presence boost
audio_tensor = F.equalizer_biquad(audio_tensor, sample_rate, center_freq=2000, gain=4, Q=1.0)   # Boost presence range around 2kHz
audio_tensor = F.equalizer_biquad(audio_tensor, sample_rate, center_freq=2500, gain=4, Q=1.4)   # Boost presence range around 2.5kHz
audio_tensor = F.equalizer_biquad(audio_tensor, sample_rate, center_freq=3000, gain=2, Q=1.4)   # Boost presence range around 3kHz
audio_tensor = F.equalizer_biquad(audio_tensor, sample_rate, center_freq=3500, gain=1, Q=1.8)   # Boost presence range around 3.5kHz

# Brilliance and Air
audio_tensor = F.equalizer_biquad(audio_tensor, sample_rate, center_freq=4000, gain=8, Q=1.4)   # Boost brilliance around 4kHz
audio_tensor = F.equalizer_biquad(audio_tensor, sample_rate, center_freq=8000, gain=8, Q=1.8)  # Boost air around 8kHz
audio_tensor = F.equalizer_biquad(audio_tensor, sample_rate, center_freq=12000, gain=8, Q=1.8)  # Add more air around 8kHz

# 5. Enharmonic Modulation (using a workaround)
# Add a very subtle sine wave modulation to the audio
mod_freq = 1  # Modulation frequency in Hz
mod_depth = 0.03 # Depth 3%
modulation = (1 + mod_depth * np.sin(2 * np.pi * mod_freq * np.arange(audio_tensor.shape[1]) / sample_rate)).astype(np.float32)
audio_tensor = audio_tensor * torch.from_numpy(modulation).unsqueeze(0)

# 6. Pitch Shift
pitch_shift = Pedalboard([
    PitchShift(semitones=-1.07)  # -0.84 semitones is approximately -7%
])
#audio_tensor = torch.from_numpy(pitch_shift(audio_tensor.numpy(), sample_rate))

# 7. Final Limiter
final_limiter = Pedalboard([
    Limiter(threshold_db=-2, release_ms=1000),  # Max 0 dB, 10ms attack, 500ms release
    Resample(target_sample_rate=44100.0, quality=Resample.Quality.WindowedSinc256)
])
audio_processed = final_limiter(audio_tensor.numpy(), sample_rate)

# 8. Write to file
torchaudio.save(output_path, torch.tensor(audio_processed), sample_rate)

print(f"Audio saved to {output_path}")