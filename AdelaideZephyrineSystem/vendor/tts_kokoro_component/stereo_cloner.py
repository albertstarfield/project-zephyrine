import argparse
import sys
import os
import numpy as np
import soundfile as sf
import traceback

# Add kokoclone to sys.path so we can import it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'kokoclone')))

from core.cloner import KokoClone

def apply_stereo_widening(audio_mono, sample_rate=24000):
    # Basic Haas effect stereo widening + slight panning
    # 1. Delay the right channel by ~15ms
    delay_samples = int(sample_rate * 0.015)
    
    # 2. Create stereo array
    stereo = np.zeros((len(audio_mono) + delay_samples, 2), dtype=np.float32)
    
    # Left channel: original audio
    stereo[:len(audio_mono), 0] = audio_mono * 0.8
    
    # Right channel: delayed audio (Haas effect)
    stereo[delay_samples:, 1] = audio_mono * 0.8
    
    # Ensure no clipping
    stereo = np.clip(stereo, -1.0, 1.0)
    return stereo

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--ref", type=str, required=True)
    parser.add_argument("--out", type=str, default="output.wav")
    args = parser.parse_args()

    try:
        cloner = KokoClone()
        # KokoClone generates and saves to output_path. We will save it to a temp file first.
        temp_mono = "mono_temp.wav"
        cloner.generate(text=args.text, lang="en", reference_audio=args.ref, output_path=temp_mono)
        
        # Load the mono audio
        audio, sr = sf.read(temp_mono)
        
        # Apply Stereo Immersion
        stereo_audio = apply_stereo_widening(audio, sr)
        
        # Save stereo output
        sf.write(args.out, stereo_audio, sr)
        
        # Cleanup temp
        if os.path.exists(temp_mono):
            os.remove(temp_mono)
            
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
