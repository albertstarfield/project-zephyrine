import sys
import os
import soundfile as sf
import warnings
warnings.filterwarnings("ignore")

# Redirect stdout to stderr so we don't corrupt any potential stdout pipes, though we're writing to a file.
sys.stdout = sys.stderr

if len(sys.argv) < 3:
    print("Usage: python cli.py <text> <output_wav>")
    sys.exit(1)

text = sys.argv[1]
output_file = sys.argv[2]

try:
    from kokoro_onnx import Kokoro
    # Resolve paths relative to this script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "..", "kokoro_models", "kokoro-v0_19.int8.onnx")
    voices_path = os.path.join(base_dir, "..", "kokoro_models", "voices-v1.0.bin")
    
    kokoro = Kokoro(model_path, voices_path)
    samples, sample_rate = kokoro.create(text, voice="af_sarah", speed=1.0, lang="en-us")
    sf.write(output_file, samples, sample_rate)
    sys.exit(0)
except Exception as e:
    print(f"Kokoro CLI Error: {e}", file=sys.stderr)
    sys.exit(1)
