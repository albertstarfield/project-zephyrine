import sys
import os

MELO_PATH = os.path.expanduser("~/Documents/misc/AdaptiveSystem/project-zephyrine/systemCore/mainEngineFrame_MacroController_EngineSharedResources/MeloAudioTTS_SubEngine")
sys.path.insert(0, MELO_PATH)

try:
    from melo.api import TTS
except ImportError as e:
    print(f"Failed to import MeloTTS: {e}")
    sys.exit(1)

text = "The quick brown fox jumps over the lazy dog."
output_path = "sampleAdeltts_blob.dat"
output_path = "sampleAdeltts_blob.wav"

print("Loading MeloTTS model...")
model = TTS(language='EN', device='cpu')
speaker_ids = model.hps.data.spk2id

speaker = 'EN-US'
if speaker not in speaker_ids:
    speaker = list(speaker_ids.keys())[0]

print(f"Generating audio with speaker {speaker}...")
model.tts_to_file(text, speaker_ids[speaker], output_path, speed=1.0)
os.rename("sampleAdeltts_blob.wav", "sampleAdeltts_blob.dat")
print(f"Successfully generated {output_path}")
