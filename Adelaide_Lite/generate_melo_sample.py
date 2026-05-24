import sys
import os

MELO_PATH = os.path.expanduser("~/Documents/misc/AdaptiveSystem/project-zephyrine/systemCore/mainEngineFrame_MacroController_EngineSharedResources/MeloAudioTTS_SubEngine")
sys.path.insert(0, MELO_PATH)

try:
    from melo.api import TTS
except ImportError as e:
    print(f"Failed to import MeloTTS: {e}")
    sys.exit(1)

text = "Artificial intelligence has seen rapid advancements in recent years, transforming the way we interact with technology. From natural language processing algorithms that can understand and generate human-like text, to computer vision systems that identify complex patterns in images, the applications are seemingly endless. In the realm of speech synthesis, modern neural networks have enabled the creation of highly realistic voices that capture the nuances of emotion, intonation, and rhythm. As we continue to push the boundaries of what is possible, it is essential to consider the ethical implications and strive for responsible development. This journey of innovation is not just about building smarter machines, but also about augmenting human potential and creating a more connected, accessible world for everyone."
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
