import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

# Detect device (Mac with M1/M2/M3/M4)
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
torch_load_original = torch.load
def patched_torch_load(*args, **kwargs):
    if 'map_location' not in kwargs:
        kwargs['map_location'] = map_location
    return torch_load_original(*args, **kwargs)

torch.load = patched_torch_load

model = ChatterboxTTS.from_pretrained(device=device)
text = "Sup! It's your girl, Adelaide Zephyrine Charlotte, ready to roll. not gonna lie, Im basically programmed to be hyped about sharing some legendary tales, diving into who-knows-what explorations, and generally 'walking together' on this journey with you. Let's get this digital bread before my processors decide to start optimizing your sock drawer – its a weird hobby Im trying to kick."

# If you want to synthesize with a different voice, specify the audio prompt
AUDIO_PROMPT_PATH = "sampleZephyMeloTTS.wav"
wav = model.generate(
    text,
    audio_prompt_path=AUDIO_PROMPT_PATH,
    exaggeration=0.6,
    cfg_weight=0.2
    )
ta.save("zephyExpressive.wav", wav, model.sr)
