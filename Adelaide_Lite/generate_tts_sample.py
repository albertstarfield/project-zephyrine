import os

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sample_wav_path = os.path.join(BASE_DIR, "sample.wav")
    blob_dat_path = os.path.join(BASE_DIR, "sampletts_blob.dat")

    try:
        from melo.api import TTS
        print("[*] MeloTTS is available. Generating sample.wav...")
        # Initialize MeloTTS (assuming English default)
        model = TTS(language='EN', device='auto')
        speaker_ids = model.hps.data.spk2id
        
        # We just need a short reference sample for Supertonic to clone the voice
        text = "Hello, I am ready to assist you. This is my reference voice."
        
        # Use first available speaker
        speaker_id = list(speaker_ids.values())[0] if speaker_ids else 0
        
        model.tts_to_file(text, speaker_id, sample_wav_path, speed=1.0)
        print(f"[*] Generated {sample_wav_path}")
        
    except ImportError:
        print("[!] MeloTTS not found in this environment. Cannot generate dynamic sample.wav")
        print("[!] If you have a sample.wav, it will be used. Otherwise, please place a sample.wav in this directory.")

    if os.path.exists(sample_wav_path):
        print(f"[*] Reading {sample_wav_path} and writing to {blob_dat_path}...")
        with open(sample_wav_path, "rb") as wav_file:
            wav_data = wav_file.read()
            
        with open(blob_dat_path, "wb") as blob_file:
            blob_file.write(wav_data)
        print(f"[*] Successfully created {blob_dat_path}")
    else:
        print(f"[!] {sample_wav_path} does not exist. Cannot create {blob_dat_path}.")

if __name__ == "__main__":
    main()
