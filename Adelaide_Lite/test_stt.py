import requests
import wave
import struct
import os

def test_transcription():
    url = "http://127.0.0.1:11420/v1/audio/transcriptions"
    
    wav_path = "../moonshine/test-assets/beckett.wav"
    if not os.path.exists(wav_path):
        print(f"Audio file not found: {wav_path}")
        return
        
    with wave.open(wav_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw_data = wf.readframes(n_frames)
        
    print(f"Read WAV: channels={n_channels}, width={sampwidth}, rate={framerate}, frames={n_frames}")
    
    # Assuming the wave is 16-bit PCM mono. We need to convert to Float32.
    shorts = struct.unpack(f"<{n_frames}h", raw_data)
    floats = [s / 32768.0 for s in shorts]
    float32_data = struct.pack(f"<{len(floats)}f", *floats)
    
    print(f"Sending {len(float32_data)} bytes of Float32 PCM data to {url}...")
    
    try:
        # We send raw payload instead of multipart for this simple test.
        response = requests.post(url, data=float32_data, headers={'Content-Type': 'text/plain'})
        print(f"Response status: {response.status_code}")
        print(f"Response text: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_transcription()
