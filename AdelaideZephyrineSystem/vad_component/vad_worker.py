import os
import urllib.request
import numpy as np
import onnxruntime as ort
import socket
import struct
import signal
import sys

# Configuration
MODEL_URL = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "silero_vad.onnx")
SAMPLE_RATE = 16000
THRESHOLD = 0.5
WINDOW_SIZE = 512
SOCKET_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "run", "adelaide_vad.sock")

# Download model if not exists
if not os.path.exists(MODEL_PATH):
    print(f"[VAD Worker] Downloading Silero VAD ONNX model to {MODEL_PATH}...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

print("[VAD Worker] Loading ONNX session...")
# Initialize ONNX session
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
session.intra_op_num_threads = 1
session.inter_op_num_threads = 1

def acoustic_dynamic_gateway(audio_floats: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Acoustic Dynamic Gateway (Inspired by Audio Technica / Tokyo Philharmonic)
    Applies a harmonic-based dynamic EQ and bandpass filter to emphasize
    human voice harmonics (300Hz-3000Hz) and suppress background noise.
    """
    if len(audio_floats) == 0:
        return audio_floats
        
    # Perform FFT
    fft_data = np.fft.rfft(audio_floats)
    freqs = np.fft.rfftfreq(len(audio_floats), d=1.0/sample_rate)
    
    # Create a dynamic EQ curve (mask)
    mask = np.ones_like(freqs, dtype=np.float32)
    
    # Low cut (attenuate rumble below 100Hz)
    mask[freqs < 100] = 0.1
    
    # Voice band enhancement (300Hz - 3000Hz)
    voice_band = (freqs >= 300) & (freqs <= 3000)
    mask[voice_band] = 1.2
    
    # High cut (attenuate hiss above 4000Hz)
    mask[freqs > 4000] = 0.2
    
    # Apply the mask (harmonic filtering)
    filtered_fft = fft_data * mask
    
    # Inverse FFT back to time domain
    filtered_audio = np.fft.irfft(filtered_fft, n=len(audio_floats))
    return filtered_audio.astype(np.float32)

def vad_process(audio_floats: np.ndarray) -> bool:
    """Run Silero VAD over the float32 array in chunks."""
    h = np.zeros((2, 1, 64), dtype=np.float32)
    c = np.zeros((2, 1, 64), dtype=np.float32)
    
    num_samples = len(audio_floats)
    for i in range(0, num_samples, WINDOW_SIZE):
        chunk = audio_floats[i : i + WINDOW_SIZE]
        if len(chunk) < WINDOW_SIZE:
            chunk = np.pad(chunk, (0, WINDOW_SIZE - len(chunk)), 'constant')
        
        chunk = np.expand_dims(chunk, axis=0).astype(np.float32)
        ort_inputs = {
            'input': chunk,
            'sr': np.array(SAMPLE_RATE, dtype=np.int64),
            'h': h,
            'c': c
        }
        
        out, h, c = session.run(None, ort_inputs)
        
        prob = float(out[0][0])
        if prob > THRESHOLD:
            return True
            
    return False

def cleanup(signum=None, frame=None):
    if os.path.exists(SOCKET_PATH):
        os.remove(SOCKET_PATH)
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

def main():
    if os.path.exists(SOCKET_PATH):
        os.remove(SOCKET_PATH)
        
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_PATH)
    server.listen(1)
    print(f"[VAD Worker] IPC Server listening on {SOCKET_PATH}")

    try:
        while True:
            conn, _ = server.accept()
            try:
                # Read 4-byte length header
                length_data = conn.recv(4)
                if not length_data or len(length_data) < 4:
                    conn.close()
                    continue
                
                # Unpack big-endian unsigned int (network byte order)
                payload_length = struct.unpack('>I', length_data)[0]
                
                # Read exactly payload_length bytes
                chunks = []
                bytes_recd = 0
                while bytes_recd < payload_length:
                    chunk = conn.recv(min(payload_length - bytes_recd, 4096))
                    if chunk == b'':
                        break
                    chunks.append(chunk)
                    bytes_recd = bytes_recd + len(chunk)
                
                raw_data = b''.join(chunks)
                if len(raw_data) == 0:
                    conn.sendall(b'0')
                else:
                    audio_floats = np.frombuffer(raw_data, dtype=np.float32)
                    
                    # Apply Acoustic Dynamic Gateway
                    filtered_floats = acoustic_dynamic_gateway(audio_floats, SAMPLE_RATE)
                    
                    is_speech = vad_process(filtered_floats)
                    
                    conn.sendall(b'1' if is_speech else b'0')
            except Exception as e:
                print(f"[VAD Worker] Error: {e}")
            finally:
                conn.close()
    except KeyboardInterrupt:
        cleanup()

if __name__ == "__main__":
    main()
