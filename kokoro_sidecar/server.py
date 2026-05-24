import os
import io
import soundfile as sf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from kokoro_onnx import Kokoro

app = FastAPI(title="Kokoro TTS Sidecar")

# Globals for the Kokoro instance
kokoro = None

class TTSRequest(BaseModel):
    text: str
    voice: str = "af_sarah"
    speed: float = 1.0
    lang: str = "en-us"

@app.on_event("startup")
async def startup_event():
    global kokoro
    model_path = os.environ.get("KOKORO_MODEL", "../kokoro_models/kokoro-v0_19.int8.onnx")
    voices_path = os.environ.get("KOKORO_VOICES", "../kokoro_models/voices-v1.0.bin")
    
    if not os.path.exists(model_path) or not os.path.exists(voices_path):
        print(f"Warning: Model files not found at {model_path} or {voices_path}. Ensure they are downloaded.")
    else:
        print("Loading Kokoro ONNX model...")
        kokoro = Kokoro(model_path, voices_path)
        print("Kokoro model loaded successfully!")

@app.post("/tts")
async def generate_tts(request: TTSRequest):
    if kokoro is None:
        raise HTTPException(status_code=500, detail="Kokoro model not loaded")
    
    try:
        # Generate audio samples
        samples, sample_rate = kokoro.create(
            request.text, 
            voice=request.voice, 
            speed=request.speed, 
            lang=request.lang
        )
        
        # Write to in-memory WAV file
        wav_io = io.BytesIO()
        sf.write(wav_io, samples, sample_rate, format='WAV', subtype='PCM_16')
        wav_io.seek(0)
        
        from fastapi.responses import Response
        return Response(content=wav_io.read(), media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=11421)
