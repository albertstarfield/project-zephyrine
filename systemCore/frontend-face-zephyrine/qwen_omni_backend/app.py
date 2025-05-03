import os
import torch
import soundfile as sf
import base64
import io
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from transformers import Qwen2_5OmniModel, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen2.5-Omni-7B"
# Set device: "cuda" for GPU (recommended), "cpu" for CPU (very slow)
# Or use "auto" to let transformers decide (usually picks GPU if available)
DEVICE_MAP = "auto"
# Set torch_dtype: "auto", torch.float16, torch.bfloat16 (check GPU compatibility)
TORCH_DTYPE = "auto"
# Enable Flash Attention 2 if supported and installed (requires compatible GPU & dtype)
USE_FLASH_ATTENTION_2 = False # Set to True if applicable
# Enable audio output from the model?
ENABLE_AUDIO_OUTPUT = True
# Default voice for audio output ("Chelsie" or "Ethan")
DEFAULT_SPEAKER = "Chelsie"
# Audio sample rate expected by the model
AUDIO_SAMPLE_RATE = 24000

# --- Global Variables ---
model = None
processor = None
device = None # Will be determined during model loading if DEVICE_MAP="auto"

# --- Model Loading ---
def load_model():
    """Loads the Qwen model and processor."""
    global model, processor, device
    print(f"Loading model: {MODEL_NAME}...")
    try:
        attn_implementation = "flash_attention_2" if USE_FLASH_ATTENTION_2 else None
        model = Qwen2_5OmniModel.from_pretrained(
            MODEL_NAME,
            torch_dtype=TORCH_DTYPE,
            device_map=DEVICE_MAP,
            attn_implementation=attn_implementation,
            enable_audio_output=ENABLE_AUDIO_OUTPUT
        )
        processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_NAME)

        # Determine the device if device_map="auto" was used
        if DEVICE_MAP == "auto":
             # Assuming the model loaded layers onto a device, find it
             # This might need adjustment depending on how device_map distributes layers
             try:
                 device = next(model.parameters()).device
                 print(f"Model loaded on device: {device}")
             except Exception as e:
                 print(f"Could not automatically determine device, defaulting to cpu. Error: {e}")
                 device = torch.device("cpu")
        elif isinstance(DEVICE_MAP, str):
             device = torch.device(DEVICE_MAP)
        else:
             # If device_map is more complex (dict), this needs refinement
             print("Complex device_map used, setting device to 'cuda' if available, else 'cpu'")
             device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Model and processor loaded successfully.")

    except Exception as e:
        print(f"Error loading model: {e}")
        # Exit if model loading fails, as the app can't function
        exit(1)

# --- Flask App Setup ---
app = Flask(__name__)
CORS(app) # Enable Cross-Origin Resource Sharing for all routes

# --- API Routes ---
@app.route('/chat', methods=['POST'])
def chat_handler():
    """Handles chat requests with multimodal input."""
    global model, processor, device

    if not model or not processor:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.json
    if not data or 'conversation' not in data:
        return jsonify({"error": "Invalid request format. 'conversation' field missing."}), 400

    conversation = data['conversation']
    # Optional parameters from request
    use_audio_in_video = data.get('use_audio_in_video', True) # Default to True
    return_audio = data.get('return_audio', True) # Default to True if model supports it
    speaker = data.get('speaker', DEFAULT_SPEAKER) # Use default if not specified

    if not ENABLE_AUDIO_OUTPUT and return_audio:
        print("Warning: Audio output requested but model loaded with enable_audio_output=False. Forcing return_audio=False.")
        return_audio = False

    try:
        # 1. Preprocess input
        print("Processing input...")
        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        # process_mm_info expects list of dicts, handles base64/urls for audio/image/video
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_audio_in_video)

        inputs = processor(text=text_prompt, audios=audios, images=images, videos=videos, return_tensors="pt", padding=True)
        inputs = inputs.to(device).to(model.dtype) # Move inputs to the correct device and dtype

        # 2. Generate response
        print("Generating response...")
        generate_kwargs = {
            "use_audio_in_video": use_audio_in_video,
            "return_audio": return_audio,
        }
        if return_audio:
             generate_kwargs["spk"] = speaker

        # model.generate can return (text_ids, audio_tensor) or just text_ids
        with torch.no_grad():
            output = model.generate(**inputs, **generate_kwargs)

        # 3. Postprocess output
        print("Processing output...")
        if return_audio:
            text_ids, audio_tensor = output
            # Reshape audio tensor and convert to numpy array
            audio_np = audio_tensor.reshape(-1).detach().cpu().numpy()

            # Encode audio to base64 WAV
            buffer = io.BytesIO()
            sf.write(buffer, audio_np, AUDIO_SAMPLE_RATE, format='WAV')
            buffer.seek(0)
            audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        else:
            text_ids = output
            audio_base64 = None

        # Decode text
        generated_text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        # Assuming batch size 1 for simplicity
        response_text = generated_text[0] if generated_text else ""

        print(f"Generated Text: {response_text[:100]}...") # Log snippet
        if audio_base64:
            print(f"Generated Audio: {len(audio_base64)} bytes (base64)")

        return jsonify({
            "text": response_text,
            "audio_base64": audio_base64 # WAV audio encoded in base64
        })

    except Exception as e:
        print(f"Error during chat processing: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback to server logs
        return jsonify({"error": f"Internal server error: {e}"}), 500

@app.route('/status', methods=['GET'])
def status():
    """Simple status check endpoint."""
    if model and processor:
        return jsonify({"status": "loaded", "model": MODEL_NAME, "device": str(device)})
    else:
        return jsonify({"status": "loading or error"})

# --- Main Execution ---
if __name__ == '__main__':
    load_model() # Load the model when the script starts
    # Run Flask app - accessible on the local network
    app.run(host='0.0.0.0', port=5000)
