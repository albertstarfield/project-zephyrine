import requests

def test_tts():
    url = "http://127.0.0.1:11420/v1/audio/speech"
    payload = {
        "model": "supertonic",
        "input": "This is a test of the Supertonic Text-to-Speech system running natively inside the Adelaide Lite stack.",
        "voice": "alloy"
    }
    print("Sending request to TTS API...")
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            if b"TTS Error" in response.content:
                print("Server returned TTS Error.")
            else:
                with open("output.pcm", "wb") as f:
                    f.write(response.content)
                print(f"Success! Saved audio to output.pcm ({len(response.content)} bytes)")
        else:
            print(f"Failed with status {response.status_code}: {response.text}")
    except Exception as e:
        print(f"Error connecting to server: {e}")

if __name__ == "__main__":
    test_tts()
