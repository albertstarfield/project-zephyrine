import requests

url = "http://127.0.0.1:11420/api/chat"
payload = {
    "model": "stella-icarus",
    "agentic": True,
    "messages": [{"role": "user", "content": "What is 58 * 14 + 19?"}],
    "stream": False
}
try:
    response = requests.post(url, json=payload)
    print("Status:", response.status_code)
    print("Response:", response.text)
except Exception as e:
    print(e)
