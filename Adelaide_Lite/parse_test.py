import requests
res = requests.post("http://127.0.0.1:11420/api/chat", json={"model": "stella-icarus", "agentic": True, "messages": [{"role": "user", "content": "What is 58 * 14 + 19?"}], "stream": False})
print("Status:", res.status_code)
print("Response:", res.text)
