import httpx
import asyncio

async def test():
    async with httpx.AsyncClient() as client:
        payload = {
            "model": "stella-icarus",
            "messages": [{"role": "user", "content": "What is anarchy?"}],
            "stream": False
        }
        response = await client.post("http://127.0.0.1:11420/api/chat", json=payload, timeout=60.0)
        print(response.status_code)
        print(response.json())

asyncio.run(test())
