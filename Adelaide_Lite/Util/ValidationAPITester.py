#!/usr/bin/env python3
import requests
import json
import time
import sys
import traceback
from datetime import datetime

# ANSI Color Codes
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
BOLD = "\033[1m"
RESET = "\033[0m"

class APIValidationException(Exception):
    pass

class ValidationAPITester:
    def __init__(self, base_url="http://localhost:11420", timeout=60):
        self.base_url = base_url
        self.timeout = timeout
        self.stats = {"passed": 0, "failed": 0, "total": 0}

    def log_success(self, msg):
        print(f"{GREEN}[PASS]{RESET} {msg}")
        self.stats["passed"] += 1
        self.stats["total"] += 1

    def log_failure(self, msg, error=None):
        print(f"{RED}[FAIL]{RESET} {msg}")
        if error:
            print(f"      Error: {error}")
        self.stats["failed"] += 1
        self.stats["total"] += 1

    def log_info(self, msg):
        print(f"{CYAN}[INFO]{RESET} {msg}")

    def assert_field(self, data, field, expected_type=None):
        if field not in data:
            raise APIValidationException(f"Missing required field: '{field}'")
        if expected_type and not isinstance(data[field], expected_type):
            raise APIValidationException(
                f"Field '{field}' has wrong type. Expected {expected_type}, got {type(data[field])}"
            )

    def test_endpoint(self, name, method, path, payload=None, is_streaming=False, is_openai=False):
        self.log_info(f"Testing {name} ({path})...")
        url = f"{self.base_url}{path}"
        start_time = time.time()
        
        try:
            if method == "GET":
                resp = requests.get(url, timeout=self.timeout)
            else:
                resp = requests.post(url, json=payload, timeout=self.timeout, stream=is_streaming)

            if resp.status_code != 200:
                self.log_failure(f"{name} returned status code {resp.status_code}")
                return

            if is_streaming:
                self.validate_streaming_response(name, resp, is_openai)
            else:
                data = resp.json()
                self.validate_json_response(name, data, is_openai, path)
            
            duration = time.time() - start_time
            self.log_success(f"{name} validated successfully in {duration:.2f}s")

        except Exception as e:
            self.log_failure(f"Exception during {name}", e)
            # traceback.print_exc()

    def validate_json_response(self, name, data, is_openai, path):
        if "models" in path or "tags" in path:
            self.assert_field(data, "models", list)
            for model in data["models"]:
                self.assert_field(model, "name", str)
        elif "embeddings" in path or "embed" in path:
            if is_openai:
                self.assert_field(data, "data", list)
                self.assert_field(data["data"][0], "embedding", list)
            else:
                self.assert_field(data, "embedding", list)
        elif "chat" in path:
            if is_openai:
                self.assert_field(data, "id", str)
                self.assert_field(data, "choices", list)
                self.assert_field(data["choices"][0], "message", dict)
                self.assert_field(data["choices"][0]["message"], "content", str)
            else:
                self.assert_field(data, "model", str)
                self.assert_field(data, "message", dict)
                self.assert_field(data["message"], "content", str)
                self.assert_field(data, "done", bool)
        elif "generate" in path:
            self.assert_field(data, "model", str)
            self.assert_field(data, "response", str)
            self.assert_field(data, "done", bool)

    def validate_streaming_response(self, name, resp, is_openai):
        chunk_count = 0
        full_content = ""
        
        for line in resp.iter_lines():
            if not line:
                continue
            
            chunk_count += 1
            decoded_line = line.decode("utf-8")
            
            if is_openai:
                if decoded_line.startswith("data: "):
                    content = decoded_line[6:]
                    if content == "[DONE]":
                        break
                    chunk_data = json.loads(content)
                    self.assert_field(chunk_data, "choices", list)
                    delta = chunk_data["choices"][0].get("delta", {})
                    if "content" in delta:
                        full_content += delta["content"]
                else:
                    # Some implementations might send empty lines or other headers
                    continue
            else:
                chunk_data = json.loads(decoded_line)
                if "message" in chunk_data:
                    full_content += chunk_data["message"].get("content", "")
                elif "response" in chunk_data:
                    full_content += chunk_data.get("response", "")
                
                if chunk_data.get("done", False):
                    break

        if chunk_count == 0:
            raise APIValidationException("No streaming chunks received")
        if not full_content:
            raise APIValidationException("Streaming content is empty")
        
        self.log_info(f"      Received {chunk_count} chunks, total length: {len(full_content)}")

    def run_all_tests(self):
        print(f"{BOLD}=== Adelaide_Lite API Aggressive Validation ==={RESET}")
        
        # 1. Model List
        self.test_endpoint("OpenAI Models List", "GET", "/v1/models")
        self.test_endpoint("Ollama Tags List", "GET", "/api/tags")

        # 2. Embeddings
        embed_payload = {"model": "adelaide-embedding", "input": "The quick brown fox jumps over the lazy dog."}
        self.test_endpoint("OpenAI Embeddings", "POST", "/v1/embeddings", embed_payload, is_openai=True)
        self.test_endpoint("Ollama Embeddings", "POST", "/api/embeddings", embed_payload)

        # 3. Chat Non-Streaming
        chat_payload = {
            "model": "adelaide-hybrid",
            "messages": [{"role": "user", "content": "Hello, who are you?"}],
            "stream": False
        }
        self.test_endpoint("OpenAI Chat Non-Streaming", "POST", "/v1/chat/completions", chat_payload, is_openai=True)
        self.test_endpoint("Ollama Chat Non-Streaming", "POST", "/api/chat", chat_payload)

        # 4. Chat Streaming
        chat_payload["stream"] = True
        self.test_endpoint("OpenAI Chat Streaming", "POST", "/v1/chat/completions", chat_payload, is_streaming=True, is_openai=True)
        self.test_endpoint("Ollama Chat Streaming", "POST", "/api/chat", chat_payload, is_streaming=True)

        # 5. Generate
        gen_payload = {
            "model": "adelaide-hybrid",
            "prompt": "Explain quantum entanglement in one sentence.",
            "stream": False
        }
        self.test_endpoint("Ollama Generate Non-Streaming", "POST", "/api/generate", gen_payload)
        gen_payload["stream"] = True
        self.test_endpoint("Ollama Generate Streaming", "POST", "/api/generate", gen_payload, is_streaming=True)

        # 6. Aggressive: Large Payload
        large_prompt = "Say 'Repeat' " * 100
        chat_payload_large = {
            "model": "adelaide-hybrid",
            "messages": [{"role": "user", "content": large_prompt}],
            "stream": False
        }
        self.test_endpoint("Large Payload Chat", "POST", "/api/chat", chat_payload_large)

        # 7. Aggressive: Malformed JSON
        self.log_info("Testing Malformed JSON (expecting failure handle)...")
        try:
            resp = requests.post(f"{self.base_url}/api/chat", data="{ malformed: json }", headers={"Content-Type": "application/json"})
            if resp.status_code >= 400:
                self.log_success("Malformed JSON correctly handled with error status")
            else:
                self.log_failure(f"Malformed JSON returned success status {resp.status_code}")
        except Exception as e:
            self.log_failure("Exception during malformed JSON test", e)

        print(f"\n{BOLD}=== Validation Summary ==={RESET}")
        print(f"Total Tests: {self.stats['total']}")
        print(f"Passed:      {GREEN}{self.stats['passed']}{RESET}")
        print(f"Failed:      {RED}{self.stats['failed']}{RESET}")
        
        if self.stats["failed"] > 0:
            sys.exit(1)

if __name__ == "__main__":
    base_url = "http://localhost:11420"
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    
    tester = ValidationAPITester(base_url=base_url)
    tester.run_all_tests()
