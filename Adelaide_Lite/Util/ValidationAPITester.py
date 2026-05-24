#!/usr/bin/env python3
import requests
import json
import time
import sys

# ANSI Color Codes
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
BOLD = "\033[1m"
MAGENTA = "\033[35m"
RESET = "\033[0m"

class APIValidationException(Exception):
    pass

class ValidationAPITester:
    def __init__(self, base_url="http://localhost:11420", timeout=420):
        self.base_url = base_url
        self.timeout = timeout
        self.stats = {"passed": 0, "failed": 0, "total": 0}
        self.server_type = "Unknown"

    def log_success(self, msg):
        print(f"{GREEN}[PASS]{RESET} {msg}")
        self.stats["passed"] += 1
        self.stats["total"] += 1

    def log_failure(self, msg, error=None):
        print(f"{RED}[FAIL]{RESET} {msg}")
        if error:
            print(f"      {YELLOW}Error:{RESET} {error}")
        self.stats["failed"] += 1
        self.stats["total"] += 1

    def log_info(self, msg):
        print(f"{CYAN}[INFO]{RESET} {msg}")

    def log_warn(self, msg):
        print(f"{YELLOW}[WARN]{RESET} {msg}")

    def assert_field(self, data, field, expected_type=None):
        if field not in data:
            raise APIValidationException(f"Missing required field: '{field}'")
        if expected_type and not isinstance(data[field], expected_type):
            raise APIValidationException(
                f"Field '{field}' has wrong type. Expected {expected_type}, got {type(data[field])}"
            )

    def detect_server(self):
        self.log_info("Detecting server type...")
        try:
            resp = requests.get(f"{self.base_url}/v1/models", timeout=5)
            server_header = resp.headers.get("Server", "")
            if "AWS" in server_header or "Ada" in server_header:
                self.server_type = "Adelaide-Ada-Core"
            else:
                # Flask usually doesn't set a Server header by default or sets Werkzeug
                self.server_type = "Adelaide-Python-Bridge (Flask)"
            self.log_info(f"Detected Server: {MAGENTA}{self.server_type}{RESET}")
        except Exception:
            self.log_warn("Could not detect server type reliably.")

    def test_endpoint(self, name, method, path, payload=None, is_streaming=False, is_openai=False):
        print(f"\n{BOLD}--- {name} ---{RESET}")
        self.log_info(f"Target: {method} {path}")
        url = f"{self.base_url}{path}"
        start_time = time.time()
        
        try:
            if method == "GET":
                resp = requests.get(url, timeout=self.timeout)
            else:
                resp = requests.post(url, json=payload, timeout=self.timeout, stream=is_streaming)

            # Aggressive Header Validation
            self.validate_headers(resp)

            if resp.status_code != 200:
                self.log_failure(f"Endpoint returned status code {resp.status_code}")
                try:
                    print(f"{MAGENTA}[RAW ERROR BODY]{RESET}\n{resp.text}")
                except Exception:
                    pass
                if resp.status_code == 404:
                    self.log_warn(f"Endpoint {path} not implemented on this server.")
                return

            if is_streaming:
                self.validate_streaming_response(name, resp, is_openai)
            else:
                data = resp.json()
                print(f"{MAGENTA}[RAW JSON RESPONSE]{RESET}")
                print(json.dumps(data, indent=2))
                self.validate_json_response(name, data, is_openai, path)
            
            duration = time.time() - start_time
            self.log_success(f"Validated in {duration:.2f}s")

        except requests.exceptions.Timeout:
            self.log_failure(f"Timeout after {self.timeout}s")
        except Exception as e:
            self.log_failure("Unexpected Exception", e)

    def validate_headers(self, resp):
        # All Adelaide APIs should support CORS
        if "Access-Control-Allow-Origin" not in resp.headers:
            self.log_warn("Missing CORS header: Access-Control-Allow-Origin")
        
        content_type = resp.headers.get("Content-Type", "")
        if "application/json" not in content_type and "text/event-stream" not in content_type and "application/x-ndjson" not in content_type:
             self.log_warn(f"Unexpected Content-Type: {content_type}")

    def validate_json_response(self, name, data, is_openai, path):
        if "models" in path or "tags" in path:
            self.assert_field(data, "models", list)
            if not data["models"]:
                 self.log_warn("Model list is empty")
            for model in data["models"]:
                self.assert_field(model, "name", str)
                self.assert_field(model, "id", str)
        elif "embeddings" in path or "embed" in path:
            if is_openai:
                self.assert_field(data, "data", list)
                self.assert_field(data["data"][0], "embedding", list)
                if len(data["data"][0]["embedding"]) == 0:
                    raise APIValidationException("Embedding vector is empty")
            else:
                self.assert_field(data, "embedding", list)
                if len(data["embedding"]) == 0:
                    raise APIValidationException("Embedding vector is empty")
        elif "chat" in path:
            if is_openai:
                self.assert_field(data, "id", str)
                self.assert_field(data, "object", str)
                self.assert_field(data, "created", int)
                self.assert_field(data, "choices", list)
                self.assert_field(data["choices"][0], "message", dict)
                msg = data["choices"][0]["message"]
                if data["choices"][0].get("finish_reason") == "tool_calls":
                    if msg.get("content") is not None:
                        raise APIValidationException(f"Field content should be None for tool_calls, got {type(msg.get('content'))}")
                    self.assert_field(msg, "tool_calls", list)
                else:
                    self.assert_field(msg, "content", str)
                self.assert_field(data, "usage", dict)
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
        first_chunk_time = None
        start_time = time.time()
        
        for line in resp.iter_lines():
            if not line:
                continue
            
            if first_chunk_time is None:
                first_chunk_time = time.time() - start_time
                self.log_info(f"      Time to first chunk: {first_chunk_time:.2f}s")
            
            chunk_count += 1
            decoded_line = line.decode("utf-8")
            
            try:
                if is_openai:
                    if decoded_line.startswith("data: "):
                        content = decoded_line[6:]
                        if content.strip() == "[DONE]":
                            print(f"{MAGENTA}[RAW CHUNK]{RESET} [DONE]")
                            break
                        chunk_data = json.loads(content)
                        print(f"{MAGENTA}[RAW CHUNK]{RESET} {json.dumps(chunk_data)}")
                        self.assert_field(chunk_data, "choices", list)
                        delta = chunk_data["choices"][0].get("delta", {})
                        if "content" in delta:
                            full_content += delta["content"]
                    else:
                        continue
                else:
                    chunk_data = json.loads(decoded_line)
                    print(f"{MAGENTA}[RAW CHUNK]{RESET} {json.dumps(chunk_data)}")
                    if "message" in chunk_data:
                        full_content += chunk_data["message"].get("content", "")
                    elif "response" in chunk_data:
                        full_content += chunk_data.get("response", "")
                    
                    if chunk_data.get("done", False):
                        # Verify final metrics if present
                        if "total_duration" in chunk_data:
                             self.log_info(f"      Server reported duration: {chunk_data['total_duration']/1e9:.2f}s")
                        break
            except json.JSONDecodeError:
                self.log_failure(f"Malformed JSON in chunk {chunk_count}: {decoded_line[:50]}...")

        if chunk_count == 0:
            raise APIValidationException("No streaming chunks received")
        if not full_content:
            self.log_warn("Streaming content is empty")
        
        self.log_info(f"      Chunks: {chunk_count}, Total length: {len(full_content)}")

    def run_all_tests(self):
        print(f"{BOLD}{MAGENTA}=================================================={RESET}")
        print(f"{BOLD}{MAGENTA}   Adelaide_Lite Aggressive API Validator         {RESET}")
        print(f"{BOLD}{MAGENTA}=================================================={RESET}")
        
        self.detect_server()
        
        # 1. Capabilities
        self.test_endpoint("OpenAI Models List", "GET", "/v1/models")
        self.test_endpoint("Ollama Tags List", "GET", "/api/tags")

        # 2. Embeddings
        embed_payload = {"model": "adelaide-embedding", "input": "Validation of semantic vectors is critical for RAG integrity."}
        self.test_endpoint("OpenAI Embeddings", "POST", "/v1/embeddings", embed_payload, is_openai=True)
        self.test_endpoint("Ollama Embeddings", "POST", "/api/embeddings", embed_payload)

        # 3. Chat Non-Streaming
        chat_payload = {
            "model": "adelaide-hybrid",
            "messages": [{"role": "user", "content": "Briefly explain why validation is important."}],
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
            "prompt": "Respond with exactly one word: Hello.",
            "stream": False
        }
        self.test_endpoint("Ollama Generate Non-Streaming", "POST", "/api/generate", gen_payload)
        gen_payload["stream"] = True
        self.test_endpoint("Ollama Generate Streaming", "POST", "/api/generate", gen_payload, is_streaming=True)

        # 6. Agentic API
        agentic_payload = {
            "model": "adelaide-hybrid",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant with access to tools."},
                {"role": "user", "content": "What is the current weather in Adelaide? Please use the search tool."}
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the web for information.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The search query"}
                        },
                        "required": ["query"]
                    }
                }
            }],
            "stream": False
        }
        self.test_endpoint("OpenAI Agentic API", "POST", "/v1/chat/completions", agentic_payload, is_openai=True)

        # 7. Aggressive: Stress Test
        self.log_info("\nStarting stress/edge-case tests...")
        
        # Large Input
        large_prompt = "Validation " * 500
        chat_payload_large = {
            "model": "adelaide-hybrid",
            "messages": [{"role": "user", "content": large_prompt}],
            "stream": False
        }
        self.test_endpoint("Large Payload Test", "POST", "/api/chat", chat_payload_large)

        # Malformed JSON
        print(f"\n{BOLD}--- Malformed JSON Test ---{RESET}")
        try:
            resp = requests.post(f"{self.base_url}/api/chat", data="{ 'missing_quotes': val }", headers={"Content-Type": "application/json"})
            if resp.status_code >= 400:
                self.log_success("Malformed JSON correctly rejected")
            else:
                self.log_failure(f"Malformed JSON accepted (Status {resp.status_code})")
        except Exception as e:
            self.log_failure("Exception during malformed JSON test", e)

        print(f"\n{BOLD}{MAGENTA}=================================================={RESET}")
        print("   Validation Summary")
        print(f"{MAGENTA}=================================================={RESET}")
        print(f"Total Tests: {self.stats['total']}")
        print(f"Passed:      {GREEN}{self.stats['passed']}{RESET}")
        print(f"Failed:      {RED}{self.stats['failed']}{RESET}")
        
        if self.stats["failed"] > 0:
            print(f"\n{RED}Validation failed with {self.stats['failed']} errors.{RESET}")
            sys.exit(1)
        else:
            print(f"\n{GREEN}All systems nominal. API is fully compliant.{RESET}")

if __name__ == "__main__":
    base_url = "http://localhost:11420"
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    
    tester = ValidationAPITester(base_url=base_url)
    tester.run_all_tests()
