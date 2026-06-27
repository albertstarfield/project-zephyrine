# Adelaide API Reference

Base URL: `http://localhost:11420`

Adelaide exposes two compatible API dialects (Ollama and OpenAI) plus native endpoints.

---

## OpenAI-Compatible Endpoints (`/v1/*`)

### Chat Completion
```
POST /v1/chat/completions
```
OpenAI-compatible chat completion with streaming support.

**Request Body:**
```json
{
  "model": "qwen3.5-0.8b",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "stream": true,
  "temperature": 0.7,
  "max_tokens": 2048
}
```

**Response:** Streaming SSE or JSON (OpenAI format).

---

### Text Completion
```
POST /v1/completions
```
Text completion (non-chat). Same request format as chat but with `prompt` instead of `messages`.

---

### Embeddings
```
POST /v1/embeddings
```
Generate vector embeddings for text.

**Request Body:**
```json
{
  "model": "qwen3-embedding-0.6b",
  "input": "text to embed"
}
```

**Response:**
```json
{
  "data": [{"embedding": [0.1, 0.2, ...], "index": 0}],
  "model": "qwen3-embedding-0.6b",
  "usage": {"prompt_tokens": 5, "total_tokens": 5}
}
```

---

### List Models
```
GET /v1/models
```
Returns available models. Alias for `/api/tags`.

---

### Messages (Claude Format)
```
POST /v1/messages
```
Claude API-compatible message endpoint.

---

### Audio Transcription (STT)
```
POST /v1/audio/transcriptions
```
Speech-to-text using Moonshine ONNX.

**Request:** Raw Float32 PCM audio data with header `Content-Type: text/plain`.

**Response:**
```json
{
  "text": "transcribed text here"
}
```

---

### Text-to-Speech (TTS)
```
POST /v1/audio/speech
```
Text-to-speech using Kokoro ONNX.

**Request Body:**
```json
{
  "input": "Hello, how are you?",
  "voice": "default",
  "response_format": "wav"
}
```

**Response:** Raw WAV audio bytes.

---

### Image Generation
```
POST /v1/images/generations
```
Image generation using FLUX Schnell (two-stage: sparse → refinement).

**Request Body:**
```json
{
  "prompt": "a sunset over mountains",
  "n": 1,
  "size": "1024x1024"
}
```

**Response:** Base64-encoded image or URL.

---

## Ollama-Compatible Endpoints (`/api/*`)

### Chat
```
POST /api/chat
```
Ollama-compatible chat. Same behavior as `/v1/chat/completions` but Ollama response format.

---

### Generate
```
POST /api/generate
```
Ollama-compatible text generation.

---

### Tags / Models
```
GET /api/tags
```
List available models with sizes and formats.

---

### Show Model Info
```
POST /api/show
```
Show detailed model information.

---

### Embeddings
```
POST /api/embeddings
POST /api/embed
```
Generate embeddings. Both endpoints are aliases.

---

### Process Status
```
GET /api/ps
```
Show currently loaded models and their memory usage.

---

### Version
```
GET /api/version
```
Return server version, git commit, and build info.

---

### Model Management (Stubs)
```
POST /api/create    — Create model (stub)
POST /api/pull      — Pull model (stub)
POST /api/push      — Push model (stub)
POST /api/copy      — Copy model (stub)
DELETE /api/delete   — Delete model (stub)
POST /api/signin     — Sign in (stub)
POST /api/signout    — Sign out (stub)
```

---

## Native Endpoints

### Health Check
```
GET /api/power
```
Returns system power state, GPU memory, and StellaIcarus telemetry.

---

### Telemetry
```
GET /api/telemetry
```
Returns detailed system telemetry: CPU usage, memory, GPU stats, uptime.

---

### Zenith Routine
```
GET /api/ZenithRoutine
```
Returns ZenithOrion pacing loop status (ELP3).

---

### Handless Mode (Voice I/O)
```
POST /api/agenticZephyHandlessMode
```
Voice interaction mode — accepts audio input, returns audio output.

---

### Server Info
```
GET /
HEAD /
```
- `HEAD /` — Heartbeat check (returns 200 if alive)
- `GET /` — Server info JSON

---

### Agent Client Protocol (ACP)
```
POST /api/acp
```

ACP is a standard JSON-RPC 2.0 interface used by clients like Zed, VS Code plugins, and `@agentclientprotocol/sdk` in Node.js.

**Connection Guide:**
1. Configure the client to use HTTP transport.
2. Set the endpoint URL to `http://<server-ip>:11420/api/acp`.
3. Send standard JSON-RPC payloads:
   ```json
   { "jsonrpc": "2.0", "method": "initialize", "params": {}, "id": 1 }
   { "jsonrpc": "2.0", "method": "chat/completion", "params": { "prompt": "Hello!" }, "id": 2 }
   ```

The local Model_Manager handles agent reasoning natively.

---

## GUI Sidecar Endpoints

These endpoints are served by the Python sidecar UI (separate process):

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/sessions` | GET/POST | List or create chat sessions |
| `/api/sessions/{id}` | PUT/DELETE | Rename or delete session |
| `/api/sessions/{id}/duplicate` | POST | Duplicate a session |
| `/api/messages` | GET | Message history |
| `/api/adelaideenginestats` | GET | Engine statistics |
| `/api/knowledgestackfrontend/upload` | POST | Upload knowledge |
| `/api/knowledgestackfrontend/search` | GET | Search knowledge |
| `/api/knowledgestackfrontend/memory/upload` | POST | Upload memory |
| `/api/knowledgestackfrontend/memory/search` | GET | Search memory |
| `/api/knowledgestackfrontend/graph` | GET | Knowledge graph |
| `/api/knowledgestackfrontend/memory/graph` | GET | Memory graph |

---

## Error Responses

All endpoints return standard HTTP status codes:
- `200` — Success
- `400` — Bad request / invalid JSON
- `404` — Unknown endpoint
- `500` — Internal server error (check server logs)

Error responses include a JSON body with `error` field:
```json
{
  "error": "description of what went wrong"
}
```
