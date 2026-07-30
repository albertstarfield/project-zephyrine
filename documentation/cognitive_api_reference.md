# Cognitive API Reference

Zephy exposes a unified API that speaks three protocols simultaneously — OpenAI, Ollama, and Anthropic (Claude). The same server handles all three. No middleware, no translation layers.

This isn't the main focus — Zephy is an adaptive GNC framework, not an API server. But the status quo trend is everyone using OpenAI-compatible endpoints, so they're here if you want to use them. The real value is the cognitive layer underneath, not the API format.

## Supported APIs

| API | Protocol | Endpoints | Status |
|-----|----------|-----------|--------|
| **OpenAI v1** | HTTP/REST | `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/models`, `/v1/audio/speech`, `/v1/audio/transcriptions`, `/v1/images/generations` | Production |
| **Ollama** | HTTP/REST | `/api/chat`, `/api/generate`, `/api/embed`, `/api/tags`, `/api/show`, `/api/create`, `/api/pull`, `/api/push`, `/api/copy`, `/api/delete`, `/api/signin`, `/api/signout`, `/api/ps` | Production |
| **Anthropic (Claude)** | HTTP/REST | `/v1/messages` | Production |

## OpenAI v1 API

Drop-in replacement for OpenAI's API. Works with existing OpenAI SDKs and tools.

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completions (streaming + non-streaming) |
| `/v1/completions` | POST | Text completions |
| `/v1/embeddings` | POST | Text embeddings |
| `/v1/models` | GET | List available models |
| `/v1/audio/speech` | POST | Text-to-speech |
| `/v1/audio/transcriptions` | POST | Speech-to-text (Whisper) |
| `/v1/images/generations` | POST | Image generation |
| `/v1/fips/status` | GET | FIPS 140-3 compliance status |

### Example (Python OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11420/v1",
    api_key="not-needed"
)

# Chat completion
response = client.chat.completions.create(
    model="zephy",
    messages=[{"role": "user", "content": "Hello!"}]
)

# Embeddings
response = client.embeddings.create(
    model="zephy-embed",
    input="Hello world"
)
```

### Example (curl)

```bash
# Chat completion
curl http://localhost:11420/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"zephy","messages":[{"role":"user","content":"Hello!"}]}'

# List models
curl http://localhost:11420/v1/models

# Embeddings
curl http://localhost:11420/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"zephy-embed","input":"Hello world"}'
```

## Ollama API

Drop-in replacement for Ollama's API. Works with Ollama clients and tools.

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/chat` | POST | Chat completions |
| `/api/generate` | POST | Text generation |
| `/api/embed` | POST | Text embeddings |
| `/api/embeddings` | POST | Text embeddings (alias) |
| `/api/tags` | GET | List available models |
| `/api/show` | POST | Show model details |
| `/api/create` | POST | Create a model |
| `/api/pull` | POST | Pull a model |
| `/api/push` | POST | Push a model |
| `/api/copy` | POST | Copy a model |
| `/api/delete` | DELETE | Delete a model |
| `/api/signin` | POST | Sign in |
| `/api/signout` | POST | Sign out |
| `/api/ps` | GET | List running models |

### Example (Ollama Python client)

```python
import ollama

response = ollama.chat(
    model="zephy",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Example (curl)

```bash
# Chat
curl http://localhost:11420/api/chat \
  -d '{"model":"zephy","messages":[{"role":"user","content":"Hello!"}]}'

# Generate
curl http://localhost:11420/api/generate \
  -d '{"model":"zephy","prompt":"Hello!"}'

# List models
curl http://localhost:11420/api/tags
```

## Anthropic (Claude) API

Works with Anthropic's SDK and tools.

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/messages` | POST | Create a message |

### Example (Anthropic Python SDK)

```python
import anthropic

client = anthropic.Anthropic(
    base_url="http://localhost:11420",
    api_key="not-needed"
)

response = client.messages.create(
    model="zephy",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Example (curl)

```bash
curl http://localhost:11420/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"model":"zephy","max_tokens":1024,"messages":[{"role":"user","content":"Hello!"}]}'
```

## Cognitive Endpoints

These are Zephy-specific endpoints for the cognitive layer.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/telemetry` | GET | Flight telemetry data |
| `/api/ZenithRoutine` | POST | Send GNC command (roll, pitch, yaw, thrust) |
| `/api/power` | GET | Power state (StellaIcarus) |
| `/api/version` | GET | System version |
| `/api/acp` | POST | Agentic control plane |
| `/api/agenticZephyHandlessMode` | POST | Agentic handless mode |
| `/api/snowballEnagaValidationBenchmark` | POST | Cognitive architecture validation |

## Port Configuration

| Port | Service |
|------|---------|
| `11420` | Default (all APIs) |
| `8080` | Custom (`--port 8080`) |
| `14580` | MAVLink (PX4) |
| `49000` | X-Plane UDP |

## Authentication

All APIs run locally — no external auth required. The `api_key` field is accepted but not validated. Set it to any string (`"not-needed"`, `"sk-any"`) for SDK compatibility.

## Streaming

All chat/completion endpoints support streaming via Server-Sent Events (SSE):

```bash
curl http://localhost:11420/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"zephy","messages":[{"role":"user","content":"Hello!"}],"stream":true}'
```
