# ELP Priority Queue — Volatus Damarae

## Overview

The **ELP (Elevated Level Privilege)** priority queue is the core scheduling mechanism in Adelaide. It ensures that user-facing tasks always get priority over background work, while deterministic hooks provide instant responses for patterned commands.

All inference flows through a **single serial queue** (parallelism = 1) to prevent heap corruption from concurrent llama.cpp FFI calls on shared contexts.

---

## Priority Levels

| Level | Name | Description | Preemptible? |
|-------|------|-------------|-------------|
| **ELP3** | Real-time | 1ms deterministic pacing loop (ZenithOrion) — always running, always on time | No |
| **ELP2** | Deterministic | StellaIcarus hooks — instant, precise answers for patterned commands | No |
| **ELP1** | Foreground | User-facing chat and API responses — high priority | No |
| **ELP0** | Background | Deep reasoning, knowledge indexing, embedding, learning — preemptible | Yes |

### Priority Rules

1. **ELP1 always preempts ELP0** — When a user sends a chat message, any running background task is paused
2. **ELP2/ELP3 are non-preemptible** — Deterministic tasks run to completion
3. **Background tasks only run when:**
   - No ELP1 tasks are pending or active
   - No ELP2/ELP3 tasks are running
   - The model is not busy

---

## How It Works

```
User Request ──→ ELP1 ──→ Model Manager ──→ llama.cpp
                         ↑
Background    ──→ ELP0 ──┘ (preempted when ELP1 arrives)

StellaIcarus  ──→ ELP2 ──→ Deterministic hook (bypasses LLM)

ZenithOrion   ──→ ELP3 ──→ 1ms pacing loop (always on)
```

### Request Flow

1. **Enqueue**: Request enters the queue at its assigned priority level
2. **Schedule**: Queue manager selects highest-priority pending request
3. **Execute**: Request is processed by the model manager
4. **Preempt**: If an ELP1 request arrives while ELP0 is running, ELP0 is paused
5. **Resume**: ELP0 resumes after ELP1 completes

---

## Pool Capacity

- **Queue Depth:** 2^63 items (4,611,686,018,427,387,904) — effectively unlimited
- **Context Paging:** 2^63 tokens (9,223,372,036,854,775,808)
- **Never blocks on enqueue**

---

## ELP0: Background Tasks

Background tasks include:
- **Knowledge indexing** — Scanning and indexing local files
- **Embedding generation** — Creating vector representations for semantic search
- **Self-reflection** — Reviewing past conversations to synthesize insights
- **Proactive caching** — Predicting follow-up questions and pre-caching responses

ELP0 tasks are designed to be **interruptible**. They save state and resume cleanly after being preempted by ELP1.

---

## ELP1: User-Facing Tasks

High-priority tasks that directly serve user requests:
- **Chat completions** — `/v1/chat/completions`, `/api/chat`
- **Text generation** — `/v1/completions`, `/api/generate`
- **Embeddings** — `/v1/embeddings`, `/api/embeddings`
- **Audio transcription** — `/v1/audio/transcriptions`
- **Text-to-speech** — `/v1/audio/speech`
- **Image generation** — `/v1/images/generations`

ELP1 tasks always run immediately and cannot be preempted.

---

## ELP2: StellaIcarus Deterministic Hooks

The **StellaIcarus** subsystem provides instant, 100% reliable answers for patterned commands without invoking the LLM:

- **Mathematical calculations** — Arithmetic, formulas, unit conversions
- **Hardware telemetry** — System stats, power state, GPU memory
- **Command patterns** — Predefined responses for specific input patterns

StellaIcarus hooks are JIT-compiled Python/C++ modules that execute in microseconds. They bypass the generative core entirely.

---

## ELP3: ZenithOrion Pacing Loop

The **ZenithOrion** module runs a deterministic 1ms pacing loop at the highest priority:

- **Heartbeat** — Maintains server health signals
- **Watchdog ping** — Reports liveness to the external watchdog
- **WCET monitoring** — Tracks worst-case execution time per ELP level

ZenithOrion runs on the Ravenscar profile (Ada 2012) for guaranteed timing behavior.

---

## Kratos Crash Isolation

If a C-level crash occurs (SIGSEGV, SIGBUS, SIGFPE, SIGTRAP, SIGABRT) during llama.cpp inference, the **Kratos** module catches it via `sigaction` + `longjmp` instead of killing the server. The external watchdog monitors heartbeat files and restarts the server if it dies.

---

## Monitoring

Queue state is reported every 5 seconds:
- Current depth (pending items)
- Active ELP level
- Utilization percentage
- Model status (loaded/unloaded)

Check queue status via:
```bash
curl http://localhost:11420/api/telemetry
```

---

## See Also

- [API Reference](API%20Reference.md) — All available endpoints
- [Developer Documentation](Developer%20Documentation/) — Architecture details
- [Troubleshooting Quick Guide](Developer%20Documentation/Troubleshooting%20Quick%20Guide.md)
