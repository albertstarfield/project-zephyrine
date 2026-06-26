<h1 align="center">

<sub>
<img src="documentation/ProjectZephy023LogoRenewal.png" height=256>
</sub>
<br>
</h1>

<h5 align="center"> </h5>


<h5 align="center">
<sub align="center">
<img src="documentation/Project%20Zephyrine%20HandDrawnPersonalized%20Logo.png" height=128>

</sub>
</h5>
<p align="center"><i>Hello there! I'm Adelaide Zephyrine Charlotte, Fascinating and a very nice moment to meet you, They usually called me Zephy. Hey are you ready to explore the aether with me?</i></p>

<p align="center"><h5>In Self-learning and Self-improvement We Trust</h5></p>
<hr>

[![Hippocratic License HL3-BDS-BOD-LAW-MEDIA-MIL-SOC-SUP-SV](https://img.shields.io/static/v1?label=Hippocratic%20License&message=HL3-BDS-BOD-LAW-MEDIA-MIL-SOC-SUP-SV&labelColor=5e2751&color=bc8c3d)](https://firstdonoharm.dev/version/3/0/bds-bod-law-media-mil-soc-sup-sv.html)

## Adelaide: The Lightweight version Core of Project Zephyrine

### A Glimpse Into the Aether: Abstract

Adelaide is an more efficient iteration of Project Zephyrine's core architecture, Adelaide distills the conversational and adaptive nature of Zephyrine into an engine designed for minimal storage footprint, lower runtime memory usage.

This is just another Adaptive Agent wrapper with determenistic response on certain question and query represents the integration of the **Stella-Icarus Deterministic Core** directly into the API routing layer. By utilizing an Elevated Level Privilege (ELP) priority thread scheduler and native OS tasking, we can ensure that deep cognitive reasoning (ELP0) never interrupts high-priority real-time responses on client response (ELP1), and finally ELP2 (StellaIcarus Determenistic API response)

### ⚙️ Core Architecture

Adelaide is specifically tailored for environments with lower storage and constrained runtime memory capacity.

1.  **Ada/SPARK Foundation:** Inspired by some paradigms (Not really compliance completely but just inspired), specifically (DO-178C, ECSS-E-ST-40C, ECSS-Q-ST-80C). the entire networking, caching, and task-scheduling layer is written in Ada, providing structured guidance against buffer overflows, race conditions, and memory leaks.
2.  **WCET Enforcements:** The Worst-Case Execution Time (WCET) manager actively monitors and terminates non-deterministic AI generation if it breaches strict latency budgets, falling back to cached or deterministic responses. for ELP0 ELP1 ELP2 and ELP 3
3.  **Hybrid Deno Global Internet Reference Hooks:** Instead of embedding fragile web drivers, Adelaide securely spawns isolated Deno/TypeScript subprocesses (`playwright_scraper.ts`) that utilize stealth-plugin techniques to breach bot-detection challenges and retrieve factual data for the generative core without compromising the stability of the Ada parent process.

### 🕊️ Volatus Damarae

The Ada-native orchestration layer of Adelaide is codenamed **Volatus Damarae** — a deliberate departure from the Python-centric architecture of Project Zephyrine.

Where Zephyrine relies on Python for scheduling, embedding math, and inference routing, Volatus Damarae replaces those layers with native Ada tasking, the ELP priority queue, and Kratos crash isolation. This yields deterministic scheduling, sub-millisecond TTFB, and memory safety guarantees that a Python orchestrator cannot provide.

The ELP queue features four priority tiers:

| Level | Role | Description |
|-------|------|-------------|
| **ELP0** | Background | Deep cognitive reasoning, indexing, embedding — preemptible |
| **ELP1** | Foreground | Real-time inference, user-facing generation — high priority |
| **ELP2** | Deterministic | Stella-Icarus Deterministic API responses |
| **ELP3** | Determenistic Life Critical | Deterministic light task — 1ms fixed nanosecond WCET, Ravenscar profile |

All inference flows through a single serial queue (parallelism = 1) to prevent heap corruption from concurrent llama.cpp FFI calls on shared contexts.

**Pool Capacity:** 4611686018427387904 items — expanded queue depth, never blocks on enqueue.
**Context Paging:** 9223372036854775808 tokens — increased context window from previous architecture.

### 🎭 API Interfaces

Adelaide maintains compatibility with standard communication dialects to ease integration.

#### Ollama-Compatible Endpoints (`/api/*`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/chat` | POST | Chat completion (Ollama format) |
| `/api/generate` | POST | Text generation (Ollama format) |
| `/api/tags` | GET | List available models |
| `/api/ps` | GET | Show loaded models |
| `/api/show` | POST | Show model information |
| `/api/version` | GET | Show server version |
| `/api/create` | POST | Create model (stub) |
| `/api/pull` | POST | Pull model (stub) |
| `/api/push` | POST | Push model (stub) |
| `/api/copy` | POST | Copy model (stub) |
| `/api/delete` | DELETE | Delete model (stub) |
| `/api/signin` | POST | Sign in (stub) |
| `/api/signout` | POST | Sign out (stub) |
| `/api/embeddings` | POST | Generate embeddings |
| `/api/embed` | POST | Generate embeddings (alias) |
| `/api/power` | GET | Health check endpoint |
| `/api/telemetry` | GET | Server telemetry and stats |
| `/api/ZenithRoutine` | GET | Zenith routine status |
| `/api/agenticZephyHandlessMode` | POST | Handless mode (voice I/O) |

#### OpenAI-Compatible Endpoints (`/v1/*`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completion (OpenAI format) |
| `/v1/completions` | POST | Text completion (OpenAI format) |
| `/v1/models` | GET | List available models |
| `/v1/embeddings` | POST | Generate embeddings |
| `/v1/messages` | POST | Messages (Claude API format) |
| `/v1/audio/transcriptions` | POST | Audio transcription (STT) |
| `/v1/audio/speech` | POST | Text-to-speech (TTS) |
| `/v1/images/generations` | POST | Image generation |

#### Server Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | HEAD | Heartbeat check |
| `/` | GET | Server info |

### 🚀 Prerequisites and Quick Start

Adelaide is designed to be highly portable but requires a specific set of tools to minimize bloat.

#### Requirements
*   **Alire (Ada LIbrary REpository):** Required to resolve Ada dependencies and build the core executable.
*   **Python 3.10+:** Required as the primary orchestrator for complex embedding mathematical operations and SQLite Knowledge Graph interfacing.
*   **OpenSSL:** Required for HTTPS support (self-signed certificate generation).
*   **Git:** Required for version control and submodule management.

#### Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/albertstarfield/OpenIntellegentiaPlatform
    cd OpenIntellegentiaPlatform
    ```

2.  **Run the Initialization Script:**
    The `run.py` script is designed to handle the Ada compilation, Python venv setup, and repository cloning (such as `llama.cpp` and `stable-diffusion.cpp`) automatically.
    ```bash
    cd Adelaide_Lite
    python3 run.py
    ```

    *The script will automatically fetch Ada Web Server (AWS) packages via Alire, build llama.cpp and stable-diffusion.cpp from source, generate SSL certificates, and start the local API listener on port `11420` (HTTP) and `11421` (HTTPS).*

## Warning
> A Warning to Adelaide users newcomers
> 
> **(Please Read Carefully)**
> 
> This is a highly experimental platform combining efficient software paradigms with generative AI. It is **NOT** a plug-and-play ChatGPT nor an Agents clone. The system expects you to actively monitor CPU bounds, configure ELP priority schedulers, and manage the SQLite knowledge graph.
>
> If you are expecting a system that follows the expectation and status quo of AI in general, This is not it. **Look somewhere else. You have been warned.**

## Development Note
> 1. For AWS API /v1/completion OpenAI API and Ollama API I/O use pragma profile Jorvik while Watchdog use Ravenscar seperate threading. We still use Ada 2012 SPARK 2014. Because I am goddamn outdated.
> 2. For development QC (Not release) we use gnatprove level=4 for detecting potential issues not level=0 nor level=2. to match the Indonesian minimum costumer & consumer responsiveness and reliance expectation.
> 3. For release builds we use gnatprove level=4 with Pragma profile ada 2012 SPARK 2014 and Pragma profile Jorvik for AWS API /v1/completion and Ollama API I/O. then test crosscompile with target Darwin XNU and Linux arm64 and Linux x86_64, with respective environment variables and toolchains. We exclude NT based system due to various of issue for now.


---
<h1 align="center">
<sub>
<img src="documentation/madeFromZephyFoundation.png" height=128>
</sub>
<h5 align="center">
Made with Love, Dreams, and Disciplines.<br>
<br>Snail Works</h5> <br>
Zephyrine Foundation 2023-2026
</h5>
<br>
</h1>
