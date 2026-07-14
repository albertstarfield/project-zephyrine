# Adelaide Zephyrine Framework Structure

This document explains the project structure to help you get started. The codebase might look daunting at first, but once you understand the layering, it becomes clear where everything lives.

---

## Top-Level Layout

```
project-zephyrine/
├── AdelaideZephyrineSystem/     # Main Ada/Python system
│   ├── src/                     # All source code (Ada, Python, C, Coq)
│   ├── vendor/                  # External dependencies (llama.cpp, PX4, ROS2, etc.)
│   ├── data/                    # Models, knowledge bases, runtime data
│   ├── tests/                   # Test suites
│   ├── run.py                   # Entry point (help screen, daemon management)
│   ├── run.sh                   # Shell wrapper for run.py
│   ├── alire.toml               # Ada dependency manager config
│   └── *.gpr                    # GNAT project files (build configs)
├── documentation/               # Docs, API reference, architecture
├── CONTRIBUTING.md              # Commit format, code of conduct
├── README.md                    # You are here
└── citations.bib                # Research citations
```

---

## Source Code (`AdelaideZephyrineSystem/src/`)

The source is organized by **responsibility**. Each directory has a clear purpose.

```
src/
├── core/                        # Server, watchdog, system init
├── engine/                      # Cognitive scheduling, ELP queue, decision engine
├── interfaces/                  # Interface.C FFI bindings (LLaMA, PX4, ROS2, TTS, ASR)
├── crypto/                      # AES-256, FIPS 140-3, key management
├── managers/                    # Database, knowledge, tools, SD
├── utils/                       # Monitoring, tracing, helpers
├── c_bindings/                  # Raw C code called by Ada via FFI
├── ModuleSensorActuator_ELP2/   # ELP2 — StellaIcarus (250µs sensor/telemetry)
├── ModuleSensorActuator_ELP3/   # ELP3 — ZenithOrion (250µs actuator control)
├── NonDeterministicGenerativeModelManager/  # LLM model management, KV cache, speculative decode
├── python/                      # Python tools (non-deterministic domain)
├── ui/                          # GUI sidecar (web frontend)
├── coq_proofs/                  # Coq formal verification proofs
├── config/                      # Runtime config (encrypted API keys)
├── data/                        # Runtime data (NetworkMemoryPool)
├── test_data/                   # Test fixtures (TTS sample audio)
├── Util/                        # Build verification (sabotage_verifier.py)
└── version.ads                  # System version constant
```

---

### `src/core/` — System Core

The heartbeat of Zephy. Server, watchdog, auto-configuration.

| File | Purpose |
|------|---------|
| `adelaide_server.adb` | Main HTTP server (binds port 11420) |
| `adelaide_server_pkg.adb` | Server package — request routing, API handlers |
| `adelaide_zephyrine_system.adb` | System initialization and main loop |
| `auto_config.adb` | Auto-detects hardware, configures LLaMA context/thread counts |
| `watchdog_manager.adb` | Monitors server health, auto-restarts on failure |
| `watchdog_ipc.adb` | Watchdog inter-process communication |
| `shutdown_manager.adb` | Graceful shutdown (SIGTERM/SIGINT handling) |
| `system_integrity.adb` | Runtime integrity checks (tamper detection) |
| `benchmark_manager.adb` | Performance benchmarking |

---

### `src/engine/` — Cognitive Engine

The cognitive and scheduling core. How Zephy thinks and prioritizes.

| File | Purpose |
|------|---------|
| `elp_queue.adb` | **ELP Priority Queue** — deterministic task scheduling (ELP0–ELP3) |
| `cronia_scheduler.adb` | Cron-like task scheduler for background operations |
| `kratos.adb` | Decision engine — GNC advisory generation |
| `proactive_engine.adb` | Proactive suggestions (anticipates pilot needs) |
| `response_cache.adb` | LRU cache for inference results |
| `streaming_queue.adb` | Streaming response queue for real-time output |
| `verification_manager.adb` | Runtime verification of safety constraints |
| `image_encoder.adb` | Multimodal image encoding (vision models) |
| `multimodal_content_parser.adb` | Parses mixed text/image/voice content |
| `scheduler_manager.adb` | High-level scheduler coordination |
| `accuracy_benchmark_manager.adb` | Model accuracy benchmarking |

---

### `src/interfaces/` — Interface.C FFI Bindings

Ada speaks to C libraries here through `Interfaces.C` pragma Import. This is how Zephy talks to the outside world — no Python middleware, direct Ada ↔ C.

| File | Purpose |
|------|---------|
| `llama_interface.ads` | **LLaMA.cpp FFI** — inference, model loading, sampling |
| `px4_ffi_bindings.ads` | **PX4 MAVLink FFI** — GNC commands to flight controller via UDP |
| `ros2_rcl_bindings.ads` | **ROS2 RCL FFI** — native DDS pub/sub (no rclpy) |
| `kokoro_interface.ads` | **Kokoro TTS** — text-to-speech synthesis |
| `moonshine_interface.ads` | **Moonshine ASR** — speech-to-text |
| `moonshine_bindings.ads` | Moonshine low-level C bindings |
| `sd_interface.ads` | **Stable Diffusion** — image generation |
| `mtmd_interface.ads` | Multimodal content handling |
| `claudealike_helper.ads` | Claude-compatible API helper |
| `supertonic_bindings.ads` | Supertonic audio processing FFI |

---

### `src/crypto/` — Security Layer

AES-256, FIPS 140-3 compliance, key management. All critical paths are SPARK-verified.

| File | Purpose |
|------|---------|
| `adelaide_crypto.adb` | Core encryption/decryption (AES-256-GCM) |
| `api_key_manager.adb` | API key CRUD (add/remove/list/edit) |
| `master_key_store.adb` | Master key storage and derivation |
| `key_derivation.adb` | Key derivation functions |
| `fips_audit.adb` | FIPS 140-3 compliance auditing |
| `identity_manager.adb` | Device identity and attestation |
| `integrity_utils.adb` | Integrity verification utilities |
| `spark_drbg.ads` | SPARK-verified deterministic random bit generator |

---

### `src/managers/` — Subsystem Managers

High-level subsystem managers.

| File | Purpose |
|------|---------|
| `database_manager.adb` | SQLite database operations |
| `knowledge_manager.adb` | RAG knowledge base (upload/search/embeddings) |
| `tool_manager.adb` | Tool registration and dispatch |
| `toolchain_manager.adb` | External toolchain integration |
| `sd_manager.adb` | Stable Diffusion model management |

---

### `src/utils/` — Utilities and Monitoring

Shared helpers and monitoring.

| File | Purpose |
|------|---------|
| `stella_icarus.adb` | StellaIcarus daemon — hardware monitor, power state |
| `performance_monitor.adb` | Runtime performance metrics |
| `log_aggregator.adb` | Centralized logging |
| `adelaide_trace.adb` | Distributed tracing |
| `fuzzy_match.adb` | Fuzzy string matching |
| `math_utils.adb` | Math helpers |
| `lsh_hash.adb` | Locality-sensitive hashing |
| `elab_probe.adb` | Elaboration probe (startup verification) |

---

### `src/c_bindings/` — Raw C Bindings

C/C++ code that Ada calls via FFI. These live here because they are tightly coupled to the Ada interface packages.

| File | Purpose |
|------|---------|
| `adl_crypto.c` | Crypto primitives (AES, HMAC) |
| `adl_tpm2.c` | TPM 2.0 interface |
| `adl_drbg_shim.c` | DRBG shim for SPARK verification |
| `adl_secure_enclave.c` | Secure enclave operations |
| `scheduling.c` | Low-level scheduling helpers |
| `sd_helper.c` | Stable Diffusion helper functions |
| `shutdown_handler.c` | Signal handlers (SIGTERM, SIGINT) |
| `kratos_signal.c` | Kratos signal processing |
| `llama_safe.cpp` | Safe LLaMA.cpp wrapper |
| `stderr_suppress.c` | stderr suppression (noisy vendor libs) |
| `stdout_unbuffer.c` | stdout unbuffering (real-time logging) |
| `supertonic/` | Supertonic audio C API (CMake build) |

---

### `src/ModuleSensorActuator_ELP2/` — ELP2 StellaIcarus (Sensors)

**250µs deterministic** sensor polling, telemetry, and hardware communication.

| Path | Purpose |
|------|---------|
| `fmc_servo_manual_hook_test.py` | Example sensor hook (Python) |
| `basic_math_hook.py` | Basic math sensor hook |
| `basic_matrix_math_hook.py` | Matrix math sensor hook |
| `hello_world_hook_example_dev_getstarted.py` | Getting-started example hook |
| `ros2_actuator_hook.py` | ROS2 actuator bridge hook |
| `ROS2/si_ros2_telemetry.ads` | Native Ada ROS2 telemetry node |
| `avionics_daemon/` | Hardware daemon (MCU communication) |
| `avionics_zephy_fmc_cpp_microcontroller_io_fmc_bridge_mk1/` | FMC bridge (Unix domain sockets to flight computer) |
| `stella_greeting/` | Stella greeting module |

---

### `src/ModuleSensorActuator_ELP3/` — ELP3 ZenithOrion (Actuators)

**250µs deterministic** actuator control and flight-critical operations.

| Path | Purpose |
|------|---------|
| `src/zenith_orion.adb` | Main pacing loop (4kHz / 250µs) |
| `src/zenith_manager.adb` | Actuator manager |
| `src/zenith_orion.ads` | Pacing loop spec |
| `src/zenith_manager.ads` | Manager spec |
| `ROS2/zo_ros2_actuator.ads` | Native Ada ROS2 actuator node |
| `ROS2/zo_ros2_actuator.adb` | ROS2 actuator implementation |
| `config/` | ELP3 configuration |
| `zenith_orion.gpr` | GNAT project file for ELP3 build |

---

### `src/NonDeterministicGenerativeModelManager/` — LLM Model Management

Manages the LLaMA inference models, KV cache, and speculative decoding.

| File | Purpose |
|------|---------|
| `model_manager.adb` | Model loading, switching, lifecycle |
| `model_types.ads` | Model type definitions |
| `kv_cache_manager.adb` | KV cache management for long contexts |
| `speculative_cache.adb` | Speculative decoding cache |
| `speculative_decode.adb` | Speculative decoding implementation |
| `embedding_batcher.adb` | Batched embedding computation |
| `reranker.adb` | Reranking for RAG retrieval |

---

### `src/python/` — Python Tools (Non-Deterministic)

Python utilities for knowledge processing, search, and tooling. These run in the non-deterministic domain (ELP0/ELP1).

| File | Purpose |
|------|---------|
| `adelaide_crypto.py` | Python crypto wrapper |
| `adelaide_bridge.py` | Ada ↔ Python bridge |
| `stellaicarus_bridge.py` | StellaIcarus Python bridge |
| `stellaicarus_daemon_runner.py` | Daemon runner |
| `memorythoughts.py` | Memory/knowledge processing |
| `searchlocalref.py` | Local document search |
| `searchglobalref.py` | Global document search |
| `code_tool.py` | Code analysis tool |
| `extract_pdf.py` | PDF extraction |
| `math_tool.py` | Math computation |
| `review.py` | Code review tool |
| `security.py` | Security analysis |
| `cat_tool.py` | File reading tool |
| `directory.py` | Directory listing tool |
| `file_edit.py` | File editing tool |
| `grep.py` | Content search tool |
| `git.py` | Git operations tool |
| `hook.py` | Hook management |
| `issue.py` | Issue tracking tool |
| `package.py` | Package management |
| `todo.py` | Todo list tool |
| `killshell.py` | Shell process management |
| `think_tag_sanitizer.py` | Think-tag output sanitizer |
| `citation_verifier.py` | Academic citation verification |
| `trace_utils.py` | Distributed tracing utils |
| `stella_icarus_utils.py` | StellaIcarus utilities |
| `lsh/` | Locality-sensitive hashing (Python) |
| `eval/` | Evaluation framework |
| `tests/` | Python unit tests |

---

### `src/ui/` — GUI Sidecar

Web-based GUI that runs alongside the server.

| Path | Purpose |
|------|---------|
| `sidecar_ui.py` | Sidecar UI server |
| `create_macos_app.py` | macOS .app bundle creator |
| `frontend/` | Web frontend (HTML/JS/CSS) |
| `.certs/` | TLS certificates for local dev |

---

### `src/coq_proofs/` — Formal Verification

Coq proof files for formal verification of critical components. Every Ada package has a corresponding `.v` proof file.

| Pattern | Purpose |
|---------|---------|
| `*.v` | Coq proof source |
| `*.vo` | Compiled Coq proof |
| `*.glob` | Coq proof indices |
| `*.aux` | Build artifacts |

Key proofs include: `elp_queue.v`, `adelaide_crypto.v`, `spark_drbg.v`, `kratos.v`, `verification_manager.v`.

---

### `src/config/` — Runtime Configuration

| File | Purpose |
|------|---------|
| `api_keys.enc` | Encrypted API key store |

---

### `src/data/` — Runtime Data

| Path | Purpose |
|------|---------|
| `NetworkMemoryPool/` | Network memory pool for distributed reasoning |

---

### `src/test_data/` — Test Fixtures

| File | Purpose |
|------|---------|
| `sampleAdeltts_blob.dat` | TTS test audio blob |
| `sampleAdeltts_refAudioSpeech.dat` | TTS reference audio |

---

### `src/Util/` — Build Verification

| File | Purpose |
|------|---------|
| `sabotage_verifier.py` | Post-build integrity verification (sabotage detection) |

---

### `src/version.ads` — Version Constant

Single source of truth for the system version number.

---

## Vendor Dependencies (`vendor/`)

External libraries compiled or cloned into the project.

| Directory | What It Is |
|-----------|------------|
| `llama.cpp/` | LLM inference engine (C++) |
| `PX4-Autopilot/` | Flight controller firmware (SITL + NuttX) |
| `mavlink_c_v2/` | MAVLink C headers (protocol for PX4) |
| `ros_env/` | ROS2 environment (RoboStack/Micromamba) |
| `kokoro-onnx/` | Kokoro TTS (ONNX runtime) |
| `kokoro_models/` | Kokoro voice models |
| `kokoclone/` | Kokoro clone |
| `moonshine/` | Moonshine ASR (speech-to-text) |
| `stable-diffusion.cpp/` | Image generation (C++) |
| `ggml/` | Tensor library (used by llama.cpp) |
| `tts_kokoro_component/` | TTS Kokoro component |
| `.micromamba/` | Micromamba environment manager |

---

## Key Concepts

### ELP Priority Queue
The heart of real-time scheduling. Tasks are assigned to priority levels:

```
ELP3 (ZenithOrion)  → 250µs  → Actuators, flight control (DETERMINISTIC)
ELP2 (StellaIcarus) → 250µs  → Sensors, telemetry (DETERMINISTIC)
ELP1 (Inference)    → on-demand → LLM inference (NON-DETERMINISTIC)
ELP0 (Background)   → preemptible → RAG indexing (NON-DETERMINISTIC)
```

### Interface.C FFI
Ada calls C libraries directly through `Interfaces.C` pragma Import. No Python middleware. This gives us:
- Deterministic timing (no GC pauses)
- Direct hardware access
- Memory safety (Ada) + performance (C)

### Deterministic vs Non-Deterministic
- **Deterministic (ELP2/ELP3):** Fixed timing, no exceptions, no dynamic allocation. Safety-critical.
- **Non-Deterministic (ELP0/ELP1):** Best-effort, preemptible. Inference and background tasks.

### Why ROS2 and PX4?
- **PX4** = flight logic interface/driver (MAVLink, attitude control, motor mixing)
- **ROS2** = actuator/sensor middleware (DDS, servos, gimbals, payloads)
- **Zephy** = sits between them, learning from PX4 telemetry and commanding actuators through ROS2

---

## Where to Start

| If you want to... | Look at... |
|-------------------|------------|
| Understand the server | `src/core/adelaide_server.adb` |
| See how inference works | `src/interfaces/llama_interface.ads` |
| Modify the help screen | `run.py` (around line 2530) |
| Add a sensor | `src/ModuleSensorActuator_ELP2/` |
| Add an actuator | `src/ModuleSensorActuator_ELP3/` |
| Understand ROS2 integration | `src/interfaces/ros2_rcl_bindings.ads` |
| See PX4/MAVLink bridging | `src/interfaces/px4_ffi_bindings.ads` |
| Review security/crypto | `src/crypto/` |
| Check LLM model management | `src/NonDeterministicGenerativeModelManager/` |
| Check test coverage | `tests/` |
| See formal proofs | `src/coq_proofs/` |
| Understand the GUI | `src/ui/` |

---

## Build System

- **Ada:** Built with `alr build` (Alire package manager)
- **GNAT Projects:** `*.gpr` files define build configurations
- **Python:** Managed via `venv/` (virtual environment)
- **C/C++:** Compiled via GNAT or CMake (for vendor libs)
- **Coq:** Proofs compiled with `coqc`

---

*For questions, see [CONTRIBUTING.md](../CONTRIBUTING.md) or open an issue.*
