---
session: ses_00d9
updated: 2026-08-11T20:06:39.043Z
---

# Session Summary

## Goal
Classify all Python files in the AdelaideZephyrineSystem project by purpose, external dependencies, and Ada porting/FFI bridging feasibility, alongside documenting the existing Ada interface layer.

## Constraints & Preferences
- Project root: `/Users/albertstarfield/Documents/misc/AdaptiveSystem/project-zephyrine/AdelaideZephyrineSystem`
- System is an adaptive GNC (Guidance, Navigation, Control) framework for unmanned aircraft/spacecraft
- Ada is the core language; Python files are "sidecars" for tooling, eval, ML, and bridging
- Existing FFI pattern: Ada uses `Interfaces.C` with `pragma Import (C, ...)` for C bindings
- No direct Ada→Python FFI exists; Python↔Ada communication is via subprocess (stdin/stdout JSON) through `AdelaideBridge`
- SPARK mode used on safety-critical Ada packages; `pragma SPARK_Mode (Off)` on C-binding wrappers

## Progress

### Done
- [x] **Full inventory of Python files** — found 34+ Python files across `src/python/`, `src/ui/`, `src/ModuleSensorActuator_ELP2/`, and root `run.py`
- [x] **Read first 50 lines of all Python files** in `src/python/` (14 files + eval/ subpackage + lsh/ subpackage)
- [x] **Read root `run.py`** (215 lines, orchestration core)
- [x] **Read all key Ada interface files** for FFI pattern documentation
- [x] **Catalogued all external dependencies** per file

### In Progress
- [ ] Reading remaining Python files in `src/ModuleSensorActuator_ELP2/` and `src/ui/`
- [ ] Final classification table output

### Blocked
- (none)

## Key Decisions
- **Subprocess-based Python↔Ada bridge (not FFI)**: `AdelaideBridge` class communicates with Ada binary via stdin/stdout JSON protocol. No `Ada → Python` FFI binding exists yet.
- **C-level FFI is well-established**: Ada binds to C (llama.cpp, PX4, cFE, ROS2 RCL) via `Interfaces.C` + `pragma Import (C, ...)`. Python would need a C shim or `PyImport` to bridge.

## Next Steps
1. Read remaining files in `src/ModuleSensorActuator_ELP2/` and `src/ui/` for completeness
2. Produce final classification table with porting recommendations per file
3. Design Ada→Python FFI bridge strategy (likely via `pybind11` C shim or subprocess continuation)

## Critical Context

### Python File Classification

#### Tier 1: Core System (Ada Porting Candidates)
| File | Purpose | External Deps | Porting Notes |
|------|---------|---------------|---------------|
| `src/python/adelaide_bridge.py` | Singleton bridge to Ada binary (cosine similarity via subprocess) | `os`, `subprocess` | **Replace with native Ada FFI** — already wraps Ada binary |
| `src/python/memorythoughts.py` | Semantic memory management, embedding search, SQLite persistence | `numpy`, `requests`, `sqlite3` | Medium — core logic could be Ada, numpy for embeddings |
| `src/python/searchlocalref.py` | Local reference indexing with LSH hashing | `numpy`, `requests`, `fitz (PyMuPDF)`, `pickle` | Hard — heavy numpy + PDF processing |
| `src/python/searchglobalref.py` | Global reference search (CrossRef API) | `numpy`, `citation_verifier` | Medium — API calls stay Python, search logic ported |
| `src/python/adelaide_crypto.py` | AES-256-GCM + HKDF crypto (mirrors C shim `adl_crypto.c`) | `cryptography (AESGCM)`, `hashlib`, `hmac` | **Already has C mirror** — Ada/SPARK version exists |
| `src/python/trace_utils.py` | Execution tracing (38 lines, trivial) | `sys`, `time` | Easy — pure stdlib |
| `src/python/build.py` | Build tool (ada/python/make/cmake) | `subprocess`, `shutil` | Stay Python — build orchestration |

#### Tier 2: ML/AI Workers (FFI Bridge Candidates)
| File | Purpose | External Deps | Porting Notes |
|------|---------|---------------|---------------|
| `src/python/lsh/lsh_qrnn_worker.py` | QRNN-based 10-bit LSH hash computation | `numpy` | **Key FFI candidate** — numpy only, stdin/stdout protocol |
| `src/python/lsh/pinn_schrodinger.py` | PINN Schrödinger Bridge solver for speculative branch prediction | `numpy`, (optional `deepxde`) | Hard — 487 lines, physics-informed NN |
| `src/python/extract_pdf.py` | PDF text/image extraction for VLM injection | `fitz (PyMuPDF)`, `json` | Stay Python — PyMuPDF is C-backed already |

#### Tier 3: Tooling & External (Stay Python)
| File | Purpose | External Deps | Porting Notes |
|------|---------|---------------|---------------|
| `src/python/citation_verifier.py` | CrossRef API querying (43 lines) | `urllib` only | Trivial, pure stdlib |
| `src/python/code_tool.py` | Sandboxed code execution (25 lines) | `io`, `exec()` | Stay Python |
| `src/python/test_adelaide.py` | Unit tests for AdelaideBridge | `numpy`, `unittest` | Stay Python |
| `src/python/stella_icarus_utils.py` | Hook manager + daemon for Ada daemon processes | `loguru`, `threading`, `subprocess` | Stay Python — orchestration glue |
| `src/python/stellaicarus_bridge.py` | Bridge to StellaIcarus hook system | `loguru` (bootstrap) | Stay Python |
| `src/python/stellaicarus_daemon_runner.py` | Daemon runner for StellaIcarus | `loguru`, `psutil` | Stay Python |

#### Tier 4: Eval Suite (All Mock, Stay Python)
| File | Purpose | External Deps |
|------|---------|---------------|
| `src/python/eval/base.py` | Base evaluator + Adelaide HTTP client | `urllib` (stdlib) |
| `src/python/eval/eval_runner.py` | Runner for all evaluators | Internal `.base`, `.mmlu`, etc. |
| `src/python/eval/mmlu.py` | MMLU benchmark (mock) | `.base` only |
| `src/python/eval/gsm8k.py` | GSM8K math benchmark (mock) | `.base` only |
| `src/python/eval/hellaswag.py` | HellaSwag NLI benchmark (mock) | `.base` only |
| `src/python/eval/humaneval.py` | HumanEval coding benchmark (mock) | `.base` only |
| `src/python/eval/mbpp.py` | MBPP programming benchmark (mock) | `.base` only |
| `src/python/eval/truthfulqa.py` | TruthfulQA benchmark (mock) | `.base` only |
| `src/python/eval/winogrande.py` | Winogrande coreference benchmark (mock) | `.base` only |
| `src/python/eval/mathqa.py` | MathQA benchmark (mock) | `.base` only |
| `src/python/eval/bbq.py` | BBQ bias benchmark (mock) | `.base` only |
| + `cmmlu.py`, `jmmlu.py`, `kmmlu.py`, `mmlu_pro.py`, `livecodebench.py` | Additional eval benchmarks (referenced in eval_runner.py) | `.base` only |

#### Tier 5: External Dependencies Summary
| Library | Files Using It | Category |
|---------|---------------|----------|
| `numpy` | memorythoughts, searchlocalref, searchglobalref, test_adelaide, lsh_qrnn_worker, pinn_schrodinger | **ML/Math** |
| `requests` | memorythoughts, searchlocalref, searchglobalref | HTTP client |
| `fitz (PyMuPDF)` | searchlocalref, extract_pdf | PDF processing |
| `loguru` | stella_icarus_utils, stellaicarus_bridge, stellaicarus_daemon_runner | Logging |
| `psutil` | stellaicarus_daemon_runner, run.py | Process management |
| `cryptography` (AESGCM) | adelaide_crypto | Crypto |
| `pickle` | searchlocalref | Serialization |
| `sqlite3` | memorythoughts | DB (stdlib) |

#### Existing Ada Interface Layer (src/interfaces/)
| Ada File | Purpose | FFI Target |
|----------|---------|------------|
| `llama_interface.ads` | llama.cpp C FFI (model, context, token types, sampling) | llama.cpp |
| `llama_cpp_linker.ads` | Linker options for llama.cpp, ggml, Metal, sqlite3 | Native linker |
| `px4_ffi_bindings.ads/.adb` | PX4 MAVLink UDP socket + GNC commands | PX4 C lib |
| `cfe_ffi_bindings.ads/.adb` | NASA cFE Software Bus (pipes, subscribe, send/recv) | cFE C API |
| `ros2_rcl_bindings.ads/.adb` | ROS2 RCL thin bindings (context, node, pub/sub, timer) | rcl C API |
| `kokoro_interface.ads/.adb` | Kokoro TTS interface | Unknown |
| `mtmd_interface.ads` | multimodal interface | Unknown |
| `moonshine_interface.ads/.adb` | Moonshine bindings | Unknown |
| `moonshine_bindings.ads` | Moonshine FFI | Unknown |
| `sd_interface.ads/.adb` | SD card interface | Unknown |
| `cfs_health_monitor.ads/.adb` | cFE health monitoring | cFE |
| `cfs_telemetry.ads` | Telemetry types | cFE |
| `cfs_tool_bridge.ads` | Tool bridge | Unknown |
| `cfs_command_router.ads` | Command routing | Unknown |
| `claudealike_helper.adb` | Claude-like helper | Unknown |
| `zephyrine_main_framedisplay.adb` | WebView integration | Unknown |

### Communication Protocol
- `AdelaideBridge` (Python) ↔ Ada binary: **subprocess stdin/stdout JSON**
- Ada binary location: `bin/AdelaideZephyrineSystem`
- Protocol methods: `cosine_similarity`, `search_similar`, `store_memory`, `get_memory_stats`

## File Operations
### Read
- `/Users/albertstarfield/Documents/misc/AdaptiveSystem/project-zephyrine/AdelaideZephyrineSystem/run.py`
- `/Users/albertstarfield/Documents/misc/AdaptiveSystem/project-zephyrine/AdelaideZephyrineSystem/src/python/adelaide_bridge.py`
- `/Users/albertstarfield/Documents/misc/AdaptiveSystem/project-zephyrine/AdelaideZephyrineSystem/src/python/adelaide_crypto.py`
- `/Users/albertstarfield/Documents/misc/AdaptiveSystem/project-zephyrine/AdelaideZephyrineSystem/src/python/build.py`
- `/Users/albertstarfield/Documents/misc/AdaptiveSystem/project-zephyrine/AdelaideZephyrineSystem/src/python/citation_verifier.py`
- `/Users/albertstarfield/Documents/misc/AdaptiveSystem/project-zephyrine/AdelaideZephyrineSystem/src/python/code_tool.py`
- `/Users/albertstarfield/Documents/misc/AdaptiveSystem/project-zephyrine/AdelaideZephyrineSystem/src/python/extract_pdf.py`
- `/Users/albertstarfield/Documents/misc/AdaptiveSystem/project-zephyrine/AdelaideZephyrineSystem/src/python/memorythoughts.py`
- `/Users/albertstarfield/Documents/misc/AdaptiveSystem/project-zephyrine/AdelaideZephyrineSystem/src/python/searchglobalref.py`
- `/Users/albertstarfield/Documents/misc/AdaptiveSystem/project-zephyrine/AdelaideZephyrineSystem/src/python/searchlocalref.py`
- `/Users/albertstarfield/Documents/misc/AdaptiveSystem/project-zephyrine/AdelaideZephyrineSystem/src/python/stella_icarus_utils.py`
- `/Users/albertstarfield/Documents/misc/AdaptiveSystem/project-zephyrine/AdelaideZephyrineSystem/src/python/stellaicarus_bridge.py`
- `/Users/albertstarfield/Documents/misc/AdaptiveSystem/project-zephyrine/AdelaideZephyrineSystem/src/python/stellaicarus_daemon_runner.py`
- `/Users/albertstarfield/Documents/misc/AdaptiveSystem/project-zephyrine/AdelaideZephyrineSystem/src/python/test_adelaide.py`
- `/Users/albertstarfield/Documents/misc/AdaptiveSystem/project-zephyrine/AdelaideZephyrineSystem/src/python/trace_utils.py`
- `/Users/albertstarfield/Documents/misc/AdaptiveSystem/project-zephyrine/AdelaideZephyrineSystem/src/python/lsh/lsh_qrnn_worker.py`
- `/Users/albertstarfield/Documents/misc/AdaptiveSystem/project-zephyrine/AdelaideZephyrineSystem/src/python/lsh/pinn_schrodinger.py`
- `/Users/albertstarfield/Documents/misc/AdaptiveSystem/project-zephyrine/AdelaideZephyrineSystem/src/python/eval/base.py`
- `/Users/albertstarfield/Documents/misc/AdaptiveSystem/project-zephyrine/AdelaideZephyrineSystem/src/python/eval/eval_runner.py`
- `/Users/albertstarfield/Documents/misc/AdaptiveSystem/project-zephyrine/AdelaideZephyrineSystem/src/python/eval/mmlu.py`
- `/Users/albertstarfield/Documents/misc/AdaptiveSystem/project-zephyrine/AdelaideZephyrineSystem/src/python/eval/gsm8k.py`
- `/Users/albertstarfield/Documents/misc/AdaptiveSystem/project-zephyrine/AdelaideZephyrineSystem/src/python/eval/hellaswag.py`
- `/Users/albertstarfield/Documents/misc/AdaptiveSystem/project-zephyrine/AdelaideZephyrineSystem/src/python/eval/humaneval.py`
- `/Users/albertstarfield/Documents/misc/AdaptiveSystem/project-zephyrine/AdelaideZephyrineSystem/src/python/eval/mbpp.py`
- `/Users/albertstarfield/Documents/misc/AdaptiveSystem/project-zephyrine/AdelaideZephyrineSystem/src/python/eval/truthfulqa.py`
- `/Users/albertstarfield/Documents/misc/AdaptiveSystem/project-zephyrine/AdelaideZephyrineSystem/src/python/eval/winogrande.py`
- `/Users/albertstarfield/Documents/misc/AdaptiveSystem/project-zephyrine/AdelaideZephyrineSystem/src/python/eval/mathqa.py`
- `/Users/albertstarfield/Documents/misc/AdaptiveSystem/project-zephyrine/AdelaideZephyrineSystem/src/python/eval/bbq.py`
- `/Users/albertstarfield/Documents/misc/AdaptiveSystem/project-zephyrine/AdelaideZephyrineSystem/src/interfaces/llama_interface.ads`
- `/Users/albertstarfield/Documents/misc/AdaptiveSystem/project-zephyrine/AdelaideZephyrineSystem/src/interfaces/px4_ffi_bindings.ads`
- `/Users/albertstarfield/Documents/misc/AdaptiveSystem/project-zephyrine/AdelaideZephyrineSystem/src/interfaces/cfe_ffi_bindings.ads`
- `/Users/albertstarfield/Documents/misc/AdaptiveSystem/project-zephyrine/AdelaideZephyrineSystem/src/interfaces/ros2_rcl_bindings.ads`
- `/Users/albertstarfield/Documents/misc/AdaptiveSystem/project-zephyrine/AdelaideZephyrineSystem/src/c_bindings/llama_cpp_linker.ads`
- `/Users/albertstarfield/Documents/misc/AdaptiveSystem/project-zephyrine/AdelaideZephyrineSystem/src/python` (directory listing)

### Modified
- (none)
