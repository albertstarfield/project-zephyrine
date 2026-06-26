# Metal Buffer Leak Issue on File Indexing

## Summary

The Adelaide server crashes with `SIGTRAP (-5)` due to Metal GPU command buffer OOM every time the embedding model (`QWEN_EMBEDDING`) performs its first `llama_decode` call during file indexing. The crash is **100% reproducible** and the server **cannot self-recover** — each crash→relaunch cycle leaks Metal backend objects, making subsequent crashes worse.

---

## Crash Signature

```
ggml_metal_synchronize: error: command buffer 0 failed with status 5
error: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)
```

**Signal**: SIGTRAP (-5) — 11 out of 11 crashes examined

**Failing operation**: First `Llama_Decode` call with batch size 30 during embedding inference

---

## Reproduction Sequence

Every crash follows this exact sequence:

```
1. Server starts
2. SNOWBALL_ENAGA_ORCHESTRATOR loads OK (~5.6GB, N_Gpu_Layers=-1, N_Ctx=8192)
3. ELP0 Knowledge-Index task triggers QWEN_EMBEDDING load
4. Embedding model loaded to GPU (all 28 layers, 639MB, N_CTX=1024)
5. Tokenize prompt (800 chars → ~286 tokens)
6. Llama_Decode batch 30 tokens → FAILS (Code: 5)
7. Metal command buffer OOM → SIGTRAP → FATAL banner → server dead
```

The crash **always** happens on the first decode of the embedding model. The orchestrator model loads and operates fine.

---

## Memory State at Crash

| Metric | Value |
|--------|-------|
| Free CPU RAM | 5123MB / 12124MB (42%) |
| GPU used by orchestrator | ~5.6GB |
| GPU used by embedding | ~639MB |
| Total GPU | ~6.2GB of 12GB |
| Free GPU (reported) | ~5.8GB |

**The system is NOT out of CPU RAM.** The Metal GPU command buffer allocator itself is failing despite having ~5.8GB free.

---

## Compute Buffer Mismatch

```
~llama_context: MTL0 compute buffer size of 151.3682 MiB, does not match expectation of 149.1143 MiB
~llama_context: CPU compute buffer size is 5.0049 MiB, matches expectation of 5.0049 MiB
```

The Metal backend allocates **2.25 MiB more** than expected. This overshoot pushes the command buffer allocation over Metal's internal limit, triggering the OOM.

---

## Escalating Leak Across Crash→Relaunch Cycles

The server auto-relaunches after each crash (via `run.py` crash recovery). Each relaunch creates a new Metal backend without fully releasing the previous one:

| Crash # | Epoch | Metal Init Calls | Llama_Decode Fails |
|---------|-------|------------------|--------------------|
| 1 | 1782460190 | 3 | 1 |
| 2 | 1782460369 | 4 | 2 |
| 3 | 1782460377 | 7 | 3 |
| 4 | 1782460388 | 10 | 4 |
| 5 | 1782460397 | 13 | 5 |
| 6 | 1782460403 | 16 | 6 |
| 7 | 1782460415 | 19 | 7 |
| 8 | 1782460422 | 22 | 8 |
| 9 | 1782460435 | 25 | 9 |
| 10 | 1782460444 | 28 | 10 |
| 11 | 1782460455 | 31 | 11 |

**Metal init calls increase by ~3 per session.** The leaking Metal backends from previous crashes consume Metal's internal GPU memory pool, making each subsequent crash worse.

---

## Why the Embedding Model Specifically

The embedding model uses `N_Gpu_Layers=-1` (all layers on GPU) and `N_CTX=1024`. When `llama_decode` is called:

1. Metal allocates compute buffers for the graph
2. The compute buffer size (151.3682 MiB) exceeds the reserved expectation (149.1143 MiB) by 2.25 MiB
3. With leaked Metal backends from previous crashes consuming part of the GPU command buffer pool, there is not enough remaining capacity
4. Metal returns `kIOGPUCommandBufferCallbackErrorOutOfMemory` (status 5)
5. llama.cpp translates this to return code 5
6. The Ada exception handler catches this and triggers SIGTRAP

The orchestrator model does not hit this because:
- It loads first (before any leaks accumulate)
- Its compute buffer reservation is larger but succeeds on the first attempt
- The leak only matters when the backend is already partially compromised

---

## WCET Telemetry Evidence

All 11 crash logs show WCET values of `0ns` for all pipeline stages:

```
[WCET] Pipeline: 0ns | ELP0: 0ns | ELP1: 0ns | ELP2: 0ns | ELP3: 0ns
```

This confirms the crash happens **before any generation completes**. The embedding model fails on its very first decode, so no pipeline timing is ever recorded.

---

## Root Cause Analysis

The root cause is a **Metal backend leak** combined with a **compute buffer size overshoot**:

1. **Leak**: When the server crashes and relaunches, the previous process's Metal backend objects are not fully released by the OS. Each relaunch creates new Metal objects on top of leaked ones.

2. **Overshoot**: The compute buffer for the embedding model is 2.25 MiB larger than the reserved expectation. This is a llama.cpp/Metal integration issue where the actual allocation exceeds the reservation.

3. **Threshold**: With ~3 leaked Metal backends (by crash #3), the combined leaked memory + new allocation exceeds Metal's command buffer limit, causing every subsequent embedding decode to fail.

4. **No recovery**: The server cannot recover because the leak persists across process restarts. Only a full system reboot or waiting for Metal to reclaim leaked objects would fix it.

---

## Affected Files

- `src/model_manager.adb` — `Load_Model`, `Get_Single_Embedding`, embedding model decode
- `src/knowledge_manager.adb` — File crawl indexing loop (triggers embedding model load)
- `src/model_manager.ads` — Model kind declarations

---

## Evidence Files

- `logs/I_am_incompetent_Panicked_and_Never_Enough_PANIC_*.log` — 11 crash logs
- `logs/I_am_incompetent_Panicked_and_Never_Enough_PANIC_*.png` — Crash plots
- `run/wcet.csv` — WCET telemetry (all zeros)
- `run/acceleration.csv` — GPU memory telemetry

---

## Potential Fix Directions

1. **Force Metal backend recreation**: After each crash, explicitly destroy and recreate the Metal backend instead of reusing the leaked one.

2. **Reduce compute buffer overshoot**: Adjust the embedding model's batch size or context size to bring the actual allocation within the reserved expectation.

3. **Skip KV save on Metal OOM**: When Metal reports OOM, do not attempt to save KV cache (which triggers more Metal operations). Just unload and reload.

4. **Rate-limit embedding loads**: Add a delay between embedding model load/unload cycles to allow Metal to reclaim leaked command buffers.

5. **Adaptive GPU layer strategy based on source**: Use CPU-only embedding for file literature indexing (burst crawl), but use GPU/Tensor Engine via the Accelerator API for all other operations:
   - **File literature index (any ELP level)** → CPU-only (`N_Gpu_Layers=0`) — avoids Metal OOM during sustained burst crawl
   - **ELP0 + other operations** → GPU via Accelerator API (`N_Gpu_Layers=-1`) — ELP0 is not restricted from GPU
   - **ELP1 user requests** → GPU via Accelerator API (`N_Gpu_Layers=-1`) — fast interactive embedding
   - The selection is based on **source type** (file index vs other), not ELP level alone

6. **Implement `FreeParallelMemory` standalone**: After each embedding operation, explicitly call `FreeParallelMemory` to release the Metal backend before the next load.
