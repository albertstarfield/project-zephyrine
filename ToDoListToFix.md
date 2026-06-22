# ToDoListToFix.md — oMLX → Adelaide + llama.cpp Implementation Plan

## Current Architecture Summary

- **llama.cpp** via C FFI (`llama_interface.ads`) — all inference through `llama_decode`
- **Qwen3.5-0.8B** (loaded permanently, ~0.5GB VRAM, Q4_K_S quantization)
- **Qwen3.5-9B** (main reasoning model, loaded on demand)
- **KV cache quantized to Q4_1** (75% memory savings, requires flash_attn=1)
- **Speculative cache** (query-level, Jaccard similarity, NOT KV cache)

---

## Feature 1: KV SSD Cache Spillover

### What
Save/load the llama.cpp KV cache to/from SSD between inference sessions.

### llama.cpp APIs (Already Bound)
- `Llama_State_Save_File(Context, Path, Tokens, N_Tokens)` — llama_interface.ads:159
- `Llama_State_Load_File(Context, Path, Tokens, N_Tokens, N_Tokens_Out)` — llama_interface.ads:163

### New Ada Module
**File:** `kv_cache_manager.ads/adb`

```ada
-- kv_cache_manager.ads
package KV_Cache_Manager is
   --  Save KV cache to SSD file
   procedure Save_To_SSD
     (Context    : Llama_Context;
      Tokens     : System.Address;
      N_Tokens   : Interfaces.C.size_t;
      File_Path  : String;
      Success    : out Boolean);
   
   --  Load KV cache from SSD file
   function Load_From_SSD
     (Context    : Llama_Context;
      File_Path  : String;
      Tokens     : out System.Address;
      N_Tokens   : out Interfaces.C.size_t) return Boolean;
   
   --  Check if SSD cache exists for a given prompt prefix
   function Has_Cached_Prefix (Prompt_Hash : String) return Boolean;
end KV_Cache_Manager;
```

### Implementation Details
- Cache directory: `cache/kv/`
- File naming: SHA-256 hash of first 128 tokens
- LRU eviction: keep most recent 10 cache files
- Prefix matching: hash first 128 tokens, check if cache starts with same prefix

### Integration Point
In `Generate` procedure, after `Llama_Memory_Clear`:
1. Check if cached prefix exists
2. If yes: load from SSD → skip re-computing KV for cached tokens
3. If no: compute normally → save to SSD after generation

### Files to Modify
| File | Change |
|------|--------|
| `kv_cache_manager.ads` | New file — KV cache save/load spec |
| `kv_cache_manager.adb` | New file — Implementation |
| `model_manager.adb` | Add cache check/save calls in `Generate` |
| `adelaide_lite.gpr` | Add new source files to project |

---

## Feature 2: Speculative Decoding (Draft Model)

### What
Use Qwen3.5-0.8B as a draft model to generate candidate tokens, then verify with Qwen3.5-9B.

### llama.cpp Support
- `common/speculative.h` — full speculative decoding infrastructure
- `COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE` — standalone draft model
- `common_params_speculative_draft` — draft parameters

### New Ada Module
**File:** `speculative_decode.ads/adb`

```ada
-- speculative_decode.ads
package Speculative_Decode is
   --  Initialize speculative decoding with draft model
   procedure Initialize
     (Draft_Model    : Llama_Model;
      Draft_Context  : Llama_Context;
      Target_Model   : Llama_Model;
      Target_Context : Llama_Context);
   
   --  Generate draft tokens (runs on draft model)
   function Generate_Draft
     (Max_Tokens : Positive := 24) return Token_Array;
   
   --  Verify draft tokens against target model
   function Verify_Draft
     (Draft_Tokens : Token_Array;
      Accepted     : out Natural) return Boolean;
   
   --  Cleanup
   procedure Finalize;
end Speculative_Decode;
```

### Implementation Approach (Pure Ada)
1. Draft model generates K=24 tokens autoregressively
2. Target model verifies all K tokens in one batch (parallel verification)
3. Accepted prefix: if target agrees with first M tokens, skip M decode steps
4. Speedup: 3-4x for draft-heavy workloads (factual, repetitive content)

### Key Insight
Unlike oMLX's non-autoregressive DFlash, llama.cpp's speculative decoding uses standard autoregressive draft generation. Simpler but still provides 3-4x speedup.

### Files to Modify
| File | Change |
|------|--------|
| `model_types.ads` | Add `Qwen_Draft` to `Model_Type` enum |
| `model_manager.ads` | Add `Enable_Speculative` flag, `Draft_Context` field |
| `model_manager.adb` | Load draft model, modify `Generate` loop |
| `speculative_decode.ads` | New file — Draft generation + verification |
| `speculative_decode.adb` | New file — Implementation |
| `adelaide_lite.gpr` | Add new source files to project |
| `llama_safe.cpp` | Add speculative decode wrapper functions |

---

## Feature 3: Draft Quantization

### Status: ALREADY DONE
- Using `Qwen3.5-0.8B-Q4_K_S.gguf` (4-bit quantization)
- Memory: ~0.5GB VRAM (already loaded permanently)
- No additional work needed

---

## Feature 4: Sparse Attention (SpecPrefill Equivalent)

### What
Reduce prefill cost by only attending to important tokens.

### llama.cpp Approach
- Use `Flash_Attn_Type := 1` (already set in `Load_Model`)
- Flash Attention already optimizes attention computation
- For true sparse attention, would need custom ggml kernel (not available in standard llama.cpp)

### Practical Implementation
1. Implement token importance scoring in Ada
2. Select top-K tokens (15% of budget) as "anchor" tokens
3. Build sparse attention mask
4. Feed mask to llama.cpp during prefill phase

### Priority
Lower priority than speculative decoding (less immediate impact). Use existing Flash Attention for now.

---

## Implementation Order

### Phase 1: KV SSD Cache (2-3 weeks)
1. Create `kv_cache_manager.ads/adb`
2. Add save/load calls in `Generate` after `Llama_Memory_Clear`
3. Add prefix hash tracking (SHA-256 of first N tokens)
4. Test with long conversations

### Phase 2: Speculative Decoding (3-4 weeks)
1. Add `Qwen_Draft` to `Model_Type` enum
2. Create `speculative_decode.ads/adb`
3. Modify `Generate` loop to use draft model
4. Test throughput improvement (tokens/sec)

### Phase 3: Sparse Attention (4-6 weeks, optional)
1. Implement token importance scoring
2. Build sparse attention mask
3. Test TTFT improvement

---

## Key Technical Details

### KV SSD Cache
- llama.cpp state files are ~100-500MB depending on context size
- Save to `cache/kv/{prompt_hash}.bin`
- LRU eviction: keep most recent 10 cache files
- Prefix matching: hash first 128 tokens, check if cache starts with same prefix

### Speculative Decoding
- Draft model generates K=24 tokens autoregressively
- Target model verifies all K tokens in one batch (parallel verification)
- Accepted prefix: if target agrees with first M tokens, skip M decode steps
- Speedup: 3-4x for draft-heavy workloads (factual, repetitive content)
- No speedup for creative/unpredictable content

### Draft Quantization
- Already using `Qwen3.5-0.8B-Q4_K_S.gguf` (4-bit)
- Memory: ~0.5GB VRAM (already loaded permanently)
- No additional work needed

---

## Open Questions

1. **Should we start with KV SSD Cache or Speculative Decoding?** — Recommend KV SSD Cache first (simpler, less risk).

2. **Draft model selection:** Qwen3.5-0.8B as draft. Should we use same quantization (`Q4_K_S`) or try `Q4_K_M` for better quality?

3. **Cache eviction policy:** LRU (keep most recent N) or LFU (keep most frequently used)?

4. **Speculative decoding K value:** Start with K=24 (oMLX default) or test different values?

5. **Priority:** Which feature matters most for immediate use case?

---

## Files Summary

| File | Status | Description |
|------|--------|-------------|
| `kv_cache_manager.ads` | NEW | KV cache save/load spec |
| `kv_cache_manager.adb` | NEW | KV cache implementation |
| `speculative_decode.ads` | NEW | Draft generation + verification spec |
| `speculative_decode.adb` | NEW | Draft generation implementation |
| `model_types.ads` | MODIFY | Add `Qwen_Draft` enum |
| `model_manager.ads` | MODIFY | Add speculative decoding support |
| `model_manager.adb` | MODIFY | Integrate KV cache + speculative decoding |
| `adelaide_lite.gpr` | MODIFY | Add new source files |
| `llama_safe.cpp` | MODIFY | Add speculative decode wrappers |
