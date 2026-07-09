# Encrypted In-Memory KV Cache Plan

The goal is to securely save and load the KV cache to/from disk by encrypting it entirely in memory, without ever writing plaintext to disk, and without modifying the `llama.cpp` library itself. 

## User Review Required

> [!WARNING]
> This approach requires allocating a buffer in RAM equal to the size of the KV cache during the save and load operations. For large models and large context sizes, the KV cache can be several gigabytes, meaning this operation will temporarily spike RAM usage. Please confirm this is acceptable.

## Proposed Changes

### `llama_interface.ads`
- Bind the following functions from `llama.h`:
  - `llama_state_get_size(llama_context * ctx)`
  - `llama_state_get_data(llama_context * ctx, uint8_t * dst, size_t size)`
  - `llama_state_set_data(llama_context * ctx, const uint8_t * src, size_t size)`

### `adl_crypto.h` & `adl_crypto.c` (or equivalent cryptography layer)
- Implement binary-to-binary encryption/decryption routines (e.g., `adl_encrypt_raw` / `adl_decrypt_raw`) using OpenSSL's `EVP_aes_256_gcm` or `EVP_aes_256_ctr`.
- Ensure these functions do not impose the hex-encoding overhead used for string-based fields, as this is unnecessary for binary files and would double memory usage.

### `kv_cache_manager.adb`
- Replace calls to `Llama_State_Save_File` with a custom save flow:
  1. Determine required size using `llama_state_get_size`.
  2. Allocate an in-memory buffer in Ada.
  3. Populate the buffer using `llama_state_get_data`.
  4. Encrypt the buffer in-memory using the new `adl_crypto` raw encryption function and the `Adelaide_Master_Key`.
  5. Write the encrypted buffer directly to the `.bin` file on disk via standard file I/O.
- Replace calls to `Llama_State_Load_File` with a custom load flow:
  1. Read the `.bin` file from disk into an allocated buffer.
  2. Decrypt the buffer in-memory using the new `adl_crypto` raw decryption function.
  3. Load the plaintext buffer into `llama.cpp` using `llama_state_set_data`.
  4. Securely clear the buffer and free the memory.

## Verification Plan

### Automated Tests
- Run existing test suites `python3 run.py --test-fips --verbose` to ensure KV cache encryption/decryption does not break system behavior or memory constraints.

### Manual Verification
- Test generating output across multiple turns in the UI to ensure the conversation context loads correctly from the encrypted cache file and that the model produces coherent continuation.
- Inspect the saved `.bin` cache files manually using `xxd` or `hexdump` to verify they contain encrypted, high-entropy data rather than raw floating-point tensors.
