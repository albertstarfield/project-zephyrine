/*
 * ── WHY THIS FILE ────────────────────────────────────────────────────────────
 * Application-level AES-256-GCM encryption shim for Ada FFI.
 *
 * Ada_Sqlite3 binds to stock libsqlite3 (alire package 0.1.1). Swapping to
 * libsqlcipher would require rebuilding the alire package — fragile across
 * alire updates. Instead, we encrypt only the sensitive content fields at the
 * application layer. The schema, indexes, timestamps, and embeddings remain
 * in plaintext (they're metadata, not secrets).
 *
 * POST-QUANTUM NOTE:
 *   AES-256 provides 128-bit post-quantum security against Grover's algorithm.
 *   128-bit security requires ~2^64 Grover iterations ≈ billions of years.
 *   SHA-384 HKDF provides 192-bit post-quantum collision resistance.
 *   No asymmetric keys → no vulnerability to Shor's algorithm.
 *   → This symmetric crypto IS post-quantum safe TODAY.
 *
 *   When NIST post-quantum symmetric standards mature, swap the cipher
 *   primitive here and in adelaide_crypto.py. The key management layer
 *   (256-bit master key, HKDF sub-keys) remains unchanged.
 *
 * ENCRYPTED BLOB FORMAT (before hex encoding):
 *   nonce(12 bytes) || AES-256-GCM ciphertext || auth_tag(16 bytes)
 *   Total overhead: 28 bytes per encrypted field.
 *
 * MASTER KEY PRIORITY (first wins):
 *   1. ADELAIDE_MASTER_KEY env var     ← portable, for CI/migration
 *   2. config/master.key               ← file, chmod 0600 (local to project)
 *   3. Generate new → write file       ← first boot (done by run.py)
 * ─────────────────────────────────────────────────────────────────────────────
 */

#ifndef ADL_CRYPTO_H
#define ADL_CRYPTO_H

#include <stddef.h>

/* ── Constants ─────────────────────────────────────────────────────────────── */
#define ADL_KEY_SIZE        32    /* 256-bit AES key */
#define ADL_KEY_HEX_SIZE    65    /* 64 hex chars + null terminator */
#define ADL_NONCE_SIZE      12    /* 96-bit random nonce for GCM */
#define ADL_TAG_SIZE        16    /* 128-bit GCM auth tag */
#define ADL_ERROR_SIZE      256   /* max error message length */
#define ADL_AAD_MAX_SIZE    256   /* max Additional Authenticated Data length */

/* ── Key Management ─────────────────────────────────────────────────────────── */

/*
 * adl_init: Load the master key.
 *
 * If key_hex_override is non-NULL and non-empty, use it directly.
 * Otherwise, read from ADELAIDE_MASTER_KEY env var.
 * Otherwise, read from config/master.key (local to project).
 * Otherwise, return -1 (caller must generate and retry).
 *
 * Returns 0 on success, -1 on error (err_buf populated).
 */
int adl_init(const char *key_hex_override, char *err_buf);

/*
 * adl_get_master_key_hex: Return pointer to the loaded master key (hex).
 * Returns NULL if adl_init() was not called or failed.
 */
const char *adl_get_master_key_hex(void);

/*
 * adl_derive_subkey: HKDF-SHA384(master_key, context) → 32-byte sub-key.
 *
 * Context examples:
 *   "adelaide:db:memory:v1"   — adelaide_memory.db
 *   "adelaide:db:literature:v1" — literatureRefIndex.db
 *   "adelaide:db:assistant:v1"  — assistant_session.db
 *   "adelaide:db:memory_index:v1" — memoryRefIndex.db
 *
 * Each DB gets its own sub-key so one DB compromised ≠ all DBs compromised.
 *
 * Returns 0 on success, -1 on error (err_buf populated).
 */
int adl_derive_subkey(const char *master_key_hex,
                      const char *context,
                      char *sub_key_hex,
                      char *err_buf);

/* ── Encrypt / Decrypt ─────────────────────────────────────────────────────── */

/*
 * adl_encrypt: AES-256-GCM encrypt plaintext → hex-encoded ciphertext.
 *
 * sub_key_hex:   64-char hex-encoded 32-byte sub-key (from adl_derive_subkey).
 * plaintext:     Raw bytes to encrypt.
 * plaintext_len: Length of plaintext.
 * aad:           Additional Authenticated Data (optional, may be NULL).
 * aad_len:       Length of AAD (0 if NULL).
 * ciphertext_hex: Output buffer. Must be at least 2*(plaintext_len + 28) + 1 bytes.
 * ciphertext_hex_len: In: size of output buffer. Out: actual length (excluding null).
 * err_buf:       256-byte error buffer.
 *
 * Returns 0 on success, -1 on error.
 */
int adl_encrypt(const char *sub_key_hex,
                const unsigned char *plaintext, size_t plaintext_len,
                const unsigned char *aad, size_t aad_len,
                char *ciphertext_hex, size_t *ciphertext_hex_len,
                char *err_buf);

/*
 * adl_decrypt: AES-256-GCM decrypt hex-encoded ciphertext → plaintext.
 *
 * sub_key_hex:     64-char hex-encoded 32-byte sub-key.
 * ciphertext_hex:  Hex-encoded ciphertext (from adl_encrypt).
 * aad:             Additional Authenticated Data (must match encryption AAD).
 * aad_len:         Length of AAD (0 if NULL).
 * plaintext:       Output buffer. Must be at least ciphertext_len/2 bytes.
 * plaintext_len:   In: size of output buffer. Out: actual plaintext length.
 * err_buf:         256-byte error buffer.
 *
 * Returns 0 on success, -1 on error (e.g., auth tag mismatch = wrong key).
 */
int adl_decrypt(const char *sub_key_hex,
                const char *ciphertext_hex,
                const unsigned char *aad, size_t aad_len,
                unsigned char *plaintext, size_t *plaintext_len,
                char *err_buf);

/*
 * adl_encrypt_string: Convenience wrapper for null-terminated strings.
 * Like adl_encrypt but takes a null-terminated string and returns
 * a hex-encoded ciphertext in a pre-allocated buffer.
 */
int adl_encrypt_string(const char *sub_key_hex,
                       const char *plaintext,
                       char *ciphertext_hex, size_t *ciphertext_hex_len,
                       char *err_buf);

/*
 * adl_decrypt_string: Convenience wrapper that guarantees null termination.
 * Like adl_decrypt but ensures the output is null-terminated.
 */
int adl_decrypt_string(const char *sub_key_hex,
                       const char *ciphertext_hex,
                       char *plaintext, size_t *plaintext_len,
                       char *err_buf);


/* ── Ada FFI Wrappers ───────────────────────────────────────────────────────── */

/*
 * Simple init: loads master key from env var or config file.
 * Returns 0 on success, -1 on error.
 */
int adl_crypto_init_wrapper(void);

/*
 * Returns 1 if master key is loaded (adl_crypto_init_wrapper was called and
 * succeeded), 0 otherwise.
 */
int adl_master_key_available(void);

/*
 * Derive sub-key, returned as malloc'd hex string (64 hex chars + null).
 * Caller must free with adl_free_cstr().
 * Returns NULL on error (writes to error_out if non-NULL).
 */
char *adl_derive_subkey_cstr(const char *context,
                              char *error_out, size_t error_out_size);

/*
 * Encrypt a plaintext field, returns malloc'd hex-encoded ciphertext blob.
 * Caller must free with adl_free_cstr().
 * Returns NULL on error (writes to error_out if non-NULL).
 */
char *adl_encrypt_field_cstr(const char *sub_key_hex, const char *plaintext,
                              char *error_out, size_t error_out_size);

/*
 * Decrypt a hex-encoded ciphertext field, returns malloc'd plaintext string.
 * Caller must free with adl_free_cstr().
 * Returns NULL on error (writes to error_out if non-NULL).
 */
char *adl_decrypt_field_cstr(const char *sub_key_hex, const char *ciphertext_hex,
                              char *error_out, size_t error_out_size);

/*
 * Free a string allocated by any adl_*_cstr() wrapper function.
 */
void adl_free_cstr(char *ptr);

/* ── FIPS 140-3 §5.1 / SP 800-90A — CTR_DRBG ────────────────────────────── */

/*
 * Deterministic Random Bit Generator (CTR_DRBG with AES-256).
 * Replaces direct RAND_bytes() calls with a FIPS-approved DRBG.
 *
 * adl_drbg_init:     Seed the DRBG from OS entropy. Call once at startup.
 * adl_drbg_generate: Generate random bytes (replaces RAND_bytes).
 * adl_drbg_reseed:   Reseed with fresh entropy.
 * adl_drbg_clear:    Zeroize DRBG state.
 */
int adl_drbg_init(size_t entropy_bytes, const char *pers_string, char *err_buf);
int adl_drbg_generate(unsigned char *out, size_t len);
int adl_drbg_reseed(const unsigned char *additional_input, size_t input_len);
void adl_drbg_clear(void);

/* ── FIPS 140-3 §5.9 Self-Tests & InferiorParadoxical Anti-Tamper ────────── */

/*
 * InferiorParadoxical — Anti-tamper dead-man's switch.
 *
 * When unauthorized modifications are detected (KAT failure, integrity
 * mismatch), the master key is zeroized and all crypto operations cease.
 * Named for the paradoxical effect: the more an attacker tampers, the more
 * they destroy what they seek.
 *
 * adl_is_poisoned:      Returns 1 if the system is poisoned (all crypto off).
 * adl_self_tests_passed: Returns 1 if power-up KATs succeeded.
 * adl_run_powerup_self_tests: Run all KATs. Returns 0 on success, -1 on
 *                        failure (also sets poisoned flag on failure).
 * adl_poison:           Force-poison the module (zeroize keys, disable crypto).
 * adl_set_expected_binary_hash: Set expected SHA-512 of compiled binary.
 * adl_set_expected_source_hash: Set expected SHA-512 of crypto source files.
 */
int adl_is_poisoned(void);
int adl_self_tests_passed(void);
int adl_run_powerup_self_tests(char *err_buf);
void adl_poison(void);
void adl_set_expected_binary_hash(const char *hash_hex);
void adl_set_expected_source_hash(const char *hash_hex);

/* ── HKDF Key Derivation ─────────────────────────────────────────────────────── */

/*
 * adl_hkdf_sha512: HKDF-SHA512 key derivation (RFC 5869).
 *
 * salt:     Salt value (may be NULL for empty salt).
 * salt_len: Length of salt in bytes.
 * ikm:      Input keying material (the user secret).
 * ikm_len:  Length of ikm in bytes.
 * info:     Context/application-specific info.
 * info_len: Length of info in bytes.
 * okm:      Output keying material buffer (must be at least okm_len bytes).
 * okm_len:  Desired output key length in bytes.
 *
 * Returns 0 on success, -1 on error.
 */
int adl_hkdf_sha512(const unsigned char *salt, size_t salt_len,
                    const unsigned char *ikm, size_t ikm_len,
                    const unsigned char *info, size_t info_len,
                    unsigned char *okm, size_t okm_len);

/*
 * adl_hkdf_sha256: HKDF-SHA256 key derivation (RFC 5869).
 *
 * salt:     Salt value (may be NULL for empty salt).
 * salt_len: Length of salt in bytes.
 * ikm:      Input keying material (the context string).
 * ikm_len:  Length of ikm in bytes.
 * info:     Context/application-specific info.
 * info_len: Length of info in bytes.
 * okm:      Output keying material buffer (must be at least okm_len bytes).
 * okm_len:  Desired output key length in bytes.
 *
 * Returns 0 on success, -1 on error.
 */
int adl_hkdf_sha256(const unsigned char *salt, size_t salt_len,
                    const unsigned char *ikm, size_t ikm_len,
                    const unsigned char *info, size_t info_len,
                    unsigned char *okm, size_t okm_len);

#endif /* ADL_CRYPTO_H */
