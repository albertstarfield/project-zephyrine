/*
 * ── Adelaide AES-256-GCM Crypto Shim ─────────────────────────────────────────
 *
 * Provides application-level field encryption for Ada FFI.
 * Uses OpenSSL 3.x EVP API for AES-256-GCM.
 *
 * BUILD REQUIREMENTS:
 *   Compiler flags: -I/opt/homebrew/opt/openssl@3.6/include
 *   Linker flags:  -L/opt/homebrew/opt/openssl@3.6/lib -lcrypto
 *
 * MASTER KEY:
 *   - 256-bit (32-byte) key, hex-encoded when stored/transmitted
 *   - Read from ADELAIDE_MASTER_KEY env var, or ~/.config/adelaide/master.key
 *   - Each DB gets a unique sub-key via HKDF-SHA384
 *
 * THREAD SAFETY:
 *   - adl_init() is NOT thread-safe (call once at startup)
 *   - All other functions are reentrant (no global state beyond the master key)
 *   - The master key pointer is set once by adl_init() and read-only after
 * ─────────────────────────────────────────────────────────────────────────────
 */

#include "adl_crypto.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

/* OpenSSL 3.x EVP API */
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <openssl/err.h>
#include <openssl/hmac.h>

/* ── Static Master Key Storage ─────────────────────────────────────────────── */
/* Set once by adl_init(), read-only thereafter. Thread-safe for reads. */
static char g_master_key_hex[ADL_KEY_HEX_SIZE] = {0};
static int g_master_key_loaded = 0;

/* ── Secure Zeroing ────────────────────────────────────────────────────────── */
/* Zero sensitive memory to prevent key material from lingering. */
static void secure_zero(void *ptr, size_t len) {
    volatile unsigned char *p = (volatile unsigned char *)ptr;
    while (len--) *p++ = 0;
}

/* ── Internal Helpers ───────────────────────────────────────────────────────── */

/* Hex encode 'len' bytes from 'in' into 'out' (out must be 2*len+1). */
static void hex_encode(const unsigned char *in, size_t len, char *out)
{
    static const char hex[] = "0123456789abcdef";
    for (size_t i = 0; i < len; i++) {
        out[i * 2]     = hex[(in[i] >> 4) & 0x0f];
        out[i * 2 + 1] = hex[in[i] & 0x0f];
    }
    out[len * 2] = '\0';
}

/* Hex decode 'hex_len' hex chars from 'in' into 'out' (out must be hex_len/2). */
/* If out is NULL, returns the expected output length (for buffer sizing). */
/* Returns number of bytes decoded, or -1 on error. */
static int hex_decode(const char *in, size_t in_len, unsigned char *out)
{
    if (in_len % 2 != 0) return -1;
    size_t out_len = in_len / 2;
    if (!out) return (int)out_len; /* length query */
    for (size_t i = 0; i < out_len; i++) {
        unsigned char hi = 0, lo = 0;
        char c = in[i * 2];
        if      (c >= '0' && c <= '9') hi = (unsigned char)(c - '0');
        else if (c >= 'a' && c <= 'f') hi = (unsigned char)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') hi = (unsigned char)(c - 'A' + 10);
        else return -1;
        c = in[i * 2 + 1];
        if      (c >= '0' && c <= '9') lo = (unsigned char)(c - '0');
        else if (c >= 'a' && c <= 'f') lo = (unsigned char)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') lo = (unsigned char)(c - 'A' + 10);
        else return -1;
        out[i] = (unsigned char)((hi << 4) | lo);
    }
    return (int)out_len;
}

/* Read file contents into buffer (up to bufsize-1 bytes, null-terminated). */
/* Returns number of bytes read (excluding null), or -1 on error. */
static int read_file(const char *path, char *buf, size_t bufsize)
{
    FILE *fp = fopen(path, "r");
    if (!fp) return -1;
    size_t n = 0;
    int c;
    while ((c = fgetc(fp)) != EOF && n < bufsize - 1) {
        /* Strip trailing whitespace/newlines */
        if (c == '\n' || c == '\r') continue;
        buf[n++] = (char)c;
    }
    buf[n] = '\0';
    fclose(fp);
    return (int)n;
}

/* Get the last OpenSSL error as a static string (for error buffers). */
static void get_openssl_error(char *buf, size_t bufsize)
{
    unsigned long err = ERR_get_error();
    if (err) {
        ERR_error_string_n(err, buf, bufsize);
    } else {
        snprintf(buf, bufsize, "Unknown OpenSSL error");
    }
}

/* ── adl_init ───────────────────────────────────────────────────────────────── */

int adl_init(const char *key_hex_override, char *err_buf)
{
    char raw_key[ADL_KEY_SIZE];
    char expanded_hex[ADL_KEY_HEX_SIZE];
    const char *src = NULL;

    err_buf[0] = '\0';

    /* Priority 1: Override (for testing) */
    if (key_hex_override && key_hex_override[0] != '\0') {
        src = key_hex_override;
        goto store;
    }

    /* Priority 2: Environment variable */
    src = getenv("ADELAIDE_MASTER_KEY");
    if (src && src[0] != '\0') {
        goto store;
    }

    /* Priority 3: Config file */
    {
        const char *home = getenv("HOME");
        if (home) {
            char path[1024];
            snprintf(path, sizeof(path), "%s/.config/adelaide/master.key", home);
            int n = read_file(path, expanded_hex, sizeof(expanded_hex));
            if (n > 0) {
                src = expanded_hex;
                goto store;
            }
        }
    }

    snprintf(err_buf, ADL_ERROR_SIZE,
             "No master key found. Set ADELAIDE_MASTER_KEY env var or "
             "create ~/.config/adelaide/master.key (run.py handles this)");
    return -1;

store:
    /* Validate hex key length (should be 64 hex chars = 32 bytes) */
    {
        size_t slen = strlen(src);
        /* Strip any trailing whitespace the file read might have left */
        while (slen > 0 && (src[slen-1] == ' ' || src[slen-1] == '\t')) slen--;
        if (slen != 64) {
            snprintf(err_buf, ADL_ERROR_SIZE,
                     "Invalid master key length: got %zu hex chars, expected 64", slen);
            return -1;
        }
        /* Decode to verify it's valid hex */
        int decoded = hex_decode(src, slen, (unsigned char*)raw_key);
        if (decoded != 32) {
            snprintf(err_buf, ADL_ERROR_SIZE,
                     "Master key is not valid hex (decoded %d bytes, expected 32)", decoded);
            return -1;
        }
        /* Store the hex-encoded key in our static buffer */
        strncpy(g_master_key_hex, src, ADL_KEY_HEX_SIZE - 1);
        g_master_key_hex[ADL_KEY_HEX_SIZE - 1] = '\0';
        g_master_key_loaded = 1;
    }

    /* Zero the raw key from stack */
    secure_zero(raw_key, ADL_KEY_SIZE);
    return 0;
}

/* ── adl_get_master_key_hex ─────────────────────────────────────────────────── */

const char *adl_get_master_key_hex(void)
{
    if (g_master_key_hex[0] == '\0') return NULL;
    return g_master_key_hex;
}

/* ── adl_derive_subkey ──────────────────────────────────────────────────────── */
/*
 * HKDF-SHA384 implementation using HMAC-SHA384.
 *
 * Implements HKDF according to RFC 5869:
 *   1. HKDF-Extract: PRK = HMAC-SHA384(salt, IKM)
 *   2. HKDF-Expand:  OKM = HKDF-Expand(PRK, info, L)
 *
 * We use an empty salt (all zeros) because the master key already has
 * sufficient entropy. Per RFC 5869 Section 3.1, using an empty salt
 * (actually a string of Hash_len zeros) is acceptable when the IKM
 * already has high entropy.
 */

#define SHA384_HASH_SIZE 48  /* SHA-384 produces 48 bytes */

/* HMAC-SHA384 helper: result must be SHA384_HASH_SIZE bytes */
static int hmac_sha384(const unsigned char *key, size_t key_len,
                       const unsigned char *data, size_t data_len,
                       unsigned char *result)
{
    unsigned int result_len = SHA384_HASH_SIZE;
    unsigned char *ret = HMAC(EVP_sha384(), key, (int)key_len,
                              data, data_len, result, &result_len);
    return (ret != NULL) ? 0 : -1;
}

int adl_derive_subkey(const char *master_key_hex,
                      const char *context,
                      char *sub_key_hex,
                      char *err_buf)
{
    unsigned char master_key[ADL_KEY_SIZE];
    unsigned char prk[SHA384_HASH_SIZE];
    unsigned char okm[ADL_KEY_SIZE];
    unsigned char salt[ADL_KEY_SIZE] = {0};  /* zero salt */

    err_buf[0] = '\0';

    /* Decode master key from hex */
    if (hex_decode(master_key_hex, 64, master_key) != ADL_KEY_SIZE) {
        snprintf(err_buf, ADL_ERROR_SIZE, "Invalid master key hex in derive_subkey");
        return -1;
    }

    /* Step 1: HKDF-Extract
     * PRK = HMAC-SHA384(salt, master_key)
     * Using zero salt (RFC 5869 Section 3.1: salt is optional when IKM has
     * sufficient entropy, but using it adds defense-in-depth) */
    if (hmac_sha384(salt, ADL_KEY_SIZE, master_key, ADL_KEY_SIZE, prk) != 0) {
        snprintf(err_buf, ADL_ERROR_SIZE, "HKDF-Extract HMAC-SHA384 failed");
        return -1;
    }

    /* Step 2: HKDF-Expand
     * T(1) = HMAC-SHA384(PRK, info || 0x01)
     * We only need one block (32 bytes output, SHA-384 can produce this in 1 round) */
    {
        unsigned char expand_input[512];
        size_t ctx_len = strlen(context);
        size_t total_len = ctx_len + 1;  /* context || 0x01 */

        if (ctx_len > 500) {
            snprintf(err_buf, ADL_ERROR_SIZE, "Context string too long (%zu)", ctx_len);
            return -1;
        }

        memcpy(expand_input, context, ctx_len);
        expand_input[ctx_len] = 0x01;  /* block counter */

        if (hmac_sha384(prk, SHA384_HASH_SIZE, expand_input, total_len, okm) != 0) {
            snprintf(err_buf, ADL_ERROR_SIZE, "HKDF-Expand HMAC-SHA384 failed");
            return -1;
        }
    }

    /* Take first 32 bytes as AES-256 key */
    hex_encode(okm, ADL_KEY_SIZE, sub_key_hex);

    return 0;
}

/* ── adl_encrypt ────────────────────────────────────────────────────────────── */

int adl_encrypt(const char *sub_key_hex,
                const unsigned char *plaintext, size_t plaintext_len,
                const unsigned char *aad, size_t aad_len,
                char *ciphertext_hex, size_t *ciphertext_hex_len,
                char *err_buf)
{
    unsigned char key[ADL_KEY_SIZE];
    unsigned char nonce[ADL_NONCE_SIZE];
    unsigned char tag[ADL_TAG_SIZE];
    unsigned char *ct_buf = NULL;
    size_t ct_len;
    int ret = -1;
    EVP_CIPHER_CTX *ctx = NULL;

    err_buf[0] = '\0';

    /* Decode sub-key */
    if (hex_decode(sub_key_hex, 64, key) != ADL_KEY_SIZE) {
        snprintf(err_buf, ADL_ERROR_SIZE, "Invalid sub-key hex in encrypt");
        return -1;
    }

    /* Generate random 96-bit nonce */
    if (RAND_bytes(nonce, ADL_NONCE_SIZE) != 1) {
        get_openssl_error(err_buf, ADL_ERROR_SIZE);
        return -1;
    }

    /* Allocate temp buffer for ciphertext (same size as plaintext for GCM) */
    ct_buf = (unsigned char*)malloc(plaintext_len ? plaintext_len : 1);
    if (!ct_buf && plaintext_len > 0) {
        snprintf(err_buf, ADL_ERROR_SIZE, "malloc failed (%zu bytes)", plaintext_len);
        return -1;
    }

    /* Encrypt with AES-256-GCM */
    ctx = EVP_CIPHER_CTX_new();
    if (!ctx) {
        snprintf(err_buf, ADL_ERROR_SIZE, "Failed to create EVP_CIPHER_CTX");
        free(ct_buf);
        return -1;
    }

    if (EVP_EncryptInit_ex(ctx, EVP_aes_256_gcm(), NULL, NULL, NULL) != 1) {
        get_openssl_error(err_buf, ADL_ERROR_SIZE);
        goto cleanup;
    }

    if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_IVLEN, ADL_NONCE_SIZE, NULL) != 1) {
        get_openssl_error(err_buf, ADL_ERROR_SIZE);
        goto cleanup;
    }

    if (EVP_EncryptInit_ex(ctx, NULL, NULL, key, nonce) != 1) {
        get_openssl_error(err_buf, ADL_ERROR_SIZE);
        goto cleanup;
    }

    /* Add AAD if provided (must be done before encrypting data) */
    if (aad && aad_len > 0) {
        int aad_out_len = 0;
        if (EVP_EncryptUpdate(ctx, NULL, &aad_out_len, aad, (int)aad_len) != 1) {
            get_openssl_error(err_buf, ADL_ERROR_SIZE);
            goto cleanup;
        }
    }

    /* Encrypt (GCM produces same-length output) */
    int out_len = 0;
    if (EVP_EncryptUpdate(ctx, ct_buf, &out_len, plaintext, (int)plaintext_len) != 1) {
        get_openssl_error(err_buf, ADL_ERROR_SIZE);
        goto cleanup;
    }
    ct_len = (size_t)out_len;

    /* Finalize (no additional ciphertext for GCM) */
    if (EVP_EncryptFinal_ex(ctx, ct_buf + out_len, &out_len) != 1) {
        get_openssl_error(err_buf, ADL_ERROR_SIZE);
        goto cleanup;
    }
    ct_len += (size_t)out_len;

    /* Get the authentication tag */
    if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_GET_TAG, ADL_TAG_SIZE, tag) != 1) {
        get_openssl_error(err_buf, ADL_ERROR_SIZE);
        goto cleanup;
    }

    /* Build output: nonce(12) || ciphertext || tag(16) */
    {
        size_t total_binary = ADL_NONCE_SIZE + ct_len + ADL_TAG_SIZE;
        size_t needed_hex = total_binary * 2 + 1;

        if (*ciphertext_hex_len < needed_hex) {
            snprintf(err_buf, ADL_ERROR_SIZE,
                     "Output buffer too small: need %zu, have %zu",
                     needed_hex, *ciphertext_hex_len);
            goto cleanup;
        }

        /* Allocate temp binary buffer */
        unsigned char *blob = (unsigned char*)malloc(total_binary);
        if (!blob) {
            snprintf(err_buf, ADL_ERROR_SIZE, "malloc failed (%zu bytes)", total_binary);
            goto cleanup;
        }

        memcpy(blob, nonce, ADL_NONCE_SIZE);
        memcpy(blob + ADL_NONCE_SIZE, ct_buf, ct_len);
        memcpy(blob + ADL_NONCE_SIZE + ct_len, tag, ADL_TAG_SIZE);

        hex_encode(blob, total_binary, ciphertext_hex);
        *ciphertext_hex_len = total_binary * 2;

        free(blob);
    }

    ret = 0;

cleanup:
    EVP_CIPHER_CTX_free(ctx);
    free(ct_buf);
    secure_zero(key, ADL_KEY_SIZE);  /* Zero key from stack */
    return ret;
}

/* ── adl_decrypt ────────────────────────────────────────────────────────────── */

int adl_decrypt(const char *sub_key_hex,
                const char *ciphertext_hex,
                const unsigned char *aad, size_t aad_len,
                unsigned char *plaintext, size_t *plaintext_len,
                char *err_buf)
{
    unsigned char key[ADL_KEY_SIZE];
    unsigned char nonce[ADL_NONCE_SIZE];
    unsigned char tag[ADL_TAG_SIZE];
    unsigned char *blob = NULL;
    int blob_len;
    size_t ct_len;
    int ret = -1;
    EVP_CIPHER_CTX *ctx = NULL;

    err_buf[0] = '\0';

    /* Decode sub-key */
    if (hex_decode(sub_key_hex, 64, key) != ADL_KEY_SIZE) {
        snprintf(err_buf, ADL_ERROR_SIZE, "Invalid sub-key hex in decrypt");
        return -1;
    }

    /* Decode the hex blob */
    size_t hex_len = strlen(ciphertext_hex);
    blob_len = hex_decode(ciphertext_hex, hex_len, NULL);
    if (blob_len < 0) {
        /* First pass just to get length */
        blob_len = (int)(hex_len / 2);
    }
    if (blob_len < (int)(ADL_NONCE_SIZE + ADL_TAG_SIZE)) {
        snprintf(err_buf, ADL_ERROR_SIZE,
                 "Ciphertext too short: %d bytes (minimum %d)",
                 blob_len, ADL_NONCE_SIZE + ADL_TAG_SIZE);
        return -1;
    }

    blob = (unsigned char*)malloc((size_t)blob_len);
    if (!blob) {
        snprintf(err_buf, ADL_ERROR_SIZE, "malloc failed (%d bytes)", blob_len);
        return -1;
    }

    if (hex_decode(ciphertext_hex, hex_len, blob) != blob_len) {
        snprintf(err_buf, ADL_ERROR_SIZE, "Failed to decode ciphertext hex");
        free(blob);
        return -1;
    }

    /* Extract nonce, ciphertext, tag */
    memcpy(nonce, blob, ADL_NONCE_SIZE);
    ct_len = (size_t)(blob_len - ADL_NONCE_SIZE - ADL_TAG_SIZE);
    memcpy(tag, blob + ADL_NONCE_SIZE + ct_len, ADL_TAG_SIZE);

    /* Check output buffer */
    if (*plaintext_len < ct_len) {
        snprintf(err_buf, ADL_ERROR_SIZE,
                 "Output buffer too small: need %zu, have %zu",
                 ct_len, *plaintext_len);
        free(blob);
        return -1;
    }

    /* Decrypt */
    ctx = EVP_CIPHER_CTX_new();
    if (!ctx) {
        snprintf(err_buf, ADL_ERROR_SIZE, "Failed to create EVP_CIPHER_CTX");
        free(blob);
        return -1;
    }

    if (EVP_DecryptInit_ex(ctx, EVP_aes_256_gcm(), NULL, NULL, NULL) != 1) {
        get_openssl_error(err_buf, ADL_ERROR_SIZE);
        goto cleanup;
    }

    if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_IVLEN, ADL_NONCE_SIZE, NULL) != 1) {
        get_openssl_error(err_buf, ADL_ERROR_SIZE);
        goto cleanup;
    }

    if (EVP_DecryptInit_ex(ctx, NULL, NULL, key, nonce) != 1) {
        get_openssl_error(err_buf, ADL_ERROR_SIZE);
        goto cleanup;
    }

    /* Add AAD if provided (must match encryption AAD) */
    if (aad && aad_len > 0) {
        int aad_out_len = 0;
        if (EVP_DecryptUpdate(ctx, NULL, &aad_out_len, aad, (int)aad_len) != 1) {
            get_openssl_error(err_buf, ADL_ERROR_SIZE);
            goto cleanup;
        }
    }

    int out_len = 0;
    if (EVP_DecryptUpdate(ctx, plaintext, &out_len, blob + ADL_NONCE_SIZE, (int)ct_len) != 1) {
        get_openssl_error(err_buf, ADL_ERROR_SIZE);
        goto cleanup;
    }

    /* Set expected tag before Final */
    if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_TAG, ADL_TAG_SIZE, tag) != 1) {
        get_openssl_error(err_buf, ADL_ERROR_SIZE);
        goto cleanup;
    }

    /* Finalize (verifies auth tag) */
    if (EVP_DecryptFinal_ex(ctx, plaintext + out_len, &out_len) != 1) {
        /* If AAD was provided, try again without AAD (backward compatibility) */
        if (aad && aad_len > 0) {
            EVP_CIPHER_CTX_free(ctx);
            ctx = EVP_CIPHER_CTX_new();
            if (!ctx) {
                snprintf(err_buf, ADL_ERROR_SIZE, "Failed to create EVP_CIPHER_CTX for retry");
                goto cleanup;
            }
            
            if (EVP_DecryptInit_ex(ctx, EVP_aes_256_gcm(), NULL, NULL, NULL) != 1) {
                get_openssl_error(err_buf, ADL_ERROR_SIZE);
                goto cleanup;
            }
            if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_IVLEN, ADL_NONCE_SIZE, NULL) != 1) {
                get_openssl_error(err_buf, ADL_ERROR_SIZE);
                goto cleanup;
            }
            if (EVP_DecryptInit_ex(ctx, NULL, NULL, key, nonce) != 1) {
                get_openssl_error(err_buf, ADL_ERROR_SIZE);
                goto cleanup;
            }
            /* Retry WITHOUT AAD */
            out_len = 0;
            if (EVP_DecryptUpdate(ctx, plaintext, &out_len, blob + ADL_NONCE_SIZE, (int)ct_len) != 1) {
                get_openssl_error(err_buf, ADL_ERROR_SIZE);
                goto cleanup;
            }
            if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_TAG, ADL_TAG_SIZE, tag) != 1) {
                get_openssl_error(err_buf, ADL_ERROR_SIZE);
                goto cleanup;
            }
            if (EVP_DecryptFinal_ex(ctx, plaintext + out_len, &out_len) != 1) {
                snprintf(err_buf, ADL_ERROR_SIZE,
                         "Decryption failed (wrong key, corrupted data, or AAD mismatch)");
                goto cleanup;
            }
            /* Success without AAD — data is legacy, will be migrated on next write */
            *plaintext_len = ct_len;
            ret = 0;
            goto cleanup;
        }
        snprintf(err_buf, ADL_ERROR_SIZE,
                 "Decryption failed (wrong key, corrupted data, or AAD mismatch)");
        goto cleanup;
    }

    *plaintext_len = ct_len;
    ret = 0;

cleanup:
    EVP_CIPHER_CTX_free(ctx);
    free(blob);
    secure_zero(key, ADL_KEY_SIZE);  /* Zero key from stack */
    return ret;
}

/* ── adl_encrypt_string ─────────────────────────────────────────────────────── */

int adl_encrypt_string(const char *sub_key_hex,
                       const char *plaintext,
                       char *ciphertext_hex, size_t *ciphertext_hex_len,
                       char *err_buf)
{
    return adl_encrypt(sub_key_hex,
                       (const unsigned char*)plaintext,
                       strlen(plaintext),
                       NULL, 0,  /* No AAD for string wrapper */
                       ciphertext_hex, ciphertext_hex_len,
                       err_buf);
}

/* ── adl_decrypt_string ─────────────────────────────────────────────────────── */

int adl_decrypt_string(const char *sub_key_hex,
                       const char *ciphertext_hex,
                       char *plaintext, size_t *plaintext_len,
                       char *err_buf)
{
    int ret = adl_decrypt(sub_key_hex,
                          ciphertext_hex,
                          NULL, 0,  /* No AAD for string wrapper */
                          (unsigned char*)plaintext, plaintext_len,
                          err_buf);
    if (ret == 0) {
        /* Guarantee null termination */
        if (*plaintext_len > 0) {
            plaintext[*plaintext_len] = '\0';
        }
    }
    return ret;
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  Ada FFI Wrappers — chars_ptr-based interface for easy Ada interop        */
/* ═══════════════════════════════════════════════════════════════════════════ */

/*
 * adl_crypto_init_wrapper: Load master key from env var or config file.
 * Returns 0 on success, -1 on error.
 * Ada: function Adl_Crypto_Init return Integer;
 *      pragma Import (C, Adl_Crypto_Init, "adl_crypto_init_wrapper");
 */
int adl_crypto_init_wrapper(void)
{
    char err[ADL_ERROR_SIZE];
    return adl_init(NULL, err);
}

/*
 * adl_master_key_available: Returns 1 if master key is loaded, 0 otherwise.
 * Ada: function Adl_Master_Key_Available return Integer;
 *      pragma Import (C, Adl_Master_Key_Available, "adl_master_key_available");
 */
int adl_master_key_available(void)
{
    return g_master_key_loaded;
}

/*
 * adl_derive_subkey_cstr: Derive sub-key, returned as malloc'd hex string.
 * Caller must free with adl_free_cstr().
 * Returns NULL on error (sets error_out if provided).
 *
 * Ada:
 *   function Adl_Derive_Subkey_Cstr
 *     (Context : chars_ptr) return chars_ptr;
 *   pragma Import (C, Adl_Derive_Subkey_Cstr, "adl_derive_subkey_cstr");
 */
char *adl_derive_subkey_cstr(const char *context, char *error_out, size_t error_out_size)
{
    char subkey[ADL_KEY_HEX_SIZE];
    char err[ADL_ERROR_SIZE];

    if (g_master_key_hex[0] == '\0') {
        if (error_out) snprintf(error_out, error_out_size, "Master key not initialized");
        return NULL;
    }

    if (adl_derive_subkey(g_master_key_hex, context, subkey, err) != 0) {
        if (error_out) snprintf(error_out, error_out_size, "%s", err);
        return NULL;
    }

    char *result = (char*)malloc(ADL_KEY_HEX_SIZE);
    if (!result) return NULL;
    memcpy(result, subkey, ADL_KEY_HEX_SIZE);
    return result;
}

/*
 * adl_encrypt_field_cstr: Encrypt a plaintext field, return malloc'd hex blob.
 * Returns NULL on error (writes to error_out).
 *
 * Ada:
 *   function Adl_Encrypt_Field_Cstr
 *     (Sub_Key : chars_ptr; Plaintext : chars_ptr) return chars_ptr;
 *   pragma Import (C, Adl_Encrypt_Field_Cstr, "adl_encrypt_field_cstr");
 */
char *adl_encrypt_field_cstr(const char *sub_key_hex, const char *plaintext,
                              char *error_out, size_t error_out_size)
{
    size_t pt_len = strlen(plaintext);
    /* Max ciphertext hex length: 2 * (pt_len + NONCE_SIZE + TAG_SIZE) + 1 */
    size_t max_ct_hex = 2 * (pt_len + ADL_NONCE_SIZE + ADL_TAG_SIZE) + 64;
    char *ct_hex = (char*)malloc(max_ct_hex);
    size_t ct_hex_len = max_ct_hex;
    char err[ADL_ERROR_SIZE];

    if (!ct_hex) {
        if (error_out) snprintf(error_out, error_out_size, "malloc failed");
        return NULL;
    }

    if (adl_encrypt_string(sub_key_hex, plaintext, ct_hex, &ct_hex_len, err) != 0) {
        if (error_out) snprintf(error_out, error_out_size, "%s", err);
        free(ct_hex);
        return NULL;
    }

    return ct_hex;
}

/*
 * adl_decrypt_field_cstr: Decrypt a hex-encoded ciphertext field, return malloc'd string.
 * Returns NULL on error (writes to error_out).
 *
 * Ada:
 *   function Adl_Decrypt_Field_Cstr
 *     (Sub_Key : chars_ptr; Ciphertext_Hex : chars_ptr) return chars_ptr;
 *   pragma Import (C, Adl_Decrypt_Field_Cstr, "adl_decrypt_field_cstr");
 */
char *adl_decrypt_field_cstr(const char *sub_key_hex, const char *ciphertext_hex,
                              char *error_out, size_t error_out_size)
{
    size_t ct_hex_len = strlen(ciphertext_hex);
    /* Max plaintext size: hex_len / 2 */
    size_t max_pt = ct_hex_len / 2 + 1;
    char *pt = (char*)calloc(max_pt, 1);
    size_t pt_len = max_pt;
    char err[ADL_ERROR_SIZE];

    if (!pt) {
        if (error_out) snprintf(error_out, error_out_size, "malloc failed");
        return NULL;
    }

    if (adl_decrypt_string(sub_key_hex, ciphertext_hex, pt, &pt_len, err) != 0) {
        if (error_out) snprintf(error_out, error_out_size, "%s", err);
        free(pt);
        return NULL;
    }

    return pt;
}

/*
 * adl_free_cstr: Free a string allocated by any adl_*_cstr function.
 *
 * Ada:
 *   procedure Adl_Free_Cstr (Ptr : chars_ptr);
 *   pragma Import (C, Adl_Free_Cstr, "adl_free_cstr");
 */
void adl_free_cstr(char *ptr)
{
    if (ptr) free(ptr);
}

/* ── Compile-time self-test (disabled by default) ──────────────────────────── */
#ifdef ADL_CRYPTO_TEST
int main(void)
{
    char err[ADL_ERROR_SIZE];
    /* 32-byte key = 64 hex chars */
    char master_hex[65] = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    char subkey[65];
    char ct[4096];
    size_t ct_len = sizeof(ct);
    char pt[4096];
    size_t pt_len;

    /* Test key derivation */
    if (adl_derive_subkey(master_hex, "adelaide:db:test:v1", subkey, err) != 0) {
        printf("FAIL: derive_subkey: %s\n", err);
        return 1;
    }
    printf("Sub-key: %s\n", subkey);

    /* Test encrypt/decrypt round-trip */
    const char *test = "Hello, Adelaide! This is sensitive data.";
    ct_len = sizeof(ct);
    if (adl_encrypt_string(subkey, test, ct, &ct_len, err) != 0) {
        printf("FAIL: encrypt: %s\n", err);
        return 1;
    }
    printf("Ciphertext (%zu hex chars): %s\n", ct_len, ct);

    pt_len = sizeof(pt);
    if (adl_decrypt_string(subkey, ct, pt, &pt_len, err) != 0) {
        printf("FAIL: decrypt: %s\n", err);
        return 1;
    }
    printf("Plaintext: %s\n", pt);

    if (strcmp(test, pt) != 0) {
        printf("FAIL: round-trip mismatch\n");
        return 1;
    }

    /* Test wrong key detection */
    const char *wrong_key = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";
    pt_len = sizeof(pt);
    if (adl_decrypt_string(wrong_key, ct, pt, &pt_len, err) == 0) {
        printf("FAIL: wrong key should have failed\n");
        return 1;
    }
    printf("Expected error with wrong key: %s\n", err);

    printf("ALL TESTS PASSED\n");
    return 0;
}
#endif /* ADL_CRYPTO_TEST */
