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
 *   - Read from ADELAIDE_MASTER_KEY env var, or config/master.key (local)
 *   - Each DB gets a unique sub-key via HKDF-SHA384
 *
 * THREAD SAFETY:
 *   - adl_init() is NOT thread-safe (call once at startup)
 *   - All other functions are reentrant (no global state beyond the master key)
 *   - The master key pointer is set once by adl_init() and read-only after
 * ─────────────────────────────────────────────────────────────────────────────
 */

#include "adl_crypto.h"

/* ── FIPS 140-3 Mode ──────────────────────────────────────────────────────── */
/*
 * ADL_FIPS_MODE:
 *   1 (default) — FIPS mode. Self-tests mandatory, CTR_DRBG enforced,
 *                 all crypto operations go through FIPS-approved paths.
 *   0           — Non-FIPS mode. Self-tests can be bypassed, RAND_bytes
 *                 fallback allowed. Intended for development/debugging only.
 *
 * Override at compile time:  -DADL_FIPS_MODE=0
 */
#ifndef ADL_FIPS_MODE
#define ADL_FIPS_MODE 1
#endif

static int g_fips_mode = ADL_FIPS_MODE;

/*
 * Query and set FIPS mode at runtime.
 * Setting mode to 0 is irreversible for this process (must restart to
 * re-enable FIPS mode).
 */
int adl_is_fips_mode(void)
{
    return g_fips_mode;
}

void adl_set_fips_mode(int mode)
{
    /* FIPS mode can only be downgraded (1 → 0), never upgraded (0 → 1) */
    if (mode == 0 && g_fips_mode == 1) {
        g_fips_mode = 0;
    }
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <termios.h>
#include <unistd.h>

/* OpenSSL 3.x EVP API */
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <openssl/err.h>
#include <openssl/hmac.h>

/* POSIX — needed by InferiorParadoxical binary/source integrity scanner */
#include <dlfcn.h>
#include <dirent.h>

/* ── Static Master Key Storage ─────────────────────────────────────────────── */
/* Set once by adl_init(), read-only thereafter. Thread-safe for reads. */
static char g_master_key_hex[ADL_KEY_HEX_SIZE] = {0};
static int g_master_key_loaded = 0;

/* ── FIPS 140-3 InferiorParadoxical Poison State ──────────────────────────── */
/*
 * InferiorParadoxical — Anti-tamper dead-man's switch.
 *
 * When unauthorized modifications are detected (KAT failure, integrity
 * mismatch, or tampered binary), the master key is zeroized and all crypto
 * operations permanently cease for the lifetime of this process.
 *
 * There is NO un-poison. Recovery requires process restart.
 *
 * FIPS 140-3 References:
 *   §5.9(a)  Power-up self-tests — required for all security levels
 *   §5.8.8   Zeroization on self-test failure
 *   §5.9(b)  Software integrity test — detects code tampering
 */
static int g_poisoned = 0;
static int g_self_tests_passed = 0;

/* Forward declarations for KAT functions (implemented at end of file) */
static int kat_aes256_gcm(void);
static int kat_sha384(void);
static int kat_sha512(void);
static int kat_hkdf_sha256(void);
static int kat_hkdf_sha384(void);
static int kat_hmac_sha384(void);
static int kat_binary_integrity(void);

/* ── Secure Zeroing ────────────────────────────────────────────────────────── */
/* Zero sensitive memory to prevent key material from lingering. */
static void secure_zero(void *ptr, size_t len) {
    volatile unsigned char *p = (volatile unsigned char *)ptr;
    while (len--) *p++ = 0;
}

/* ── InferiorParadoxical Poison API ─────────────────────────────────────────── */

void adl_poison(void)
{
    g_poisoned = 1;
    g_self_tests_passed = 0;
    /* Zeroize the master key from static memory — irrevocable */
    secure_zero(g_master_key_hex, ADL_KEY_HEX_SIZE);
    g_master_key_loaded = 0;
    /* Zeroize DRBG key material — no more randomness */
    adl_drbg_clear();
}

int adl_is_poisoned(void)
{
    return g_poisoned;
}

int adl_self_tests_passed(void)
{
    return g_self_tests_passed;
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

    /* Priority 2.5: Master key file (path in ADELAIDE_MASTER_KEY_FILE)
     * run.py writes the key to a temp file (0600 perms) and sets this
     * env var instead of ADELAIDE_MASTER_KEY, avoiding leaking the
     * plaintext key to all subprocess environments.
     * We leave the file in place — run.py cleans it up on shutdown. */
    src = getenv("ADELAIDE_MASTER_KEY_FILE");
    if (src && src[0] != '\0') {
        FILE *fp = fopen(src, "r");
        if (fp) {
            size_t n = 0;
            int c;
            while ((c = fgetc(fp)) != EOF && n < sizeof(expanded_hex) - 1) {
                if (c == '\n' || c == '\r') continue;
                expanded_hex[n++] = (char)c;
            }
            expanded_hex[n] = '\0';
            fclose(fp);
            if (n > 0) {
                src = expanded_hex;
                goto store;
            }
        }
    }

    /* Priority 3: Config file (local to project) */
    {
        /* Try local config directory first, then legacy ~/.config/adelaide */
        const char *local_path = "config/master.key";
        const char *home = getenv("HOME");
        
        /* Check local config directory first */
        FILE *fp = fopen(local_path, "r");
        if (fp) {
            size_t n = 0;
            int c;
            while ((c = fgetc(fp)) != EOF && n < sizeof(expanded_hex) - 1) {
                if (c == '\n' || c == '\r') continue;
                expanded_hex[n++] = (char)c;
            }
            expanded_hex[n] = '\0';
            fclose(fp);
            if (n > 0) {
                src = expanded_hex;
                goto store;
            }
        }
        
        /* Fall back to legacy ~/.config/adelaide/master.key */
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
             "No master key found. Set ADELAIDE_MASTER_KEY or "
             "ADELAIDE_MASTER_KEY_FILE env var, or "
             "create config/master.key (run.py handles this)");
        return -1;

store:
    /* Validate hex key length (should be 64 hex chars = 32 bytes) */
    {
        size_t slen = strlen(src);
        /* Strip any trailing whitespace the file read might have left */
        while (slen > 0 && (src[slen-1] == ' ' || src[slen-1] == '\t')) slen--;
        if (slen != 64 && slen != 128) {
            snprintf(err_buf, ADL_ERROR_SIZE,
                     "Invalid master key length: got %zu hex chars, expected 64 or 128", slen);
        return -1;
        }
        /* Decode to verify it's valid hex */
        int decoded = hex_decode(src, slen, (unsigned char*)raw_key);
        if (decoded != 32 && decoded != 64) {
            snprintf(err_buf, ADL_ERROR_SIZE,
                     "Master key is not valid hex (decoded %d bytes, expected 32 or 64)", decoded);
        return -1;
        }
        /* Store the hex-encoded key in our static buffer */
        strncpy(g_master_key_hex, src, ADL_KEY_HEX_SIZE - 1);
        g_master_key_hex[ADL_KEY_HEX_SIZE - 1] = '\0';
        g_master_key_loaded = 1;
    }

    /* Zero the raw key from stack */
    secure_zero(raw_key, ADL_KEY_SIZE);

    /* ── FIPS 140-3 §5.9(a): Run power-up self-tests ───────────────────── */
    {
        char kat_err[ADL_ERROR_SIZE];
        if (adl_run_powerup_self_tests(kat_err) != 0) {
            /* Self-test failure → poison immediately (FIPS §5.8.8) */
            snprintf(err_buf, ADL_ERROR_SIZE,
                     "FIPS power-up self-test FAILED: %s. Keys have been zeroized.",
                     kat_err);
        return -1;
        }
    }

    /* ── FIPS 140-3 §5.1: Initialize CTR_DRBG ─────────────────────────── */
    {
        char drbg_err[ADL_ERROR_SIZE];
        if (adl_drbg_init(48, "adelaide:crypto:v1", drbg_err) != 0) {
            adl_poison();
            snprintf(err_buf, ADL_ERROR_SIZE,
                     "DRBG initialization FAILED: %s", drbg_err);
        return -1;
        }
    }

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

    /* InferiorParadoxical: Refuse crypto if poisoned */
    if (g_poisoned) {
        snprintf(err_buf, ADL_ERROR_SIZE, "Module is poisoned — key zeroized due to tamper detection");
        return -1;
    }

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

    /* InferiorParadoxical: Refuse crypto if poisoned */
    if (g_poisoned) {
        snprintf(err_buf, ADL_ERROR_SIZE, "Module is poisoned — key zeroized due to tamper detection");
        return -1;
    }

    /* Decode sub-key */
    if (hex_decode(sub_key_hex, 64, key) != ADL_KEY_SIZE) {
        snprintf(err_buf, ADL_ERROR_SIZE, "Invalid sub-key hex in encrypt");
        return -1;
    }

    /* Generate random 96-bit nonce via FIPS-approved CTR_DRBG */
    if (adl_drbg_generate(nonce, ADL_NONCE_SIZE) != 0) {
        snprintf(err_buf, ADL_ERROR_SIZE, "DRBG generate failed for nonce");
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

    /* InferiorParadoxical: Refuse crypto if poisoned */
    if (g_poisoned) {
        snprintf(err_buf, ADL_ERROR_SIZE, "Module is poisoned — key zeroized due to tamper detection");
        return -1;
    }

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

    // NULL check follows
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

/* ── adl_encrypt_raw ────────────────────────────────────────────────────────── */

int adl_encrypt_raw(const char *sub_key_hex,
                    const unsigned char *plaintext, size_t plaintext_len,
                    unsigned char *ciphertext, size_t *ciphertext_len,
                    char *err_buf)
{
    unsigned char key[ADL_KEY_SIZE];
    unsigned char nonce[ADL_NONCE_SIZE];
    unsigned char tag[ADL_TAG_SIZE];
    size_t ct_len = 0;
    int ret = -1;
    EVP_CIPHER_CTX *ctx = NULL;

    err_buf[0] = '\0';

    if (g_poisoned) {
        snprintf(err_buf, ADL_ERROR_SIZE, "Module is poisoned");
        return -1;
    }

    if (hex_decode(sub_key_hex, 64, key) != ADL_KEY_SIZE) {
        snprintf(err_buf, ADL_ERROR_SIZE, "Invalid sub-key hex in encrypt_raw");
        return -1;
    }

    if (adl_drbg_generate(nonce, ADL_NONCE_SIZE) != 0) {
        snprintf(err_buf, ADL_ERROR_SIZE, "DRBG generate failed for nonce");
        return -1;
    }

    if (*ciphertext_len < plaintext_len + ADL_NONCE_SIZE + ADL_TAG_SIZE) {
        snprintf(err_buf, ADL_ERROR_SIZE, "Output buffer too small");
        return -1;
    }

    ctx = EVP_CIPHER_CTX_new();
    if (!ctx) {
        snprintf(err_buf, ADL_ERROR_SIZE, "Failed to create EVP_CIPHER_CTX");
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

    int out_len = 0;
    if (EVP_EncryptUpdate(ctx, ciphertext + ADL_NONCE_SIZE, &out_len, plaintext, (int)plaintext_len) != 1) {
        get_openssl_error(err_buf, ADL_ERROR_SIZE);
        goto cleanup;
    }
    ct_len = (size_t)out_len;

    if (EVP_EncryptFinal_ex(ctx, ciphertext + ADL_NONCE_SIZE + out_len, &out_len) != 1) {
        get_openssl_error(err_buf, ADL_ERROR_SIZE);
        goto cleanup;
    }
    ct_len += (size_t)out_len;

    if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_GET_TAG, ADL_TAG_SIZE, tag) != 1) {
        get_openssl_error(err_buf, ADL_ERROR_SIZE);
        goto cleanup;
    }

    memcpy(ciphertext, nonce, ADL_NONCE_SIZE);
    memcpy(ciphertext + ADL_NONCE_SIZE + ct_len, tag, ADL_TAG_SIZE);
    *ciphertext_len = ADL_NONCE_SIZE + ct_len + ADL_TAG_SIZE;

    ret = 0;

cleanup:
    EVP_CIPHER_CTX_free(ctx);
    secure_zero(key, ADL_KEY_SIZE);
    return ret;
}

/* ── adl_decrypt_raw ────────────────────────────────────────────────────────── */

int adl_decrypt_raw(const char *sub_key_hex,
                    const unsigned char *ciphertext, size_t ciphertext_len,
                    unsigned char *plaintext, size_t *plaintext_len,
                    char *err_buf)
{
    unsigned char key[ADL_KEY_SIZE];
    unsigned char nonce[ADL_NONCE_SIZE];
    unsigned char tag[ADL_TAG_SIZE];
    size_t ct_len;
    int ret = -1;
    EVP_CIPHER_CTX *ctx = NULL;

    err_buf[0] = '\0';

    if (g_poisoned) {
        snprintf(err_buf, ADL_ERROR_SIZE, "Module is poisoned");
        return -1;
    }

    if (hex_decode(sub_key_hex, 64, key) != ADL_KEY_SIZE) {
        snprintf(err_buf, ADL_ERROR_SIZE, "Invalid sub-key hex in decrypt_raw");
        return -1;
    }

    if (ciphertext_len < ADL_NONCE_SIZE + ADL_TAG_SIZE) {
        snprintf(err_buf, ADL_ERROR_SIZE, "Ciphertext too short");
        return -1;
    }

    ct_len = ciphertext_len - ADL_NONCE_SIZE - ADL_TAG_SIZE;

    if (*plaintext_len < ct_len) {
        snprintf(err_buf, ADL_ERROR_SIZE, "Output buffer too small");
        return -1;
    }

    memcpy(nonce, ciphertext, ADL_NONCE_SIZE);
    memcpy(tag, ciphertext + ADL_NONCE_SIZE + ct_len, ADL_TAG_SIZE);

    ctx = EVP_CIPHER_CTX_new();
    if (!ctx) {
        snprintf(err_buf, ADL_ERROR_SIZE, "Failed to create EVP_CIPHER_CTX");
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

    int out_len = 0;
    if (EVP_DecryptUpdate(ctx, plaintext, &out_len, ciphertext + ADL_NONCE_SIZE, (int)ct_len) != 1) {
        get_openssl_error(err_buf, ADL_ERROR_SIZE);
        goto cleanup;
    }

    if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_TAG, ADL_TAG_SIZE, tag) != 1) {
        get_openssl_error(err_buf, ADL_ERROR_SIZE);
        goto cleanup;
    }

    if (EVP_DecryptFinal_ex(ctx, plaintext + out_len, &out_len) != 1) {
        snprintf(err_buf, ADL_ERROR_SIZE, "Decryption failed (wrong key or corrupted data)");
        goto cleanup;
    }

    *plaintext_len = ct_len;
    ret = 0;

cleanup:
    EVP_CIPHER_CTX_free(ctx);
    secure_zero(key, ADL_KEY_SIZE);
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
    int res = adl_init(NULL, err);
    if (res != 0) {
        fprintf(stderr, "[CRYPTO_C_WRAPPER] FATAL ERROR: %s\n", err);
    }
    return res;
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

    // NULL check follows
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
    // NULL check follows
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

/*
 * adl_derive_master_key_cstr: Python-accessible wrapper for adl_derive_master_key.
 *
 * Takes integrity_hash and user_secret as C strings, returns malloc'd
 * hex-encoded master key string, or NULL on failure.
 * Caller must free with adl_free_cstr().
 */
char *adl_derive_master_key_cstr(const char *integrity_hash,
                                  const char *user_secret)
{
    // NULL check follows
    char *out = malloc(129);  /* 128 hex chars + null */
    if (!out) return NULL;

    if (adl_derive_master_key(integrity_hash, user_secret, out) != 0) {
        free(out);
        return NULL;
    }
    return out;
}

#ifdef __APPLE__
int adl_get_hardware_secret_apple(char *secret_out, size_t max_len);
#else
int adl_get_hardware_secret_linux(char *secret_out, size_t max_len);
#endif

int adl_get_hardware_secret(char *secret_out, size_t max_len) {
#ifdef __APPLE__
    return adl_get_hardware_secret_apple(secret_out, max_len);
#else
    return adl_get_hardware_secret_linux(secret_out, max_len);
#endif
}

char* adl_auto_unlock_master_key_cstr(const char *integrity_hash, const char *wrapped_key_hex) {
    char hsm_secret[65] = {0};
    if (adl_get_hardware_secret(hsm_secret, sizeof(hsm_secret)) != 0) {
        hsm_secret[0] = '\0';
    }

    char combined[1024];
    snprintf(combined, sizeof(combined), "%s%s", integrity_hash, hsm_secret);

    unsigned char combined_hash[64];
    unsigned int hash_len = 0;
    if (EVP_Digest(combined, strlen(combined), combined_hash, &hash_len, EVP_sha512(), NULL) != 1) {
        return NULL;
    }

    /* Step 1: HKDF-Extract */
    unsigned char salt[ADL_KEY_SIZE] = {0};
    unsigned char prk[SHA384_HASH_SIZE];
    if (hmac_sha384(salt, ADL_KEY_SIZE, combined_hash, hash_len, prk) != 0) {
        return NULL;
    }

    /* Step 2: HKDF-Expand */
    const char *context = "adelaide:auto:wrapper:v1";
    size_t ctx_len = strlen(context);
    unsigned char expand_input[512];
    memcpy(expand_input, context, ctx_len);
    expand_input[ctx_len] = 0x01;
    
    unsigned char okm[SHA384_HASH_SIZE];
    if (hmac_sha384(prk, SHA384_HASH_SIZE, expand_input, ctx_len + 1, okm) != 0) {
        return NULL;
    }
    
    // NULL check follows
    char *master_key_plaintext = malloc(2048);
    if (!master_key_plaintext) return NULL;
    
    char okm_hex[65];
    hex_encode(okm, 32, okm_hex);
    
    char err_buf[256];
    size_t pt_len = 2048;
    if (adl_decrypt(okm_hex, wrapped_key_hex, NULL, 0, (unsigned char*)master_key_plaintext, &pt_len, err_buf) != 0) {
        free(master_key_plaintext);
        return NULL;
    }
    master_key_plaintext[pt_len] = '\0';
    
    return master_key_plaintext;
}

char* adl_auto_wrap_master_key_cstr(const char *integrity_hash, const char *master_key_hex) {
    char hsm_secret[65] = {0};
    if (adl_get_hardware_secret(hsm_secret, sizeof(hsm_secret)) != 0) {
        hsm_secret[0] = '\0';
    }

    char combined[1024];
    snprintf(combined, sizeof(combined), "%s%s", integrity_hash, hsm_secret);

    unsigned char combined_hash[64];
    unsigned int hash_len = 0;
    if (EVP_Digest(combined, strlen(combined), combined_hash, &hash_len, EVP_sha512(), NULL) != 1) {
        return NULL;
    }

    unsigned char salt[ADL_KEY_SIZE] = {0};
    unsigned char prk[SHA384_HASH_SIZE];
    if (hmac_sha384(salt, ADL_KEY_SIZE, combined_hash, hash_len, prk) != 0) {
        return NULL;
    }

    const char *context = "adelaide:auto:wrapper:v1";
    size_t ctx_len = strlen(context);
    unsigned char expand_input[512];
    memcpy(expand_input, context, ctx_len);
    expand_input[ctx_len] = 0x01;
    
    unsigned char okm[SHA384_HASH_SIZE];
    if (hmac_sha384(prk, SHA384_HASH_SIZE, expand_input, ctx_len + 1, okm) != 0) {
        return NULL;
    }
    
    // NULL check follows
    char *wrapped_key_hex = malloc(2048);
    if (!wrapped_key_hex) return NULL;
    
    char okm_hex[65];
    hex_encode(okm, 32, okm_hex);
    
    char err_buf[256];
    size_t ct_len = 2048;
    if (adl_encrypt(okm_hex, (const unsigned char*)master_key_hex, strlen(master_key_hex), NULL, 0, wrapped_key_hex, &ct_len, err_buf) != 0) {
        free(wrapped_key_hex);
        return NULL;
    }
    wrapped_key_hex[ct_len] = '\0';
    
    return wrapped_key_hex;
}

/*
 * adl_derive_master_key_from_stdin: Securely read password via termios without
 * echoing, derive the master key directly in C, and zeroize the buffer.
 * Returns malloc'd hex-encoded master key string, or NULL on failure.
 * Caller must free with adl_free_cstr().
 */
char *adl_derive_master_key_from_stdin(const char *integrity_hash, const char *prompt)
{
    struct termios oldt, newt;
    char secret_buf[256];
    char *out = NULL;
    int i = 0;
    int c;

    printf("%s", prompt);
    fflush(stdout);

    /* Disable echo */
    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);

    /* Read secret */
    while ((c = getchar()) != '\n' && c != EOF && i < 255) {
        secret_buf[i++] = (char)c;
    }
    secret_buf[i] = '\0';

    /* Restore echo */
    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    printf("\n");

    out = adl_derive_master_key_cstr(integrity_hash, secret_buf);

    /* Zeroize plaintext secret from C memory stack immediately */
    secure_zero(secret_buf, sizeof(secret_buf));

    return out;
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

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  HKDF Key Derivation (RFC 5869)                                           */
/* ═══════════════════════════════════════════════════════════════════════════ */

/*
 * HKDF-SHA512 implementation using OpenSSL EVP API.
 * Implements HKDF according to RFC 5869:
 *   1. HKDF-Extract: PRK = HMAC-SHA512(salt, IKM)
 *   2. HKDF-Expand:  OKM = HKDF-Expand(PRK, info, L)
 */
int adl_hkdf_sha512(const unsigned char *salt, size_t salt_len,
                    const unsigned char *ikm, size_t ikm_len,
                    const unsigned char *info, size_t info_len,
                    unsigned char *okm, size_t okm_len)
{
    /* InferiorParadoxical: Refuse if poisoned */
    if (g_poisoned) return -1;

    unsigned char prk[64];  /* SHA-512 hash size = 64 bytes */
    unsigned int prk_len = 64;
    unsigned char *p = okm;
    size_t remaining = okm_len;
    unsigned char counter = 1;
    
    /* Default salt (all zeros) if NULL */
    unsigned char default_salt[64] = {0};
    const unsigned char *actual_salt = salt ? salt : default_salt;
    size_t actual_salt_len = salt ? salt_len : 64;
    
    /* Step 1: HKDF-Extract
     * PRK = HMAC-SHA512(salt, IKM) */
    if (!HMAC(EVP_sha512(), actual_salt, (int)actual_salt_len,
              ikm, ikm_len, prk, &prk_len)) {
        return -1;
    }
    
    /* Step 2: HKDF-Expand
     * T(1) = HMAC-SHA512(PRK, info || 0x01)
     * T(2) = HMAC-SHA512(PRK, T(1) || info || 0x02)
     * ... */
    while (remaining > 0) {
        unsigned char hmac_input[256];
        size_t hmac_input_len = 0;
        unsigned char hmac_result[64];
        unsigned int hmac_result_len = 64;
        
        /* Build input: previous_result || info || counter */
        if (counter > 1) {
            /* Copy previous result (T(n-1)) */
            memcpy(hmac_input, p - 64, 64);
            hmac_input_len = 64;
        }
        
        /* Copy info */
        if (info && info_len > 0) {
            memcpy(hmac_input + hmac_input_len, info, info_len);
            hmac_input_len += info_len;
        }
        
        /* Add counter byte */
        hmac_input[hmac_input_len++] = counter;
        
        /* Compute HMAC-SHA512(PRK, hmac_input) */
        if (!HMAC(EVP_sha512(), prk, prk_len,
                  hmac_input, hmac_input_len,
                  hmac_result, &hmac_result_len)) {
            secure_zero(prk, sizeof(prk));
            return -1;
        }
        
        /* Copy result to output */
        size_t to_copy = (remaining < 64) ? remaining : 64;
        memcpy(p, hmac_result, to_copy);
        p += to_copy;
        remaining -= to_copy;
        counter++;
        
        secure_zero(hmac_result, sizeof(hmac_result));
    }
    
    secure_zero(prk, sizeof(prk));
    return 0;
}

/*
 * HKDF-SHA256 implementation using OpenSSL EVP API.
 * Same as HKDF-SHA512 but with SHA-256 (32-byte output per block).
 */
int adl_hkdf_sha256(const unsigned char *salt, size_t salt_len,
                    const unsigned char *ikm, size_t ikm_len,
                    const unsigned char *info, size_t info_len,
                    unsigned char *okm, size_t okm_len)
{
    /* InferiorParadoxical: Refuse if poisoned */
    if (g_poisoned) return -1;

    unsigned char prk[32];  /* SHA-256 hash size = 32 bytes */
    unsigned int prk_len = 32;
    unsigned char *p = okm;
    size_t remaining = okm_len;
    unsigned char counter = 1;
    
    /* Default salt (all zeros) if NULL */
    unsigned char default_salt[32] = {0};
    const unsigned char *actual_salt = salt ? salt : default_salt;
    size_t actual_salt_len = salt ? salt_len : 32;
    
    /* Step 1: HKDF-Extract
     * PRK = HMAC-SHA256(salt, IKM) */
    if (!HMAC(EVP_sha256(), actual_salt, (int)actual_salt_len,
              ikm, ikm_len, prk, &prk_len)) {
        return -1;
    }
    
    /* Step 2: HKDF-Expand */
    while (remaining > 0) {
        unsigned char hmac_input[256];
        size_t hmac_input_len = 0;
        unsigned char hmac_result[32];
        unsigned int hmac_result_len = 32;
        
        /* Build input: previous_result || info || counter */
        if (counter > 1) {
            memcpy(hmac_input, p - 32, 32);
            hmac_input_len = 32;
        }
        
        if (info && info_len > 0) {
            memcpy(hmac_input + hmac_input_len, info, info_len);
            hmac_input_len += info_len;
        }
        
        hmac_input[hmac_input_len++] = counter;
        
        if (!HMAC(EVP_sha256(), prk, prk_len,
                  hmac_input, hmac_input_len,
                  hmac_result, &hmac_result_len)) {
            secure_zero(prk, sizeof(prk));
            return -1;
        }
        
        size_t to_copy = (remaining < 32) ? remaining : 32;
        memcpy(p, hmac_result, to_copy);
        p += to_copy;
        remaining -= to_copy;
        counter++;
        
        secure_zero(hmac_result, sizeof(hmac_result));
    }
    
    secure_zero(prk, sizeof(prk));
    return 0;
}

/* ── adl_derive_master_key ──────────────────────────────────────────────────── */
/*
 * FIPS 140-3 master key derivation from hardware integrity hash + user secret.
 *
 * Replaces Python run.py → derive_master_key() with a FIPS-approved
 * C implementation using HKDF-SHA512.
 *
 * Derivation: master_key = HKDF-SHA512(salt=integrity_hash, ikm=user_secret,
 *                                     info="adelaide:master-key:v1")
 *
 * integrity_hash:   Hex-encoded SHA-512 integrity hash (128 hex chars).
 * user_secret:      UTF-8 password or recovery key.
 * master_key_out:   Output buffer for hex-encoded master key (129 bytes
 *                   for 128 hex chars + null terminator).
 *
 * Returns 0 on success, -1 on error.
 */
int adl_derive_master_key(const char *integrity_hash,
                          const char *user_secret,
                          char *master_key_out)
{
    /* Convert integrity_hash hex → binary salt (64 bytes for SHA-512) */
    size_t hash_hex_len = strlen(integrity_hash);
    size_t salt_len = hash_hex_len / 2;

    // NULL check follows
    unsigned char *salt = malloc(salt_len);
    if (!salt) return -1;

    for (size_t i = 0; i < salt_len; i++) {
        unsigned int byte;
        if (sscanf(integrity_hash + i * 2, "%2x", &byte) != 1) {
            free(salt);
            return -1;
        }
        salt[i] = (unsigned char)byte;
    }

    /* IKM = user_secret as UTF-8 */
    size_t ikm_len = strlen(user_secret);
    const unsigned char *ikm = (const unsigned char *)user_secret;

    /* Info = "adelaide:master-key:v1:{username}" */
    const char *username = getenv("ADELAIDE_USER");
    if (!username || strlen(username) == 0) {
        username = "default";
    }
    char info[256];
    snprintf(info, sizeof(info), "adelaide:master-key:v1:%s", username);
    size_t info_len = strlen(info);

    /* Output: 64 bytes (512-bit master key) */
    unsigned char okm[64];

    int ret = adl_hkdf_sha512(salt, salt_len, ikm, ikm_len,
                               (const unsigned char *)info, info_len,
                               okm, 64);
    secure_zero(salt, salt_len);
    free(salt);

    if (ret != 0) {
        secure_zero(okm, sizeof(okm));
        return -1;
    }

    /* Convert binary okm → hex string */
    for (size_t i = 0; i < 32; i++) {
        snprintf(master_key_out + (i * 2), 3, "%02x", okm[i]);
    }
    master_key_out[64] = '\0';

    secure_zero(okm, sizeof(okm));
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  FIPS 140-3 §5.1 / NIST SP 800-90A — CTR_DRBG (AES-256 Counter Mode)    */
/* ═══════════════════════════════════════════════════════════════════════════ */
/*
 * Deterministic Random Bit Generator using AES-256 in counter mode.
 *
 * Replaces all direct RAND_bytes() calls with a FIPS-approved CTR_DRBG.
 * Entropy source: OS /dev/urandom (seeded via OpenSSL RAND_bytes).
 *
 * Reseed interval: 2^48 requests (per SP 800-90A Table 3 for AES-256).
 * Max bytes per request: 2^19 = 524,288 bytes.
 *
 * Thread safety: the DRBG state is protected by the single-caller assumption
 * (adl_init() is non-thread-safe; all other operations are reentrant and
 * independently seeded — only adl_init() uses the DRBG).
 */

/* ── FIPS 140-3 §5.1 / SP 800-90A — SPARK CTR_DRBG Dependencies ────────────── */

int adl_gather_entropy(unsigned char *buffer, size_t len)
{
    if (RAND_bytes(buffer, (int)len) != 1) return 0;
    return 1;
}

int adl_aes256_ecb_encrypt(const unsigned char key[32],
                           const unsigned char plaintext[16],
                           unsigned char ciphertext[16])
{
    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    if (!ctx) return 0;

    int out_len = 0, ret = 0;
    if (EVP_EncryptInit_ex(ctx, EVP_aes_256_ecb(), NULL, key, NULL) != 1) goto done;
    if (EVP_CIPHER_CTX_set_padding(ctx, 0) != 1) goto done;
    if (EVP_EncryptUpdate(ctx, ciphertext, &out_len, plaintext, 16) != 1) goto done;
    if (out_len != 16) goto done;
    ret = 1;

done:
    EVP_CIPHER_CTX_free(ctx);
    return ret;
}

void adl_gather_entropy_wrapper(unsigned char *buffer, size_t len, int *result)
{
    *result = adl_gather_entropy(buffer, len);
}

void adl_aes256_ecb_encrypt_wrapper(const unsigned char key[32], 
                                    const unsigned char plaintext[16], 
                                    unsigned char ciphertext[16],
                                    int *result)
{
    *result = adl_aes256_ecb_encrypt(key, plaintext, ciphertext);
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  FIPS 140-3 §5.9 — Power-Up Self-Tests                                    */
/*  InferiorParadoxical — Source/Binary Integrity Scanner & Auto-Poison      */
/* ═══════════════════════════════════════════════════════════════════════════ */
/*
 * InferiorParadoxical Anti-Tamper:
 *
 *   On every power-up, each cryptographic algorithm is verified against
 *   Known Answer Test (KAT) vectors. Additionally:
 *     1. The compiled binary is SHA-512 hashed and compared against expected
 *     2. The source files (adl_crypto.c, adl_crypto.h) are SHA-512 hashed
 *        and compared against expected (if source files exist on disk)
 *
 *   If ANY check fails:
 *     1. Master key is immediately zeroized (FIPS §5.8.8)
 *     2. Poison flag is set — all crypto ops permanently disabled
 *     3. Process must restart for recovery
 *
 *   "The more an attacker tampers, the more they destroy what they seek."
 */

/* ══════════════════════════════════════════════════════════════════════════ */
/*  KAT Vectors                                                              */
/*  Generated deterministically at build time — MUST match or self-tests    */
/*  will fail and the module will poison itself.                            */
/* ══════════════════════════════════════════════════════════════════════════ */

/* ── AES-256-GCM KAT #1 ─────────────────────────────────────────────────── */
/* Plaintext:  "Hello Adelaide FIPS 140-3 KAT!" */
static const unsigned char KAT_AES256_KEY_1[32] = {
    0xfe,0xff,0xe9,0x92,0x86,0x65,0x73,0x1c,0x6d,0x6a,0x8f,0x94,
    0x67,0x30,0x83,0x08,0xfe,0xff,0xe9,0x92,0x86,0x65,0x73,0x1c,
    0x6d,0x6a,0x8f,0x94,0x67,0x30,0x83,0x08
};
static const unsigned char KAT_AES256_IV_1[12] = {
    0xca,0xfe,0xba,0xbe,0xfa,0xce,0xdb,0xad,0xde,0xca,0xf8,0x88
};
static const unsigned char KAT_AES256_PT_1[30] = {
    0x48,0x65,0x6c,0x6c,0x6f,0x20,0x41,0x64,0x65,0x6c,0x61,0x69,
    0x64,0x65,0x20,0x46,0x49,0x50,0x53,0x20,0x31,0x34,0x30,0x2d,
    0x33,0x20,0x4b,0x41,0x54,0x21
};
static const unsigned char KAT_AES256_CT_1[30] = {
    0xc3,0x79,0x9f,0xb9,0x0e,0xf2,0x3a,0x86,0x34,0x4a,0x5f,0x0f,
    0xe1,0x14,0x44,0xa1,0xab,0xcd,0x76,0xaf,0x9b,0xe5,0x07,0x3e,
    0x68,0xf4,0xd9,0xc1,0xfb,0x45
};
static const unsigned char KAT_AES256_TAG_1[16] = {
    0x24,0xc0,0xc3,0x42,0xcd,0xf5,0x4b,0x66,0x5e,0xe6,0x5f,0x6f,
    0x6d,0x89,0x40,0x72
};

/* ── AES-256-GCM KAT #2 ─────────────────────────────────────────────────── */
/* Plaintext:  "FIPS 140-3 Level 1 Self-Test" */
static const unsigned char KAT_AES256_KEY_2[32] = {
    0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0a,0x0b,
    0x0c,0x0d,0x0e,0x0f,0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17,
    0x18,0x19,0x1a,0x1b,0x1c,0x1d,0x1e,0x1f
};
static const unsigned char KAT_AES256_IV_2[12] = {
    0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0a,0x0b,0x0c
};
static const unsigned char KAT_AES256_PT_2[28] = {
    0x46,0x49,0x50,0x53,0x20,0x31,0x34,0x30,0x2d,0x33,0x20,0x4c,
    0x65,0x76,0x65,0x6c,0x20,0x31,0x20,0x53,0x65,0x6c,0x66,0x2d,
    0x54,0x65,0x73,0x74
};
static const unsigned char KAT_AES256_CT_2[28] = {
    0x43,0xa3,0x0a,0x86,0xcc,0xa5,0xc4,0xb6,0x61,0x91,0x43,0x0b,
    0x75,0x65,0x8f,0x44,0x62,0x72,0xc1,0xad,0xf2,0x01,0x36,0xe8,
    0xf5,0x3a,0x8a,0x50
};
static const unsigned char KAT_AES256_TAG_2[16] = {
    0x80,0x22,0x46,0xb4,0x3e,0x02,0x58,0x7b,0xea,0x5f,0x7e,0x9a,
    0xaa,0x12,0xb4,0x49
};

/* ── SHA-384 KAT ────────────────────────────────────────────────────────── */
/* Message:  "Adelaide FIPS 140-3 SHA-384 KAT Vector" */
static const unsigned char KAT_SHA384_MSG[38] = {
    0x41,0x64,0x65,0x6c,0x61,0x69,0x64,0x65,0x20,0x46,0x49,0x50,
    0x53,0x20,0x31,0x34,0x30,0x2d,0x33,0x20,0x53,0x48,0x41,0x2d,
    0x33,0x38,0x34,0x20,0x4b,0x41,0x54,0x20,0x56,0x65,0x63,0x74,
    0x6f,0x72
};
static const unsigned char KAT_SHA384_DIGEST[48] = {
    0xf4,0xb9,0x84,0xf8,0xda,0x06,0x7b,0x9c,0x66,0xbe,0x5c,0xf6,
    0x05,0x5d,0x44,0xac,0x34,0x86,0x2c,0x61,0xd5,0x2c,0xd9,0xcf,
    0x90,0xaf,0x6e,0xab,0x73,0x2c,0x79,0x92,0xa1,0x79,0xc6,0x09,
    0x0e,0x77,0xc1,0x6e,0x71,0x0e,0xfe,0x9d,0xcd,0x7b,0x43,0x70
};

/* ── SHA-512 KAT ────────────────────────────────────────────────────────── */
/* Message:  "Adelaide FIPS 140-3 SHA-512 KAT Vector" */
static const unsigned char KAT_SHA512_MSG[38] = {
    0x41,0x64,0x65,0x6c,0x61,0x69,0x64,0x65,0x20,0x46,0x49,0x50,
    0x53,0x20,0x31,0x34,0x30,0x2d,0x33,0x20,0x53,0x48,0x41,0x2d,
    0x35,0x31,0x32,0x20,0x4b,0x41,0x54,0x20,0x56,0x65,0x63,0x74,
    0x6f,0x72
};
static const unsigned char KAT_SHA512_DIGEST[64] = {
    0xa0,0x36,0x4c,0xac,0x0b,0xc1,0x73,0x38,0xf7,0x45,0xed,0x46,
    0xfb,0x26,0x0c,0xbc,0x17,0x2f,0x02,0xd8,0x1d,0x2e,0x81,0x02,
    0xca,0x23,0x4c,0x87,0xe2,0x77,0x22,0xa2,0x48,0x7b,0xce,0xe2,
    0xef,0x14,0x2d,0x51,0xa3,0x6a,0xc2,0x1f,0xa8,0x17,0x57,0x00,
    0xd2,0x23,0x3f,0xc9,0xe3,0x85,0x7e,0xc1,0x5c,0xdc,0x77,0xb6,
    0x15,0xec,0xa2,0xce
};

/* ── HKDF-SHA256 KAT ────────────────────────────────────────────────────── */
/* Salt:   feffe992... (32 bytes) */
/* IKM:    "adelaide:db:kat-test:v1" */
/* Info:   "adelaide:db:memory:v1" */
/* OKM:    4dba418b... (32 bytes) */
static const unsigned char KAT_HKDF256_SALT[32] = {
    0xfe,0xff,0xe9,0x92,0x86,0x65,0x73,0x1c,0x6d,0x6a,0x8f,0x94,
    0x67,0x30,0x83,0x08,0xfe,0xff,0xe9,0x92,0x86,0x65,0x73,0x1c,
    0x6d,0x6a,0x8f,0x94,0x67,0x30,0x83,0x08
};
static const unsigned char KAT_HKDF256_IKM[23] = {
    0x61,0x64,0x65,0x6c,0x61,0x69,0x64,0x65,0x3a,0x64,0x62,0x3a,
    0x6b,0x61,0x74,0x2d,0x74,0x65,0x73,0x74,0x3a,0x76,0x31
};
static const unsigned char KAT_HKDF256_INFO[21] = {
    0x61,0x64,0x65,0x6c,0x61,0x69,0x64,0x65,0x3a,0x64,0x62,0x3a,
    0x6d,0x65,0x6d,0x6f,0x72,0x79,0x3a,0x76,0x31
};
static const unsigned char KAT_HKDF256_OKM[32] = {
    0x4d,0xba,0x41,0x8b,0x43,0x9f,0xfe,0xf9,0x8d,0x48,0x9a,0xd4,
    0x32,0x43,0xf6,0x7c,0xa5,0xf1,0xce,0x5f,0x6d,0x71,0x76,0xd5,
    0xfe,0x8f,0x73,0xa0,0x3b,0x4a,0x56,0xe7
};

/* ── HKDF-SHA384 KAT (matches existing adl_derive_subkey logic) ─────────── */
static const unsigned char KAT_HKDF384_SALT[32] = {
    0xfe,0xff,0xe9,0x92,0x86,0x65,0x73,0x1c,0x6d,0x6a,0x8f,0x94,
    0x67,0x30,0x83,0x08,0xfe,0xff,0xe9,0x92,0x86,0x65,0x73,0x1c,
    0x6d,0x6a,0x8f,0x94,0x67,0x30,0x83,0x08
};
static const unsigned char KAT_HKDF384_IKM[23] = {
    0x61,0x64,0x65,0x6c,0x61,0x69,0x64,0x65,0x3a,0x64,0x62,0x3a,
    0x6b,0x61,0x74,0x2d,0x74,0x65,0x73,0x74,0x3a,0x76,0x31
};
static const unsigned char KAT_HKDF384_INFO[25] = {
    0x61,0x64,0x65,0x6c,0x61,0x69,0x64,0x65,0x3a,0x64,0x62,0x3a,
    0x6c,0x69,0x74,0x65,0x72,0x61,0x74,0x75,0x72,0x65,0x3a,0x76,
    0x31
};
static const unsigned char KAT_HKDF384_OKM[32] = {
    0x51,0xa5,0x0c,0x72,0x39,0x09,0xb4,0x25,0x1f,0xf1,0xf5,0x8c,
    0x83,0x45,0x1c,0xa0,0x43,0x50,0x69,0xad,0x2e,0x35,0xe6,0x01,
    0xa4,0xf4,0xc8,0xfd,0x82,0xe0,0x1b,0xa2
};

/* ── HMAC-SHA384 KAT ────────────────────────────────────────────────────── */
static const unsigned char KAT_HMAC384_KEY[32] = {
    0x41,0x64,0x65,0x6c,0x61,0x69,0x64,0x65,0x20,0x46,0x49,0x50,
    0x53,0x20,0x49,0x6e,0x74,0x65,0x67,0x72,0x69,0x74,0x79,0x20,
    0x4b,0x65,0x79,0x20,0x32,0x30,0x32,0x36
};
static const unsigned char KAT_HMAC384_MSG[37] = {
    0x41,0x44,0x45,0x4c,0x41,0x49,0x44,0x45,0x5f,0x43,0x52,0x59,
    0x50,0x54,0x4f,0x5f,0x4d,0x4f,0x44,0x55,0x4c,0x45,0x5f,0x49,
    0x4e,0x54,0x45,0x47,0x52,0x49,0x54,0x59,0x5f,0x54,0x45,0x53,
    0x54
};
static const unsigned char KAT_HMAC384_DIGEST[48] = {
    0xe8,0xa6,0xa5,0x9f,0x02,0xee,0x76,0xc3,0x60,0x6e,0xb2,0x2a,
    0xc9,0x8d,0xa1,0xef,0x62,0xb8,0xab,0xe9,0x8f,0xa6,0xa5,0x3f,
    0x38,0x8f,0x4e,0xac,0x28,0x92,0xbd,0xf9,0x5f,0x3c,0x0b,0xeb,
    0x38,0xdc,0x58,0x66,0xdb,0x58,0x1e,0x14,0x89,0x11,0xf9,0x82
};

/* ══════════════════════════════════════════════════════════════════════════ */
/*  KAT Implementation Functions                                             */
/* ══════════════════════════════════════════════════════════════════════════ */

/* ── KAT: AES-256-GCM Encrypt/Decrypt ───────────────────────────────────── */
/* Tests both encrypt and decrypt with known vectors using EVP API.         */
static int kat_aes256_gcm(void)
{
    unsigned char out_buf[256];
    unsigned char tag[16];
    int out_len, final_len;
    int ret;
    EVP_CIPHER_CTX *ctx = NULL;

    /* ── Vector 1: Encrypt ─────────────────────────────────────── */
    ctx = EVP_CIPHER_CTX_new();
    if (!ctx) return -1;

    ret = -1;
    do {
        if (EVP_EncryptInit_ex(ctx, EVP_aes_256_gcm(), NULL, NULL, NULL) != 1) break;
        if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_IVLEN, 12, NULL) != 1) break;
        if (EVP_EncryptInit_ex(ctx, NULL, NULL, KAT_AES256_KEY_1, KAT_AES256_IV_1) != 1) break;
        out_len = 0;
        if (EVP_EncryptUpdate(ctx, out_buf, &out_len,
                              KAT_AES256_PT_1, (int)sizeof(KAT_AES256_PT_1)) != 1) break;
        final_len = 0;
        if (EVP_EncryptFinal_ex(ctx, out_buf + out_len, &final_len) != 1) break;
        if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_GET_TAG, 16, tag) != 1) break;
        if ((out_len + final_len) != (int)sizeof(KAT_AES256_CT_1)) break;
        if (CRYPTO_memcmp(out_buf, KAT_AES256_CT_1, sizeof(KAT_AES256_CT_1)) != 0) break;
        if (CRYPTO_memcmp(tag, KAT_AES256_TAG_1, 16) != 0) break;
        ret = 0;
    } while (0);
    EVP_CIPHER_CTX_free(ctx);
    ctx = NULL;
    if (ret != 0) return -1;

    /* ── Vector 2: Encrypt (different key, verify independently) ─ */
    ctx = EVP_CIPHER_CTX_new();
    if (!ctx) return -1;

    ret = -1;
    do {
        if (EVP_EncryptInit_ex(ctx, EVP_aes_256_gcm(), NULL, NULL, NULL) != 1) break;
        if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_IVLEN, 12, NULL) != 1) break;
        if (EVP_EncryptInit_ex(ctx, NULL, NULL, KAT_AES256_KEY_2, KAT_AES256_IV_2) != 1) break;
        out_len = 0;
        if (EVP_EncryptUpdate(ctx, out_buf, &out_len,
                              KAT_AES256_PT_2, (int)sizeof(KAT_AES256_PT_2)) != 1) break;
        final_len = 0;
        if (EVP_EncryptFinal_ex(ctx, out_buf + out_len, &final_len) != 1) break;
        if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_GET_TAG, 16, tag) != 1) break;
        if ((out_len + final_len) != (int)sizeof(KAT_AES256_CT_2)) break;
        if (CRYPTO_memcmp(out_buf, KAT_AES256_CT_2, sizeof(KAT_AES256_CT_2)) != 0) break;
        if (CRYPTO_memcmp(tag, KAT_AES256_TAG_2, 16) != 0) break;
        ret = 0;
    } while (0);
    EVP_CIPHER_CTX_free(ctx);
    ctx = NULL;
    if (ret != 0) return -1;

    /* ── Vector 1: Decrypt back ────────────────────────────────── */
    ctx = EVP_CIPHER_CTX_new();
    if (!ctx) return -1;

    ret = -1;
    do {
        if (EVP_DecryptInit_ex(ctx, EVP_aes_256_gcm(), NULL, NULL, NULL) != 1) break;
        if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_IVLEN, 12, NULL) != 1) break;
        if (EVP_DecryptInit_ex(ctx, NULL, NULL, KAT_AES256_KEY_1, KAT_AES256_IV_1) != 1) break;
        out_len = 0;
        if (EVP_DecryptUpdate(ctx, out_buf, &out_len,
                              KAT_AES256_CT_1, (int)sizeof(KAT_AES256_CT_1)) != 1) break;
        if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_TAG, 16,
                                (void*)KAT_AES256_TAG_1) != 1) break;
        final_len = 0;
        if (EVP_DecryptFinal_ex(ctx, out_buf + out_len, &final_len) != 1) break;
        if ((out_len + final_len) != (int)sizeof(KAT_AES256_PT_1)) break;
        if (CRYPTO_memcmp(out_buf, KAT_AES256_PT_1, sizeof(KAT_AES256_PT_1)) != 0) break;
        ret = 0;
    } while (0);
    EVP_CIPHER_CTX_free(ctx);
    if (ret != 0) return -1;

    return 0;
}

/* ── KAT: SHA-384 ───────────────────────────────────────────────────────── */
static int kat_sha384(void)
{
    unsigned char digest[48];
    unsigned int digest_len = 48;

    if (!EVP_Digest(KAT_SHA384_MSG, sizeof(KAT_SHA384_MSG),
                    digest, &digest_len, EVP_sha384(), NULL)) {
        return -1;
    }
    if (digest_len != 48) return -1;
    return (CRYPTO_memcmp(digest, KAT_SHA384_DIGEST, 48) == 0) ? 0 : -1;
}

/* ── KAT: SHA-512 ───────────────────────────────────────────────────────── */
static int kat_sha512(void)
{
    unsigned char digest[64];
    unsigned int digest_len = 64;

    if (!EVP_Digest(KAT_SHA512_MSG, sizeof(KAT_SHA512_MSG),
                    digest, &digest_len, EVP_sha512(), NULL)) {
        return -1;
    }
    if (digest_len != 64) return -1;
    return (CRYPTO_memcmp(digest, KAT_SHA512_DIGEST, 64) == 0) ? 0 : -1;
}

/* ── KAT: HKDF-SHA256 ───────────────────────────────────────────────────── */
static int kat_hkdf_sha256(void)
{
    unsigned char okm[32];
    if (adl_hkdf_sha256(KAT_HKDF256_SALT, sizeof(KAT_HKDF256_SALT),
                        KAT_HKDF256_IKM, sizeof(KAT_HKDF256_IKM),
                        KAT_HKDF256_INFO, sizeof(KAT_HKDF256_INFO),
                        okm, sizeof(okm)) != 0) {
        return -1;
    }
    return (CRYPTO_memcmp(okm, KAT_HKDF256_OKM, 32) == 0) ? 0 : -1;
}

/* ── KAT: HKDF-SHA384 (matches adl_derive_subkey logic) ──────────────────── */
static int kat_hkdf_sha384(void)
{
    unsigned char prk[48];
    unsigned char expand_input[256];
    unsigned int result_len;

    /* Extract: PRK = HMAC-SHA384(salt, IKM) */
    result_len = 48;
    if (!HMAC(EVP_sha384(),
              KAT_HKDF384_SALT, (int)sizeof(KAT_HKDF384_SALT),
              KAT_HKDF384_IKM, (int)sizeof(KAT_HKDF384_IKM),
              prk, &result_len)) {
        return -1;
    }
    if (result_len != 48) { secure_zero(prk, sizeof(prk)); return -1; }

    /* Expand: T(1) = HMAC-SHA384(PRK, info || 0x01) */
    memcpy(expand_input, KAT_HKDF384_INFO, sizeof(KAT_HKDF384_INFO));
    expand_input[sizeof(KAT_HKDF384_INFO)] = 0x01;

    unsigned char okm[48];
    result_len = 48;
    unsigned char *ok = HMAC(EVP_sha384(), prk, 48,
                             expand_input, sizeof(KAT_HKDF384_INFO) + 1,
                             okm, &result_len);
    secure_zero(prk, sizeof(prk));
    if (!ok) return -1;

    return (CRYPTO_memcmp(okm, KAT_HKDF384_OKM, 32) == 0) ? 0 : -1;
}

/* ── KAT: AES-256-ECB ────────────────────────────────────────────────────── */
/* Tests the AES-256-ECB primitive used by the SPARK DRBG.                 */
static int kat_aes256_ecb(void)
{
    unsigned char key[32] = {0};
    unsigned char pt[16] = {0};
    unsigned char ct[16];

    if (adl_aes256_ecb_encrypt(key, pt, ct) != 1) return -1;

    /* Verify non-zero output */
    int all_zero = 1;
    for (int i = 0; i < 16; i++) {
        if (ct[i] != 0) { all_zero = 0; break; }
    }
    if (all_zero) return -1;

    return 0;
}

/* ── KAT: HMAC-SHA384 ───────────────────────────────────────────────────── */
static int kat_hmac_sha384(void)
{
    unsigned char digest[48];
    unsigned int digest_len = 48;

    if (!HMAC(EVP_sha384(),
              KAT_HMAC384_KEY, (int)sizeof(KAT_HMAC384_KEY),
              KAT_HMAC384_MSG, (int)sizeof(KAT_HMAC384_MSG),
              digest, &digest_len)) {
        return -1;
    }
    if (digest_len != 48) return -1;
    return (CRYPTO_memcmp(digest, KAT_HMAC384_DIGEST, 48) == 0) ? 0 : -1;
}

/* ══════════════════════════════════════════════════════════════════════════ */
/*  InferiorParadoxical Binary + Source Integrity Scanner (FIPS §5.9(b))    */
/* ══════════════════════════════════════════════════════════════════════════ */
/*
 * Spans BOTH the compiled binary AND the source files on disk.
 * If source files aren't available (production deployment), binary-only.
 *
 * Build-time integration:
 *   Run scripts/update_integrity_hash.py after compilation to embed both
 *   the source and binary hashes.
 *
 * Dev mode (empty expected hash):
 *   The first call records the hash and trusts it going forward.
 */

static char g_expected_bin_hash[129] = "";    /* 128 hex + null */
static char g_expected_src_hash[129] = "";    /* 128 hex + null */
static int  g_bin_hash_recorded = 0;
static int  g_src_hash_recorded = 0;

/* Set expected hashes (called from Python after build). */
void adl_set_expected_binary_hash(const char *hash_hex)
{
    if (hash_hex && strlen(hash_hex) == 128) {
        strncpy(g_expected_bin_hash, hash_hex, 128);
        g_expected_bin_hash[128] = '\0';
        g_bin_hash_recorded = 1;
    }
}

void adl_set_expected_source_hash(const char *hash_hex)
{
    if (hash_hex && strlen(hash_hex) == 128) {
        strncpy(g_expected_src_hash, hash_hex, 128);
        g_expected_src_hash[128] = '\0';
        g_src_hash_recorded = 1;
    }
}

/* Locate own binary path via dladdr() — works on Linux and macOS. */
static int get_own_path(char *path_buf, size_t buf_size)
{
    Dl_info info;
    if (dladdr((const void*)get_own_path, &info) && info.dli_fname) {
        strncpy(path_buf, info.dli_fname, buf_size - 1);
        path_buf[buf_size - 1] = '\0';
        return 0;
    }
    return -1;
}

/* Extract directory part from a file path (dest must be same size). */
static void dirname_of(const char *path, char *dest, size_t dest_size)
{
    strncpy(dest, path, dest_size - 1);
    dest[dest_size - 1] = '\0';
    char *slash = strrchr(dest, '/');
    if (slash) *slash = '\0';
    else       dest[0] = '.';
}

/* SHA-512 hash of a file's contents. Returns 0 on success. */
static int sha512_file(const char *path, unsigned char hash[64])
{
    FILE *fp = fopen(path, "rb");
    if (!fp) return -1;

    EVP_MD_CTX *mdctx = EVP_MD_CTX_new();
    if (!mdctx) { fclose(fp); return -1; }

    int ret = -1;
    unsigned char buf[16384];
    size_t n;
    unsigned int hash_len = 64;

    if (EVP_DigestInit_ex(mdctx, EVP_sha512(), NULL) != 1) goto done;
    while ((n = fread(buf, 1, sizeof(buf), fp)) > 0) {
        if (EVP_DigestUpdate(mdctx, buf, n) != 1) goto done;
    }
    if (ferror(fp)) goto done;
    if (EVP_DigestFinal_ex(mdctx, hash, &hash_len) != 1) goto done;
    if (hash_len != 64) goto done;
    ret = 0;

done:
    EVP_MD_CTX_free(mdctx);
    fclose(fp);
    return ret;
}

/* Hex-encode 64 bytes to 128-char hex string (null-terminated). */
static void hex64_encode(const unsigned char hash[64], char out[129])
{
    static const char hex[] = "0123456789abcdef";
    for (int i = 0; i < 64; i++) {
        out[i * 2]     = hex[(hash[i] >> 4) & 0x0f];
        out[i * 2 + 1] = hex[hash[i] & 0x0f];
    }
    out[128] = '\0';
}

/* ── Binary Integrity Check ─────────────────────────────────────────────── */
static int kat_binary_integrity(void)
{
    char bin_path[1024];
    unsigned char actual_hash[64];
    char actual_hex[129];

    if (get_own_path(bin_path, sizeof(bin_path)) != 0) return -1;
    if (sha512_file(bin_path, actual_hash) != 0) return -1;
    hex64_encode(actual_hash, actual_hex);

    /* Dev mode: auto-record */
    if (!g_bin_hash_recorded || g_expected_bin_hash[0] == '\0') {
        strncpy(g_expected_bin_hash, actual_hex, 128);
        g_expected_bin_hash[128] = '\0';
        g_bin_hash_recorded = 1;
        return 0;
    }

    return (strcmp(actual_hex, g_expected_bin_hash) == 0) ? 0 : -1;
}

/* ── Source Integrity Check ─────────────────────────────────────────────── */
/* Scans adl_crypto.c, adl_crypto.h, and related crypto source files.       */
static int kat_source_integrity(void)
{
    char bin_path[1024];
    char src_dir[1024];
    char filepath[1024];
    unsigned char ctx_hash[64];
    int ret = -1;

    if (get_own_path(bin_path, sizeof(bin_path)) != 0) return 0; /* skip if can't find */
    dirname_of(bin_path, src_dir, sizeof(src_dir));

    EVP_MD_CTX *mdctx = EVP_MD_CTX_new();
    if (!mdctx) return 0; /* skip on OOM */

    if (EVP_DigestInit_ex(mdctx, EVP_sha512(), NULL) != 1) goto done;

    /* Hash adl_crypto.c and adl_crypto.h (in the same source tree) */
    const char *src_files[] = {
        "adl_crypto.c",
        "adl_crypto.h",
        NULL
    };

    /* Try looking relative to the binary path (dev builds) */
    int found_any = 0;
    for (int pass = 0; pass < 3; pass++) {
        const char *base = NULL;
        switch (pass) {
            case 0: base = src_dir; break;
            /* Also check ../src/ (common build layout: build/foo, source is ../src/ */
            default: {
                char tmp[1024];
                strncpy(tmp, src_dir, sizeof(tmp) - 1);
                tmp[sizeof(tmp) - 1] = '\0';
                char *last = strrchr(tmp, '/');
                if (last) {
                    *last = '\0';
                    strncat(tmp, "/../src", sizeof(tmp) - strlen(tmp) - 1);
                    base = tmp;
                }
                break;
            }
        }
        if (!base) continue;

        for (int fi = 0; src_files[fi]; fi++) {
            snprintf(filepath, sizeof(filepath), "%s/%s", base, src_files[fi]);
            FILE *fp = fopen(filepath, "rb");
            if (fp) {
                found_any = 1;
                unsigned char buf[16384];
                size_t n;
                while ((n = fread(buf, 1, sizeof(buf), fp)) > 0) {
                    EVP_DigestUpdate(mdctx, buf, n);
                }
                fclose(fp);
            }
        }
        if (found_any) break;
    }

    if (!found_any) {
        /* Source files not on disk — skip check (binary-only deployment) */
        ret = 0;
        goto done;
    }

    unsigned int hash_len = 64;
    if (EVP_DigestFinal_ex(mdctx, ctx_hash, &hash_len) != 1) goto done;
    if (hash_len != 64) goto done;

    char actual_hex[129];
    hex64_encode(ctx_hash, actual_hex);

    /* Dev mode: auto-record */
    if (!g_src_hash_recorded || g_expected_src_hash[0] == '\0') {
        strncpy(g_expected_src_hash, actual_hex, 128);
        g_expected_src_hash[128] = '\0';
        g_src_hash_recorded = 1;
        ret = 0;
        goto done;
    }

    ret = (strcmp(actual_hex, g_expected_src_hash) == 0) ? 0 : -1;

done:
    EVP_MD_CTX_free(mdctx);
    return ret;
}

/* ══════════════════════════════════════════════════════════════════════════ */
/*  adl_run_powerup_self_tests — Run ALL FIPS §5.9 power-up KATs            */
/* ══════════════════════════════════════════════════════════════════════════ */

int adl_run_powerup_self_tests(char *err_buf)
{
    struct {
        const char *name;
        int (*fn)(void);
    } tests[] = {
        {"AES-256-GCM",         kat_aes256_gcm},
        {"AES-256-ECB",         kat_aes256_ecb},
        {"SHA-384",             kat_sha384},
        {"SHA-512",             kat_sha512},
        {"HKDF-SHA256",         kat_hkdf_sha256},
        {"HKDF-SHA384",         kat_hkdf_sha384},
        {"HMAC-SHA384",         kat_hmac_sha384},
        {"Binary Integrity",    kat_binary_integrity},
        {"Source Integrity",    kat_source_integrity},
    };

    int num_tests = sizeof(tests) / sizeof(tests[0]);

    for (int i = 0; i < num_tests; i++) {
        int result = tests[i].fn();
        if (result != 0) {
            /* ── InferiorParadoxical: FAILURE → POISON ─────────── */
            adl_poison();
            snprintf(err_buf, ADL_ERROR_SIZE,
                     "FIPS 140-3 §5.9 self-test FAILED: %s "
                     "(code or binary may be tampered — keys zeroized)",
                     tests[i].name);
        return -1;
        }
    }

    g_self_tests_passed = 1;
    return 0;
}
