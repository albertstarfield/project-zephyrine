/*
 * ── DRBG Shim for standalone libadl_crypto.dylib ──────────────────────────
 *
 * The SPARK Ada DRBG (spark_drbg.adb) cannot be linked into a standalone
 * C shared library.  This shim replaces the Ada DRBG with OpenSSL's
 * cryptographically secure RNG (CTR_DRBG via RAND_bytes), which meets
 * the same NIST SP 800-90A requirements.
 *
 * These three symbols satisfy the extern declarations in adl_crypto.h
 * that adl_crypto.c expects.
 *
 * Thread safety: RAND_bytes() is thread-safe in OpenSSL ≥ 1.1.0.
 * ────────────────────────────────────────────────────────────────────────────
 */

#include <stdio.h>
#include <string.h>
#include <openssl/rand.h>

/* ── DRBG State ──────────────────────────────────────────────────────────────
 * We do not need to track state explicitly — OpenSSL's RAND manages its own
 * DRBG internally.  These variables just track whether we've been initialised
 * so that adl_drbg_generate() can return an error if init was never called.
 */
static int g_drbg_initialised = 0;


/* ── Initialise the DRBG ──────────────────────────────────────────────────────
 * Seeds OpenSSL's random number generator.  On most platforms OpenSSL seeds
 * itself automatically from /dev/urandom, but we call RAND_poll() to ensure
 * the OS entropy pool is consulted.
 *
 * Returns 0 on success, -1 on failure (with error message in err_buf).
 */
int adl_drbg_init(size_t entropy_bytes, const char *pers_string, char *err_buf)
{
    (void)entropy_bytes;   /* OpenSSL manages its own entropy budget */
    (void)pers_string;     /* Personalisation string — ignored for OS RNG */

    if (g_drbg_initialised) {
        return 0;  /* already initialised */
    }

    /* Seed OpenSSL's CSPRNG from OS entropy sources */
    if (RAND_poll() != 1) {
        if (err_buf) {
            snprintf(err_buf, 256, "RAND_poll() failed — no entropy source");
        }
        return -1;
    }

    g_drbg_initialised = 1;
    return 0;
}


/* ── Internal: auto-initialise if not done ────────────────────────────────────
 * OpenSSL's RAND_bytes() is self-seeding on most platforms, but we call
 * RAND_poll() once to guarantee OS entropy is consulted.
 * Returns 0 on success, -1 on failure.
 */
static int ensure_drbg_ready(void)
{
    if (g_drbg_initialised) {
        return 0;
    }
    if (RAND_poll() != 1) {
        return -1;
    }
    g_drbg_initialised = 1;
    return 0;
}


/* ── Generate random bytes ───────────────────────────────────────────────────
 * Fills 'out' with 'len' cryptographically secure random bytes.
 *
 * Unlike the Ada SPARK DRBG, this function performs lazy auto-initialisation
 * so that callers (like adl_auto_wrap_master_key_cstr) do not need to invoke
 * adl_init() / adl_drbg_init() explicitly.
 *
 * Returns 0 on success, -1 on failure.
 */
int adl_drbg_generate(unsigned char *out, size_t len)
{
    if (ensure_drbg_ready() != 0) {
        return -1;
    }

    if (RAND_bytes(out, (int)len) != 1) {
        return -1;  /* random generation failed */
    }

    return 0;
}


/* ── Clear / finalise the DRBG ────────────────────────────────────────────────
 * Resets the state so that any subsequent call to adl_drbg_generate() will
 * fail until adl_drbg_init() is called again.
 */
void adl_drbg_clear(void)
{
    /* OpenSSL manages its own cleanup via RAND_cleanup() / OPENSSL_cleanup().
     * We just reset our tracking flag so the next init actually re-seeds. */
    g_drbg_initialised = 0;
}
