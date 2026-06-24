/* [DO NOT REMOVE] Force unbuffered stdout for Ada Text_IO.
 *
 * Problem: When run.py launches the Ada server via subprocess.Popen(),
 * stdout becomes a pipe (not a terminal).  C stdio defaults to full
 * buffering (8KB) on pipes, so Ada.Text_IO.Put_Line output sits in
 * the C buffer and is never flushed.  The server runs fine but is
 * completely invisible -- no banner, no init logs, no API responses.
 * The watchdog health checks all fail because it can't see the server.
 *
 * Fix: Call force_stdout_unbuffered() as the VERY FIRST thing in
 * the Ada main(), before any Put_Line.  This sets stdout to unbuffered
 * via setvbuf(), ensuring every Put_Line appears immediately.
 *
 * force_stderr_unbuffered() does the same for stderr, useful for
 * llama.cpp / ggml diagnostic output.
 *
 * The __attribute__((constructor)) ensures startup_marker runs BEFORE
 * any Ada elaboration code, so we can see if the binary even starts.
 *
 * [DO NOT REMOVE THIS PRINT VERBOSITY]
 * elab_trace() functions write directly to fd 2 (stderr) using raw
 * POSIX write(), bypassing ALL buffering (C stdio AND Ada.Text_IO).
 * This is the ONLY way to get diagnostic output during Ada elaboration,
 * because Ada.Text_IO may not be initialized yet, and C stdio may
 * be fully buffered on pipes.  Raw write(2,...) always works.
 */
#include <stdio.h>
#include <unistd.h>

/* [DO NOT REMOVE] Runs before Ada elaboration. Confirms binary startup. */
__attribute__((constructor))
static void startup_marker(void) {
    const char msg[] = "[BOOT] adelaide_server binary started (C constructor)\n";
    /* Write directly to fd 2 (stderr) bypassing C stdio buffering entirely. */
    write(2, msg, sizeof(msg) - 1);
}

void force_stdout_unbuffered(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
}

void force_stderr_unbuffered(void) {
    setvbuf(stderr, NULL, _IONBF, 0);
}

/* [DO NOT REMOVE THIS PRINT VERBOSITY]
 * Raw trace functions for Ada elaboration debugging.
 * These write directly to fd 2 (stderr) using POSIX write(), which bypasses
 * both C stdio buffering AND Ada.Text_IO buffering.
 *
 * Ada String FFI convention: GNAT passes (const char* ptr, ptrdiff_t len)
 * for unconstrained array parameters.  So these functions accept both
 * pointer and length, NOT null-terminated strings.
 */
/* [DO NOT REMOVE THIS PRINT VERBOSITY]
 * ABI FIX (2026-06-24): GNAT passes Ada unconstrained array (String) as
 * a FAT POINTER: (data_ptr, bounds_ptr) — two pointers, NOT (ptr, len).
 * The second value in the register is a BOUNDS DESCRIPTOR ADDRESS, not
 * a byte length.  Using it as a write() length dumped the binary's
 * entire __cstring section to stderr (10MB of symbols).
 * FIX: Use strlen() to measure the string at runtime.  GNAT string
 * literals are null-terminated in practice, so this is safe.
 * SAFETY: Guard against strings > 10000 bytes to prevent regressions.
 */
#include <string.h>
void elab_trace_c(const char* label) {
    const char prefix[] = "[ElabTrace-C] ";
    size_t plen = sizeof(prefix) - 1;
    write(2, prefix, plen);
    if (label != NULL) {
        size_t len = strlen(label);
        if (len > 10000) len = 10000;
        write(2, label, len);
    }
    write(2, "\n", 1);
}

void elab_trace_c2(const char* label1, const char* label2) {
    const char prefix[] = "[ElabTrace-C] ";
    size_t plen = sizeof(prefix) - 1;
    write(2, prefix, plen);
    if (label1 != NULL) {
        size_t len = strlen(label1);
        if (len > 10000) len = 10000;
        write(2, label1, len);
    }
    write(2, " ", 1);
    if (label2 != NULL) {
        size_t len = strlen(label2);
        if (len > 10000) len = 10000;
        write(2, label2, len);
    }
    write(2, "\n", 1);
}
