/*
 * Jorvik Crash Isolation Layer — Signal Handler
 *
 * Catches SIGSEGV, SIGBUS, SIGFPE during llama.cpp inference
 * and longjmps back to a safe recovery point instead of
 * terminating the entire Ada server process.
 *
 * Named after the Viking settlement that endured.
 */
#include <signal.h>
#include <setjmp.h>
#include <stdio.h>
#include <string.h>

/* Recovery point — set by Ada before each C FFI call */
static jmp_buf jorvik_recovery;
static volatile sig_atomic_t jorvik_crash_signal = 0;
static volatile sig_atomic_t jorvik_installed = 0;

/* Which threads are in a protected region */
static volatile sig_atomic_t jorvik_guard_depth = 0;

/* Previous handlers for chain-calling */
static struct sigaction jorvik_prev_segv;
static struct sigaction jorvik_prev_bus;
static struct sigaction jorvik_prev_fpe;
static struct sigaction jorvik_prev_trap;

static void jorvik_handler(int sig) {
    jorvik_crash_signal = sig;
    if (jorvik_guard_depth > 0) {
        longjmp(jorvik_recovery, sig);
    }
    /* Not in a protected region — re-raise with default handler */
    struct sigaction *prev;
    switch (sig) {
        case SIGSEGV: prev = &jorvik_prev_segv; break;
        case SIGBUS:  prev = &jorvik_prev_bus;  break;
        case SIGFPE:  prev = &jorvik_prev_fpe;  break;
        case SIGTRAP: prev = &jorvik_prev_trap; break;
        default: return;
    }
    if (prev->sa_handler != SIG_DFL && prev->sa_handler != SIG_IGN) {
        prev->sa_handler(sig);
    } else {
        signal(sig, SIG_DFL);
        raise(sig);
    }
}

void jorvik_install_handlers(void) {
    if (jorvik_installed) return;

    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = jorvik_handler;
    sa.sa_flags = SA_RESTART;
    sigemptyset(&sa.sa_mask);

    sigaction(SIGSEGV, &sa, &jorvik_prev_segv);
    sigaction(SIGBUS,  &sa, &jorvik_prev_bus);
    sigaction(SIGFPE,  &sa, &jorvik_prev_fpe);
    sigaction(SIGTRAP, &sa, &jorvik_prev_trap);

    jorvik_installed = 1;
}

/* Enter a protected region. Returns 0 on normal entry, nonzero if recovering from crash. */
int jorvik_guard_enter(void) {
    jorvik_guard_depth++;
    return setjmp(jorvik_recovery);
}

/* Exit a protected region */
void jorvik_guard_exit(void) {
    if (jorvik_guard_depth > 0)
        jorvik_guard_depth--;
}

/* Query crash state */
int jorvik_crash_occurred(void) {
    return jorvik_crash_signal != 0;
}

int jorvik_get_crash_signal(void) {
    return (int)jorvik_crash_signal;
}

/* Clear crash state after recovery */
void jorvik_clear_crash(void) {
    jorvik_crash_signal = 0;
}
