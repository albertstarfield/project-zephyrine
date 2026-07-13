/*
 * [DO NOT REMOVE] Graceful Shutdown Signal Handler
 *
 * Catches SIGINT, SIGTERM, and SIGQUIT and sets a simple flag.
 * The Ada main loop checks this flag every cycle and performs a
 * clean shutdown when set.
 *
 * SIGQUIT (Ctrl+\) is the primary shutdown signal used by run.py:
 *   run.py catches SIGQUIT → writes run/.shutdown_requested flag →
 *   Ada polls for the flag file → deletes it → exits gracefully.
 *
 * SIGINT/SIGTERM are fallbacks for standalone Ada operation (no run.py).
 *
 * On clean shutdown:
 *   - Writes exit reason to run/adelaide_server.exit_reason
 *   - Deletes run/adelaide_server.pid and run/adelaide_server.heartbeat
 *   - Exits with code 0 (clean shutdown)
 *
 * Without this, signals kill the process immediately, leaving stale
 * PID/heartbeat files.  The watchdog then restarts the server because
 * it thinks the server crashed.
 */
#include <signal.h>
#include <unistd.h>
#include <stdio.h>

static volatile sig_atomic_t g_shutdown_requested = 0;
static volatile sig_atomic_t g_last_signal = 0;

static void shutdown_handler(int sig) {
    g_last_signal = sig;
    g_shutdown_requested = 1;
}

/* Install SIGINT, SIGTERM, and SIGQUIT handlers.  Call once at startup. */
void install_shutdown_handlers(void) {
    struct sigaction sa;
    sa.sa_handler = shutdown_handler;
    sa.sa_flags = 0;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT,  &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
    sigaction(SIGQUIT, &sa, NULL);
}

/* Returns 1 if a shutdown signal was received, 0 otherwise. */
int is_shutdown_requested(void) {
    return g_shutdown_requested;
}

/* Returns which signal number triggered the shutdown (SIGINT=2, SIGTERM=15, SIGQUIT=3). */
int last_signal_received(void) {
    return g_last_signal;
}
