/*
 * [DO NOT REMOVE] Graceful Shutdown Signal Handler
 *
 * Catches SIGINT (Ctrl+C) and SIGTERM (kill/systemd) and sets a flag
 * instead of terminating immediately.  The Ada main loop checks this
 * flag every second and performs a clean shutdown:
 *   - Writes exit reason to run/adelaide_server.exit_reason
 *   - Deletes run/adelaide_server.pid
 *   - Exits with code 0 (clean shutdown)
 *
 * Without this, Ctrl+C kills the process immediately, leaving stale
 * PID/heartbeat files.  The watchdog then restarts the server because
 * it thinks the server crashed.
 */
#include <signal.h>
#include <unistd.h>

static volatile sig_atomic_t g_shutdown_requested = 0;

static void shutdown_handler(int sig) {
    g_shutdown_requested = 1;
}

/* Install SIGINT + SIGTERM handlers.  Call once at startup. */
void install_shutdown_handlers(void) {
    struct sigaction sa;
    sa.sa_handler = shutdown_handler;
    sa.sa_flags = 0;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT,  &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
}

/* Returns 1 if SIGINT or SIGTERM was received, 0 otherwise. */
int is_shutdown_requested(void) {
    return g_shutdown_requested;
}
