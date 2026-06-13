/* [DO NOT REMOVE] Stderr suppression for llama.cpp model loading.
 *
 * llama.cpp prints hundreds of verbose lines to stderr during model load
 * (create_tensor, repack, load_tensors, print_info, etc.).  This clutter
 * makes the server output unreadable.  These functions redirect stderr to
 * /dev/null during loading, then restore it.
 *
 * suppress_dup: Save stderr fd and redirect to /dev/null.
 *   Returns the saved fd (pass to suppress_restore to undo).
 *
 * suppress_restore: Restore stderr from a saved fd.
 */
#include <unistd.h>
#include <fcntl.h>

/* Save current stderr (fd 2), then redirect stderr to /dev/null.
 * Returns the saved fd so the caller can pass it to suppress_restore. */
int suppress_dup(void) {
    int saved = dup(2);            /* save original stderr */
    int null_fd = open("/dev/null", O_WRONLY);
    if (null_fd >= 0) {
        dup2(null_fd, 2);          /* redirect stderr → /dev/null */
        close(null_fd);
    }
    return saved;
}

/* Restore stderr from a saved fd returned by suppress_dup. */
int suppress_restore(int saved_fd) {
    if (saved_fd >= 0) {
        dup2(saved_fd, 2);         /* restore original stderr */
        close(saved_fd);
    }
    return 0;
}
