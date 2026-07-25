/* KNOWN GOOD C: Every function here should NOT trigger SMT violations. */

#include <stdlib.h>

/* CHECK 1: Null pointer dereference — GUARDED. */
int null_safe(int *ptr) {
    if (ptr != NULL) {
        return *ptr;
    }
    return 0;
}

/* CHECK 2: Integer overflow — GUARDED. */
int overflow_safe(int a, int b) {
    if (a > 0 && b > 0 && a <= 2147483647 / b) {
        return a * b;
    }
    return 0;
}

/* CHECK 3: Buffer overflow — GUARDED (no memcpy, manual copy). */
void buffer_safe(char *dst, char *src, int n) {
    for (int i = 0; i < n && dst[i] != '\0'; i++) {
        dst[i] = src[i];
    }
}

/* CHECK 4: Division by zero — GUARDED. */
int div_safe(int a, int b) {
    if (b != 0) {
        return a / b;
    }
    return 0;
}
