/* KNOWN BAD C: Every function here SHOULD trigger SMT violations. */

#include <stdlib.h>

/* CHECK 1: Null pointer dereference — ptr used without NULL check. */
int null_deref(int *ptr) {
    return *ptr;
}

/* CHECK 2: Integer overflow — multiplication unchecked. */
int overflow_demo(int a, int b) {
    return a * b;
}

/* CHECK 3: Buffer overflow — memcpy without size check. */
void buffer_overflow(char *dst, char *src, int n) {
    memcpy(dst, src, n);
}

/* CHECK 4: Division by zero — divisor unchecked. */
int div_by_zero(int a, int b) {
    return a / b;
}
