#include <stdio.h>

#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <mach/thread_policy.h>
#include <pthread.h>

void set_darwin_realtime() {
    thread_time_constraint_policy_data_t policy;
    mach_timebase_info_data_t timebase_info;
    mach_timebase_info(&timebase_info);

    // Convert nanoseconds to Mach absolute time units
    // absolute_time = ns * (numer / denom)
    // ns = absolute_time * (denom / numer)
    double ns_to_absolute = (double)timebase_info.denom / timebase_info.numer;

    // Period: 20ms
    policy.period = (uint32_t)(20000000.0 * ns_to_absolute);
    // Computation: 10ms
    policy.computation = (uint32_t)(10000000.0 * ns_to_absolute);
    // Constraint: 15ms
    policy.constraint = (uint32_t)(15000000.0 * ns_to_absolute);
    policy.preemptible = 1;

    kern_return_t kr = thread_policy_set(
        mach_thread_self(),
        THREAD_TIME_CONSTRAINT_POLICY,
        (thread_policy_t)&policy,
        THREAD_TIME_CONSTRAINT_POLICY_COUNT
    );
    if (kr != KERN_SUCCESS) {
        fprintf(stderr, "[!] scheduling.c: Failed to set THREAD_TIME_CONSTRAINT_POLICY: %d\n", kr);
    } else {
        fprintf(stderr, "[+] scheduling.c: THREAD_TIME_CONSTRAINT_POLICY set successfully on Apple Silicon\n");
    }
}
#else
void set_darwin_realtime() {
    // No-op on non-macOS platforms
}
#endif
