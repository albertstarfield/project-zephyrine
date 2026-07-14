#include <stdio.h>
#include <stdlib.h>

extern int adl_run_powerup_self_tests(char* err);

int main() {
    char err[256];
    int result = adl_run_powerup_self_tests(err);
    if (result != 0) {
        printf("FIPS Power-Up Self-Test Failed: %s\n", err);
        return 1;
    }
    printf("FIPS Power-Up Self-Test Passed.\n");
    return 0;
}
