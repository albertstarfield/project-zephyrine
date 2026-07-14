#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

extern int adl_hkdf_sha256(const unsigned char *salt, size_t salt_len,
                           const unsigned char *ikm, size_t ikm_len,
                           const unsigned char *info, size_t info_len,
                           unsigned char *okm, size_t okm_len);

extern int adl_run_powerup_self_tests(char* err);

int main(int argc, char** argv) {
    char err[256];
    adl_run_powerup_self_tests(err);

#ifdef __AFL_HAVE_MANUAL_CONTROL
    __AFL_INIT();
#endif

    unsigned char buffer[4096];
    ssize_t len = read(0, buffer, sizeof(buffer));
    
    if (len > 32) {
        unsigned char okm[32];
        adl_hkdf_sha256(buffer, 16, buffer + 16, len - 16, (const unsigned char *)"info", 4, okm, 32);
    }
    
    return 0;
}
