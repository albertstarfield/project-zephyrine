/*
 * ── adl_tpm2.c ───────────────────────────────────────────────────────────────
 * C bindings for Linux TPM2 interaction.
 * Used by adl_crypto.c to fetch the hardware-backed secret for HKDF.
 * ─────────────────────────────────────────────────────────────────────────────
 */

#ifndef __APPLE__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <uuid/uuid.h>

/*
 * adl_get_hardware_secret_linux:
 * Fetches the 'adelaide_hsm_secret' from TPM2 NVRAM (index 0x1500015).
 * If it does not exist, generates a new UUID and stores it via tpm2_tools.
 *
 * secret_out: buffer to store the hex string.
 * max_len: maximum length of the buffer.
 *
 * Returns 0 on success, -1 on failure.
 */
int adl_get_hardware_secret_linux(char *secret_out, size_t max_len) {
    char cmd_buf[512];
    FILE *fp;
    const char *nv_index = "0x1500015";

    // 1. Check if tpm2_nvread exists
    if (system("which tpm2_nvread > /dev/null 2>&1") != 0) {
        return -1;
    }

    // 2. Try to read existing secret
    snprintf(cmd_buf, sizeof(cmd_buf), "tpm2_nvread %s 2>/dev/null", nv_index);
    fp = popen(cmd_buf, "r");
    if (fp) {
        if (fgets(secret_out, max_len, fp) != NULL) {
            pclose(fp);
            // Trim newline
            size_t len = strlen(secret_out);
            if (len > 0 && secret_out[len-1] == '\n') {
                secret_out[len-1] = '\0';
            }
            if (strlen(secret_out) > 0) {
                return 0; // Found it
            }
        } else {
            pclose(fp);
        }
    }

    // 3. Generate a new UUID
    uuid_t uuid;
    uuid_generate(uuid);
    char uuid_str[37];
    uuid_unparse_lower(uuid, uuid_str);
    
    // Remove hyphens
    char clean_uuid[33];
    int j = 0;
    for (int i = 0; i < 36 && j < 32; i++) {
        if (uuid_str[i] != '-') {
            clean_uuid[j++] = uuid_str[i];
        }
    }
    clean_uuid[32] = '\0';

    // 4. Define NVRAM space
    snprintf(cmd_buf, sizeof(cmd_buf), "tpm2_nvdefine -C o -s 32 %s > /dev/null 2>&1", nv_index);
    if (system(cmd_buf) != 0) {
        return -1;
    }

    // 5. Write to NVRAM
    // We create a temporary file to pipe into tpm2_nvwrite
    char tmp_template[] = "/tmp/adl_tpm_XXXXXX";
    int fd = mkstemp(tmp_template);
    if (fd == -1) {
        return -1;
    }
    if (write(fd, clean_uuid, 32) != 32) {
        close(fd);
        unlink(tmp_template);
        return -1;
    }
    close(fd);

    snprintf(cmd_buf, sizeof(cmd_buf), "tpm2_nvwrite -C o -i %s %s > /dev/null 2>&1", tmp_template, nv_index);
    int ret = system(cmd_buf);
    unlink(tmp_template);

    if (ret == 0) {
        strncpy(secret_out, clean_uuid, max_len);
        return 0;
    }

    return -1;
}

#endif /* !__APPLE__ */
