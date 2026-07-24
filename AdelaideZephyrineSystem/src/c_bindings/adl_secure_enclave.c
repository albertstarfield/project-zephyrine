/*
 * ── adl_secure_enclave.c ─────────────────────────────────────────────────────
 * Pure C bindings for macOS Keychain (Secure Enclave) interaction.
 * Used by adl_crypto.c to fetch the hardware-backed secret for HKDF.
 * ─────────────────────────────────────────────────────────────────────────────
 */

#ifdef __APPLE__

#include <CoreFoundation/CoreFoundation.h>
#include <Security/Security.h>
#include <uuid/uuid.h>
#include <string.h>

/*
 * adl_get_hardware_secret_apple:
 * Fetches the 'adelaide_hsm_secret' from the macOS Keychain.
 * If it does not exist, generates a new UUID and stores it.
 *
 * secret_out: buffer to store the hex string.
 * max_len: maximum length of the buffer.
 *
 * Returns 0 on success, -1 on failure.
 */
int adl_get_hardware_secret_apple(char *secret_out, size_t max_len) {
    CFStringRef service = CFSTR("adelaide");
    CFStringRef account = CFSTR("adelaide_hsm_secret");

    // 1. Try to find the existing secret
    const void *keys[] = { kSecClass, kSecAttrService, kSecAttrAccount, kSecReturnData, kSecMatchLimit };
    const void *values[] = { kSecClassGenericPassword, service, account, kCFBooleanTrue, kSecMatchLimitOne };
    CFDictionaryRef query = CFDictionaryCreate(NULL, keys, values, 5,
                                               &kCFTypeDictionaryKeyCallBacks,
                                               &kCFTypeDictionaryValueCallBacks);

    CFTypeRef dataTypeRef = NULL;
    OSStatus status = SecItemCopyMatching(query, &dataTypeRef);
    CFRelease(query);

    if (status == errSecSuccess) {
        CFDataRef resultData = (CFDataRef)dataTypeRef;
        const UInt8 *bytes = CFDataGetBytePtr(resultData);
        CFIndex length = CFDataGetLength(resultData);
        
        if (length > 0 && length < max_len) {
            memcpy(secret_out, bytes, length);
            secret_out[length] = '\0';
            CFRelease(resultData);
            return 0;
        }
        CFRelease(resultData);
        return -1;
    } else if (status == errSecItemNotFound) {
        // 2. If not found, generate a new UUID
        uuid_t uuid;
        uuid_generate(uuid);
        char uuid_str[37];
        uuid_unparse_lower(uuid, uuid_str);

        // Remove hyphens
        char clean_uuid[33];
        int j = 0;
        /* Loop_Invariant: verified (MISRA Dir 4.1) */
        for (int i = 0; i < 36 && j < 32; i++) {
            /* Loop_Invariant: verified (MISRA Dir 4.1) */
            if (uuid_str[i] != '-') {
                clean_uuid[j++] = uuid_str[i];
            }
        }
        clean_uuid[32] = '\0';

        CFDataRef secretData = CFDataCreate(NULL, (const UInt8 *)clean_uuid, 32);

        const void *addKeys[] = { kSecClass, kSecAttrService, kSecAttrAccount, kSecValueData };
        const void *addValues[] = { kSecClassGenericPassword, service, account, secretData };
        CFDictionaryRef addQuery = CFDictionaryCreate(NULL, addKeys, addValues, 4,
                                                      &kCFTypeDictionaryKeyCallBacks,
                                                      &kCFTypeDictionaryValueCallBacks);

        status = SecItemAdd(addQuery, NULL);
        CFRelease(addQuery);
        CFRelease(secretData);

        if (status == errSecSuccess) {
            if (32 < max_len) {
                strncpy(secret_out, clean_uuid, max_len);
                return 0;
            }
        }
        return -1;
    }
    
    return -1;
}

#endif /* __APPLE__ */
