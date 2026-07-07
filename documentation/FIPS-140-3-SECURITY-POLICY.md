# FIPS 140-3 Security Policy — Adelaide Crypto Subsystem

**Document Version:** 1.0  
**Date:** 2026-07-07  
**Module Name:** Adelaide Crypto Subsystem  
**FIPS 140-3 Security Level:** 1 (Target)  
**Status:** Implementation Complete — Pending Formal Validation  

---

## 1. Cryptographic Boundary

### 1.1 Module Definition

The Adelaide Crypto Subsystem is a **software-only, multi-chip standalone** cryptographic module. All cryptographic operations execute within the address space of the Adelaide Zephyrine System process on a general-purpose computer (macOS or Linux).

**Physical Boundary:** The module consists of compiled object code linked into the main Ada binary (`adelaide_zephyrine_system`). No dedicated hardware cryptographic accelerator is required. The module relies on the host CPU and operating system for general computing services (memory, process isolation, entropy via `/dev/urandom`).

**Logical Boundary:**  
| Component | Language | Files | Role |
|-----------|----------|-------|------|
| Crypto Primitives | C | `adl_crypto.c`, `adl_crypto.h` | AES-256-GCM, HKDF, CTR_DRBG, SHA-384/512, HMAC, KATs |
| Ada FFI Layer | Ada | `adelaide_crypto.ads`, `.adb` | Ada-to-C bridge, string conversion |
| Key Derivation | C → Ada | `adl_crypto.c`, `key_derivation.adb` | HKDF-SHA384 sub-key derivation |
| Key Storage | Ada | `master_key_store.ads`, `.adb` | Master key lifecycle (load, verify, zeroize) |
| API Key Mgmt | Ada | `api_key_manager.ads`, `.adb` | Crypto Officer role, key validation |
| Integrity | Ada | `system_integrity.ads`, `.adb` | Binary/source integrity scanner |
| Shutdown | Ada | `shutdown_manager.ads`, `.adb` | Secure shutdown coordination |

**Excluded from boundary:** SQLite database, Python `run.py` orchestrator, operating system, network stack.

### 1.2 Approved Cryptographic Algorithms

| Algorithm | Standard | Use | Key Size | Cert. Status |
|-----------|----------|-----|----------|--------------|
| AES-256-GCM | NIST SP 800-38D | Field encryption | 256-bit | Pending |
| AES-256-ECB | NIST SP 800-38A | CTR_DRBG primitive | 256-bit | Pending |
| CTR_DRBG | NIST SP 800-90A | Random bit generation | 256-bit (AES) | Pending |
| HMAC-SHA-384 | NIST FIPS 198-1 | HKDF-Extract, integrity | 384-bit | Pending |
| HMAC-SHA-512 | NIST FIPS 198-1 | HKDF-Extract, master key | 512-bit | Pending |
| HKDF-SHA-256 | NIST SP 800-56C | Sub-key derivation (AES) | 256-bit | Pending |
| HKDF-SHA-384 | NIST SP 800-56C | Sub-key derivation (Ada) | 384-bit | Pending |
| HKDF-SHA-512 | NIST SP 800-56C | Master key derivation | 512-bit | Pending |
| SHA-384 | NIST FIPS 180-4 | Hashing, KATs | 384-bit | Pending |
| SHA-512 | NIST FIPS 180-4 | Integrity scans, InferiorParadoxical | 512-bit | Pending |

### 1.3 Non-Approved but Allowed Functions

| Function | Purpose | Justification |
|----------|---------|---------------|
| `RAND_bytes()` | Initial entropy gathering for DRBG seed | Used only at init for seed material; not used for operational randomness |
| `getentropy()` / `getrandom()` | OS entropy source | Kernel-provided entropy — equivalent to `/dev/urandom` |

### 1.4 FIPS 140-3 Security Level per Area

| Area | Level | Notes |
|------|-------|-------|
| Cryptographic Module Specification | 1 | Software module, logical boundary |
| Cryptographic Module Ports and Interfaces | 1 | All I/O through API calls |
| Roles, Services, and Authentication | 1 | Crypto Officer + User role |
| Finite State Model | 1 | Power-on, Init, Crypto-Operational, Poisoned, Shutdown |
| Physical Security | N/A | Software-only module |
| Operational Environment | 1 | GP-OS (macOS/Linux) with process isolation |
| Cryptographic Key Management | 1 | Key generation, storage, zeroization |
| EMI/EMC | N/A | Software-only module |
| Self-Tests | 1 | Power-up KATs + continuous health tests |
| Design Assurance | 1 | Documented design and source code |
| Mitigation of Other Attacks | 1 | Timing attack mitigation via constant-time comparisons |

---

## 2. Module Interfaces

### 2.1 API Categories

**C API (crypto primitives):**
| Function | Category | Data Path | Control Path | Status Path |
|----------|----------|-----------|--------------|-------------|
| `adl_init()` | Control | — | Initialization | Self-test status |
| `adl_encrypt()` | Data I/O | Plaintext → Ciphertext | Encryption | Error code |
| `adl_decrypt()` | Data I/O | Ciphertext → Plaintext | Decryption | Error code |
| `adl_derive_subkey()` | Data I/O | HKDF key derivation | Key derivation | Error code |
| `adl_hkdf_sha256/384/512()` | Data I/O | IKM → OKM | Key derivation | Error code |
| `adl_drbg_init()` | Control | Entropy → DRBG state | Initialization | Error code |
| `adl_drbg_generate()` | Data I/O | DRBG state → Random bytes | Generation | Error code |
| `adl_run_powerup_self_tests()` | Control | — | Self-test execution | Pass/Fail |
| `adl_is_poisoned()` | Status | — | — | Poison state |
| `adl_poison()` | Control | Key zeroization | Anti-tamper | — |
| `adl_derive_master_key()` | Data I/O | Master key derivation | Key derivation | Error code |

**Ada API (application layer):**
| Function | Category | Description |
|----------|----------|-------------|
| `Initialize_Crypto()` | Control | Load master key, run self-tests, seed DRBG |
| `Is_FIPS_Ready()` | Status | Combined check: initialized + tests passed + not poisoned |
| `Is_Poisoned()` | Status | InferiorParadoxical anti-tamper status |
| `Self_Tests_Passed()` | Status | Power-up KAT result |
| `Encrypt_Field()` | Data I/O | AES-256-GCM field encryption |
| `Decrypt_Field()` | Data I/O | AES-256-GCM field decryption |
| `Derive_Subkey()` | Data I/O | HKDF per-DB sub-key derivation |
| `API_Key_Manager.Initialize()` | Control | Load API keys |
| `API_Key_Manager.Initialize_Crypto_Officer()` | Control | Load Crypto Officer key |
| `API_Key_Manager.Enable_Enforcement()` | Control | Enable key enforcement (CO only) |
| `API_Key_Manager.Validate_API_Key()` | Data I/O | Constant-time key validation |

### 2.2 Data Flow

```
┌───────────────────────────────────────────────────────────┐
│                    Adelaide Crypto Subsystem               │
│                                                           │
│  Master Key (env)                                          │
│       │                                                    │
│       ▼                                                    │
│  ┌──────────┐    ┌──────────────┐    ┌────────────────┐   │
│  │ adl_init  │───▶│ Self-Tests   │───▶│ InferiorParadox│   │
│  │ (key load)│    │ (KATs +      │    │  xical Poison  │   │
│  └──────────┘    │  integrity)   │    │  (auto-trip)   │   │
│                  └──────────────┘    └────────────────┘   │
│                         │               │                  │
│                         ▼               ▼                  │
│                  ┌──────────────┐    ┌────────────────┐   │
│                  │ CTR_DRBG     │    │ Zeroize Keys   │   │
│                  │ (AES-256)    │    │ (irrevocable)  │   │
│                  └──────────────┘    └────────────────┘   │
│                         │                                   │
│                         ▼                                   │
│  ┌──────────────┐    ┌──────────────┐    ┌────────────────┐ │
│  │ AES-256-GCM  │◀───│ HKDF-SHA{256,│───▶│ Master Key     │ │
│  │ Encrypt/Dec  │    │   384,512}   │    │ Derivation     │ │
│  └──────────────┘    └──────────────┘    └────────────────┘ │
│                                                             │
│  Ada FFI Wrapper (adelaide_crypto.adb)                      │
│  API Key Manager (api_key_manager.adb)                      │
│  Shutdown Manager (shutdown_manager.adb)                    │
└───────────────────────────────────────────────────────────┘
```

---

## 3. Roles, Services, and Authentication

### 3.1 Roles (FIPS 140-3 §5.3.1)

| Role | ID | Authority | Authentication |
|------|----|-----------|----------------|
| **Crypto Officer** | CO | Enable/disable enforcement, reload keys, manage crypto policy | `ADELAIDE_CRYPTO_OFFICER_KEY` env var |
| **Crypto User** | CU | Regular crypto operations (encrypt, decrypt, key validation) | API key from `ADELAIDE_API_KEYS` or key file |
| **Unauthenticated** | — | No crypto operations; returns errors | No credential |
| **Maintenance** | — | Process shutdown, key zeroization | No credential (requires OS-level access) |

### 3.2 Services per Role

| Service | CO | CU | Description |
|---------|----|----|-------------|
| Initialize Crypto | ✅ | — | Load master key, run KATs, seed DRBG |
| Initialize API Keys | ✅ | — | Load API keys from env/file |
| Enable Enforcement | ✅ | — | Activate API key validation |
| Disable Enforcement | ✅ | — | Deactivate API key validation |
| Reload API Keys | ✅ | — | Re-read key file |
| Encrypt Field | ✅ | ✅ | AES-256-GCM encryption |
| Decrypt Field | ✅ | ✅ | AES-256-GCM decryption |
| Derive Sub-Key | ✅ | ✅ | HKDF per-DB sub-key |
| Validate API Key | ✅ | ✅ | Constant-time key check |
| Status Queries | ✅ | ✅ | Poison, self-test, readiness |

### 3.3 Authentication Mechanisms

**Crypto Officer:**  
- Pre-shared key from `ADELAIDE_CRYPTO_OFFICER_KEY` environment variable
- Compared using constant-time comparison (FIPS §5.7)
- Only loaded at process start; never written to disk

**Crypto User:**  
- API key from `ADELAIDE_API_KEYS` env var (memory-only) or legacy key file
- Validated via `Validate_API_Key` using constant-time comparison
- Key file decrypted outside the boundary (Python `run.py`)

---

## 4. Key Management Lifecycle

### 4.1 Key Types

| Key Type | Algorithm | Size | Origin | Storage | Lifecycle |
|----------|-----------|------|--------|---------|-----------|
| **Master Key** | HKDF-SHA512-derived | 512 bits | key_gen_via_HKDF (C `adl_derive_master_key`) | Static RAM, zeroized on poison/exit | Load at boot → derive children → zeroize at exit |
| **Per-DB Sub-Key** | HKDF-SHA384 | 256 bits | Derivation from Master Key | Stack/local variables | Created per operation → discarded after use |
| **DRBG Key** | AES-256 | 256 bits | `adl_drbg_init` → OS entropy + derivation | Static RAM, cleared on poison | Init at boot → updated each generate → zeroized on clear |
| **DRBG V (counter)** | AES-128 block | 128 bits | `adl_drbg_init` | Static RAM, cleared on poison | Init at boot → incremented each generate → zeroized on clear |
| **Crypto Officer Key** | Pre-shared | Variable | `ADELAIDE_CRYPTO_OFFICER_KEY` env | Static RAM (Ada `Co_Key`) | Load at boot → used for auth → not zeroized (env var copy) |
| **API Keys** | Pre-shared | Variable | `ADELAIDE_API_KEYS` env or file | Static RAM (Ada `Key_Sets`) | Load at boot → validated at request → reloadable by CO |

### 4.2 Key Generation

- **Master key:** HKDF-SHA512(salt=integrity_hash, ikm=user_secret, info="adelaide:master-key:v1") — implemented in C (`adl_derive_master_key`)
- **Entropy source:** OS kernel entropy pool (`getentropy()` / OpenSSL `RAND_bytes()`)
- **DRBG seeding:** Initial entropy from OS (48 bytes) + nonce (16 bytes) → SHA-256 condensation → CTR_DRBG update
- **All operational randomness:** CTR_DRBG (AES-256 counter mode, SP 800-90A)

### 4.3 Key Zeroization

| Trigger | Action | Timing |
|---------|--------|--------|
| KAT failure | `adl_poison()` → zeroize master key + DRBG state + set poison flag | Immediate, during `adl_run_powerup_self_tests()` |
| Integrity mismatch | Same as KAT failure | Immediate, during binary/source scan |
| Continuous RNG health test failure | `adl_poison()` on stuck-at detection | Immediate, during `adl_drbg_generate()` |
| Process exit | OS memory deallocation | At process termination |
| Explicit `adl_drbg_clear()` | Zeroize DRBG Key + V + last_block | On demand (called by `adl_poison()`) |

### 4.4 Key Entry and Output

- **Plaintext key entry:** Master key is never entered directly; it is derived from integrity hash + user secret.  
- **Plaintext key output:** No keys are output in plaintext outside the boundary.  
- **Encrypted key output:** Master key may be stored as AES-256-GCM-wrapped blob in `system_state` table (InferiorParadoxical auto-decrypt).  

---

## 5. Self-Tests (FIPS 140-3 §5.9)

### 5.1 Power-Up Self-Tests

All power-up tests run automatically during `adl_init()` → `adl_run_powerup_self_tests()`. Any failure triggers `adl_poison()`.

| Test ID | Algorithm | Type | Vectors | Section in Code |
|---------|-----------|------|---------|-----------------|
| KAT-01 | AES-256-GCM | Known Answer Test | 2 vectors (encrypt) | `kat_aes256_gcm()` |
| KAT-02 | SHA-384 | Known Answer Test | 1 vector | `kat_sha384()` |
| KAT-03 | SHA-512 | Known Answer Test | 1 vector | `kat_sha512()` |
| KAT-04 | HKDF-SHA256 | Known Answer Test | 1 vector | `kat_hkdf_sha256()` |
| KAT-05 | HKDF-SHA384 | Known Answer Test | 1 vector | `kat_hkdf_sha384()` |
| KAT-06 | HMAC-SHA384 | Known Answer Test | 1 vector | `kat_hmac_sha384()` |
| KAT-07 | CTR_DRBG | Known Answer Test | AES-ECB block (known DRBG state) | `kat_drbg()` |
| INT-01 | Binary Integrity | SHA-512 of compiled `.so` | Auto-recorded or build-time | `kat_binary_integrity()` |
| INT-02 | Source Integrity | SHA-512 of `adl_crypto.c`/`.h` | Auto-recorded or build-time | `kat_source_integrity()` |

### 5.2 Conditional Tests

| Test ID | Algorithm | Frequency | Trigger | Section |
|---------|-----------|-----------|---------|---------|
| DRBG-HT | CTR_DRBG Continuous Health | Every `adl_drbg_generate()` call | Stuck-at failure → poison | FIPS 140-3 IG 9.8 |

### 5.3 Test Failure Response

On any self-test failure:
1. `adl_poison()` sets `g_poisoned = 1`
2. Master key zeroized: `secure_zero(g_master_key_hex, ...)`
3. DRBG state zeroized: `adl_drbg_clear()`
4. All subsequent crypto operations return errors
5. Ada `Is_Poisoned()` returns True
6. Recovery requires process restart

**There is no un-poison mechanism.** This is intentional (InferiorParadoxical design).

---

## 6. Operational Environment

### 6.1 Supported Platforms

| Platform | Minimum OS Version | Arch | Status |
|----------|-------------------|------|--------|
| macOS | 13.0 (Ventura) | arm64, x86_64 | Production |
| Linux | Kernel 5.x | x86_64, arm64 | Development |

### 6.2 Compiler Requirements

- Ada: GNAT (GCC) 13+
- C: Clang 14+ (macOS) / GCC 12+ (Linux)
- OpenSSL 3.x (`libcrypto` for AES/GCM/HASH/HMAC primitives)

### 6.3 Process Isolation

The module relies on the host OS for process isolation (standard GP-OS security assumptions). Running as a dedicated user/group is recommended but not enforced.

---

## 7. Physical and Operational Security

### 7.1 Physical Security (N/A)

The Adelaide Crypto Subsystem is a **software-only module**. Physical security is provided by the host platform (TPM/Secure Enclave on supported hardware for InferiorParadoxical identity storage).

### 7.2 Operational Security

| Control | Implementation |
|---------|---------------|
| Zeroization on tamper | KAT failure → poison → zeroize keys (FIPS §5.8.8) |
| Zeroization on process end | All keys are in process memory → destroyed on exit |
| Timing attack mitigation | Constant-time comparison in `api_key_manager.adb` |
| Side-channel resistance | AES-256-GCM is resistant to known side-channel attacks in software |

---

## 8. Cryptographic Algorithm Implementation Details

### 8.1 AES-256-GCM (SP 800-38D)

- Implementation: OpenSSL EVP API (`EVP_EncryptInit_ex` with `EVP_aes_256_gcm`)
- Key size: 256 bits
- Nonce size: 96 bits (generated via CTR_DRBG)
- Tag size: 128 bits
- Additional authenticated data (AAD): Currently zero-length (field-level encryption)

### 8.2 CTR_DRBG (SP 800-90A §10.2.1)

- Cipher: AES-256-ECB
- Derivation function: Enabled (df = 1 per §10.2.1.3)
- Key length: 256 bits
- Block length: 128 bits
- Seed length: 384 bits (Key + V)
- Reseed interval: 2^48 requests
- Max bytes per request: 524,288
- Entropy source: OS kernel (`RAND_bytes` at init only)
- Continuous health test: FIPS 140-3 IG 9.8

### 8.3 HKDF (RFC 5869)

| Variant | Hash | Salt Length | OKM Length | Use |
|---------|------|-------------|------------|-----|
| HKDF-SHA256 | SHA-256 | 32 bytes | Variable (sub-key for AES) | `adl_hkdf_sha256` |
| HKDF-SHA384 | SHA-384 | 32 bytes | Variable (sub-key for Ada) | `adl_hkdf_sha384` |
| HKDF-SHA512 | SHA-512 | 64 bytes | Variable (master key) | `adl_hkdf_sha512` |

### 8.4 InferiorParadoxical Anti-Tamper

See `/documentation/FIPS-140-3-GAP-ANALYSIS.md` for full description.

**Summary:**
- Binary integrity: SHA-512 of compiled `.so` file (via `dladdr` + file read)
- Source integrity: SHA-512 of `adl_crypto.c`/`.h` on disk
- Poison triggers: any KAT failure, integrity mismatch, RNG health test failure
- Poison action: zeroize master key + DRBG, set poison flag, reject all crypto

---

## 9. References

- [FIPS 140-3] NIST FIPS PUB 140-3, "Security Requirements for Cryptographic Modules"
- [SP 800-38A] NIST SP 800-38A, "Recommendation for Block Cipher Modes of Operation"
- [SP 800-38D] NIST SP 800-38D, "Recommendation for Block Cipher Modes: GCM"
- [SP 800-56C] NIST SP 800-56C, "Recommendation for Key Derivation through Extraction-then-Expansion"
- [SP 800-90A] NIST SP 800-90A Rev. 1, "Recommendation for Random Number Generation Using Deterministic Random Bit Generators"
- [FIPS 180-4] NIST FIPS PUB 180-4, "Secure Hash Standard (SHS)"
- [FIPS 198-1] NIST FIPS PUB 198-1, "The Keyed-Hash Message Authentication Code (HMAC)"
- [RFC 5869] IETF RFC 5869, "HMAC-based Extract-and-Expand Key Derivation Function (HKDF)"
- [IG 9.8] FIPS 140-3 Implementation Guidance 9.8, "Continuous RNG Health Tests"
- [Gap Analysis] `/documentation/FIPS-140-3-GAP-ANALYSIS.md` — Detailed gap analysis and remediation plan
- [Security Architecture] `/documentation/SECURITY_ARCHITECTURE.md` — Hardware-bound key derivation design

---

## 10. Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-07-07 | OpenAgent (InferiorParadoxical) | Initial FIPS 140-3 Security Policy — Phase 1-3 implementation complete |
