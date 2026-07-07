# FIPS 140-3 Gap Analysis — Adelaide Crypto Module

**Date:** 2026-07-07  
**Author:** OpenAgent (Automated Analysis)  
**Standard:** NIST FIPS PUB 140-3 (March 22, 2019) per ISO/IEC 19790:2012  
**Scope:** `src/adl_crypto.c`, `src/adl_crypto.h`, `src/adelaide_crypto.ads/.adb`, `src/master_key_store.ads/.adb`, `src/key_derivation.ads/.adb`, `src/system_integrity.ads/.adb`, `src/integrity_utils.ads/.adb`, `src/api_key_manager.ads/.adb`, `src/shutdown_manager.ads/.adb`  
**Target Level:** This analysis applies across all four FIPS 140-3 security levels (Level 1–4)

---

## 1. Background

NIST FIPS 140-3 (*Security Requirements for Cryptographic Modules*) is the U.S. federal standard that defines security requirements for cryptographic modules protecting sensitive but unclassified information. It supersedes FIPS 140-2 and aligns with ISO/IEC 19790:2012.

The Adelaide crypto subsystem provides application-layer AES-256-GCM field encryption for SQLite databases, with HKDF key derivation bound to system integrity state. This document maps the current implementation against each of the 11 requirement areas in FIPS 140-3 and identifies gaps.

### Security Levels

| Level | Description |
|-------|-------------|
| **Level 1** | Production-grade algorithms on a general-purpose computer. Minimum requirement. |
| **Level 2** | Adds role-based authentication, tamper-evidence (opaque coating/seals), and audit. |
| **Level 3** | Adds identity-based authentication, tamper-response (zeroization), physical separation of ports, and FIPS-approved DRBG. |
| **Level 4** | Adds environmental fault protection (temperature/voltage), complete envelope of protection. |

---

## 2. Summary of Findings

### ✅ What We Do Well

| Requirement Area | Strength | Details |
|---|---|---|
| **Key Zeroization** | **Strong** | C `secure_zero()` uses volatile-cast pointer to prevent compiler elision; Ada `Clear_Key` uses `Volatile` pragma in SPARK-verified package |
| **Key Storage** | **Good** | Master key is memory-only, never written to disk; SPARK-verified `Master_Key_Store` with `Global => null` contracts |
| **Key Derivation** | **Good** | HKDF-SHA384/512/256 via OpenSSL EVP API (RFC 5869 compliant, NIST SP 800-56C) |
| **Approved Algorithms** | **Good** | AES-256-GCM (FIPS 197, SP 800-38D), SHA-512/384/256 (FIPS 180-4), HKDF (SP 800-56C) |
| **SPARK Formal Verification** | **Good** | `Master_Key_Store` and `Integrity_Utils` are SPARK-proved with formal contracts |
| **Symmetric-Only Design** | **Good** | No asymmetric keys → no vulnerability to Shor's algorithm (post-quantum native) |

### ❌ Critical Gaps (Block All Levels)

| # | Gap | FIPS Section | Impact |
|---|-----|-------------|--------|
| **G1** | No power-up Known Answer Tests (KAT) for AES-256-GCM, SHA-384/512, HKDF | **§5.9(a)** | Module cannot self-verify algorithm integrity at startup |
| **G2** | No FIPS-approved DRBG — uses `RAND_bytes()` without SP 800-90A compliance | **§5.8.1, §5.9(c)** | Random number generation is not approved |
| **G3** | No software/firmware integrity test of module binary at load time | **§5.9(b)** | Module cannot prove it hasn't been tampered with |

### ❌ High-Priority Gaps (Level 2+)

| # | Gap | FIPS Section |
|---|-----|-------------|
| **G4** | No Crypto Officer role — all operations use a single implicit role | **§5.3.1** |
| **G5** | API key enforcement disabled by default (`ADELAIDE_API_KEY_ENFORCE=0`) | **§5.3.2** |
| **G6** | Master key generated outside cryptographic boundary (Python `run.py`) | **§5.8.2** |
| **G7** | No continuous RNG health test around `RAND_bytes()` | **§5.9(c)** |
| **G8** | No FIPS Security Policy document | **§5.10.5** |
| **G9** | No cryptographic boundary definition | **§5.1.1** |
| **G10** | No "FIPS Mode" / approved mode indicator | **§5.1.4** |

### ❌ Medium-Priority Gaps (Level 3+)

| # | Gap | FIPS Section |
|---|-----|-------------|
| **G11** | `Clear_Key` not automatically called on all error/shutdown paths | **§5.8.8** |
| **G12** | No constant-time comparison for auth tags or API keys (`memcmp`, `strcmp`) | **§5.7** |
| **G13** | No timing attack mitigations documented or implemented | **§5.7** |
| **G14** | No trusted channel for key entry/output | **§5.2.2** |
| **G15** | No audit logging of security-relevant events | **§5.3.3** |

---

## 3. Detailed Gap Analysis by FIPS 140-3 Section

### 3.1 §5.1 — Cryptographic Module Specification

| Requirement | What FIPS 140-3 Mandates | Current State | Gap |
|-------------|--------------------------|---------------|-----|
| 5.1.1 Boundary | Define module boundary — which components are in-scope | No boundary diagram or spec exists | **G9** |
| 5.1.2 Approved Algorithms | All cryptographic functions must be FIPS-approved | AES-256-GCM ✓, SHA-512/384/256 ✓, HKDF ✓ | ✅ Good |
| 5.1.3 Non-Approved Algorithms | Non-approved functions must not interfere with security | LSH hash and CRC-32 are for non-crypto purposes | ✅ Acceptable |
| 5.1.4 Approved Mode | Module must operate in an approved mode where only approved algos are used | No FIPS/non-FIPS mode switch | **G10** |

#### Current Module Boundary

```
┌─────────────────────────────────────────────────┐
│                adelaide_server                    │
│  ┌───────────────────────────────────────────┐   │
│  │         Adelaide Crypto Subsystem          │   │
│  │  ┌──────────┐  ┌──────────────┐           │   │
│  │  │ HKDF C   │  │ AES-256-GCM  │           │   │
│  │  │ (OpenSSL)│  │ (OpenSSL)    │           │   │
│  │  └──────────┘  └──────────────┘           │   │
│  │  ┌─────────────────┐  ┌──────────────┐    │   │
│  │  │ Master_Key_Store │  │ Integrity    │    │   │
│  │  │ (SPARK Ada)      │  │ Hashing     │    │   │
│  │  └─────────────────┘  └──────────────┘    │   │
│  └───────────────────────────────────────────┘   │
│  ┌───────────────────────────────────────────┐   │
│  │         Non-FIPS Components                │   │
│  │  LSH hash, CRC-32 (integrity_utils.ads)  │   │
│  └───────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

### 3.2 §5.2 — Cryptographic Module Interfaces

| Requirement | Mandate | Current State | Gap |
|-------------|---------|---------------|-----|
| Data Input | Logical interface for data to be processed | Ada `Encrypt_Field`, `Decrypt_Field` functions | OK |
| Data Output | Logical interface for processed data | Return values from crypto functions | OK |
| Control Input | Commands to the module | `Initialize_Crypto`, `Derive_Subkey` | OK |
| Status Output | Module status indicators | `Is_Crypto_Ready` exists but no error state reporting | **Partial** |
| Trusted Channel (Level 3+) | Protected SSP entry/output | All in-process memory, no external channels | N/A |

### 3.3 §5.3 — Roles, Services, and Authentication

| Requirement | Mandate | Current State | Gap |
|-------------|---------|---------------|-----|
| 5.3.1 Roles | User + Crypto Officer role separation | Only one implicit role | **G4** |
| 5.3.2 Authentication | Identity-based authentication per role | API key enforcement disabled by default; keys in plaintext file | **G5** |
| 5.3.3 Services | Defined services per role | No service/role matrix documented | **Partial** |

#### Required Role Separation

The Crypto Officer role must be able to:
- Initialize the module and load keys
- Configure FIPS mode
- Trigger zeroization
- Run self-tests manually
- View status

The User role must be limited to:
- Encrypt/decrypt data with provisioned keys
- Query module status (non-sensitive)

### 3.4 §5.4 — Software/Firmware Security

| Requirement | Mandate | Current State | Gap |
|-------------|---------|---------------|-----|
| 5.4.1 Software | Integrity verification of all software components | No HMAC of module binary at load | **G3** |
| 5.4.2 Firmware | Same for firmware components | N/A (software module) | N/A |

### 3.5 §5.5 — Operating Environment

| Requirement | Level | Current State | Gap |
|-------------|-------|---------------|-----|
| Level 1 | Single operator, general purpose OS | Runs on macOS/Linux | ✅ OK |
| Level 2 | Limited OS, CCEVS-evaluated OS preferred | No OS security evaluation | **G16** |
| Level 3+ | Mandatory access control (SELinux/AppArmor) | No security policy for adelaide_server | **G16** |

### 3.6 §5.6 — Physical Security

N/A for pure software module. Physical security requirements apply to hardware and hybrid modules. For software, the operating environment security (§5.5) covers platform-level protection.

### 3.7 §5.7 — Non-Invasive Security

| Requirement | Level | Current State | Gap |
|-------------|-------|---------------|-----|
| Timing attack mitigation | Level 3+ | `memcmp` used for auth tag comparison (not constant-time) | **G12** |
| Power analysis mitigation | Level 4 | Not addressed | **G13** |
| TEMPEST | Level 4 | Not addressed | **G13** |

### 3.8 §5.8 — Sensitive Security Parameter (SSP) Management

| Requirement | Mandate | Current State | Gap |
|-------------|---------|---------------|-----|
| 5.8.1 RNG | Approved SP 800-90A DRBG | `RAND_bytes()` — not necessarily FIPS-indicated DRBG | **G2** |
| 5.8.2 Key Generation | Approved method inside module boundary | Master key generated in Python (`run.py`), not inside crypto module | **G6** |
| 5.8.3 Key Establishment | Approved key agreement | HKDF derivation uses OpenSSL (SP 800-56C) | ✅ Good |
| 5.8.6 Key Entry/Output | Plaintext keys never output outside boundary | No explicit restriction on debug output of keys | **G14** |
| 5.8.7 Key Storage | Keys stored in plaintext only within boundary | Memory-only storage in SPARK-verified package | ✅ Good |
| 5.8.8 Zeroization | Automated zeroization on shutdown/error | `Clear_Key` exists but not called on all error paths | **G11** |

#### Current Key Hierarchy

```
System Integrity Hash (SHA-512 of HW + binary state)
        │
        ▼  (salt)
User Secret ──► HKDF-SHA512 ──► Master Key (512-bit, memory-only)
                                       │
                                       ▼  (salt)
                              Context String ──► HKDF-SHA256 ──► AES-256 Key
                                                                      │
                                                                      ▼
                                                              AES-256-GCM
                                                          (field encryption)
```

### 3.9 §5.9 — Self-Tests

| Requirement | Mandate | Current State | Gap |
|-------------|---------|---------------|-----|
| 5.9(a) Power-Up | Algorithm KATs run on every power-up BEFORE any crypto operation | Compile-time KAT only (`#ifdef ADL_CRYPTO_TEST`), not at runtime | **G1** |
| 5.9(b) Integrity | HMAC integrity check of module code at load | CRC-32 in `Integrity_Utils` is not cryptographically strong; own binary not checked | **G3** |
| 5.9(c) Conditional — RNG | Continuous RNG test (stuck bit detection) | No wrapper around `RAND_bytes()` | **G7** |
| 5.9(c) Conditional — Pair-wise | For asymmetric key gen | No asymmetric keys | N/A |
| 5.9(c) Conditional — Bypass | Test bypass capability if exists | No bypass mechanism | N/A |
| 5.9(c) Conditional — Firmware Load | Verify loaded firmware | N/A | N/A |

#### Required Power-Up Self-Tests

A compliant module must perform the following before allowing any crypto operation:

```
Initialize_Crypto():
  1. Run AES-256-GCM KAT:
     - Encrypt known plaintext with known key
     - Verify ciphertext + tag matches expected value
     - Decrypt and verify round-trip
  2. Run SHA-384 KAT:
     - Hash known message
     - Verify digest matches expected value
  3. Run HKDF KAT:
     - Derive key from known salt + IKM + info
     - Verify output matches expected value
  4. Run software integrity test:
     - Compute HMAC-SHA384 of module binary
     - Compare against stored value
  5. If any test fails → enter error state, refuse all crypto ops
```

### 3.10 §5.10 — Life-Cycle Assurance

| Requirement | Mandate | Current State | Gap |
|-------------|---------|---------------|-----|
| 5.10.1 CM | Unique version ID, CM system | `version.ads` exists, no formal CM for crypto | ✅ Partial |
| 5.10.2 Design | Specification, high/low-level design | No crypto design document | **G8** |
| 5.10.3 Distribution | Secure delivery | Not addressed | **G17** |
| 5.10.4 Development | Secure coding practices | SPARK for Ada, but no secure dev policy for C code | ✅ Partial |
| 5.10.5 Guidance | Security policy document | **No FIPS Security Policy document exists** | **G8** |
| 5.10.6 Life-Cycle | EOL, migration procedures | Not documented | **G17** |

### 3.11 §5.11 — Mitigation of Other Attacks

| Requirement | Mandate | Current State | Gap |
|-------------|---------|---------------|-----|
| Specific attacks | Document and implement mitigations for module-specific threats | No attack model documented | **G13** |

---

## 4. Prioritized Remediation Plan

### Phase 1 — Foundation (Blocks ALL Levels)

Estimated effort: **3–5 days**

| Task | Files | FIPS Section | Effort |
|------|-------|-------------|--------|
| 1.1 Add power-up KATs for AES-256-GCM, SHA-384/512, HKDF | `src/adl_crypto.c` (+150 lines) | §5.9(a) | Low |
| 1.2 Wire KATs into `adl_init()` / `Initialize_Crypto()` | `src/adl_crypto.c`, `src/adelaide_crypto.adb` | §5.9(a) | Low |
| 1.3 Add software integrity test (HMAC-SHA384 of module binary at load) | `src/adl_crypto.c`, `src/system_integrity.adb` | §5.9(b) | Medium |
| 1.4 Replace `RAND_bytes()` with FIPS-indicated DRBG (CTR_DRBG) | `src/adl_crypto.c` | §5.8.1 | Medium |

### Phase 2 — Level 2 Compliance

Estimated effort: **5–7 days**

| Task | Files | FIPS Section | Effort |
|------|-------|-------------|--------|
| 2.1 Add Crypto Officer role with authentication | `src/api_key_manager.ads/.adb` | §5.3.1 | Low |
| 2.2 Make API key enforcement mandatory in FIPS mode | `src/api_key_manager.adb` | §5.3.2 | Low |
| 2.3 Generate master key inside crypto module (not Python) | `src/adl_crypto.c`, `run.py` | §5.8.2 | Medium |
| 2.4 Add continuous RNG test wrapper | `src/adl_crypto.c` | §5.9(c) | Low |
| 2.5 Add conditional self-tests for HKDF | `src/key_derivation.adb` | §5.9(c) | Low |
| 2.6 Wire `Clear_Key` into all exit/error paths | `src/shutdown_manager.adb` | §5.8.8 | Low |

### Phase 3 — Documentation (Required for Any Validation)

Estimated effort: **3–4 days**

| Task | Files | FIPS Section | Effort |
|------|-------|-------------|--------|
| 3.1 Write FIPS Security Policy document | `documentation/FIPS-SECURITY-POLICY.md` | §5.10.5 | Medium |
| 3.2 Define cryptographic boundary | `documentation/FIPS-SECURITY-POLICY.md` | §5.1.1 | Low |
| 3.3 Implement FIPS/non-FIPS mode switch | `src/adelaide_crypto.ads/.adb` | §5.1.4 | Low |

### Phase 4 — Level 3 Targeting

Estimated effort: **5–8 days**

| Task | Files | FIPS Section | Effort |
|------|-------|-------------|--------|
| 4.1 Replace `memcmp`/`strcmp` with constant-time equivalents | `src/adl_crypto.c`, `src/api_key_manager.adb` | §5.7 | Low |
| 4.2 Audit and document timing attack mitigations | `src/adl_crypto.c` | §5.7 | Medium |
| 4.3 Add trusted channel for key entry | New IPC module | §5.2.2 | High |
| 4.4 Implement OS-level process isolation (SELinux/AppArmor) | Deployment config | §5.5 | Medium |
| 4.5 Add audit logging for crypto operations | New audit module | §5.3.3 | Medium |

---

## 5. Key Code Locations

| File | Role | Needs Change For |
|------|------|------------------|
| `src/adl_crypto.c` | C crypto shim — AES-256-GCM, HKDF, master key load | **Phase 1** (KATs, DRBG, integrity test), **Phase 2** (RNG test, key gen) |
| `src/adl_crypto.h` | C header — API declarations | Sync with new functions |
| `src/adelaide_crypto.ads` | Ada crypto wrapper — FFI to C shim | **Phase 1** (wire KATs), **Phase 3** (FIPS mode flag) |
| `src/adelaide_crypto.adb` | Ada crypto implementation | **Phase 1** (KAT integration), **Phase 3** (FIPS mode) |
| `src/master_key_store.ads` | SPARK-verified key storage | **Phase 2** (auto-zeroization) |
| `src/master_key_store.adb` | Key storage implementation | **Phase 2** (error path zeroization) |
| `src/key_derivation.ads` | HKDF key derivation | **Phase 2** (conditional self-tests) |
| `src/key_derivation.adb` | Key derivation implementation | **Phase 2** (in-process key gen, HKDF KAT) |
| `src/system_integrity.ads` | Integrity hash computation | **Phase 1** (binary self-check) |
| `src/system_integrity.adb` | Integrity implementation | **Phase 1** (own-binary HMAC) |
| `src/api_key_manager.ads` | API key validation | **Phase 2** (Crypto Officer role, constant-time) |
| `src/api_key_manager.adb` | Key manager implementation | **Phase 2** (enforcement default, role separation) |
| `src/shutdown_manager.adb` | Graceful shutdown signaling | **Phase 2** (trigger key zeroization) |
| `run.py` | Build orchestration, master key init | **Phase 2** (move key gen to Ada/C boundary) |

---

## 6. References

1. NIST FIPS PUB 140-3, *Security Requirements for Cryptographic Modules*, March 22, 2019. https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.140-3.pdf
2. ISO/IEC 19790:2012, *Security Requirements for Cryptographic Modules*
3. ISO/IEC 24759:2017, *Test Requirements for Cryptographic Modules*
4. NIST SP 800-38D, *Recommendation for Block Cipher Modes of Operation: Galois/Counter Mode (GCM)*
5. NIST SP 800-56C, *Recommendation for Key Derivation through Extraction-then-Expansion*
6. NIST SP 800-90A Rev. 1, *Recommendation for Random Number Generation Using Deterministic Random Bit Generators*
7. RFC 5869, *HMAC-based Extract-and-Expand Key Derivation Function (HKDF)*
8. FIPS 180-4, *Secure Hash Standard (SHA-512 family)*
9. FIPS 197, *Advanced Encryption Standard (AES)*

---

## 7. Appendix: Algorithm Status Table

| Algorithm | FIPS Approved? | Used For | Status |
|-----------|---------------|----------|--------|
| AES-256-GCM | ✅ (FIPS 197, SP 800-38D) | Field encryption | ✅ Good |
| SHA-512 | ✅ (FIPS 180-4) | Integrity hashing | ✅ Good |
| SHA-384 | ✅ (FIPS 180-4) | HKDF-SHA384 sub-key derivation | ✅ Good |
| SHA-256 | ✅ (FIPS 180-4) | HKDF-SHA256 AES key derivation | ✅ Good |
| HKDF-SHA512 | ✅ (SP 800-56C) | Master key derivation | ✅ Good |
| HKDF-SHA384 | ✅ (SP 800-56C) | Sub-key derivation | ✅ Good |
| HKDF-SHA256 | ✅ (SP 800-56C) | AES key derivation | ✅ Good |
| CTR_DRBG | ✅ (SP 800-90A) | Random number generation | **G2 — Not implemented** |
| CRC-32 | ❌ Non-crypto | Integrity check (non-security) | ✅ Acceptable |
| LSH hash | ❌ Non-crypto | Locality-sensitive hashing | ✅ Acceptable |
