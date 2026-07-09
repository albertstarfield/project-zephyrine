# SECURITY.md

Well even our face is cute, we still have responsibility since we have a lot of information and data being in and out. and perhaps even personal or even something that not meant to be public thus it's better to implement it like this.

## FIPS 140-3 Compliance Status

The Adelaide Zephyrine System implements a high-security cryptographic boundary conforming to FIPS 140-3 standards. 

* **Validation Level**: Designed to meet Level 1-4 formal validation criteria.
* **Tamper Evidence**: While this is a software-only module, it implements cryptographic continuous health tests, source and binary integrity tests on boot (`kat_binary_integrity`, `kat_source_integrity`), and utilizes a formally verified SPARK Ada `CTR_DRBG` for memory-safe deterministic random bit generation.
* **Authentication**: The system utilizes strict **Identity-based Authentication** using PBKDF2 with HKDF-SHA512 hashing to fulfill Level 3 and 4 access control requirements. 
* **Key Generation**: 256-bit high-entropy Recovery Keys are generated via the SPARK DRBG backed by AES-256 and `/dev/urandom`. OpenSSL's `RAND_bytes()` reliance for key generation has been fully replaced with a 100% formally verified, memory-safe SPARK Random Number Generator.

## Human Identity Verification & Compartmentalization

To guarantee strict data isolation, the Adelaide Zephyrine System employs **compartmentalization based on unique human identities**:

1. **Identity Cryptography**: User identities (e.g., username/email/identity) are registered and verified using PBKDF2/HKDF-SHA512 password hashing. Upon verification, the user is assigned a unique, cryptographically secure 128-bit hash representing their `Session_ID`.
2. **Database Isolation**: The `identity_store.db` maintains a strict boundary mapping human identities to their secure hashes and credentials.
3. **KV Cache Compartmentalization**: The system leverages the `Session_ID` to strictly isolate Virtual Context memory on disk. Every user's KV cache is sandboxed into dedicated namespaces (`cache/kv/{Session_ID}/`), ensuring that a single human identity can NEVER access, override, or read the contextual memory of another human identity.

## Security Practices

All cryptographic operations are performed within a tightly bound C/Ada boundary with automatic memory zeroization on context clear. Any failure of power-up self-tests results in immediate process poisoning (`adl_poison`), ensuring that no cryptographic operations are allowed until the environment is healthy.
