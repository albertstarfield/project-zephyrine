#!/usr/bin/env python3
"""
Adelaide Encryption Module — Python side of the AES-256-GCM + HKDF crypto layer.

DESIGN
------
Mirrors the C shim (src/adl_crypto.c) exactly — same key derivation, same
encrypted blob format (nonce||ciphertext||tag), same hex encoding. Ensures
that Ada-encrypted fields can be decrypted by Python and vice-versa.

MASTER KEY PRIORITY (first wins):
  1. ADELAIDE_MASTER_KEY env var       ← portable, for CI/migration
  2. ~/.config/adelaide/master.key     ← file, chmod 0600
  3. Generate new → write to both      ← first boot

SUB-KEY DERIVATION (per DB):
  HKDF-SHA384(master_key, context_string) → 32-byte AES-256 sub-key

  Context strings:
    "adelaide:db:memory:v1"        — adelaide_memory.db (Ada field-level)
    "adelaide:db:literature:v1"    — literatureRefIndex.db (Ada+Python field-level)
    "adelaide:db:assistant:v1"     — assistant_session.db (Python field-level)
    "adelaide:db:memory_index:v1"  — memoryRefIndex.db (Python field-level)

POST-QUANTUM NOTE:
  AES-256 → 128-bit post-quantum via Grover's (still safe)
  SHA-384 HKDF → 192-bit post-quantum collision resistance
  No asymmetric keys → no Shor vulnerability
"""

import os
import sys
import base64
import hashlib
import hmac
import struct

# ── cryptography library (already installed per system audit) ────────────
try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    HAS_AESGCM = True
except ImportError:
    HAS_AESGCM = False

# ── Constants (MUST match adl_crypto.h exactly) ──────────────────────────
KEY_SIZE = 32          # 256-bit AES key
NONCE_SIZE = 12        # 96-bit random nonce for GCM
TAG_SIZE = 16          # 128-bit GCM auth tag
KEY_HEX_SIZE = 64      # 32 bytes = 64 hex chars

# HKDF context strings — one per database
CTX_MEMORY       = "adelaide:db:memory:v1"
CTX_LITERATURE   = "adelaide:db:literature:v1"
CTX_ASSISTANT    = "adelaide:db:assistant:v1"
CTX_MEMORY_INDEX = "adelaide:db:memory_index:v1"

CONFIG_DIR = os.path.expanduser("~/.config/adelaide")
KEY_FILE = os.path.join(CONFIG_DIR, "master.key")

# ── Key Management ───────────────────────────────────────────────────────

def load_master_key() -> str:
    """
    Load the master key following the priority chain.
    Returns hex-encoded 256-bit key (64 hex chars).
    Raises RuntimeError if no key found.
    """
    # Priority 1: Environment variable
    key = os.environ.get("ADELAIDE_MASTER_KEY", "").strip()
    if key and len(key) == 64:
        _validate_hex(key, "ADELAIDE_MASTER_KEY")
        return key

    # Priority 2: Config file
    if os.path.exists(KEY_FILE):
        try:
            with open(KEY_FILE, "r") as f:
                key = f.read().strip()
            if key and len(key) == 64:
                _validate_hex(key, KEY_FILE)
                return key
        except (OSError, IOError) as e:
            raise RuntimeError(f"Cannot read master key file {KEY_FILE}: {e}")

    raise RuntimeError(
        "No master key found. Set ADELAIDE_MASTER_KEY env var "
        "or run bootstrap_crypto() first."
    )


def generate_master_key() -> str:
    """
    Generate a new 256-bit cryptographically random master key.
    Returns hex-encoded key (64 hex chars).
    """
    raw = os.urandom(KEY_SIZE)
    return raw.hex()


def bootstrap_crypto() -> str:
    """
    First-boot key bootstrap.
    1. Try to load existing key (from env or file).
    2. If none found, generate new key and persist to file.

    Returns the hex-encoded master key.
    """
    # Try loading first
    try:
        return load_master_key()
    except RuntimeError:
        pass

    # Generate new key
    print("[CRYPTO] No master key found. Generating new 256-bit AES key...")
    master_hex = generate_master_key()

    # Persist to file
    try:
        os.makedirs(CONFIG_DIR, mode=0o700, exist_ok=True)
        with open(KEY_FILE, "w") as f:
            f.write(master_hex + "\n")
        os.chmod(KEY_FILE, 0o600)
        print(f"[CRYPTO] Master key written to {KEY_FILE} (chmod 0600)")
    except (OSError, IOError) as e:
        print(f"[CRYPTO] WARNING: Could not persist master key to {KEY_FILE}: {e}")
        print("[CRYPTO] You must set ADELAIDE_MASTER_KEY env var on next boot.")

    # Also set env var for child processes
    os.environ["ADELAIDE_MASTER_KEY"] = master_hex

    return master_hex


def save_master_key_to_env(master_hex: str) -> None:
    """Set the ADELAIDE_MASTER_KEY env var for child processes."""
    os.environ["ADELAIDE_MASTER_KEY"] = master_hex


# ── HKDF-SHA384 Sub-Key Derivation (MUST match C shim) ────────────────────

def derive_sub_key(master_key_hex: str, context: str) -> bytes:
    """
    HKDF-SHA384 derivation returning 32-byte AES-256 sub-key.

    Implements RFC 5869 HKDF:
      1. Extract: PRK = HMAC-SHA384(zero_salt, master_key_bytes)
      2. Expand:  OKM = HMAC-SHA384(PRK, context || 0x01)

    Context examples:
      "adelaide:db:memory:v1"
      "adelaide:db:literature:v1"
      "adelaide:db:assistant:v1"
      "adelaide:db:memory_index:v1"

    Returns 32 raw bytes.
    """
    raw_master = bytes.fromhex(master_key_hex)

    # Step 1: Extract — HMAC-SHA384 with zero salt
    salt = b'\x00' * KEY_SIZE
    prk = hmac.new(salt, raw_master, hashlib.sha384).digest()  # 48 bytes

    # Step 2: Expand — T(1) = HMAC-SHA384(PRK, context || 0x01)
    expand_input = context.encode("utf-8") + b'\x01'
    okm = hmac.new(prk, expand_input, hashlib.sha384).digest()  # 48 bytes

    # Take first 32 bytes as AES-256 key
    return okm[:KEY_SIZE]


# ── AES-256-GCM Encrypt / Decrypt (MUST match C shim) ────────────────────

def encrypt_field(sub_key: bytes, plaintext: str) -> str:
    """
    Encrypt a string field with AES-256-GCM.

    Args:
        sub_key:  32-byte AES-256 sub-key (from derive_sub_key).
        plaintext: UTF-8 text to encrypt.

    Returns:
        Hex-encoded ciphertext blob: nonce(12) || ciphertext || tag(16).
    """
    if not HAS_AESGCM:
        raise RuntimeError("cryptography library not available (pip install cryptography)")

    aesgcm = AESGCM(sub_key)
    nonce = os.urandom(NONCE_SIZE)
    pt_bytes = plaintext.encode("utf-8") if isinstance(plaintext, str) else plaintext

    # AESGCM.encrypt returns ciphertext + tag (16 bytes appended)
    ct_with_tag = aesgcm.encrypt(nonce, pt_bytes, None)

    # Build blob: nonce(12) || ciphertext || tag(16)
    blob = nonce + ct_with_tag
    return blob.hex()


def decrypt_field(sub_key: bytes, ciphertext_hex: str) -> str:
    """
    Decrypt a hex-encoded field from AES-256-GCM.

    Args:
        sub_key:        32-byte AES-256 sub-key.
        ciphertext_hex: Hex-encoded blob: nonce(12) || ciphertext || tag(16).

    Returns:
        Decrypted UTF-8 plaintext string.

    Raises:
        ValueError: If auth tag verification fails (wrong key or corrupted data).
    """
    if not HAS_AESGCM:
        raise RuntimeError("cryptography library not available (pip install cryptography)")

    blob = bytes.fromhex(ciphertext_hex)

    if len(blob) < NONCE_SIZE + TAG_SIZE:
        raise ValueError(
            f"Ciphertext too short: {len(blob)} bytes "
            f"(minimum {NONCE_SIZE + TAG_SIZE})"
        )

    nonce = blob[:NONCE_SIZE]
    ct_with_tag = blob[NONCE_SIZE:]

    aesgcm = AESGCM(sub_key)
    # AESGCM.decrypt expects ciphertext||tag combined
    try:
        plaintext = aesgcm.decrypt(nonce, ct_with_tag, None)
    except Exception as e:
        raise ValueError(
            f"Decryption failed (wrong key or corrupted data): {e}"
        ) from e
    return plaintext.decode("utf-8")


# ── Migration Helpers ────────────────────────────────────────────────────

def is_field_encrypted(value: str) -> bool:
    """
    Heuristic: check if a DB field value looks like our encrypted format.
    Encrypted hex blobs are always longer than 28*2=56 hex chars (nonce+tag)
    and contain only hex characters.
    """
    if not value or len(value) < (NONCE_SIZE + TAG_SIZE) * 2:
        return False
    # Must be valid lowercase hex
    try:
        bytes.fromhex(value)
        return True
    except ValueError:
        return False


def migrate_database(db_path: str, sub_key: bytes, field_map: dict) -> None:
    """
    Migrate a plaintext SQLite database to encrypted fields in-place.

    Args:
        db_path:   Path to the SQLite database file.
        sub_key:   32-byte AES-256 sub-key.
        field_map: {
            "table_name": {
                "key_column": "id",         # column used in WHERE
                "encrypt_columns": ["col1", "col2"],
            }
        }

    This reads each row, encrypts the specified columns, and writes them back.
    Unencrypted plaintext columns are detected by is_field_encrypted check.
    """
    import sqlite3

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    for table_name, config in field_map.items():
        key_col = config["key_column"]
        encrypt_cols = config["encrypt_columns"]

        # Check if table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,)
        )
        if not cursor.fetchone():
            continue

        # Build query
        cols = ", ".join([key_col] + encrypt_cols)
        cursor.execute(f"SELECT {cols} FROM {table_name}")
        rows = cursor.fetchall()

        migrated = 0
        for row in rows:
            row_key = row[0]
            needs_update = False
            new_values = []

            for i, col in enumerate(encrypt_cols):
                val = row[i + 1]
                if val and not is_field_encrypted(str(val)):
                    # Encrypt it
                    encrypted = encrypt_field(sub_key, str(val))
                    new_values.append(encrypted)
                    needs_update = True
                else:
                    new_values.append(val)

            if needs_update:
                set_clause = ", ".join(
                    f"{col}=?" for col in encrypt_cols
                )
                cursor.execute(
                    f"UPDATE {table_name} SET {set_clause} WHERE {key_col}=?",
                    new_values + [row_key]
                )
                migrated += 1

        if migrated > 0:
            print(f"[CRYPTO] {table_name}: migrated {migrated} rows to encrypted")

    conn.commit()
    conn.close()


# ── Internal Helpers ─────────────────────────────────────────────────────

def _validate_hex(key: str, source: str) -> None:
    """Validate that a string is 64-char hex."""
    if len(key) != KEY_HEX_SIZE:
        raise RuntimeError(
            f"Invalid key length from {source}: "
            f"got {len(key)} chars, expected {KEY_HEX_SIZE}"
        )
    try:
        bytes.fromhex(key)
    except ValueError as e:
        raise RuntimeError(f"Invalid hex key from {source}: {e}")


# ── Standalone Test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    print("=== Adelaide Crypto Module Self-Test ===\n")

    # Test key generation
    master_hex = generate_master_key()
    print(f"Master key: {master_hex} ({len(master_hex)} hex chars)")

    # Test HKDF derivation
    sub_key = derive_sub_key(master_hex, CTX_MEMORY)
    print(f"Memory sub-key: {sub_key.hex()} ({len(sub_key)} bytes)")

    sub_key2 = derive_sub_key(master_hex, CTX_LITERATURE)
    print(f"Literature sub-key: {sub_key2.hex()} ({len(sub_key2)} bytes)")
    assert sub_key != sub_key2, "Sub-keys should differ for different contexts"

    # Test encrypt/decrypt round-trip
    test_text = "Hello, Adelaide! This is sensitive conversation data."
    ct = encrypt_field(sub_key, test_text)
    print(f"\nCiphertext ({len(ct)} hex chars): {ct[:40]}...{ct[-40:]}")

    pt = decrypt_field(sub_key, ct)
    print(f"Decrypted: {pt}")
    assert pt == test_text, "Round-trip failed!"

    # Test wrong key detection
    wrong_sub_key = derive_sub_key(master_hex, CTX_ASSISTANT)
    try:
        decrypt_field(wrong_sub_key, ct)
        print("FAIL: Wrong key should have raised ValueError")
        sys.exit(1)
    except ValueError as e:
        print(f"Wrong key correctly rejected: {e}")

    # Test is_field_encrypted heuristic
    assert is_field_encrypted(ct), "Should detect encrypted field"
    assert not is_field_encrypted("Hello, plaintext!"), "Should reject plaintext"
    assert not is_field_encrypted(""), "Should reject empty string"
    assert not is_field_encrypted("abc"), "Should reject short string"

    # Test bootstrap_crypto (will generate key if none exists)
    print("\n=== Testing bootstrap_crypto ===")
    # Save current env, restore after
    saved_env = os.environ.get("ADELAIDE_MASTER_KEY")
    if "ADELAIDE_MASTER_KEY" in os.environ:
        del os.environ["ADELAIDE_MASTER_KEY"]

    # Temporarily remove key file if exists
    key_backup = None
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE) as f:
            key_backup = f.read().strip()
        os.remove(KEY_FILE)

    try:
        boot_key = bootstrap_crypto()
        print(f"Bootstrapped key: {boot_key}")
        assert len(boot_key) == 64, "Bootstrap should return valid key"
        assert os.path.exists(KEY_FILE), "Key file should exist after bootstrap"

        # Second call should load existing
        boot_key2 = bootstrap_crypto()
        assert boot_key == boot_key2, "Second bootstrap should return same key"
        print("Bootstrap idempotency: OK")
    finally:
        # Restore
        if saved_env:
            os.environ["ADELAIDE_MASTER_KEY"] = saved_env
        if key_backup:
            with open(KEY_FILE, "w") as f:
                f.write(key_backup + "\n")

    print("\n=== ALL TESTS PASSED ===")
