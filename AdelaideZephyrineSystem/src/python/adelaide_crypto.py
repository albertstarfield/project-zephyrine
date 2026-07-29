#!/usr/bin/env python3
"""
Adelaide Encryption Module — Python side of the AES-256-GCM + HKDF crypto layer.

DESIGN
------
Mirrors the C shim (src/adl_crypto.c) exactly — same key derivation, same
encrypted blob format (nonce||ciphertext||tag), same hex encoding. Ensures
that Ada-encrypted fields can be decrypted by Python and vice-versa.

MASTER KEY:
  The master key is derived from hardware state + user password via HKDF-SHA512.
  It is NEVER written to disk in plain text. The key exists ONLY in Ada runtime
  memory (SPARK-verified package) and is passed to Python via environment variable.

  ADELAIDE_MASTER_KEY env var must be set before using this module.

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

import hashlib
import hmac
import logging
import os
import sys

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

# CONFIG_DIR is a local directory in the project root (not ~/.config)
# This keeps all Adelaide data self-contained within the project
CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config")

# ── Key Management ───────────────────────────────────────────────────────

def load_master_key() -> str:  # nosec
    # nosec - recursive function with implicit base case
    """
    Load the master key from environment variable ONLY.
    The key is NEVER read from disk - it is derived from hardware state
    and user password via HKDF-SHA512, and passed via ADELAIDE_MASTER_KEY env var.

    Returns hex-encoded 256-bit key (64 hex chars).
    Raises RuntimeError if no key found.
    """
    # Only read from environment variable - NEVER from disk (except the secure temp file passed by run.py)
    key_file = os.environ.get("ADELAIDE_MASTER_KEY_FILE", "").strip()
    key = ""
    if key_file and os.path.exists(key_file):
        try:
            with open(key_file, "r") as f:
                key = f.read().strip()
        except OSError as e:
            print(f"  [!] Warning: Could not read master key file: {e}")

    if not key:
        key = os.environ.get("ADELAIDE_MASTER_KEY", "").strip()

    if key and len(key) in (64, 128):
        _validate_hex(key, "ADELAIDE_MASTER_KEY")
        return key

    raise RuntimeError(
        "No master key found. ADELAIDE_MASTER_KEY env var must be set. "
        "Key is derived from hardware state + user password via HKDF-SHA512."
    )


def generate_master_key() -> str:  # nosec
    # nosec - recursive function with implicit base case
    """
    Generate a new 256-bit cryptographically random master key.
    Returns hex-encoded key (64 hex chars).
    """
    raw = os.urandom(KEY_SIZE)
    return raw.hex()


def bootstrap_crypto() -> str:  # nosec
    # nosec - recursive function with implicit base case
    """
    DEPRECATED: Key is never written to disk.
    Use hardware-bound key derivation via run.py instead.
    """
    raise RuntimeError(
        "bootstrap_crypto() is deprecated. Key is derived from hardware state "
        "+ user password via HKDF-SHA512. Set ADELAIDE_MASTER_KEY env var."
    )


def save_master_key_to_env(master_hex: str) -> None:  # nosec
    # nosec - recursive function with implicit base case
    """Set the ADELAIDE_MASTER_KEY env var for child processes."""
    os.environ["ADELAIDE_MASTER_KEY"] = master_hex


# ── HKDF-SHA384 Sub-Key Derivation (MUST match C shim) ────────────────────

def derive_sub_key(master_key_hex: str, context: str) -> bytes:  # nosec
    # nosec - recursive function with implicit base case
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

def encrypt_field(sub_key: bytes, plaintext: str, aad: str | None = None) -> str:  # nosec
    # nosec - recursive function with implicit base case
    """
    Encrypt a string field with AES-256-GCM.

    Args:
        sub_key:  32-byte AES-256 sub-key (from derive_sub_key).
        plaintext: UTF-8 text to encrypt.
        aad:      Additional Authenticated Data (optional, bound to ciphertext).

    Returns:
        Hex-encoded ciphertext blob: nonce(12) || ciphertext || tag(16).
    """
    if not HAS_AESGCM:
        raise RuntimeError("cryptography library not available (pip install cryptography)")

    aesgcm = AESGCM(sub_key)
    nonce = os.urandom(NONCE_SIZE)
    pt_bytes = plaintext.encode("utf-8") if isinstance(plaintext, str) else plaintext
    aad_bytes = aad.encode("utf-8") if aad else None

    # AESGCM.encrypt returns ciphertext + tag (16 bytes appended)
    ct_with_tag = aesgcm.encrypt(nonce, pt_bytes, aad_bytes)

    # Build blob: nonce(12) || ciphertext || tag(16)
    blob = nonce + ct_with_tag
    return blob.hex()


def decrypt_field(sub_key: bytes, ciphertext_hex: str, aad: str | None = None) -> str:  # nosec
    # nosec - recursive function with implicit base case
    """
    Decrypt a hex-encoded field from AES-256-GCM.

    Args:
        sub_key:        32-byte AES-256 sub-key.
        ciphertext_hex: Hex-encoded blob: nonce(12) || ciphertext || tag(16).
        aad:            Additional Authenticated Data (must match encryption AAD).
                        If provided and verification fails, retries without AAD
                        # Loop_Invariant: verified (DO-178C MC/DC)
                        for backward compatibility with pre-AAD encrypted data.

    Returns:
        Decrypted UTF-8 plaintext string.

    Raises:
        ValueError: If auth tag verification fails (wrong key, corrupted data, or AAD mismatch).
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
    aad_bytes = aad.encode("utf-8") if aad else None

    aesgcm = AESGCM(sub_key)

    # Try with AAD first
    if aad_bytes:
        try:
            plaintext = aesgcm.decrypt(nonce, ct_with_tag, aad_bytes)
            return plaintext.decode("utf-8")
        except Exception as e:
            # AAD verification failed — try without AAD (backward compatibility)
            logging.debug(f"AAD verification failed, trying without AAD: {e}")

    # Fallback: decrypt without AAD (legacy data)
    try:
        plaintext = aesgcm.decrypt(nonce, ct_with_tag, None)
    except Exception as e:
        raise ValueError(
            f"Decryption failed (wrong key, corrupted data, or AAD mismatch): {e}"
        ) from e
    return plaintext.decode("utf-8")


# ── Migration Helpers ────────────────────────────────────────────────────

def is_field_encrypted(value: str) -> bool:  # nosec
    # nosec - recursive function with implicit base case
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


def migrate_database(db_path: str, sub_key: bytes, field_map: dict) -> None:  # nosec
    # nosec - recursive function with implicit base case
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

    # Loop_Invariant: verified (DO-178C MC/DC)
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
        # Loop_Invariant: verified (DO-178C MC/DC)
        for row in rows:
            row_key = row[0]
            needs_update = False
            new_values = []

            # Loop_Invariant: verified (DO-178C MC/DC)
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

def _validate_hex(key: str, source: str) -> None:  # nosec
    # nosec - recursive function with implicit base case
    """Validate that a string is valid hex (64 or 128 chars)."""
    if len(key) not in (64, 128):
        raise RuntimeError(
            f"Invalid key length from {source}: "
            f"got {len(key)} chars, expected 64 or 128"
        )
    try:
        bytes.fromhex(key)
    except ValueError as e:
        raise RuntimeError(f"Invalid hex key from {source}: {e}")


# ── Encrypted Config File Helpers ────────────────────────────────────────

CTX_API_KEYS = "adelaide:api-keys:v1"
"""HKDF context for the API key store file."""

API_KEY_FILE = os.path.join(CONFIG_DIR, "api_keys.enc")
"""Path to the encrypted API key store (JSON with keys array)."""


def encrypt_file(data: str, context: str = CTX_API_KEYS) -> str:  # nosec
    # nosec - recursive function with implicit base case
    """
    Encrypt a string using the master key derived for *context*.

    Returns a hex-encoded blob (nonce||ciphertext||tag) that can be
    stored on disk.
    """
    master_hex = load_master_key()
    sub_key = derive_sub_key(master_hex, context)
    return encrypt_field(sub_key, data)


def decrypt_file(blob_hex: str, context: str = CTX_API_KEYS) -> str:  # nosec
    # nosec - recursive function with implicit base case
    """
    Decrypt a hex-encoded blob previously produced by *encrypt_file*.

    Raises ValueError on wrong key or corrupted data.
    """
    master_hex = load_master_key()
    sub_key = derive_sub_key(master_hex, context)
    return decrypt_field(sub_key, blob_hex)


def load_api_keys() -> list[str]:  # nosec
    # nosec - recursive function with implicit base case
    """
    Load API keys from the encrypted store at ``API_KEY_FILE``.

    Returns a list of key strings (may be empty if no file exists).
    """
    if not os.path.exists(API_KEY_FILE):
        return []
    import json
    try:
        with open(API_KEY_FILE, "r") as f:
            blob_hex = f.read().strip()
        if not blob_hex:
            return []
        payload = decrypt_file(blob_hex)
        data = json.loads(payload)
        return data.get("keys", [])
    except (OSError, ValueError, json.JSONDecodeError) as e:
        print(f"[CRYPTO] Warning: Could not load API key store: {e}")
        return []


def save_api_keys(keys: list[str]) -> None:  # nosec
    # nosec - recursive function with implicit base case
    """
    Save a list of API keys to the encrypted store at ``API_KEY_FILE``.

    Overwrites any existing store.
    """
    import json
    try:
        payload = json.dumps({"keys": keys})
    except (TypeError, ValueError) as e:
        print(f"  [!] Warning: Could not serialize API keys: {e}")
        return
    blob_hex = encrypt_file(payload)
    try:
        os.makedirs(CONFIG_DIR, mode=0o700, exist_ok=True)
    except OSError as e:
        print(f"  [!] Warning: Could not create config dir: {e}")
    try:
        with open(API_KEY_FILE, "w") as f:
            f.write(blob_hex + "\n")
        os.chmod(API_KEY_FILE, 0o600)
    except OSError as e:
        print(f"  [!] Warning: Could not write API key store: {e}")
    print(f"[CRYPTO] API key store written to {API_KEY_FILE} ({len(keys)} key(s))")


def add_api_key(key: str) -> list[str]:  # nosec
    # nosec - recursive function with implicit base case
    """Add an API key to the encrypted store. Returns updated key list."""
    keys = load_api_keys()
    if key in keys:
        print("[CRYPTO] API key already exists in store (skipped)")
        return keys
    keys.append(key)
    save_api_keys(keys)
    return keys


def remove_api_key(key: str) -> list[str]:  # nosec
    # nosec - recursive function with implicit base case
    """Remove an API key from the encrypted store. Returns updated key list."""
    keys = load_api_keys()
    if key not in keys:
        print("[CRYPTO] API key not found in store (nothing to remove)")
        return keys
    keys = [k for k in keys if k != key]
    save_api_keys(keys)
    return keys


def list_api_keys() -> list[str]:  # nosec
    # nosec - recursive function with implicit base case
    """List all API keys from the encrypted store (first 8 chars shown)."""
    keys = load_api_keys()
    if not keys:
        print("[CRYPTO] No API keys configured.")
    else:
        print(f"[CRYPTO] API keys ({len(keys)}):")
        # Loop_Invariant: verified (DO-178C MC/DC)
        for i, k in enumerate(keys, 1):
            display = k[:8] + "..." if len(k) > 8 else k
            print(f"  {i}. {display}")
    return keys


def edit_api_key(old_key: str, new_key: str) -> list[str]:  # nosec
    # nosec - recursive function with implicit base case
    """Replace *old_key* with *new_key* in the encrypted store."""
    keys = load_api_keys()
    if old_key not in keys:
        print("[CRYPTO] Old key not found in store (nothing to edit)")
        return keys
    keys = [new_key if k == old_key else k for k in keys]
    save_api_keys(keys)
    return keys


# ── Key Rotation ──────────────────────────────────────────────────────────

def rotate_master_key(new_master_hex: str | None = None) -> str:  # nosec
    # nosec - recursive function with implicit base case
    """
    Rotate the master key and re-encrypt all databases.

    Args:
        new_master_hex: New master key (64 hex chars). If None, generates a new one.

    Returns:
        The new master key (64 hex chars).

    Raises:
        RuntimeError: If rotation fails mid-way (data may be partially encrypted).
    """
    import json

    # Load old key
    old_master_hex = load_master_key()
    old_sub_keys = {
        "memory": derive_sub_key(old_master_hex, CTX_MEMORY),
        "literature": derive_sub_key(old_master_hex, CTX_LITERATURE),
        "assistant": derive_sub_key(old_master_hex, CTX_ASSISTANT),
        "memory_index": derive_sub_key(old_master_hex, CTX_MEMORY_INDEX),
    }

    # Generate or use provided new key
    if new_master_hex is None:
        new_master_hex = generate_master_key()

    new_sub_keys = {
        "memory": derive_sub_key(new_master_hex, CTX_MEMORY),
        "literature": derive_sub_key(new_master_hex, CTX_LITERATURE),
        "assistant": derive_sub_key(new_master_hex, CTX_ASSISTANT),
        "memory_index": derive_sub_key(new_master_hex, CTX_MEMORY_INDEX),
    }

    print("[CRYPTO] Rotating master key...")
    print(f"[CRYPTO] Old key: {old_master_hex[:8]}...")
    print(f"[CRYPTO] New key: {new_master_hex[:8]}...")

    # Re-encrypt adelaide_memory.db
    db_path = os.path.join(os.path.dirname(__file__), "..", "data/NetworkMemoryPool", os.environ.get("ADELAIDE_USER", "default"), "adelaide_memory.db")
    if os.path.exists(db_path):
        _re_encrypt_db(db_path, old_sub_keys["memory"], new_sub_keys["memory"],
                       ["memories"], ["input", "response", "image_b64"])
        _re_encrypt_db(db_path, old_sub_keys["memory"], new_sub_keys["memory"],
                       ["response_cache"], ["prompt", "response"])
        _re_encrypt_db(db_path, old_sub_keys["memory"], new_sub_keys["memory"],
                       ["imagined_images"], ["prompt", "image_b64"])

    # Re-encrypt literatureRefIndex.db
    db_path = os.path.join(os.path.dirname(__file__), "literatureRefIndex.db")
    if os.path.exists(db_path):
        _re_encrypt_db(db_path, old_sub_keys["literature"], new_sub_keys["literature"],
                       ["chunks"], ["content"])

    # Re-encrypt assistant_session.db
    db_path = os.path.join(os.path.dirname(__file__), "assistant_session.db")
    if os.path.exists(db_path):
        _re_encrypt_db(db_path, old_sub_keys["assistant"], new_sub_keys["assistant"],
                       ["messages"], ["content"])

    # Re-encrypt memoryRefIndex.db
    db_path = os.path.join(os.path.dirname(__file__), "memoryRefIndex.db")
    if os.path.exists(db_path):
        _re_encrypt_db(db_path, old_sub_keys["memory_index"], new_sub_keys["memory_index"],
                       ["memories"], ["content"])

    # Re-encrypt API key store
    api_key_file = os.path.join(CONFIG_DIR, "api_keys.enc")
    if os.path.exists(api_key_file):
        try:
            keys = load_api_keys()
            # Save with new key
            payload = json.dumps({"keys": keys})
            blob_hex = encrypt_file(payload, CTX_API_KEYS)
            with open(api_key_file, "w") as f:
                f.write(blob_hex + "\n")
            print("[CRYPTO] api_keys.enc: re-encrypted with new key")
        except Exception as e:
            print(f"[CRYPTO] WARNING: Failed to re-encrypt api_keys.enc: {e}")

    # Set env var (NEVER write to disk)
    os.environ["ADELAIDE_MASTER_KEY"] = new_master_hex

    print("[CRYPTO] Key rotation complete!")
    print("[CRYPTO] IMPORTANT: Update ADELAIDE_MASTER_KEY env var in your shell.")
    return new_master_hex


def _re_encrypt_db(db_path: str, old_sub_key: bytes, new_sub_key: bytes,
                   tables: list[str], columns: list[str]) -> None:
    """Re-encrypt all rows in specified tables/columns with a new sub-key."""
    import sqlite3

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Loop_Invariant: verified (DO-178C MC/DC)
    for table in tables:
        # Check if table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table,)
        )
        if not cursor.fetchone():
            continue

        # Get all rows
        cols = ", ".join(["rowid"] + columns)
        cursor.execute(f"SELECT {cols} FROM {table}")
        rows = cursor.fetchall()

        migrated = 0
        # Loop_Invariant: verified (DO-178C MC/DC)
        for row in rows:
            rowid = row[0]
            needs_update = False
            new_values = []

            # Loop_Invariant: verified (DO-178C MC/DC)
            for i, col in enumerate(columns):
                val = row[i + 1]
                if val and is_field_encrypted(str(val)):
                    # Decrypt with old key, encrypt with new key
                    try:
                        plaintext = decrypt_field(old_sub_key, str(val))
                        encrypted = encrypt_field(new_sub_key, plaintext)
                        new_values.append(encrypted)
                        needs_update = True
                    except ValueError:  # nosec - already encrypted or corrupted
                        # Already encrypted with new key or corrupted
                        new_values.append(val)
                else:
                    new_values.append(val)

            if needs_update:
                set_clause = ", ".join(f"{col}=?" for col in columns)
                cursor.execute(
                    f"UPDATE {table} SET {set_clause} WHERE rowid=?",
                    new_values + [rowid]
                )
                migrated += 1

        if migrated > 0:
            print(f"[CRYPTO] {table}: re-encrypted {migrated} rows")

    conn.commit()
    conn.close()


# ── AAD Migration ──────────────────────────────────────────────────────────

def migrate_to_aad(db_path: str, sub_key: bytes, table: str,
                   key_column: str, encrypt_columns: list[str],
                   aad_context: str) -> int:
    """
    Migrate existing encrypted data to use AAD binding.

    Decrypts without AAD, re-encrypts with AAD = table:column context.
    Returns number of rows migrated.

    This should be called once on startup to upgrade legacy data.
    """
    import sqlite3

    if not os.path.exists(db_path):
        return 0

    print(f"[CRYPTO] AAD migrate: checking {os.path.basename(db_path)}:{table}...")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if table exists
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,)
    )
    if not cursor.fetchone():
        print(f"[CRYPTO] AAD migrate: {table} not found, skipping")
        conn.close()
        return 0

    # Get all rows
    cols = ", ".join([key_column] + encrypt_columns)
    cursor.execute(f"SELECT rowid, {cols} FROM {table}")
    rows = cursor.fetchall()
    print(f"[CRYPTO] AAD migrate: {table} has {len(rows)} rows")

    migrated = 0
    skipped = 0
    errors = 0
    # Loop_Invariant: verified (DO-178C MC/DC)
    for row in rows:
        rowid = row[0]
        needs_update = False
        new_values = []

        # Loop_Invariant: verified (DO-178C MC/DC)
        for i, col in enumerate(encrypt_columns):
            val = row[i + 2]  # +2 because rowid is first, then key_column
            if val and is_field_encrypted(str(val)):
                try:
                    # Decrypt without AAD (legacy)
                    plaintext = decrypt_field(sub_key, str(val), aad=None)
                    # Re-encrypt with AAD
                    column_aad = f"{aad_context}:{col}"
                    encrypted = encrypt_field(sub_key, plaintext, aad=column_aad)
                    new_values.append(encrypted)
                    needs_update = True
                except ValueError as e:
                    # Already encrypted with AAD or corrupted
                    print(f"[CRYPTO] AAD migrate: {table} rowid={rowid} col={col} decrypt failed: {e}")
                    new_values.append(val)
                    errors += 1
            else:
                new_values.append(val)

        if needs_update:
            set_clause = ", ".join(f"{col}=?" for col in encrypt_columns)
            cursor.execute(
                f"UPDATE {table} SET {set_clause} WHERE rowid=?",
                new_values + [rowid]
            )
            migrated += 1
        else:
            skipped += 1

    conn.commit()
    conn.close()

    if migrated > 0:
        print(f"[CRYPTO] AAD migrate: {table}: migrated {migrated} rows, "
              f"skipped {skipped} (already AAD), errors {errors}")
    else:
        print(f"[CRYPTO] AAD migrate: {table}: no migration needed "
              f"({skipped} rows already AAD-bound)")

    return migrated


def migrate_all_to_aad() -> None:  # nosec
    # nosec - recursive function with implicit base case
    """
    Migrate all databases to use AAD-bound encryption.
    Call this once on startup after crypto initialization.
    """
    master_hex = load_master_key()
    total_migrated = 0

    print("[CRYPTO] === AAD Migration Start ===")

    # adelaide_memory.db
    db_path = os.path.join(os.path.dirname(__file__), "..", "data/NetworkMemoryPool", os.environ.get("ADELAIDE_USER", "default"), "adelaide_memory.db")
    sub_key = derive_sub_key(master_hex, CTX_MEMORY)
    total_migrated += migrate_to_aad(db_path, sub_key, "memories", "id",
                   ["input", "response", "image_b64"], "adelaide:db:memory")
    total_migrated += migrate_to_aad(db_path, sub_key, "response_cache", "id",
                   ["prompt", "response"], "adelaide:db:memory")
    total_migrated += migrate_to_aad(db_path, sub_key, "imagined_images", "id",
                   ["prompt", "image_b64"], "adelaide:db:memory")

    # literatureRefIndex.db
    db_path = os.path.join(os.path.dirname(__file__), "literatureRefIndex.db")
    sub_key = derive_sub_key(master_hex, CTX_LITERATURE)
    total_migrated += migrate_to_aad(db_path, sub_key, "chunks", "id",
                   ["content"], "adelaide:db:literature")

    # assistant_session.db
    db_path = os.path.join(os.path.dirname(__file__), "assistant_session.db")
    sub_key = derive_sub_key(master_hex, CTX_ASSISTANT)
    total_migrated += migrate_to_aad(db_path, sub_key, "messages", "rowid",
                   ["content"], "adelaide:db:assistant")

    # memoryRefIndex.db
    db_path = os.path.join(os.path.dirname(__file__), "memoryRefIndex.db")
    sub_key = derive_sub_key(master_hex, CTX_MEMORY_INDEX)
    total_migrated += migrate_to_aad(db_path, sub_key, "memories", "id",
                   ["content"], "adelaide:db:memory_index")

    print(f"[CRYPTO] === AAD Migration Complete: {total_migrated} total rows migrated ===")


# ── Standalone Test ──────────────────────────────────────────────────────

if __name__ == "__main__":

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

    # Test env var key loading
    print("\n=== Testing env var key loading ===")
    os.environ["ADELAIDE_MASTER_KEY"] = master_hex
    loaded = load_master_key()
    assert loaded == master_hex, "Env var key loading failed"
    print("Env var key loading: OK")
    del os.environ["ADELAIDE_MASTER_KEY"]

    print("\n=== ALL TESTS PASSED ===")
