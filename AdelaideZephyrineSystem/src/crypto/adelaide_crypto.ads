pragma SPARK_Mode (Off);
-- c_binding: OpenSSL FFI for cryptographic operations
with Ada.Strings.Unbounded;

--  ── Adelaide Crypto Wrapper ───────────────────────────────────────────────
--  Ada interface to the C AES-256-GCM + HKDF crypto shim (adl_crypto.c).
--
--  Provides Encrypt/Decrypt for field-level encryption of sensitive columns.
--  The C shim manages the master key (from env var or config file). This
--  package derives per-DB sub-keys and exposes encrypt/decrypt operations.
--
--  USAGE:
--    1. Call Initialize_Crypto early in boot (before any DB access).
--    2. Call Derive_Subkey for each database context.
--    3. Encrypt fields before INSERT/UPDATE, decrypt after SELECT.
--
--  POST-QUANTUM NOTE: AES-256-GCM is post-quantum safe. Grover's algorithm
--  halves effective key size to 128 bits, which is still sufficient. See
--  adl_crypto.c header for full rationale.
--  ────────────────────────────────────────────────────────────────────────────

package Adelaide_Crypto is

   --  Result type for crypto operations
   type Crypto_Result is record
      Success : Boolean;
      Data    : Ada.Strings.Unbounded.Unbounded_String;
      Error   : Ada.Strings.Unbounded.Unbounded_String;
   end record;

   --  Initialize: loads master key from ADELAIDE_MASTER_KEY env var or
   --  config/master.key (local to project). Must be called once at startup.
   --  Runs FIPS 140-3 §5.9 power-up self-tests (KATs + integrity scan).
   --  On test failure, keys are zeroized and crypto is permanently disabled
   --  for the lifetime of this process (InferiorParadoxical anti-tamper).
   --  Returns True if initialization succeeded.
   function Initialize_Crypto return Boolean;

   --  Returns True if crypto is initialized and ready. Use this to skip
   --  encryption if no key is available (e.g., after graceful fallback).
   function Is_Crypto_Ready return Boolean;

   --  FIPS 140-3 InferiorParadoxical status checks:
   --  Is_Poisoned:       Returns True if anti-tamper tripped (keys zeroized).
   --  Self_Tests_Passed: Returns True if power-up KATs all succeeded.
   --  Is_FIPS_Ready:     Returns True if crypto is ready AND self-tests passed
   --                     AND module is not poisoned (one combined check).
   function Is_Poisoned return Boolean;
   function Self_Tests_Passed return Boolean;
   function Is_FIPS_Ready return Boolean;

   --  FIPS 140-3 mode control:
   --  Is_FIPS_Mode:  Returns True if operating in FIPS mode.
   --  Set_FIPS_Mode: Disable FIPS mode (Crypto Officer operation).
   --                  Can only disable, never re-enable without restart.
   function Is_FIPS_Mode return Boolean;
   procedure Set_FIPS_Mode (Enabled : Boolean);

   --  Derive a per-DB sub-key from the master key.
   --  Context examples:
   --    "adelaide:db:memory:v1"     -- adelaide_memory.db
   --    "adelaide:db:literature:v1" -- literatureRefIndex.db
   --    "adelaide:db:assistant:v1"  -- assistant_session.db
   --  Sub_Key output is 64 hex characters.
   function Derive_Subkey
     (Context : String) return Crypto_Result;

   --  Encrypt a plaintext string field.
   --  Returns hex-encoded ciphertext blob: nonce(12) || AES-GCM ciphertext || tag(16).
   function Encrypt_Field
     (Sub_Key_Hex : String;
      Plaintext   : String) return Crypto_Result;

   --  Decrypt a hex-encoded ciphertext field.
   --  Returns the original plaintext UTF-8 string.
   function Decrypt_Field
     (Sub_Key_Hex   : String;
      Ciphertext_Hex : String) return Crypto_Result;

   --  Convenience: encrypt, returning the hex string or Plaintext on failure.
   --  Use this for write paths where you want graceful fallback.
   function Try_Encrypt
     (Sub_Key_Hex : String;
      Plaintext   : String) return String;

   --  Convenience: decrypt, returning the plaintext or Ciphertext_Hex on failure.
   --  Use this for read paths where you want graceful fallback.
   function Try_Decrypt
     (Sub_Key_Hex   : String;
      Ciphertext_Hex : String) return String;

   --  Check if a hex-encoded value looks like an encrypted blob
   --  (minimum length = nonce(12) + tag(16) = 28 bytes = 56 hex chars)
   function Is_Encrypted (Value : String) return Boolean;

end Adelaide_Crypto;
