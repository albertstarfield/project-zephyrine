pragma SPARK_Mode (Off);

--  ============================================================================
--  API Key Manager — loads API keys from a plaintext file (one per line) and
--  validates x-api-key headers against them during request dispatch.
--
--  FIPS 140-3 §5.3.1 — Crypto Officer Role Separation
--  ---------------------------------------------------
--  Two distinct roles:
--    Crypto Officer   — administrator who can enable/disable enforcement,
--                       reload keys, and manage crypto policy. Authenticated
--                       via ADELAIDE_CRYPTO_OFFICER_KEY env var.
--    Crypto User      — regular API consumer. Validated against the loaded
--                       key file during request dispatch.
--
--  DESIGN
--  ------
--  API keys come from one of two sources (tried in order):
--    1. ADELAIDE_API_KEYS env var  — semicolon-separated keys (memory-only,
--       never touches disk. FIPS 140-3 §5.8.2 compliant).
--    2. ADELAIDE_API_KEY_FILE env var — plaintext file path (legacy).
--
--  Enforcement defaults to ON (FIPS 140-3 §5.3.2). Override via
--  ADELAIDE_API_KEY_ENFORCE env var set to "0".
--  Use Enable_Enforcement(Co_Key) / Disable_Enforcement(Co_Key) at runtime.
--
--  The key file (legacy) is written by run.py (decrypted from the encrypted
--  store at config/api_keys.enc) just before spawning the server.
--
--  USAGE
--  -----
--     API_Key_Manager.Initialize;                       -- load keys at startup
--     API_Key_Manager.Initialize_Crypto_Officer;         -- load CO key from env
--     if API_Key_Manager.Is_Crypto_Officer (key) then
--        API_Key_Manager.Enable_Enforcement (key);        -- CO only
--     end if;
--     if API_Key_Manager.Is_Enforcement_Enabled then
--        if not API_Key_Manager.Validate_API_Key (key) then
--           --  return 401
--        end if;
--     end if;
--  ============================================================================

with Ada.Containers.Ordered_Sets;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;

package API_Key_Manager is

   --  ── Initialization ──────────────────────────────────────────────────────

   --  Load the key file.  Must be called once at server startup.
   --  If ADELAIDE_API_KEY_ENFORCE is not set or is "0", enforcement
   --  remains disabled regardless of the key file contents.
   procedure Initialize;

   --  Load the Crypto Officer authentication key from the environment
   --  variable ADELAIDE_CRYPTO_OFFICER_KEY. Must be called at startup.
   --  If the env var is not set, Crypto Officer operations are unavailable.
   procedure Initialize_Crypto_Officer;

   --  ── Enforcement Control (Crypto Officer only) ───────────────────────────

   --  Return True if API key enforcement is active.
   function Is_Enforcement_Enabled return Boolean;

   --  Enable API key enforcement. Requires Crypto Officer authentication.
   --  Co_Key must match the ADELAIDE_CRYPTO_OFFICER_KEY that was loaded
   --  at startup. Returns True on success.
   function Enable_Enforcement (Co_Key : String) return Boolean;

   --  Disable API key enforcement. Requires Crypto Officer authentication.
   --  Co_Key must match the Crypto Officer key. Returns True on success.
   function Disable_Enforcement (Co_Key : String) return Boolean;

   --  Reload API keys from the key file. Requires Crypto Officer auth.
   function Reload_Keys (Co_Key : String) return Boolean;

   --  ── API Key Validation ──────────────────────────────────────────────────

   --  Validate an x-api-key value against the loaded keys (Crypto User auth).
   --  Uses constant-time comparison to prevent timing side-channels.
   function Validate_API_Key (Key : String) return Boolean;

   --  ── Utility ─────────────────────────────────────────────────────────────

   --  Number of loaded API keys (0 if file missing / empty).
   function Key_Count return Natural;

   --  Return True if the given key matches the Crypto Officer key.
   function Is_Crypto_Officer (Key : String) return Boolean;

private

   --  Ordered set of allowed API keys (Crypto Users).
   package Key_Sets is new Ada.Containers.Ordered_Sets
     (Element_Type => Unbounded_String);

   Loaded_Keys   : Key_Sets.Set;
   Enforcement   : Boolean := False;
   Co_Key        : Unbounded_String;  -- Crypto Officer auth key
   Co_Initialized : Boolean := False;

end API_Key_Manager;
