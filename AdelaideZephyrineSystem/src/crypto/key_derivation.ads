--  ── Key Derivation (HKDF-SHA512) ────────────────────────────────────────────
--  Derives encryption keys from integrity hash and user secret.
--  Uses HKDF-SHA512 (RFC 5869) for key derivation.
--
--  KEY DERIVATION CHAIN:
--    1. integrity_hash = SHA512(hw_hash || binary_hash) from System_Integrity
--    2. master_key = HKDF-SHA512(salt=integrity_hash, ikm=user_secret,
--                                info="adelaide:master-key:v1")
--    3. aes_key = HKDF-SHA256(salt=master_key, ikm=context_string,
--                             info="adelaide:db:memory:v1")
--
--  SECURITY PROPERTIES:
--    - HKDF-SHA512 provides 512-bit derived key
--    - Integrity hash binds to hardware state
--    - User secret provides authentication factor
--    - Context string binds to specific database
--  ──────────────────────────────────────────────────────────────────────────────

with Interfaces;          use Interfaces;
with System_Integrity;    use System_Integrity;
with Master_Key_Store;    use Master_Key_Store;

package Key_Derivation
  with SPARK_Mode => Off  --  Requires C FFI for HKDF
is
   --  Master key size (512 bits = 64 bytes)
   subtype Master_Key_Index is Positive range 1 .. 64;
   type Master_Key_Type is array (Master_Key_Index) of Interfaces.Unsigned_8
     with Pack;

   --  AES key size (256 bits = 32 bytes)
   subtype AES_Key_Index is Positive range 1 .. 32;
   type AES_Key_Type is array (AES_Key_Index) of Interfaces.Unsigned_8
     with Pack;

   --  Empty keys
   Empty_Master_Key : constant Master_Key_Type := (others => 0);
   Empty_AES_Key    : constant AES_Key_Type := (others => 0);

   --  Derive master key from integrity hash and user secret
   --  master_key = HKDF-SHA512(salt=integrity_hash, ikm=user_secret,
   --                           info="adelaide:master-key:v1")
   function Derive_Master_Key
     (Integrity_Hash : Hash_Type;
      User_Secret    : String) return Master_Key_Type;

   --  Derive AES-256 key for specific database context
   --  aes_key = HKDF-SHA256(salt=master_key, ikm=context,
   --                        info="adelaide:db:" & context & ":v1")
   function Derive_AES_Key
     (Master_Key : Master_Key_Type;
      Context    : String) return AES_Key_Type;

   --  Convert Master_Key_Type to hex string (128 hex chars)
   function Master_Key_To_Hex (K : Master_Key_Type) return String;

   --  Convert hex string to Master_Key_Type
   function Hex_To_Master_Key (S : String) return Master_Key_Type;

   --  Convert AES_Key_Type to hex string (64 hex chars)
   function AES_Key_To_Hex (K : AES_Key_Type) return String;

   --  Convert hex string to AES_Key_Type
   function Hex_To_AES_Key (S : String) return AES_Key_Type;

   --  Initialize key derivation (compute integrity hash and store)
   --  Returns True if initialization succeeded
   function Initialize_Key_Derivation return Boolean;

   --  Derive and store master key from user secret
   --  Uses stored integrity hash
   procedure Derive_And_Store_Master_Key (Password_Salt : Hash_Type; User_Secret : String);

   --  Get current master key (from Master_Key_Store)
   function Get_Master_Key return Master_Key_Type;

   --  Clear master key from memory
   procedure Clear_Master_Key;

end Key_Derivation;
