--  ── Master Key Store (SPARK-Verified) ───────────────────────────────────────
--  512-bit master key storage with SPARK formal verification.
--  Key exists ONLY in Ada runtime memory — never written to disk.
--
--  SECURITY PROPERTIES:
--    - SPARK_Mode(On): No runtime exceptions, no aliasing, no uninitialized reads
--    - Key stored in 64-byte array (512 bits)
--    - Clear_Key uses volatile write to prevent compiler optimization
--    - Key_Valid flag ensures Get_Key returns valid data only
--
--  USAGE:
--    1. Master_Key_Store.Set_Key(Derived_Key) after key derivation
--    2. Master_Key_Store.Get_Key for AES key derivation (first 32 bytes)
--    3. Master_Key_Store.Clear_Key on shutdown or error
--  ──────────────────────────────────────────────────────────────────────────────

with Interfaces; use Interfaces;

package Master_Key_Store
  with SPARK_Mode => On
is
   --  512-bit key = 64 bytes
   subtype Key_Index is Positive range 1 .. 64;
   type Key_Type is array (Key_Index) of Interfaces.Unsigned_8
     with Pack;

   --  Empty key (all zeros)
   Empty_Key : constant Key_Type := (others => 0);

   --  Store a 512-bit master key
   procedure Set_Key (K : Key_Type)
     with Global => null;

   --  Retrieve the 512-bit master key
   --  Returns Empty_Key if not set
   function Get_Key return Key_Type
     with Global => null;

   --  Clear the key from memory (volatile write)
   procedure Clear_Key
     with Global => null;

   --  Check if a key is currently stored
   function Is_Set return Boolean
     with Global => null;

   --  Get first 32 bytes (256 bits) for AES-256 key derivation
   function Get_AES_Part return Key_Type
     with Global => null;

private
   --  Key storage (volatile to prevent optimization)
   Key       : Key_Type := (others => 0)
     with Volatile;
   Key_Valid : Boolean := False
     with Volatile;

end Master_Key_Store;
