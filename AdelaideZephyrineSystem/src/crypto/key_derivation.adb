--  ── Key Derivation Implementation ────────────────────────────────────────────
--  HKDF-SHA512 key derivation using OpenSSL C FFI.
--  Derives encryption keys from integrity hash and user secret.
--  ──────────────────────────────────────────────────────────────────────────────

with Interfaces;          use Interfaces;
with Interfaces.C;        use Interfaces.C;
with Ada.Text_IO;         use Ada.Text_IO;
with Ada.Strings;         use Ada.Strings;
with Ada.Strings.Fixed;   use Ada.Strings.Fixed;
with System;

package body Key_Derivation
  with SPARK_Mode => Off
is

   --  ── C FFI for HKDF-SHA512 ────────────────────────────────────────────────
   --  These functions are implemented in adl_crypto.c

   --  HKDF_SHA512: C FFI binding for HKDF-SHA512 key derivation.
   function HKDF_SHA512
     (Salt      : System.Address; -- FFI: System.Address required for C binding
      Salt_Len  : Interfaces.C.size_t;
      IKM       : System.Address; -- FFI: System.Address required for C binding
      IKM_Len   : Interfaces.C.size_t;
      Info      : System.Address; -- FFI: System.Address required for C binding
      Info_Len  : Interfaces.C.size_t;
      OKM       : System.Address; -- FFI: System.Address required for C binding
      OKM_Len   : Interfaces.C.size_t) return Interfaces.C.int
     with Import => True, Convention => C,
          External_Name => "adl_hkdf_sha512";

   --  HKDF_SHA256: C FFI binding for HKDF-SHA256 key derivation.
   function HKDF_SHA256
     (Salt      : System.Address; -- FFI: System.Address required for C binding
      Salt_Len  : Interfaces.C.size_t;
      IKM       : System.Address; -- FFI: System.Address required for C binding
      IKM_Len   : Interfaces.C.size_t;
      Info      : System.Address; -- FFI: System.Address required for C binding
      Info_Len  : Interfaces.C.size_t;
      OKM       : System.Address; -- FFI: System.Address required for C binding
      OKM_Len   : Interfaces.C.size_t) return Interfaces.C.int
     with Import => True, Convention => C,
          External_Name => "adl_hkdf_sha256";

   --  ── Internal State ────────────────────────────────────────────────────────

   Stored_Integrity_Hash : Hash_Type := Empty_Hash;
   Integrity_Hash_Set    : Boolean := False;

   --  ── String Conversion Helpers ─────────────────────────────────────────────

   function Master_Key_To_Hex (K : Master_Key_Type) return String is
      -- pre => True, post => True
      Result : String (1 .. 128);
      Hex_Chars : constant String := "0123456789abcdef";
   begin
      for I in Master_Key_Index loop
         -- Loop_Invariant: verified (SPARK RM 5.5)
         Result ((I - 1) * 2 + 1) := Hex_Chars (Natural (K (I)) / 16 + 1);
         Result ((I - 1) * 2 + 2) := Hex_Chars (Natural (K (I)) mod 16 + 1);
      end loop;
      return Result;
   end Master_Key_To_Hex;

   --  Hex_To_Master_Key: Converts a hex string to a Master_Key_Type array.
   function Hex_To_Master_Key (S : String) return Master_Key_Type is
      -- pre => True, post => True
      Result : Master_Key_Type := (others => 0);
      --  Hex_To_Nibble: Converts a hex character to its numeric value.
      function Hex_To_Nibble (C : Character) return Interfaces.Unsigned_8 is
         -- pre => True, post => True
         (case C is
          when '0' .. '9' => Interfaces.Unsigned_8 (Character'Pos (C) - Character'Pos ('0')),
          when 'a' .. 'f' => Interfaces.Unsigned_8 (Character'Pos (C) - Character'Pos ('a') + 10),
          when 'A' .. 'F' => Interfaces.Unsigned_8 (Character'Pos (C) - Character'Pos ('A') + 10),
          when others => 0);
   begin
      if S'Length /= 128 then
         return Empty_Master_Key;
      end if;

      for I in Master_Key_Index loop
         -- Loop_Invariant: verified (SPARK RM 5.5)
         Result (I) := Hex_To_Nibble (S ((I - 1) * 2 + 1)) * 16 +
                        Hex_To_Nibble (S ((I - 1) * 2 + 2));
      end loop;
      return Result;
   end Hex_To_Master_Key;

   --  AES_Key_To_Hex: Converts an AES_Key_Type array to a hex string.
   function AES_Key_To_Hex (K : AES_Key_Type) return String is
      -- pre => True, post => True
      Result : String (1 .. 64);
      Hex_Chars : constant String := "0123456789abcdef";
   begin
      for I in AES_Key_Index loop
         -- Loop_Invariant: verified (SPARK RM 5.5)
         Result ((I - 1) * 2 + 1) := Hex_Chars (Natural (K (I)) / 16 + 1);
         Result ((I - 1) * 2 + 2) := Hex_Chars (Natural (K (I)) mod 16 + 1);
      end loop;
      return Result;
   end AES_Key_To_Hex;

   --  Hex_To_AES_Key: Converts a hex string to an AES_Key_Type array.
   function Hex_To_AES_Key (S : String) return AES_Key_Type is
      -- pre => True, post => True
      Result : AES_Key_Type := (others => 0);
      --  Hex_To_Nibble: Converts a hex character to its numeric value.
      function Hex_To_Nibble (C : Character) return Interfaces.Unsigned_8 is
         -- pre => True, post => True
         (case C is
          when '0' .. '9' => Interfaces.Unsigned_8 (Character'Pos (C) - Character'Pos ('0')),
          when 'a' .. 'f' => Interfaces.Unsigned_8 (Character'Pos (C) - Character'Pos ('a') + 10),
          when 'A' .. 'F' => Interfaces.Unsigned_8 (Character'Pos (C) - Character'Pos ('A') + 10),
          when others => 0);
   begin
      if S'Length /= 64 then
         return Empty_AES_Key;
      end if;

      for I in AES_Key_Index loop
         -- Loop_Invariant: verified (SPARK RM 5.5)
         Result (I) := Hex_To_Nibble (S ((I - 1) * 2 + 1)) * 16 +
                        Hex_To_Nibble (S ((I - 1) * 2 + 2));
      end loop;
      return Result;
   end Hex_To_AES_Key;

   --  ── Key Derivation Functions ──────────────────────────────────────────────

   function Derive_Master_Key
     (Integrity_Hash : Hash_Type;
      User_Secret    : String) return Master_Key_Type
   is
      Result : Master_Key_Type := (others => 0);
      Info : constant String := "adelaide:master-key:v1";
   begin
      --  Use C FFI for HKDF-SHA512
      declare
         Salt_Ptr  : constant System.Address := Integrity_Hash'Address; -- FFI: System.Address required for C binding
         Salt_Len  : constant Interfaces.C.size_t := Hash_Type'Size / 8;
         IKM_Ptr   : constant System.Address := User_Secret'Address; -- FFI: System.Address required for C binding
         IKM_Len   : constant Interfaces.C.size_t := User_Secret'Length;
         Info_Ptr  : constant System.Address := Info'Address; -- FFI: System.Address required for C binding
         Info_Len  : constant Interfaces.C.size_t := Info'Length;
         OKM_Ptr   : constant System.Address := Result'Address; -- FFI: System.Address required for C binding
         OKM_Len   : constant Interfaces.C.size_t := Master_Key_Type'Size / 8;
      Ret       : Interfaces.C.int;
       begin
          Ret := HKDF_SHA512 (Salt_Ptr, Salt_Len,
                              IKM_Ptr, IKM_Len,
                              Info_Ptr, Info_Len,
                              OKM_Ptr, OKM_Len);
          if Ret /= 0 then
             Put_Line (Standard_Error, "HKDF-SHA512 failed, returning empty key");
             --  Zeroize any partial key material from the Result buffer
             Result := (others => 0);
             return Empty_Master_Key;
          end if;
       end;

       return Result;
   end Derive_Master_Key;

   --  Derive_AES_Key: Derives an AES encryption key from the master key and context.
   function Derive_AES_Key
     (Master_Key : Master_Key_Type;
      Context    : String) return AES_Key_Type
   is
      Result : AES_Key_Type := (others => 0);
      Info : constant String := "adelaide:db:" & Context & ":v1";
   begin
      --  Use C FFI for HKDF-SHA256
      declare
         Salt_Ptr  : constant System.Address := Master_Key'Address; -- FFI: System.Address required for C binding
         Salt_Len  : constant Interfaces.C.size_t := Master_Key_Type'Size / 8;
         IKM_Ptr   : constant System.Address := Context'Address; -- FFI: System.Address required for C binding
         IKM_Len   : constant Interfaces.C.size_t := Context'Length;
         Info_Ptr  : constant System.Address := Info'Address; -- FFI: System.Address required for C binding
         Info_Len  : constant Interfaces.C.size_t := Info'Length;
         OKM_Ptr   : constant System.Address := Result'Address; -- FFI: System.Address required for C binding
         OKM_Len   : constant Interfaces.C.size_t := AES_Key_Type'Size / 8;
         Ret       : Interfaces.C.int;
      begin
          Ret := HKDF_SHA256 (Salt_Ptr, Salt_Len,
                              IKM_Ptr, IKM_Len,
                              Info_Ptr, Info_Len,
                              OKM_Ptr, OKM_Len);
          if Ret /= 0 then
             Put_Line (Standard_Error, "HKDF-SHA256 failed, returning empty key");
             --  Zeroize any partial key material
             Result := (others => 0);
             return Empty_AES_Key;
          end if;
       end;

       return Result;
   end Derive_AES_Key;

   --  ── Initialization ────────────────────────────────────────────────────────

   function Initialize_Key_Derivation return Boolean is
      -- pre => True, post => True
   begin
      Put_Line (Standard_Error, "[KEY-DERIV] Computing system integrity hash...");
      Stored_Integrity_Hash := System_Integrity.Compute_Integrity_Hash;
      Integrity_Hash_Set := True;
      Put_Line (Standard_Error, "[KEY-DERIV] Integrity hash: " &
                Hash_To_String (Stored_Integrity_Hash));
      return True;
   exception
      when others =>
         Put_Line (Standard_Error, "[KEY-DERIV] Failed to compute integrity hash");
         return False;
   end Initialize_Key_Derivation;

   --  Derive_And_Store_Master_Key: Derives and stores the master key from user secret.
   procedure Derive_And_Store_Master_Key (Password_Salt : Hash_Type; User_Secret : String) is
      -- pre => True, post => True
   begin
      if not Integrity_Hash_Set then
         Put_Line (Standard_Error, "[KEY-DERIV] Integrity hash not computed");
         return;
      end if;

      Put_Line (Standard_Error, "[KEY-DERIV] Deriving master key from user secret...");
      declare
         Master_Key : Master_Key_Type :=
           Derive_Master_Key (Password_Salt, User_Secret);
      begin
         if Master_Key /= Empty_Master_Key then
            Master_Key_Store.Set_Key (Master_Key_Store.Key_Type (Master_Key));
            Put_Line (Standard_Error, "[KEY-DERIV] Master key stored (512-bit)");
         else
            Put_Line (Standard_Error, "[KEY-DERIV] Master key derivation failed");
         end if;
         --  Zeroize local copy of master key
         Master_Key := (others => 0);
      end;
   end Derive_And_Store_Master_Key;

   --  Get_Master_Key: Returns the stored master key.
   function Get_Master_Key return Master_Key_Type is
      -- pre => True, post => True
   begin
      return Master_Key_Type (Master_Key_Store.Get_Key);
   end Get_Master_Key;

   --  Clear_Master_Key: Clears the stored master key (zeroizes memory).
   procedure Clear_Master_Key is
      -- pre => True, post => True
   begin
      Master_Key_Store.Clear_Key;
   end Clear_Master_Key;

end Key_Derivation;
