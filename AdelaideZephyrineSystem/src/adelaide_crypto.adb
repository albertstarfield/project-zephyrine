pragma SPARK_Mode (Off);

--  ── Adelaide Crypto Wrapper Implementation ─────────────────────────────────
--  Wraps the C adl_crypto shim (AES-256-GCM + HKDF) for Ada FFI.
--
--  The C shim handles master key loading and encryption operations.
--  This package provides Ada-friendly wrappers with proper string conversion.
--
--  THREAD SAFETY: The C shim's adl_init() is NOT thread-safe (called once at
--  startup in Initialize). All subsequent operations are reentrant.
--  ────────────────────────────────────────────────────────────────────────────

with Interfaces.C.Strings;
with Ada.Exceptions;
with Ada.Text_IO;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;

package body Adelaide_Crypto is

   use Interfaces.C;
   use Interfaces.C.Strings;

   --  ── Crypto Ready Flag ──────────────────────────────────────────────────
   Crypto_Initialized : Boolean := False;

   --  ── C FFI: adl_crypto.h wrappers ───────────────────────────────────────
   --  These map to the chars_ptr-based C wrapper functions in adl_crypto.c

   function Adl_Crypto_Init_Wrapper return int;
   pragma Import (C, Adl_Crypto_Init_Wrapper, "adl_crypto_init_wrapper");

   function Adl_Master_Key_Available return int;
   pragma Import (C, Adl_Master_Key_Available, "adl_master_key_available");

   --  These return malloc'd strings (chars_ptr). Must be freed with Adl_Free_Cstr.
   function Adl_Derive_Subkey_Cstr
     (Context : chars_ptr) return chars_ptr;
   pragma Import (C, Adl_Derive_Subkey_Cstr, "adl_derive_subkey_cstr");

   function Adl_Encrypt_Field_Cstr
     (Sub_Key  : chars_ptr;
      Plaintext : chars_ptr) return chars_ptr;
   pragma Import (C, Adl_Encrypt_Field_Cstr, "adl_encrypt_field_cstr");

   function Adl_Decrypt_Field_Cstr
     (Sub_Key       : chars_ptr;
      Ciphertext_Hex : chars_ptr) return chars_ptr;
   pragma Import (C, Adl_Decrypt_Field_Cstr, "adl_decrypt_field_cstr");

   procedure Adl_Free_Cstr (Ptr : chars_ptr);
   pragma Import (C, Adl_Free_Cstr, "adl_free_cstr");

   --  ── Internal: Call a C wrapper that returns a malloc'd chars_ptr ───────
   --  Takes one or two Ada String inputs, converts to chars_ptr, calls C,
   --  converts result back to Ada String, frees C allocations.
   --  Returns Success = False on any failure.

   type C_String_Func is access function (Arg1 : chars_ptr) return chars_ptr;
   pragma Convention (C, C_String_Func);

   type C_String2_Func is access function (Arg1 : chars_ptr; Arg2 : chars_ptr) return chars_ptr;
   pragma Convention (C, C_String2_Func);

   function Call_C_String
     (Fn         : C_String_Func;
      Arg1       : String) return Crypto_Result
   is
      C_Arg1 : chars_ptr := New_String (Arg1);
      C_Res  : chars_ptr;
   begin
      C_Res := Fn (C_Arg1);
      Free (C_Arg1);
      if C_Res = Null_Ptr then
         return (Success => False, others => <>);
      end if;
      declare
         Ada_Res : constant String := Value (C_Res);
      begin
         Adl_Free_Cstr (C_Res);
         return (Success => True,
                 Data    => To_Unbounded_String (Ada_Res),
                 Error   => Null_Unbounded_String);
      end;
   exception
      when E : others =>
         if C_Res /= Null_Ptr then
            Adl_Free_Cstr (C_Res);
         end if;
         return (Success => False,
                 Data    => Null_Unbounded_String,
                 Error   => To_Unbounded_String (Ada.Exceptions.Exception_Message (E)));
   end Call_C_String;

   function Call_C_String2
     (Fn         : C_String2_Func;
      Arg1, Arg2 : String) return Crypto_Result
   is
      C_Arg1 : chars_ptr := New_String (Arg1);
      C_Arg2 : chars_ptr := New_String (Arg2);
      C_Res  : chars_ptr;
   begin
      C_Res := Fn (C_Arg1, C_Arg2);
      Free (C_Arg1);
      Free (C_Arg2);
      if C_Res = Null_Ptr then
         return (Success => False, others => <>);
      end if;
      declare
         Ada_Res : constant String := Value (C_Res);
      begin
         Adl_Free_Cstr (C_Res);
         return (Success => True,
                 Data    => To_Unbounded_String (Ada_Res),
                 Error   => Null_Unbounded_String);
      end;
   exception
      when E : others =>
         if C_Res /= Null_Ptr then
            Adl_Free_Cstr (C_Res);
         end if;
         return (Success => False,
                 Data    => Null_Unbounded_String,
                 Error   => To_Unbounded_String (Ada.Exceptions.Exception_Message (E)));
   end Call_C_String2;

   --  ── Public API ─────────────────────────────────────────────────────────

   function Initialize_Crypto return Boolean is
   begin
      if Crypto_Initialized then
         return True;
      end if;

      if Adl_Crypto_Init_Wrapper = 0 then
         Crypto_Initialized := True;
         Ada.Text_IO.Put_Line ("[CRYPTO] Master key loaded successfully.");
      else
         Ada.Text_IO.Put_Line ("[CRYPTO] WARNING: No master key available. " &
                               "Encryption disabled.");
         Crypto_Initialized := False;
      end if;
      return Crypto_Initialized;
   end Initialize_Crypto;

   function Is_Crypto_Ready return Boolean is
   begin
      return Crypto_Initialized and then Adl_Master_Key_Available = 1;
   end Is_Crypto_Ready;

   function Derive_Subkey (Context : String) return Crypto_Result is
   begin
      if not Crypto_Initialized then
         return (Success => False,
                 Data    => Null_Unbounded_String,
                 Error   => To_Unbounded_String ("Crypto not initialized"));
      end if;
      return Call_C_String (Adl_Derive_Subkey_Cstr'Access, Context);
   end Derive_Subkey;

   function Encrypt_Field
     (Sub_Key_Hex : String;
      Plaintext   : String) return Crypto_Result
   is
   begin
      if not Crypto_Initialized then
         return (Success => False,
                 Data    => Null_Unbounded_String,
                 Error   => To_Unbounded_String ("Crypto not initialized"));
      end if;
      if Plaintext'Length = 0 then
         return (Success => True, Data => Null_Unbounded_String, Error => Null_Unbounded_String);
      end if;
      return Call_C_String2 (Adl_Encrypt_Field_Cstr'Access, Sub_Key_Hex, Plaintext);
   end Encrypt_Field;

   function Decrypt_Field
     (Sub_Key_Hex   : String;
      Ciphertext_Hex : String) return Crypto_Result
   is
   begin
      if not Crypto_Initialized then
         return (Success => False,
                 Data    => Null_Unbounded_String,
                 Error   => To_Unbounded_String ("Crypto not initialized"));
      end if;
      if Ciphertext_Hex'Length = 0 then
         return (Success => True, Data => Null_Unbounded_String, Error => Null_Unbounded_String);
      end if;
      return Call_C_String2 (Adl_Decrypt_Field_Cstr'Access, Sub_Key_Hex, Ciphertext_Hex);
   end Decrypt_Field;

   function Try_Encrypt
     (Sub_Key_Hex : String;
      Plaintext   : String) return String
   is
      Res : constant Crypto_Result := Encrypt_Field (Sub_Key_Hex, Plaintext);
   begin
      if Res.Success and then Res.Data /= Null_Unbounded_String then
         return To_String (Res.Data);
      end if;
      --  WARNING: Encryption failed, storing plaintext!
      Ada.Text_IO.Put_Line (Ada.Text_IO.Standard_Error,
        "[CRYPTO] WARNING: Encryption failed for field (" &
        Positive'Image (Plaintext'Length) & " bytes). " &
        "Data stored in PLAINTEXT. Error: " &
        (if Res.Error /= Null_Unbounded_String then To_String (Res.Error)
         else "unknown"));
      return Plaintext;  -- fallback (best effort)
   end Try_Encrypt;

   function Try_Decrypt
     (Sub_Key_Hex   : String;
      Ciphertext_Hex : String) return String
   is
      Res : constant Crypto_Result := Decrypt_Field (Sub_Key_Hex, Ciphertext_Hex);
   begin
      if Res.Success and then Res.Data /= Null_Unbounded_String then
         return To_String (Res.Data);
      end if;
      --  WARNING: Decryption failed, returning ciphertext!
      Ada.Text_IO.Put_Line (Ada.Text_IO.Standard_Error,
        "[CRYPTO] WARNING: Decryption failed for field (" &
        Positive'Image (Ciphertext_Hex'Length) & " hex chars). " &
        "Returning raw ciphertext. Error: " &
        (if Res.Error /= Null_Unbounded_String then To_String (Res.Error)
         else "unknown"));
      return Ciphertext_Hex;  -- fallback (best effort)
   end Try_Decrypt;

   function Is_Encrypted (Value : String) return Boolean is
      --  Minimum encrypted blob = nonce(12) + tag(16) = 28 bytes = 56 hex chars
      Min_Hex_Length : constant Natural := 28 * 2;  -- 56
   begin
      if Value'Length < Min_Hex_Length then
         return False;
      end if;
      --  Check it's valid lowercase hex
      for I in Value'Range loop
         case Value (I) is
            when '0' .. '9' | 'a' .. 'f' =>
               null;
            when others =>
               return False;
         end case;
      end loop;
      return True;
   end Is_Encrypted;

end Adelaide_Crypto;
