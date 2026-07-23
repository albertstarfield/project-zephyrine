--  =============================================================================
--  Architectural Foundation & Security Subsystem:
--  - Cryptographic Validation: FIPS 140-3 [NIST2019FIPS1403]
--  - Hardware Constraints: DO-254 [RTCA2000DO254]
--  - Zero-Trust Posture: Mitigates catastrophic physical data breaches as 
--    modeled by [AppliedSci2025ZeroTrust, Schneier2018Click].
--  =============================================================================
pragma SPARK_Mode (Off);
-- c_binding: OpenSSL FFI for cryptographic operations

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

with Spark_Drbg; -- Force linkage of the exported C symbols

package body Adelaide_Crypto is

   use Interfaces.C;
   use Interfaces.C.Strings;

   --  ── Crypto Ready Flag ──────────────────────────────────────────────────
   Crypto_Initialized : Boolean := False;

   --  ── C FFI: adl_crypto.h wrappers ───────────────────────────────────────
   --  These map to the chars_ptr-based C wrapper functions in adl_crypto.c

   function Adl_Crypto_Init_Wrapper return int;
   pragma Import (C, Adl_Crypto_Init_Wrapper, "adl_crypto_init_wrapper");

   --  Adl_Master_Key_Available: C FFI binding to check if master key is available.
   function Adl_Master_Key_Available return int;
   pragma Import (C, Adl_Master_Key_Available, "adl_master_key_available");

   --  Adl_Is_Poisoned: C FFI binding to check if crypto is poisoned.
   function Adl_Is_Poisoned return int;
   pragma Import (C, Adl_Is_Poisoned, "adl_is_poisoned");

   --  Adl_Self_Tests_Passed: C FFI binding to check if self-tests passed.
   function Adl_Self_Tests_Passed return int;
   pragma Import (C, Adl_Self_Tests_Passed, "adl_self_tests_passed");

   --  Adl_Is_FIPS_Mode: C FFI binding to check if FIPS mode is enabled.
   function Adl_Is_FIPS_Mode return int;
   pragma Import (C, Adl_Is_FIPS_Mode, "adl_is_fips_mode");

   --  Adl_Set_FIPS_Mode: C FFI binding to enable or disable FIPS mode.
   procedure Adl_Set_FIPS_Mode (Mode : int);
   pragma Import (C, Adl_Set_FIPS_Mode, "adl_set_fips_mode");

   --  These return malloc'd strings (chars_ptr). Must be freed with Adl_Free_Cstr.
   function Adl_Derive_Subkey_Cstr
     (Context : chars_ptr) return chars_ptr;
   pragma Import (C, Adl_Derive_Subkey_Cstr, "adl_derive_subkey_cstr");

   --  Adl_Encrypt_Field_Cstr: C FFI binding to encrypt a field with AES-GCM.
   function Adl_Encrypt_Field_Cstr
     (Sub_Key  : chars_ptr;
      Plaintext : chars_ptr) return chars_ptr;
   pragma Import (C, Adl_Encrypt_Field_Cstr, "adl_encrypt_field_cstr");

   --  Adl_Decrypt_Field_Cstr: C FFI binding to decrypt a field with AES-GCM.
   function Adl_Decrypt_Field_Cstr
     (Sub_Key       : chars_ptr;
      Ciphertext_Hex : chars_ptr) return chars_ptr;
   pragma Import (C, Adl_Decrypt_Field_Cstr, "adl_decrypt_field_cstr");

   --  Adl_Free_Cstr: C FFI binding to free a C string allocated by malloc.
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

   --  Call_C_String: Calls a C function that returns a malloc'd string, with error handling.
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

   --  Call_C_String2: Calls a C function with two string arguments, with error handling.
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
         if Adl_Is_Poisoned = 1 then
            Ada.Text_IO.Put_Line ("[CRYPTO] FATAL: InferiorParadoxical anti-tamper " &
                                  "tripped on power-up. Keys zeroized. Exiting.");
            return False;
         end if;
         if Adl_Self_Tests_Passed = 1 then
            Ada.Text_IO.Put_Line ("[CRYPTO] FIPS 140-3 power-up self-tests: PASSED.");
         else
            Ada.Text_IO.Put_Line ("[CRYPTO] FATAL: FIPS 140-3 power-up self-tests: FAILED. " &
                                  "Anti-tamper engaged. Exiting.");
            return False;
         end if;
      else
         Ada.Text_IO.Put_Line ("[CRYPTO] WARNING: No master key available. " &
                               "Encryption disabled.");
         Crypto_Initialized := False;
      end if;
      return Crypto_Initialized;
   end Initialize_Crypto;

   --  Is_Crypto_Ready: Returns True if crypto is initialized and master key is available.
   function Is_Crypto_Ready return Boolean is
   begin
      return Crypto_Initialized and then Adl_Master_Key_Available = 1;
   end Is_Crypto_Ready;

   --  Is_Poisoned: Returns True if crypto is poisoned (zeroized).
   function Is_Poisoned return Boolean is
   begin
      return Adl_Is_Poisoned = 1;
   end Is_Poisoned;

   --  Self_Tests_Passed: Returns True if FIPS self-tests have passed.
   function Self_Tests_Passed return Boolean is
   begin
      return Crypto_Initialized and then Adl_Self_Tests_Passed = 1;
   end Self_Tests_Passed;

   --  Is_FIPS_Ready: Returns True if crypto is ready for FIPS operations.
   function Is_FIPS_Ready return Boolean is
   begin
      return Crypto_Initialized
         and then Adl_Master_Key_Available = 1
         and then Adl_Self_Tests_Passed = 1
         and then Adl_Is_Poisoned = 0;
   end Is_FIPS_Ready;

   --  Is_FIPS_Mode: Returns True if FIPS mode is currently enabled.
   function Is_FIPS_Mode return Boolean is
   begin
      return Adl_Is_FIPS_Mode = 1;
   end Is_FIPS_Mode;

   --  Set_FIPS_Mode: Enables or disables FIPS mode (disable only, no re-enable without restart).
   procedure Set_FIPS_Mode (Enabled : Boolean) is
   begin
      if not Enabled then
         Adl_Set_FIPS_Mode (0);
         Ada.Text_IO.Put_Line ("[CRYPTO] FIPS mode disabled (Crypto Officer override).");
      end if;
      --  If Enabled = True, this is a no-op (FIPS mode cannot be re-enabled
      --  without a process restart).
   end Set_FIPS_Mode;

   function Derive_Subkey (Context : String) return Crypto_Result is
   begin
      if not Crypto_Initialized then
         return (Success => False,
                 Data    => Null_Unbounded_String,
                 Error   => To_Unbounded_String ("Crypto not initialized"));
      end if;
      return Call_C_String (Adl_Derive_Subkey_Cstr'Access, Context);
   end Derive_Subkey;

   --  Encrypt_Field: Encrypts a field using AES-GCM with the given sub-key.
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

   --  Decrypt_Field: Decrypts a field using AES-GCM with the given sub-key.
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

   --  Try_Encrypt: Attempts encryption, falls back to plaintext on failure.
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

   --  Try_Decrypt: Attempts decryption, falls back to ciphertext on failure.
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

   --  Is_Encrypted: Returns True if the value appears to be an encrypted hex string.
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
