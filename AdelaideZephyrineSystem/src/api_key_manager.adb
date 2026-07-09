pragma SPARK_Mode (Off);

--  ============================================================================
--  Implementation of API_Key_Manager.
--
--  FIPS 140-3 §5.3.1 — Crypto Officer Role Separation:
--    - Crypto Officer controls enforcement (Enable/Disable/Reload)
--    - Crypto User validation uses constant-time comparison (§5.7)
--
--  READS:
--    ADELAIDE_API_KEY_FILE        — path to plaintext key file (one key per line)
--    ADELAIDE_API_KEY_ENFORCE     — set to "1" to enable enforcement
--    ADELAIDE_CRYPTO_OFFICER_KEY  — Crypto Officer authentication key
--
--  The key file is expected to be a simple text file with one API key per
--  line.  Blank lines and lines starting with '#' are ignored.
--  ============================================================================

with Ada.Text_IO;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Ada.Environment_Variables;
with FIPS_Audit;
with Ada.Strings.Fixed;
with Interfaces; use Interfaces;

package body API_Key_Manager is

   --  ── Constant-time Comparison (FIPS 140-3 §5.7) ──────────────────────────
   --  Prevents timing side-channel attacks on API key and Crypto Officer
   --  key comparisons.  Always compares every byte, even after a mismatch.
   --  Uses Unsigned_32 for bitwise XOR/OR operations (modular type).

   function Constant_Time_Compare (A, B : String) return Boolean is
      Result : Unsigned_32 :=
        Unsigned_32 (A'Length) xor Unsigned_32 (B'Length);
   begin
      for I in 1 .. A'Length loop
         declare
            A_Char : Character := (if I <= A'Length then A (A'First + I - 1) else ' ');
            B_Char : Character := (if I <= B'Length then B (B'First + I - 1) else ' ');
         begin
            Result := Result or
              (Unsigned_32 (Character'Pos (A_Char)) xor
               Unsigned_32 (Character'Pos (B_Char)));
         end;
      end loop;
      for I in 1 .. B'Length loop
         declare
            A_Char : Character := (if I <= A'Length then A (A'First + I - 1) else ' ');
            B_Char : Character := (if I <= B'Length then B (B'First + I - 1) else ' ');
         begin
            Result := Result or
              (Unsigned_32 (Character'Pos (A_Char)) xor
               Unsigned_32 (Character'Pos (B_Char)));
         end;
      end loop;
      return Result = 0;
   end Constant_Time_Compare;

   ---------------
   -- Initialize --
   ---------------

   procedure Initialize is
      use Ada.Text_IO;
      Key_File  : constant String :=
        Ada.Environment_Variables.Value ("ADELAIDE_API_KEY_FILE", "");
      Key_Env   : constant String :=
        Ada.Environment_Variables.Value ("ADELAIDE_API_KEYS", "");
      Enforce   : constant String :=
        Ada.Environment_Variables.Value ("ADELAIDE_API_KEY_ENFORCE", "1");
      F         : File_Type;
      Line_Buf  : String (1 .. 4096);
      Last      : Natural;
   begin
      --  Determine enforcement mode (default ON per FIPS 140-3 §5.3.2)
      Enforcement := (Enforce = "1");

      --  Clear any previously loaded keys
      Loaded_Keys.Clear;

      --  ── Source 1: Memory-only keys from env var (FIPS §5.8.2) ─────────────
      if Key_Env'Length > 0 then
         declare
            Start : Positive := Key_Env'First;
         begin
            for I in Key_Env'Range loop
               if Key_Env (I) = ';' then
                  declare
                     Key : constant String :=
                       Ada.Strings.Fixed.Trim (Key_Env (Start .. I - 1),
                                                Ada.Strings.Both);
                  begin
                     if Key'Length > 0 then
                        Loaded_Keys.Insert (To_Unbounded_String (Key));
                     end if;
                  end;
                  Start := I + 1;
               end if;
            end loop;
            --  Last key (or only key if no semicolons)
            declare
               Key : constant String :=
                 Ada.Strings.Fixed.Trim (Key_Env (Start .. Key_Env'Last),
                                          Ada.Strings.Both);
            begin
               if Key'Length > 0 then
                  Loaded_Keys.Insert (To_Unbounded_String (Key));
               end if;
            end;
         end;
         Put_Line ("[API_KEY] Loaded "
                   & Natural'Image (Natural (Loaded_Keys.Length))
                   & " API key(s) from environment variable"
                   & (if Enforcement then "." else " (enforcement OFF)."));
         return;
      end if;

      --  ── Source 2: Legacy key file ─────────────────────────────────────────
      if Key_File'Length = 0 then
         if Enforcement then
            Put_Line ("[API_KEY] Enforcement enabled but no keys configured. "
                      & "Set ADELAIDE_API_KEYS or ADELAIDE_API_KEY_FILE.");
         end if;
         return;
      end if;

      --  Read key file (one key per line)
      begin
         Open (F, In_File, Key_File);
         while not End_Of_File (F) loop
            Get_Line (F, Line_Buf, Last);
            declare
               Line : constant String := Line_Buf (1 .. Last);
               Trimmed : constant String := Ada.Strings.Fixed.Trim (Line, Ada.Strings.Both);
            begin
               --  Skip blanks and comments
               if Trimmed'Length > 0 and then Trimmed (Trimmed'First) /= '#' then
                  Loaded_Keys.Insert (To_Unbounded_String (Trimmed));
               end if;
            end;
         end loop;
         Close (F);

         Put_Line ("[API_KEY] Loaded "
                   & Natural'Image (Natural (Loaded_Keys.Length))
                   & " API key(s) from "
                   & Key_File);
      exception
         when others =>
            Put_Line ("[API_KEY] WARNING: Could not read key file: " & Key_File);
            Loaded_Keys.Clear;
      end;
   end Initialize;

   ------------------------------
   -- Initialize_Crypto_Officer --
   ------------------------------

   procedure Initialize_Crypto_Officer is
      use Ada.Text_IO;
   begin
      declare
         Co_Key_Str : constant String :=
           Ada.Environment_Variables.Value ("ADELAIDE_CRYPTO_OFFICER_KEY", "");
      begin
         if Co_Key_Str'Length = 0 then
            Put_Line ("[API_KEY] Crypto Officer key not set. "
                      & "Crypto Officer operations unavailable.");
            Co_Initialized := False;
            return;
         end if;

         Co_Key := To_Unbounded_String (Co_Key_Str);
         Co_Initialized := True;
         Put_Line ("[API_KEY] Crypto Officer key loaded.");
         FIPS_Audit.Log_Event ("Crypto Officer key initialized successfully.");
      end;
   end Initialize_Crypto_Officer;

   ---------------------------
   -- Is_Enforcement_Enabled --
   ---------------------------

   function Is_Enforcement_Enabled return Boolean is
   begin
      return Enforcement;
   end Is_Enforcement_Enabled;

   -------------------------
   -- Enable_Enforcement --
   -------------------------

   function Enable_Enforcement (Co_Key : String) return Boolean is
      use Ada.Text_IO;
   begin
      if not Co_Initialized then
         Put_Line ("[API_KEY] Cannot enable enforcement: "
                   & "Crypto Officer not initialized.");
         return False;
      end if;

      if not Constant_Time_Compare (Co_Key, To_String (API_Key_Manager.Co_Key)) then
         Put_Line ("[API_KEY] Unauthorized attempt to enable enforcement.");
         return False;
      end if;

      Enforcement := True;
      Put_Line ("[API_KEY] Enforcement enabled by Crypto Officer.");
      return True;
   end Enable_Enforcement;

   --------------------------
   -- Disable_Enforcement --
   --------------------------

   function Disable_Enforcement (Co_Key : String) return Boolean is
      use Ada.Text_IO;
   begin
      if not Co_Initialized then
         Put_Line ("[API_KEY] Cannot disable enforcement: "
                   & "Crypto Officer not initialized.");
         return False;
      end if;

      if not Constant_Time_Compare (Co_Key, To_String (API_Key_Manager.Co_Key)) then
         Put_Line ("[API_KEY] Unauthorized attempt to disable enforcement.");
         return False;
      end if;

      Enforcement := False;
      Put_Line ("[API_KEY] Enforcement disabled by Crypto Officer.");
      return True;
   end Disable_Enforcement;

   -----------------
   -- Reload_Keys --
   -----------------

   function Reload_Keys (Co_Key : String) return Boolean is
      use Ada.Text_IO;
   begin
      if not Co_Initialized then
         Put_Line ("[API_KEY] Cannot reload keys: "
                   & "Crypto Officer not initialized.");
         return False;
      end if;

      if not Constant_Time_Compare (Co_Key, To_String (API_Key_Manager.Co_Key)) then
         Put_Line ("[API_KEY] Unauthorized attempt to reload keys.");
         return False;
      end if;

      --  Re-read the key file (same logic as Initialize)
      declare
         Key_File : constant String :=
           Ada.Environment_Variables.Value ("ADELAIDE_API_KEY_FILE", "");
         F        : File_Type;
         Line_Buf : String (1 .. 4096);
         Last     : Natural;
      begin
         Loaded_Keys.Clear;

         if Key_File'Length = 0 then
            Put_Line ("[API_KEY] Reload: no key file set.");
            return True;
         end if;

         Open (F, In_File, Key_File);
         while not End_Of_File (F) loop
            Get_Line (F, Line_Buf, Last);
            declare
               Line    : constant String := Line_Buf (1 .. Last);
               Trimmed : constant String :=
                 Ada.Strings.Fixed.Trim (Line, Ada.Strings.Both);
            begin
               if Trimmed'Length > 0 and then Trimmed (Trimmed'First) /= '#' then
                  Loaded_Keys.Insert (To_Unbounded_String (Trimmed));
               end if;
            end;
         end loop;
         Close (F);

         Put_Line ("[API_KEY] Reloaded "
                   & Natural'Image (Natural (Loaded_Keys.Length))
                   & " API key(s) from "
                   & Key_File);
         return True;

      exception
         when others =>
            Put_Line ("[API_KEY] Reload failed: could not read key file.");
            return False;
      end;
   end Reload_Keys;

   ----------------------
   -- Validate_API_Key --
   ----------------------

   function Validate_API_Key (Key : String) return Boolean is
   begin
      if not Enforcement then
         --  When enforcement is off, ALL requests pass through
         --  (backward compatible with Ollama/OpenWebUI clients)
         return True;
      end if;

      --  Constant-time scan through all keys (FIPS 140-3 §5.7),
      --  always checks every key, even after finding a match.
      declare
         Found : Boolean := False;
      begin
         for Cursor in Loaded_Keys.Iterate loop
             declare
                Stored_Key : constant String :=
                  To_String (Key_Sets.Element (Cursor));
             begin
                if Constant_Time_Compare (Key, Stored_Key) then
                   Found := True;
                end if;
             end;
          end loop;

         if not Found then
            FIPS_Audit.Log_Event ("Failed API Key authentication attempt.");
         end if;
         return Found;
      end;
   end Validate_API_Key;

   ---------------------
   -- Is_Crypto_Officer --
   ---------------------

   function Is_Crypto_Officer (Key : String) return Boolean is
   begin
      if not Co_Initialized then
         return False;
      end if;
      return Constant_Time_Compare (Key, To_String (Co_Key));
   end Is_Crypto_Officer;

   ---------------
   -- Key_Count --
   ---------------

   function Key_Count return Natural is
   begin
      return Natural (Loaded_Keys.Length);
   end Key_Count;

end API_Key_Manager;
