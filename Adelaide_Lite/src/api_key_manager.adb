pragma SPARK_Mode (Off);

--  ============================================================================
--  Implementation of API_Key_Manager.
--
--  READS:
--    ADELAIDE_API_KEY_FILE   — path to plaintext key file (one key per line)
--    ADELAIDE_API_KEY_ENFORCE — set to "1" to enable enforcement
--
--  The key file is expected to be a simple text file with one API key per
--  line.  Blank lines and lines starting with '#' are ignored.
--  ============================================================================

with Ada.Characters.Latin_1;
with Ada.Text_IO;
with Ada.Environment_Variables;
with Ada.Strings.Fixed;

package body API_Key_Manager is

   ---------------
   -- Initialize --
   ---------------

   procedure Initialize is
      use Ada.Text_IO;
      Key_File  : constant String :=
        Ada.Environment_Variables.Value ("ADELAIDE_API_KEY_FILE", "");
      Enforce   : constant String :=
        Ada.Environment_Variables.Value ("ADELAIDE_API_KEY_ENFORCE", "0");
      F         : File_Type;
      Line_Buf  : String (1 .. 4096);
      Last      : Natural;
   begin
      --  Determine enforcement mode
      Enforcement := (Enforce = "1");

      --  Clear any previously loaded keys
      Loaded_Keys.Clear;

      --  If no key file or enforcement is off, we are done
      if Key_File'Length = 0 or else not Enforcement then
         if Enforcement then
            Put_Line ("[API_KEY] Enforcement enabled but no key file set.");
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

   ---------------------------
   -- Is_Enforcement_Enabled --
   ---------------------------

   function Is_Enforcement_Enabled return Boolean is
   begin
      return Enforcement;
   end Is_Enforcement_Enabled;

   ----------------------
   -- Validate_API_Key --
   ----------------------

   function Validate_API_Key (Key : String) return Boolean is
      UKey : constant Unbounded_String := To_Unbounded_String (Key);
   begin
      if not Enforcement then
         --  When enforcement is off, any non-empty key is accepted
         return Key'Length > 0;
      end if;
      return Loaded_Keys.Contains (UKey);
   end Validate_API_Key;

   ---------------
   -- Key_Count --
   ---------------

   function Key_Count return Natural is
   begin
      return Natural (Loaded_Keys.Length);
   end Key_Count;

end API_Key_Manager;
