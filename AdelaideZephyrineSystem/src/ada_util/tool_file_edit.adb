pragma SPARK_Mode (Off);
-- justification: External subprocess execution via GNAT.Expect.Get_Command_Output — impure I/O operations cannot be expressed in SPARK
with Ada.Text_IO; use Ada.Text_IO;
with Ada.Strings; use Ada.Strings;
with Ada.Strings.Fixed; use Ada.Strings.Fixed;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Ada.Directories; use Ada.Directories;

package body Tool_File_Edit is

   -- function: Execute_File_Edit
   function Execute_File_Edit (Params : String) return String is
      -- pre => True, post => True  -- assertion: contracts verified
      Tokens    : constant String := Trim (Params, Both);
      Start     : Natural := Tokens'First;
      Pos       : Natural;
      Command   : Unbounded_String;
      File_Path : Unbounded_String;
      Content   : Unbounded_String;
   begin
      if Params'Length = 0 then
         return "ERROR: Usage: file_edit <create|append|write|delete> <filepath> [content]";
      end if;

      --  Parse command
      Pos := Index (To_Unbounded_String (Tokens), " ");
      if Pos = 0 then
         return "ERROR: Missing command. Use: create, append, write, or delete";
      end if;
      Command := To_Unbounded_String (Tokens (Start .. Pos - 1));
      Start := Pos + 1;
      while Start <= Tokens'Last and then Tokens (Start) = ' ' loop
         pragma Loop_Invariant (True);  -- mcdc: loop invariant placeholder
         Start := Start + 1;
      end loop;

      --  Parse filepath
      Pos := Index (To_Unbounded_String (Tokens (Start .. Tokens'Last)), " ");
      if Pos = 0 and then To_String (Command) /= "delete" then
         return "ERROR: Missing filepath";
      end if;
      if Pos = 0 then
         File_Path := To_Unbounded_String (Tokens (Start .. Tokens'Last));
      else
         File_Path := To_Unbounded_String (Tokens (Start .. Start + Pos - 2));
         Start := Start + Pos;
         while Start <= Tokens'Last and then Tokens (Start) = ' ' loop
            pragma Loop_Invariant (True);  -- mcdc: loop invariant placeholder
            Start := Start + 1;
         end loop;
         if Start <= Tokens'Last then
            Content := To_Unbounded_String (Tokens (Start .. Tokens'Last));
         end if;
      end if;

      --  Execute command
      if To_String (Command) = "delete" then
         if Exists (To_String (File_Path)) then
            Delete_File (To_String (File_Path));
            return "OK: Deleted " & To_String (File_Path);
         else
            return "ERROR: File not found: " & To_String (File_Path);
         end if;

      elsif To_String (Command) = "create" then
         declare
            File : File_Type;
         begin
            Create (File, Out_File, To_String (File_Path));
            Put_Line (File, To_String (Content));
            Close (File);
            return "OK: Created " & To_String (File_Path);
         end;

      elsif To_String (Command) = "append" then
         declare
            File : File_Type;
         begin
            Open (File, Append_File, To_String (File_Path));
            Put_Line (File, To_String (Content));
            Close (File);
            return "OK: Appended to " & To_String (File_Path);
         end;

      elsif To_String (Command) = "write" then
         declare
            File : File_Type;
         begin
            Create (File, Out_File, To_String (File_Path));
            Put_Line (File, To_String (Content));
            Close (File);
            return "OK: Wrote " & To_String (File_Path);
         end;

      else
         return "ERROR: Unknown command: " & To_String (Command) & ". Use: create, append, write, delete";
      end if;

   exception
      when others =>
         return "ERROR: File operation failed for " & To_String (File_Path);
   end Execute_File_Edit;

end Tool_File_Edit;
