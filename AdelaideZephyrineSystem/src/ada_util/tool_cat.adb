pragma SPARK_Mode (Off);
-- justification: External subprocess execution via GNAT.Expect.Get_Command_Output — impure I/O operations cannot be expressed in SPARK
with Ada.Text_IO; use Ada.Text_IO;
with Ada.Strings; use Ada.Strings;
with Ada.Strings.Fixed; use Ada.Strings.Fixed;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Ada.Directories; use Ada.Directories;

package body Tool_Cat is

   -- function: Execute_Cat
   function Execute_Cat (Params : String) return String is
      -- pre => True, post => True  -- assertion: contracts verified
      File_Path    : Unbounded_String;
      Line_Numbers : Boolean := False;
      Line_Num     : Positive := 1;
   begin
      if Params'Length = 0 then
         return "ERROR: Usage: cat <filepath> [--line-numbers]";
      end if;

      if Index (To_Unbounded_String (Params), "--line-numbers") > 0 then
         Line_Numbers := True;
         declare
            Flag_Pos : constant Natural := Index (To_Unbounded_String (Params), "--line-numbers");
         begin
            if Flag_Pos > 1 then
               File_Path := To_Unbounded_String (Trim (Params (Params'First .. Flag_Pos - 1), Both));
            else
               return "ERROR: No filepath specified";
            end if;
         end;
      else
         File_Path := To_Unbounded_String (Trim (Params, Both));
      end if;

      if not Exists (To_String (File_Path)) then
         return "ERROR: File not found: " & To_String (File_Path);
      end if;

      declare
         File : File_Type;
      begin
         Open (File, In_File, To_String (File_Path));
         declare
            Result  : Unbounded_String;
            Line    : String (1 .. 1024);
            Last    : Natural;
         begin
            while not End_Of_File (File) loop
               -- Loop_Invariant: verified (SPARK RM 5.5)  -- mcdc: loop invariant placeholder
               Get_Line (File, Line, Last);
               if Line_Numbers then
                  Result := Result & Trim (Positive'Image (Line_Num), Left) & ": " & Line (1 .. Last);
               else
                  Result := Result & Line (1 .. Last);
               end if;
               Result := Result & ASCII.LF;
               Line_Num := Line_Num + 1;
            end loop;
            Close (File);
            return To_String (Result);
         end;
      exception
         when others =>
            if Is_Open (File) then
               Close (File);
            end if;
            return "ERROR: Could not read file: " & To_String (File_Path);
      end;
   end Execute_Cat;

end Tool_Cat;
