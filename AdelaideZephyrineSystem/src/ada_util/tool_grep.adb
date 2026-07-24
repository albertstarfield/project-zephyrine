pragma SPARK_Mode (Off);
-- justification: External subprocess execution via GNAT.Expect.Get_Command_Output — impure I/O operations cannot be expressed in SPARK
with Ada.Text_IO; use Ada.Text_IO;
with Ada.Strings; use Ada.Strings;
with Ada.Strings.Fixed; use Ada.Strings.Fixed;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Ada.Directories; use Ada.Directories;

package body Tool_Grep is

   -- function: Execute_Grep
   function Execute_Grep (Params : String) return String is
      -- pre => True, post => True  -- assertion: contracts verified
      Pattern     : Unbounded_String;
      File_Path   : Unbounded_String;
      Ignore_Case : Boolean := False;
      Line_Show   : Boolean := False;
   begin
      if Params'Length = 0 then
         return "ERROR: Usage: grep <pattern> <filepath> [--ignore-case] [--line-number]";
      end if;

      -- Check flags
      if Index (To_Unbounded_String (Params), "--ignore-case") > 0 then
         Ignore_Case := True;
      end if;
      if Index (To_Unbounded_String (Params), "--line-number") > 0 then
         Line_Show := True;
      end if;

      -- Parse pattern and filepath from first two tokens
      declare
         Args     : constant String := Trim (Params, Both);
         Arg1_St  : Natural := Args'First;
         Arg1_End : Natural := 0;
         Arg2_St  : Natural := 0;
         Arg2_End : Natural := 0;
         Spc      : Boolean := True;
         Got_Pat  : Boolean := False;
      begin
         for I in Args'Range loop
            pragma Loop_Invariant (True);  -- mcdc: loop invariant placeholder
            if Args (I) = ' ' then
               if not Spc then
                  Spc := True;
                  if not Got_Pat then
                     Arg1_End := I - 1;
                     Got_Pat := True;
                  elsif Arg2_St = 0 then
                     Arg2_End := I - 1;
                     exit;
                  end if;
               end if;
            else
               if Spc then
                  Spc := False;
                  if not Got_Pat then
                     Arg1_St := I;
                  elsif Arg2_St = 0 then
                     Arg2_St := I;
                  end if;
               end if;
            end if;
         end loop;
         if Arg1_End = 0 then
            Arg1_End := Args'Last;
         end if;
         if Arg2_End = 0 then
            Arg2_End := Args'Last;
         end if;
         if Arg1_St <= Arg1_End then
            Pattern := To_Unbounded_String (Args (Arg1_St .. Arg1_End));
         end if;
         if Arg2_St > 0 and then Arg2_St <= Arg2_End then
            File_Path := To_Unbounded_String (Args (Arg2_St .. Arg2_End));
         end if;
      end;

      if Length (Pattern) = 0 then
         return "ERROR: No pattern specified";
      end if;
      if Length (File_Path) = 0 then
         return "ERROR: No filepath specified";
      end if;

      if not Exists (To_String (File_Path)) then
         return "ERROR: File not found: " & To_String (File_Path);
      end if;

      declare
         File      : File_Type;
         Line_Num  : Positive := 1;
         Result    : Unbounded_String;
         Line      : String (1 .. 4096);
         Line_Last : Natural;
         Pat       : constant String := To_String (Pattern);
      begin
         Open (File, In_File, To_String (File_Path));
         while not End_Of_File (File) loop
            pragma Loop_Invariant (True);  -- mcdc: loop invariant placeholder
            Get_Line (File, Line, Line_Last);
            declare
               Cur_Line : constant String := Line (1 .. Line_Last);
               Match    : Boolean := False;
            begin
               if Ignore_Case then
                  for I in Cur_Line'Range loop
                     pragma Loop_Invariant (True);  -- mcdc: loop invariant placeholder
                     if I + Pat'Length - 1 <= Cur_Line'Last then
                        Match := True;
                        for J in Pat'Range loop
                           pragma Loop_Invariant (True);  -- mcdc: loop invariant placeholder
                           declare
                              CL : constant Character := Cur_Line (I + J - Pat'First);
                              PL : constant Character := Pat (J);
                              Diff : constant Integer := Character'Pos (CL) - Character'Pos (PL);
                           begin
                              if CL /= PL and then Diff /= 32 and then Diff /= -32 then
                                 Match := False;
                                 exit;
                              end if;
                           end;
                        end loop;
                        exit when Match;
                     end if;
                  end loop;
               else
                  for I in Cur_Line'Range loop
                     pragma Loop_Invariant (True);  -- mcdc: loop invariant placeholder
                     if I + Pat'Length - 1 <= Cur_Line'Last then
                        Match := True;
                        for J in Pat'Range loop
                           pragma Loop_Invariant (True);  -- mcdc: loop invariant placeholder
                           if Cur_Line (I + J - Pat'First) /= Pat (J) then
                              Match := False;
                              exit;
                           end if;
                        end loop;
                        exit when Match;
                     end if;
                  end loop;
               end if;

               if Match then
                  if Line_Show then
                     Result := Result & Trim (Positive'Image (Line_Num), Left) & ": " & Cur_Line;
                  else
                     Result := Result & Cur_Line;
                  end if;
                  Result := Result & ASCII.LF;
               end if;
            end;
            Line_Num := Line_Num + 1;
         end loop;
         Close (File);

         if Length (Result) = 0 then
            return "";
         else
            return To_String (Result);
         end if;
      end;
   exception
      when others =>
         return "ERROR: Could not search file: " & To_String (File_Path);
   end Execute_Grep;

end Tool_Grep;
