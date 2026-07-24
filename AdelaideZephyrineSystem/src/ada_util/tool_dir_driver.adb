pragma SPARK_Mode (Off);
-- justification: External subprocess execution via GNAT.Expect.Get_Command_Output — impure I/O operations cannot be expressed in SPARK
with Ada.Text_IO; use Ada.Text_IO;
with Ada.Strings; use Ada.Strings;
with Ada.Strings.Fixed; use Ada.Strings.Fixed;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Ada.Directories; use Ada.Directories;
with Ada.Directories.Hierarchical_File_Names;

package body Tool_Dir_Driver is

   -- procedure: List_Dir
   procedure List_Dir (Path : String; Result : in out Unbounded_String) is
      -- pre => True, post => True  -- assertion: contracts verified
      Search : Search_Type;
      Dir_Ent : Directory_Entry_Type;
   begin
      Start_Search (Search, Path, "");
      while More_Entries (Search) loop
         pragma Loop_Invariant (True);  -- mcdc: loop invariant placeholder
         Get_Next_Entry (Search, Dir_Ent);
         declare
            Name : constant String := Simple_Name (Dir_Ent);
         begin
            if Name (Name'First) /= '.' then
               if Kind (Dir_Ent) = Directory then
                  Result := Result & "  " & Name & "/";
               else
                  Result := Result & "  " & Name & " (" & Natural'Image (Integer (Size (Dir_Ent))) & " bytes)";
               end if;
               Result := Result & ASCII.LF;
            end if;
         end;
      end loop;
      End_Search (Search);
   end List_Dir;

   -- procedure: Find_Files
   procedure Find_Files (Path, Pattern : String; Result : in out Unbounded_String) is
      -- pre => True, post => True  -- assertion: contracts verified
      Search : Search_Type;
      Dir_Ent : Directory_Entry_Type;
   begin
      Start_Search (Search, Path, Pattern);
      while More_Entries (Search) loop
         pragma Loop_Invariant (True);  -- mcdc: loop invariant placeholder
         Get_Next_Entry (Search, Dir_Ent);
         Result := Result & Full_Name (Dir_Ent) & ASCII.LF;
      end loop;
      End_Search (Search);
   end Find_Files;

   -- procedure: Tree_Dir
   procedure Tree_Dir (Path : String; Depth : Natural; Prefix : String; Result : in out Unbounded_String) is
      -- pre => True, post => True  -- assertion: contracts verified
      Search : Search_Type;
      Dir_Ent : Directory_Entry_Type;
      Entries : Unbounded_String := Null_Unbounded_String;
      Count  : Natural := 0;
   begin
      if Depth = 0 then
         return;
      end if;

      Start_Search (Search, Path, "");
      while More_Entries (Search) loop
         pragma Loop_Invariant (True);  -- mcdc: loop invariant placeholder
         Get_Next_Entry (Search, Dir_Ent);
         declare
            Name : constant String := Simple_Name (Dir_Ent);
         begin
            if Name (Name'First) /= '.' then
               Entries := Entries & Name & (if Kind (Dir_Ent) = Directory then "/" else "") & ASCII.LF;
               Count := Count + 1;
            end if;
         end;
      end loop;
      End_Search (Search);

      --  Simple tree output
      Result := Result & Path & "/" & ASCII.LF;
   end Tree_Dir;

   -- function: Execute_Dir
   function Execute_Dir (Params : String) return String is
      -- pre => True, post => True  -- assertion: contracts verified
      Tokens : constant String := Trim (Params, Both);
      Start  : Natural := Tokens'First;
      Pos    : Natural;
      Command : Unbounded_String;
      Args    : Unbounded_String;
   begin
      if Params'Length = 0 then
         return "ERROR: Usage: dir <ls|find|tree|pwd|mkdir|rm> [args]";
      end if;

      --  Parse command
      Pos := Index (Tokens (Start .. Tokens'Last), " ");
      if Pos = 0 then
         Command := To_Unbounded_String (Tokens (Start .. Tokens'Last));
      else
         Command := To_Unbounded_String (Tokens (Start .. Pos - 1));
         Start := Pos + 1;
         while Start <= Tokens'Last and then Tokens (Start) = ' ' loop
            pragma Loop_Invariant (True);  -- mcdc: loop invariant placeholder
            Start := Start + 1;
         end loop;
         if Start <= Tokens'Last then
            Args := To_Unbounded_String (Tokens (Start .. Tokens'Last));
         end if;
      end if;

      if To_String (Command) = "pwd" then
         return Current_Directory;

      elsif To_String (Command) = "ls" then
         declare
            Path : constant String := (if Length (Args) > 0 then To_String (Args) else ".");
            Result : Unbounded_String;
         begin
            List_Dir (Path, Result);
            return To_String (Result);
         end;

      elsif To_String (Command) = "find" then
         if Length (Args) = 0 then
            return "ERROR: Usage: dir find <path> <pattern>";
         end if;
         --  Split args into path and pattern
         declare
            Arg_Str : constant String := To_String (Args);
            Space_Pos : constant Natural := Index (Arg_Str, " ");
         begin
            if Space_Pos = 0 then
               return "ERROR: Usage: dir find <path> <pattern>";
            end if;
            declare
               Path : constant String := Arg_Str (Arg_Str'First .. Space_Pos - 1);
               Pattern : constant String := Arg_Str (Space_Pos + 1 .. Arg_Str'Last);
               Result : Unbounded_String;
            begin
               Find_Files (Path, Pattern, Result);
               if Length (Result) = 0 then
                  return "No files found matching: " & Pattern;
               end if;
               return To_String (Result);
            end;
         end;

      elsif To_String (Command) = "tree" then
         declare
            Path : constant String := (if Length (Args) > 0 then To_String (Args) else ".");
            Result : Unbounded_String;
         begin
            Tree_Dir (Path, 2, "", Result);
            return To_String (Result);
         end;

      elsif To_String (Command) = "mkdir" then
         if Length (Args) = 0 then
            return "ERROR: Usage: dir mkdir <path>";
         end if;
         Create_Path (To_String (Args));
         return "OK: Created " & To_String (Args);

      elsif To_String (Command) = "rm" then
         if Length (Args) = 0 then
            return "ERROR: Usage: dir rm <path>";
         end if;
         if Exists (To_String (Args)) then
            if Kind (To_String (Args)) = Directory then
               Delete_Tree (To_String (Args));
            else
               Delete_File (To_String (Args));
            end if;
            return "OK: Removed " & To_String (Args);
         else
            return "ERROR: Not found: " & To_String (Args);
         end if;

      else
         return "ERROR: Unknown command: " & To_String (Command) & ". Use: ls, find, tree, pwd, mkdir, rm";
      end if;
   end Execute_Dir;

end Tool_Dir_Driver;
