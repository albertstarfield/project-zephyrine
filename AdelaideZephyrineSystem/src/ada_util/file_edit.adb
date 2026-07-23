-- File: file_edit.adb
-- File Edit Tool - Edit files for Adelaide Lite.

--  SPARK_Mode(off)
--  Justification: Standalone CLI procedure. Performs filesystem read/write
--  via Ada.Text_IO and Ada.Directories, accesses command-line arguments
--  via Ada.Command_Line, converts strings via Ada.Strings.Unbounded.
--  All operations are impure I/O.

with Ada.Text_IO;
with Ada.Directories;
with Ada.Command_Line;
with Ada.Strings;
with Ada.Strings.Unbounded;
with Trace_Utils;

--  File_Edit: Main entry point. Dispatches file operations (read, write,
--  edit, append, exists, head, tail) to filesystem via Ada.Text_IO.
procedure File_Edit is
   use Ada.Text_IO;
   use Ada.Strings.Unbounded;

   --  Args: Concatenate command-line arguments 2..N into a single string.
   function Args return Unbounded_String is
      Result : Unbounded_String := Null_Unbounded_String;
   begin
      for I in 2 .. Ada.Command_Line.Argument_Count loop
         if I > 2 then
            Append(Result, " ");
         end if;
         Append(Result, Ada.Command_Line.Argument(I));
      end loop;
      return Result;
   end Args;

   --  Do_Read: Read and print file contents line by line.
   procedure Do_Read (Path : in String) is
   begin
      Trace_Utils.Trace_Print("file_edit", "read", "file: " & Path);
      if Ada.Directories.Exists(Path) then
         declare
            File : File_Type;
         begin
            Open(File, In_File, Path);
            while not End_Of_File(File) loop
               Put_Line(Get_Line(File));
            end loop;
            Close(File);
         end;
      else
         Put_Line("ERROR: File not found: " & Path);
      end if;
   end Do_Read;

   --  Do_Write: Create/overwrite a file with the given content string.
   procedure Do_Write (Path : in String; Content : in String) is
   begin
      Trace_Utils.Trace_Print("file_edit", "write", "file: " & Path);
      declare
         File : File_Type;
      begin
         Create(File, Out_File, Path);
         Put_Line(File, Content);
         Close(File);
         Put_Line("OK: Written to " & Path);
      end;
   end Do_Write;

   --  Do_Edit: Find and replace the first occurrence of Old with New in file.
   procedure Do_Edit (Path, Old, New : in String) is
   begin
      Trace_Utils.Trace_Print("file_edit", "edit", "file: " & Path);
      if not Ada.Directories.Exists(Path) then
         Put_Line("ERROR: File not found: " & Path);
         return;
      end if;

      declare
         File : File_Type;
         Content : Unbounded_String := Null_Unbounded_String;
      begin
         Open(File, In_File, Path);
         while not End_Of_File(File) loop
            Append(Content, Get_Line(File));
            if not End_Of_File(File) then
               Append(Content, ASCII.LF);
            end if;
         end loop;
         Close(File);

         declare
            Content_Str : constant String := To_String(Content);
            Pos : Natural;
         begin
            Pos := Ada.Strings.Fixed.Index(Content_Str, Old);
            if Pos = 0 then
               Put_Line("ERROR: Old text not found in " & Path);
            else
               declare
                  Result : Unbounded_String :=
                    To_Unbounded_String(Content_Str(1 .. Pos - 1));
               begin
                  Append(Result, New);
                  Append(Result, Content_Str(Pos + Old'Length .. Content_Str'Length));

                  Create(File, Out_File, Path);
                  Put_Line(File, To_String(Result));
                  Close(File);
                  Put_Line("OK: Edited " & Path);
               end;
            end if;
         end;
      end;
   end Do_Edit;

   --  Do_Exists: Print "true" if file exists, "false" otherwise.
   procedure Do_Exists (Path : in String) is
   begin
      Trace_Utils.Trace_Print("file_edit", "exists", "file: " & Path);
      if Ada.Directories.Exists(Path) then
         Put_Line("true");
      else
         Put_Line("false");
      end if;
   end Do_Exists;

   --  Do_Head: Print the first N lines of a file (default 10).
   procedure Do_Head (Path : in String; N : in Positive := 10) is
   begin
      Trace_Utils.Trace_Print("file_edit", "head", "file: " & Path);
      declare
         File : File_Type;
         Count : Natural := 0;
      begin
         Open(File, In_File, Path);
         while not End_Of_File(File) and Count < N loop
            Put_Line(Get_Line(File));
            Count := Count + 1;
         end loop;
         Close(File);
      end;
   end Do_Head;

   --  Do_Tail: Print the last N lines of a file (default 10).
   procedure Do_Tail (Path : in String; N : in Positive := 10) is
   begin
      Trace_Utils.Trace_Print("file_edit", "tail", "file: " & Path);
      declare
         File : File_Type;
         All_Lines : Unbounded_String := Null_Unbounded_String;
         Line_Count : Natural := 0;
      begin
         Open(File, In_File, Path);
         while not End_Of_File(File) loop
            declare
               Line : constant String := Get_Line(File);
            begin
               if Line_Count > 0 then
                  Append(All_Lines, ASCII.LF);
               end if;
               Append(All_Lines, Line);
               Line_Count := Line_Count + 1;
            end;
         end loop;
         Close(File);

         --  Print last N lines (simplified: print all if fewer than N)
         if Line_Count <= N then
            Put_Line(To_String(All_Lines));
         else
            --  Find the Nth line from end
            declare
               Str : constant String := To_String(All_Lines);
               Start : Natural := Str'First;
            begin
               for I in 1 .. Line_Count - N loop
                  Start := Ada.Strings.Fixed.Index(Str, ASCII.LF & "", Start) + 1;
               end loop;
               Put_Line(Str(Start .. Str'Last));
            end;
         end if;
      end;
   end Do_Tail;

begin
   Trace_Utils.Init_Trace;

   if Ada.Command_Line.Argument_Count < 1 then
      Put_Line("Usage: file_edit <command> [args...]");
      Put_Line("Commands: read, write, edit, append, exists, head, tail");
      Ada.Command_Line.Set_Exit_Status(1);
      return;
   end if;

   declare
      Cmd : constant String := Ada.Command_Line.Argument(1);
   begin
      if Cmd = "read" then
         if Ada.Command_Line.Argument_Count < 2 then
            Put_Line("ERROR: Usage: file_edit read <file>");
            Ada.Command_Line.Set_Exit_Status(1);
         else
            Do_Read(Ada.Command_Line.Argument(2));
         end if;

      elsif Cmd = "write" then
         if Ada.Command_Line.Argument_Count < 3 then
            Put_Line("ERROR: Usage: file_edit write <file> <content>");
            Ada.Command_Line.Set_Exit_Status(1);
         else
            Do_Write(Ada.Command_Line.Argument(2), Ada.Command_Line.Argument(3));
         end if;

      elsif Cmd = "edit" then
         if Ada.Command_Line.Argument_Count < 4 then
            Put_Line("ERROR: Usage: file_edit edit <file> <old> <new>");
            Ada.Command_Line.Set_Exit_Status(1);
         else
            Do_Edit(Ada.Command_Line.Argument(2),
                     Ada.Command_Line.Argument(3),
                     Ada.Command_Line.Argument(4));
         end if;

      elsif Cmd = "exists" then
         if Ada.Command_Line.Argument_Count < 2 then
            Put_Line("ERROR: Usage: file_edit exists <file>");
            Ada.Command_Line.Set_Exit_Status(1);
         else
            Do_Exists(Ada.Command_Line.Argument(2));
         end if;

      elsif Cmd = "head" then
         if Ada.Command_Line.Argument_Count < 2 then
            Put_Line("ERROR: Usage: file_edit head <file> [n]");
            Ada.Command_Line.Set_Exit_Status(1);
         else
            Do_Head(Ada.Command_Line.Argument(2));
         end if;

      elsif Cmd = "tail" then
         if Ada.Command_Line.Argument_Count < 2 then
            Put_Line("ERROR: Usage: file_edit tail <file> [n]");
            Ada.Command_Line.Set_Exit_Status(1);
         else
            Do_Tail(Ada.Command_Line.Argument(2));
         end if;

      else
         Put_Line("ERROR: Unknown command: " & Cmd);
         Ada.Command_Line.Set_Exit_Status(1);
      end if;
   end;

   Trace_Utils.Trace_Result("file_edit", True, "cmd: " &
     (if Ada.Command_Line.Argument_Count > 0
      then Ada.Command_Line.Argument(1)
      else ""));
end File_Edit;
