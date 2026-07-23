-- File: cat_tool.adb
-- Cat Tool - Read and print file contents for Adelaide Lite.

--  SPARK_Mode(off)
--  Justification: Standalone CLI procedure. Reads files via Ada.Text_IO
--  and Ada.Directories, accesses command-line arguments via
--  Ada.Command_Line. All operations are impure I/O with filesystem
--  and external process interaction.

with Ada.Text_IO;
with Ada.Directories;
with Ada.Command_Line;
with Trace_Utils;

procedure Cat_Tool is
   use Ada.Text_IO;
   use Ada.Directories;
begin
   Trace_Utils.Init_Trace;

   if Ada.Command_Line.Argument_Count < 1 then
      Put_Line("Usage: cat_tool <file>");
      Ada.Command_Line.Set_Exit_Status(1);
      return;
   end if;

   declare
      Path : constant String := Ada.Command_Line.Argument(1);
   begin
      Trace_Utils.Trace_Print("cat", "read", "file: " & Path);

      if Exists(Path) then
         declare
            File : File_Type;
         begin
            Open(File, In_File, Path);
            while not End_Of_File(File) loop
               Put_Line(Get_Line(File));
            end loop;
            Close(File);
            Trace_Utils.Trace_Result("cat", True, "read " & Path);
         end;
      else
         Put_Line("File not found: " & Path);
         Trace_Utils.Trace_Result("cat", False, "file not found: " & Path);
      end if;
   end;
end Cat_Tool;
