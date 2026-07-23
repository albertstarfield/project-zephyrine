-- File: grep_tool.adb
-- Grep Tool - Search file contents for Adelaide Lite.
-- Uses shell execution to invoke system grep.

--  SPARK_Mode(off)
--  Justification: Standalone CLI procedure. Executes external processes
--  via Ada.Processes.Command_Line (system grep), accesses command-line
--  arguments via Ada.Command_Line, writes output via Ada.Text_IO.
--  External subprocess interaction cannot be expressed in SPARK.

with Ada.Text_IO;
with Ada.Command_Line;
with Ada.Strings;
with Ada.Strings.Unbounded;
with Ada.Processes;
with Trace_Utils;

procedure Grep_Tool is
   use Ada.Text_IO;
   use Ada.Strings.Unbounded;

   function Run_Grep (Pattern, Path : in String;
                      Ignore_Case   : Boolean := False;
                      Count_Mode    : Boolean := False;
                      Files_Only    : Boolean := False)
     return String
   is
      Cmd : Unbounded_String := "grep -r";
   begin
      if Ignore_Case then
         Append(Cmd, " -i");
      end if;

      if Count_Mode then
         Append(Cmd, " -c");
      end if;

      if Files_Only then
         Append(Cmd, " -l");
      end if;

      Append(Cmd, " " & Pattern & " " & Path);

      --  Execute command via shell
      begin
         Ada.Processes.Command_Line(
           Command_Line => To_String(Cmd),
           Output       => True);
         return "";
      exception
         when others =>
            return "ERROR: Grep failed";
      end;
   end Run_Grep;

begin
   Trace_Utils.Init_Trace;

   if Ada.Command_Line.Argument_Count < 2 then
      Put_Line("Usage: grep_tool <command> <pattern> [path]");
      Put_Line("Commands: search, regex, fixed, count, files");
      Ada.Command_Line.Set_Exit_Status(1);
      return;
   end if;

   declare
      Cmd     : constant String := Ada.Command_Line.Argument(1);
      Pattern : constant String := Ada.Command_Line.Argument(2);
      Path    : constant String :=
        (if Ada.Command_Line.Argument_Count >= 3
         then Ada.Command_Line.Argument(3)
         else ".");
   begin
      Trace_Utils.Trace_Print("grep", Cmd,
        "pattern: " & Pattern & ", path: " & Path);

      if Cmd = "search" or Cmd = "regex" then
         Put_Line(Run_Grep(Pattern, Path));

      elsif Cmd = "fixed" then
         Put_Line(Run_Grep(Pattern, Path));

      elsif Cmd = "count" then
         Put_Line(Run_Grep(Pattern, Path, Count_Mode => True));

      elsif Cmd = "files" then
         Put_Line(Run_Grep(Pattern, Path, Files_Only => True));

      else
         Put_Line("ERROR: Unknown command: " & Cmd);
         Ada.Command_Line.Set_Exit_Status(1);
      end if;
   end;
end Grep_Tool;
