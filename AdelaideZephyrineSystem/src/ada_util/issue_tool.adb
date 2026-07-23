-- File: issue_tool.adb
-- Issue Tool - Manage GitHub issues for Adelaide Lite.

--  SPARK_Mode(off)
--  Justification: Standalone CLI procedure. Executes external processes
--  via Ada.Processes.Command_Line (gh CLI), accesses command-line
--  arguments via Ada.Command_Line, writes output via Ada.Text_IO.
--  External subprocess interaction cannot be expressed in SPARK.

with Ada.Text_IO;
with Ada.Command_Line;
with Ada.Strings;
with Ada.Strings.Unbounded;
with Ada.Processes;
with Trace_Utils;

--  Issue_Tool: Main entry point. Dispatches GitHub issue commands
--  (list, view, create, close, comment, search) via gh CLI.
procedure Issue_Tool is
   use Ada.Text_IO;
   use Ada.Strings.Unbounded;

   --  Run_Gh: Execute a gh CLI command via subprocess and return output.
   function Run_Gh (Args : in String) return String is
      Cmd : constant String := "gh " & Args;
   begin
      begin
         Ada.Processes.Command_Line(
           Command_Line => Cmd,
           Output       => True);
         return "";
      exception
         when others =>
            return "ERROR: gh CLI not found or failed";
      end;
   end Run_Gh;

begin
   Trace_Utils.Init_Trace;

   if Ada.Command_Line.Argument_Count < 1 then
      Put_Line("Usage: issue_tool <command> [args...]");
      Put_Line("Commands: list, view, create, close, comment, search");
      Ada.Command_Line.Set_Exit_Status(1);
      return;
   end if;

   declare
      Cmd  : constant String := Ada.Command_Line.Argument(1);
      Args : Unbounded_String := Null_Unbounded_String;
   begin
      for I in 2 .. Ada.Command_Line.Argument_Count loop
         if I > 2 then
            Append(Args, " ");
         end if;
         Append(Args, Ada.Command_Line.Argument(I));
      end loop;

      Trace_Utils.Trace_Print("issue", Cmd, To_String(Args));

      if Cmd = "list" then
         Put_Line(Run_Gh("issue list"));

      elsif Cmd = "view" then
         if Ada.Command_Line.Argument_Count < 2 then
            Put_Line("ERROR: Usage: issue_tool view <number>");
            Ada.Command_Line.Set_Exit_Status(1);
         else
            Put_Line(Run_Gh("issue view " & To_String(Args)));
         end if;

      elsif Cmd = "create" then
         if Ada.Command_Line.Argument_Count < 2 then
            Put_Line("ERROR: Usage: issue_tool create <title> [body]");
            Ada.Command_Line.Set_Exit_Status(1);
         else
            Put_Line(Run_Gh("issue create --title """ &
              Ada.Command_Line.Argument(2) & """"));
         end if;

      elsif Cmd = "close" then
         if Ada.Command_Line.Argument_Count < 2 then
            Put_Line("ERROR: Usage: issue_tool close <number>");
            Ada.Command_Line.Set_Exit_Status(1);
         else
            Put_Line(Run_Gh("issue close " & To_String(Args)));
         end if;

      elsif Cmd = "comment" then
         if Ada.Command_Line.Argument_Count < 3 then
            Put_Line("ERROR: Usage: issue_tool comment <number> <message>");
            Ada.Command_Line.Set_Exit_Status(1);
         else
            Put_Line(Run_Gh("issue comment " &
              Ada.Command_Line.Argument(2) &
              " --body """ & Ada.Command_Line.Argument(3) & """"));
         end if;

      elsif Cmd = "search" then
         if Ada.Command_Line.Argument_Count < 2 then
            Put_Line("ERROR: Usage: issue_tool search <query>");
            Ada.Command_Line.Set_Exit_Status(1);
         else
            Put_Line(Run_Gh("issue list --search " & To_String(Args)));
         end if;

      else
         Put_Line("ERROR: Unknown command: " & Cmd);
         Ada.Command_Line.Set_Exit_Status(1);
      end if;
   end;
end Issue_Tool;
