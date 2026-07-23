-- File: killshell.adb
-- KillShell Tool - Kill processes for Adelaide Lite.

--  SPARK_Mode(off)
--  Justification: Standalone CLI procedure. Executes external processes
--  via Ada.Processes.Command_Line (kill, killall, pkill, ps), accesses
--  command-line arguments via Ada.Command_Line, writes output via
--  Ada.Text_IO. External subprocess interaction cannot be expressed in SPARK.

with Ada.Text_IO;
with Ada.Command_Line;
with Ada.Strings;
with Ada.Strings.Unbounded;
with Ada.Processes;
with Trace_Utils;

--  KillShell: Main entry point. Dispatches process management commands
--  (kill, killall, pkill, ps, top) to system shell.
procedure KillShell is
   use Ada.Text_IO;
   use Ada.Strings.Unbounded;

   --  Run_Cmd: Execute a shell command via subprocess and return output.
   function Run_Cmd (Cmd : in String) return String is
   begin
      begin
         Ada.Processes.Command_Line(
           Command_Line => Cmd,
           Output       => True);
         return "";
      exception
         when others =>
            return "ERROR: Command failed";
      end;
   end Run_Cmd;

begin
   Trace_Utils.Init_Trace;

   if Ada.Command_Line.Argument_Count < 1 then
      Put_Line("Usage: killshell <command> [args...]");
      Put_Line("Commands: kill, killall, pkill, ps, top");
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

      if Cmd = "kill" then
         if Ada.Command_Line.Argument_Count < 2 then
            Put_Line("ERROR: Usage: killshell kill <pid>");
            Ada.Command_Line.Set_Exit_Status(1);
         else
            declare
               Pid : constant String := Ada.Command_Line.Argument(2);
            begin
               Trace_Utils.Trace_Print("killshell", "kill", "pid=" & Pid);
               Put_Line(Run_Cmd("kill " & Pid));
            end;
         end if;

      elsif Cmd = "killall" then
         if Ada.Command_Line.Argument_Count < 2 then
            Put_Line("ERROR: Usage: killshell killall <name>");
            Ada.Command_Line.Set_Exit_Status(1);
         else
            Put_Line(Run_Cmd("killall " & To_String(Args)));
         end if;

      elsif Cmd = "pkill" then
         if Ada.Command_Line.Argument_Count < 2 then
            Put_Line("ERROR: Usage: killshell pkill <pattern>");
            Ada.Command_Line.Set_Exit_Status(1);
         else
            Put_Line(Run_Cmd("pkill -f " & To_String(Args)));
         end if;

      elsif Cmd = "ps" then
         if Ada.Command_Line.Argument_Count >= 2 then
            Put_Line(Run_Cmd("ps aux | grep " & To_String(Args)));
         else
            Put_Line(Run_Cmd("ps aux"));
         end if;

      elsif Cmd = "top" then
         declare
            N : constant String :=
              (if Ada.Command_Line.Argument_Count >= 2
               then Ada.Command_Line.Argument(2)
               else "10");
         begin
            Put_Line(Run_Cmd("ps aux --sort=-pcpu --rows=" & N));
         end;

      else
         Put_Line("ERROR: Unknown command: " & Cmd);
         Ada.Command_Line.Set_Exit_Status(1);
      end if;
   end;
end KillShell;
