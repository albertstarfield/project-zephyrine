-- File: git_tool.adb
-- Git Tool - Execute git operations for Adelaide Lite.

with Ada.Text_IO;
with Ada.Command_Line;
with Ada.Strings;
with Ada.Strings.Unbounded;
with Ada.Processes;
with Trace_Utils;

procedure Git_Tool is
   use Ada.Text_IO;
   use Ada.Strings.Unbounded;

   function Run_Git (Args : in String) return String is
      Cmd : constant String := "git " & Args;
   begin
      begin
         Ada.Processes.Command_Line(
           Command_Line => Cmd,
           Output       => True);
         return "";
      exception
         when others =>
            return "ERROR: Git command failed";
      end;
   end Run_Git;

begin
   Trace_Utils.Init_Trace;

   if Ada.Command_Line.Argument_Count < 1 then
      Put_Line("Usage: git_tool <command> [args...]");
      Put_Line("Commands: status, diff, commit, push, pull, log, branch, checkout");
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

      Trace_Utils.Trace_Print("git", Cmd, To_String(Args));

      if Cmd = "status" then
         Put_Line(Run_Git("status"));

      elsif Cmd = "diff" then
         Put_Line(Run_Git("diff"));

      elsif Cmd = "commit" then
         if Ada.Command_Line.Argument_Count < 2 then
            Put_Line("ERROR: Usage: git_tool commit <message>");
            Ada.Command_Line.Set_Exit_Status(1);
         else
            Put_Line(Run_Git("add ."));
            Put_Line(Run_Git("commit -m """ & To_String(Args) & """"));
         end if;

      elsif Cmd = "push" then
         Put_Line(Run_Git("push"));

      elsif Cmd = "pull" then
         Put_Line(Run_Git("pull"));

      elsif Cmd = "log" then
         declare
            N : constant String :=
              (if Ada.Command_Line.Argument_Count >= 2
               then Ada.Command_Line.Argument(2)
               else "10");
         begin
            Put_Line(Run_Git("log --oneline -" & N));
         end;

      elsif Cmd = "branch" then
         Put_Line(Run_Git("branch -a"));

      elsif Cmd = "checkout" then
         if Ada.Command_Line.Argument_Count < 2 then
            Put_Line("ERROR: Usage: git_tool checkout <branch>");
            Ada.Command_Line.Set_Exit_Status(1);
         else
            Put_Line(Run_Git("checkout " & To_String(Args)));
         end if;

      else
         Put_Line("ERROR: Unknown command: " & Cmd);
         Ada.Command_Line.Set_Exit_Status(1);
      end if;
   end;

   Trace_Utils.Trace_Result("git", True, "command: " &
     Ada.Command_Line.Argument(1));
end Git_Tool;
