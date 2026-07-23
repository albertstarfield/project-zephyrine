-- File: hook_tool.adb
-- Hook System Tool - Pre/post tool execution hooks for Adelaide Lite.
-- Note: JSON handling is simplified; full JSON parsing would require a library.

--  SPARK_Mode(off)
--  Justification: Standalone CLI procedure. Executes external processes
--  via Ada.Processes.Command_Line (python3 hook scripts), reads files
--  via Ada.Directories and Ada.Text_IO, accesses command-line arguments
--  via Ada.Command_Line. External subprocess and filesystem interaction
--  cannot be expressed in SPARK.

with Ada.Text_IO;
with Ada.Command_Line;
with Ada.Strings;
with Ada.Strings.Unbounded;
with Ada.Processes;
with Ada.Directories;
with Trace_Utils;

--  Hook_Tool: Main entry point. Manages pre/post tool execution hooks
--  via .hooks.json configuration file.
procedure Hook_Tool is
   use Ada.Text_IO;
   use Ada.Strings.Unbounded;

   Hooks_File : constant String := ".hooks.json";

   --  Run_Hook: Execute a Python hook script via subprocess.
   function Run_Hook (Script : in String) return Boolean is
      Cmd : constant String := "python3 " & Script;
   begin
      begin
         Ada.Processes.Command_Line(
           Command_Line => Cmd,
           Output       => True);
         return True;
      exception
         when others =>
            return False;
      end;
   end Run_Hook;

begin
   Trace_Utils.Init_Trace;

   if Ada.Command_Line.Argument_Count < 1 then
      Put_Line("Usage: hook_tool <command> [args...]");
      Put_Line("Commands: list, add, remove, run");
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

      if Cmd = "list" then
         if Ada.Directories.Exists(Hooks_File) then
            --  Read and display hooks file
            declare
               File : File_Type;
            begin
               Open(File, In_File, Hooks_File);
               while not End_Of_File(File) loop
                  Put_Line(Get_Line(File));
               end loop;
               Close(File);
            end;
         else
            Put_Line("No hooks configured");
         end if;

      elsif Cmd = "run" then
         if Ada.Command_Line.Argument_Count < 2 then
            Put_Line("ERROR: Usage: hook_tool run <event>");
            Ada.Command_Line.Set_Exit_Status(1);
         else
            declare
               Event : constant String := Ada.Command_Line.Argument(2);
            begin
              Trace_Utils.Trace_Print("hook", "run", Event);
              --  Would parse .hooks.json and execute matching hooks
              Put_Line("Hook run: " & Event);
            end;
         end if;

      else
         Put_Line("ERROR: Unknown command: " & Cmd);
         Ada.Command_Line.Set_Exit_Status(1);
      end if;
   end;
end Hook_Tool;
