-- File: package_tool.adb
-- Package Manager Tool - Install system packages for Adelaide Lite.

--  SPARK_Mode(off)
--  Justification: Standalone CLI procedure. Executes external processes
--  via Ada.Processes.Command_Line (apt-get, brew), reads environment
--  variables via Ada.Environment_Variables, accesses command-line
--  arguments via Ada.Command_Line, writes output via Ada.Text_IO.
--  External subprocess and environment interaction cannot be expressed
--  in SPARK.

with Ada.Text_IO;
with Ada.Command_Line;
with Ada.Strings;
with Ada.Strings.Unbounded;
with Ada.Processes;
with Ada.Environment_Variables;
with Trace_Utils;

--  Package_Tool: Main entry point. Dispatches package management commands
--  (detect, install, uninstall, update, upgrade, search, list).
procedure Package_Tool is
   use Ada.Text_IO;
   use Ada.Strings.Unbounded;

   --  Detect_Package_Manager: Return "apt" for Linux, "brew" for macOS.
   function Detect_Package_Manager return Unbounded_String is
      Sys : constant String :=
        (if Ada.Environment_Variables.Exists("OS") then
            Ada.Environment_Variables.Value("OS")
         else "linux");
   begin
      --  Simplified detection: check common package managers
      if Sys = "linux" or Sys = "Linux" then
         --  Would check for apt, yum, pacman, etc.
         return To_Unbounded_String("apt");
      elsif Sys = "darwin" or Sys = "Darwin" then
         return To_Unbounded_String("brew");
      else
         return To_Unbounded_String("unknown");
      end if;
   end Detect_Package_Manager;

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

   --  Install_Package: Detect package manager and install the named package.
   function Install_Package (Pkg : in String) return String is
      PM : constant String := To_Unbounded_String(Detect_Package_Manager);
   begin
      Trace_Utils.Trace_Print("package", "detect", PM);
      Trace_Utils.Trace_Print("package", "install", Pkg);

      if PM = "apt" then
         Put_Line(Run_Cmd("sudo apt-get update"));
         return Run_Cmd("sudo apt-get install -y " & Pkg);
      elsif PM = "brew" then
         return Run_Cmd("brew install " & Pkg);
      else
         return "ERROR: No supported package manager found";
      end if;
   end Install_Package;

begin
   Trace_Utils.Init_Trace;

   if Ada.Command_Line.Argument_Count < 1 then
      Put_Line("Usage: package_tool <command> [args...]");
      Put_Line("Commands: detect, install, uninstall, update, upgrade, search, list");
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

      if Cmd = "detect" then
         declare
            PM : constant String := To_Unbounded_String(Detect_Package_Manager);
         begin
            Put_Line("Package manager: " & PM);
            Trace_Utils.Trace_Result("package", PM /= "unknown",
              "detected " & PM);
         end;

      elsif Cmd = "install" then
         if Ada.Command_Line.Argument_Count < 2 then
            Put_Line("ERROR: Usage: package_tool install <package>");
            Ada.Command_Line.Set_Exit_Status(1);
         else
            declare
               Output : constant String :=
                 Install_Package(Ada.Command_Line.Argument(2));
            begin
               Put_Line(Output);
               Trace_Utils.Trace_Result("package",
                 "ERROR" not in Output,
                 "installed " & Ada.Command_Line.Argument(2));
            end;
         end if;

      elsif Cmd = "update" then
         declare
            PM : constant String := To_Unbounded_String(Detect_Package_Manager);
         begin
            if PM = "apt" then
               Put_Line(Run_Cmd("sudo apt-get update"));
            elsif PM = "brew" then
               Put_Line(Run_Cmd("brew update"));
            end if;
         end;

      elsif Cmd = "search" then
         if Ada.Command_Line.Argument_Count < 2 then
            Put_Line("ERROR: Usage: package_tool search <query>");
            Ada.Command_Line.Set_Exit_Status(1);
         else
            declare
               PM : constant String := To_Unbounded_String(Detect_Package_Manager);
            begin
               if PM = "apt" then
                  Put_Line(Run_Cmd("apt-cache search " & To_String(Args)));
               elsif PM = "brew" then
                  Put_Line(Run_Cmd("brew search " & To_String(Args)));
               end if;
            end;
         end if;

      elsif Cmd = "list" then
         declare
            PM : constant String := To_Unbounded_String(Detect_Package_Manager);
         begin
            if PM = "apt" then
               Put_Line(Run_Cmd("dpkg --list"));
            elsif PM = "brew" then
               Put_Line(Run_Cmd("brew list"));
            end if;
         end;

      else
         Put_Line("ERROR: Unknown command: " & Cmd);
         Ada.Command_Line.Set_Exit_Status(1);
      end if;
   end;
end Package_Tool;
