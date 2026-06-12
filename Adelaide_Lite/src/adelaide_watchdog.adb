pragma SPARK_Mode (Off);

--  [DO NOT REMOVE THIS]
--  ===========================================================================
--  Adelaide Watchdog — Separate Process Monitor
--  ===========================================================================
--  A dedicated external process that monitors the health of the
--  main adelaide_server process.  If the server dies (e.g. from
--  a Kratos-escaped SIGTRAP, kernel OOM, etc.) or its heartbeat
--  goes stale, the watchdog restarts it automatically.
--
--  Communication: file-based IPC via the run/ directory.
--
--  [DO NOT REMOVE THIS] LAUNCH GUARD:
--  This watchdog MUST be launched exclusively through run.py (or run.sh).
--  Direct execution of bin/adelaide_watchdog is PROHIBITED because it
--  bypasses the orchestration layer that manages:
--    - Environment setup (DYLD_LIBRARY_PATH for onnxruntime)
--    - Server process lifecycle (PID tracking, graceful shutdown)
--    - Health ping coordination (startup watchdog timeout)
--    - Resource cleanup on exit (SIGTERM propagation)
--  Running the watchdog directly will cause orphaned processes, broken
--  health monitoring, and resource leaks.  run.py sets the environment
--  variable ADLAIDE_WATCHDOG_ORCHESTRATED=1 before spawning this process.
--  If that variable is not present, this binary will refuse to start.
--  ===========================================================================

with Ada.Text_IO;           use Ada.Text_IO;
with Ada.Exceptions;        use Ada.Exceptions;
with Ada.Directories;       use Ada.Directories;
with Ada.Real_Time;         use Ada.Real_Time;
with Ada.Command_Line;
with Ada.Environment_Variables;
with GNAT.OS_Lib;           use GNAT.OS_Lib;

procedure Adelaide_Watchdog is

   Run_Dir    : constant String := "run";
   PID_File   : constant String := Run_Dir & "/adelaide_server.pid";
   HB_File    : constant String := Run_Dir & "/adelaide_server.heartbeat";
   Server_Bin : constant String := "bin/adelaide_server";

   --  Timeouts
   HB_Stale_Limit : constant Duration := 10.0;
   Check_Interval : constant Duration := 1.0;

   --  After restarting, wait this long before considering another restart.
   --  Prevents re-spawn loops when the server is still starting up.
   Restart_Cooldown : constant Duration := 30.0;

   --  C FFI for POSIX kill(2).  We keep this at package level because
   --  we read PIDs as plain Integers from the IPC file and GNAT.OS_Lib.Kill
   --  requires the private Process_Id type (no Integer-to-Process_Id conversion).
   function Sys_Kill (P : Integer; Sig : Integer) return Integer;
   pragma Import (C, Sys_Kill, "kill");

   Oneshot       : Boolean := False;
   Last_Restart  : Ada.Real_Time.Time := Time_Of (0, Time_Span_Zero);

   -------------------
   -- Read_PID --
   -------------------

   function Read_PID return Integer is
      F : File_Type;
      S : String (1 .. 16);
      L : Natural;
   begin
      if not Exists (PID_File) then
         return -1;
      end if;
      Open (F, In_File, PID_File);
      Get_Line (F, S, L);
      Close (F);
      return Integer'Value (S (1 .. L));
   exception
      when others =>
         return -1;
   end Read_PID;

   --------------------
   -- Is_Process_Alive --
   --------------------

   function Is_Process_Alive (Pid : Integer) return Boolean is
   begin
      if Pid <= 0 then
         return False;
      end if;
      return Sys_Kill (Pid, 0) = 0;
   end Is_Process_Alive;

   -------------------------
   -- Get_Heartbeat_Age_S --
   -------------------------

   function Get_Heartbeat_Age_S return Duration is
      F : File_Type;
      S : String (1 .. 32);
      L : Natural;
      Stored_Dur : Duration;
      Now_Time   : constant Time := Clock;
      Now_Dur    : constant Duration :=
        To_Duration (Now_Time - Time_Of (0, Time_Span_Zero));
   begin
      if not Exists (HB_File) then
         return Duration'Last;
      end if;
      Open (F, In_File, HB_File);
      Get_Line (F, S, L);
      Close (F);
      Stored_Dur := Duration'Value (S (1 .. L));
      return Now_Dur - Stored_Dur;
   exception
      when others =>
         return Duration'Last;
   end Get_Heartbeat_Age_S;

   ----------------------
   -- Restart_Server --
   ----------------------

   procedure Restart_Server (Old_Pid : Integer) is
      Alr     : String_Access;
      Cmd     : String_Access;
      Args    : Argument_List (1 .. 3);
      New_Pid : Process_Id;
      N_Args  : Natural := 0;
   begin
      Put_Line (Standard_Error,
        "[Watchdog] Server (PID" & Integer'Image (Old_Pid) &
        ") is dead or frozen. Restarting...");

      --  Kill old process if still hanging around
      if Old_Pid > 0 and then Is_Process_Alive (Old_Pid) then
         Put_Line (Standard_Error,
           "[Watchdog] Sending SIGTERM to old PID" &
           Integer'Image (Old_Pid));
         if Sys_Kill (Old_Pid, 15) /= 0 then    --  SIGTERM
            declare
               R : Integer := Sys_Kill (Old_Pid, 9);  --  SIGKILL
            begin
               null;
            end;
         end if;
         delay 1.0;
      end if;

      --  Prefer alr exec for proper library paths (DYLD_LIBRARY_PATH etc.)
      Alr := Locate_Exec_On_Path ("alr");
      if Alr /= null then
         Cmd := Alr;
         Args (1) := new String'("exec");
         Args (2) := new String'("--");
         Args (3) := new String'(Server_Bin);
         N_Args := 3;
      else
         Cmd := new String'(Server_Bin);
         Args (1) := new String'("--no-gui");
         N_Args := 1;
      end if;

      Put_Line (Standard_Error,
        "[Watchdog] Spawning: " & Cmd.all);

      --  Start server as a background (non-blocking) process
      New_Pid := Non_Blocking_Spawn (Cmd.all, Args (1 .. N_Args));

      if New_Pid = Invalid_Pid then
         Put_Line (Standard_Error,
           "[Watchdog] FAILED to start server process.");
      else
         Put_Line (Standard_Error,
           "[Watchdog] Server started, PID:" &
           Integer'Image (Pid_To_Integer (New_Pid)));
      end if;

      Free (Cmd);
      for I in 1 .. N_Args loop
         Free (Args (I));
      end loop;
   exception
      when E : others =>
         Put_Line (Standard_Error,
           "[Watchdog] Failed to restart server: " &
           Exception_Message (E));
   end Restart_Server;

   -------------------
   -- Check_Server --
   -------------------

   procedure Check_Server is
      Pid         : constant Integer := Read_PID;
      Alive       : constant Boolean := Is_Process_Alive (Pid);
      HB_Age      : constant Duration := Get_Heartbeat_Age_S;
      Since_RS    : constant Time_Span := Clock - Last_Restart;
      Since_RS_D  : constant Duration := To_Duration (Since_RS);
   begin
      if Since_RS_D < Restart_Cooldown then
         return;  --  Still in cooldown after previous restart
      end if;

      if not Alive then
         Last_Restart := Clock;
         Restart_Server (Pid);
      elsif HB_Age > HB_Stale_Limit then
         Last_Restart := Clock;
         Put_Line (Standard_Error,
           "[Watchdog] Heartbeat stale for" & Duration'Image (HB_Age) &
           "s, PID" & Integer'Image (Pid) & " appears frozen.");
         Restart_Server (Pid);
      end if;
   end Check_Server;

--  Program start
begin
   --  [DO NOT REMOVE THIS] LAUNCH GUARD
   --  Refuse to run if not launched through run.py orchestration.
   --  This prevents orphaned processes, broken health monitoring, and
   --  resource leaks from direct binary execution.
   if not Ada.Environment_Variables.Exists ("ADLAIDE_WATCHDOG_ORCHESTRATED") then
      Put_Line (Standard_Error,
        "[Watchdog] FATAL: Cannot run adelaide_watchdog directly.");
      Put_Line (Standard_Error,
        "[Watchdog] This binary MUST be launched through run.py (or run.sh).");
      Put_Line (Standard_Error,
        "[Watchdog] Direct execution bypasses orchestration and causes resource leaks.");
      Put_Line (Standard_Error,
        "[Watchdog] Use: ./run.py --no-gui   OR   python3 run.py");
      Ada.Command_Line.Set_Exit_Status (Ada.Command_Line.Failure);
      return;
   end if;

   --  Parse --oneshot flag
   if Ada.Command_Line.Argument_Count > 0
     and then Ada.Command_Line.Argument (1) = "--oneshot"
   then
      Oneshot := True;
   end if;

   if not Oneshot then
      Put_Line (Standard_Error,
        "[Watchdog] Adelaide Watchdog process started. " &
        "Monitoring server via run/ directory...");
   end if;

   loop
      Check_Server;

      exit when Oneshot;

      delay Check_Interval;
   end loop;
exception
   when E : others =>
      Put_Line (Standard_Error,
        "[Watchdog] Fatal error: " & Exception_Message (E));
      Ada.Command_Line.Set_Exit_Status (Ada.Command_Line.Failure);
end Adelaide_Watchdog;
