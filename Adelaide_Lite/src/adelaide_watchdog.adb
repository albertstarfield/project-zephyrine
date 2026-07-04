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
with Interfaces.C;          use Interfaces.C;

procedure Adelaide_Watchdog is

   --  [DO NOT REMOVE] C FFI for graceful shutdown (SIGINT/SIGTERM)
   procedure Install_Shutdown_Handlers;
   pragma Import (C, Install_Shutdown_Handlers, "install_shutdown_handlers");
   function Is_Shutdown_Requested return Interfaces.C.int;
   pragma Import (C, Is_Shutdown_Requested, "is_shutdown_requested");
   function Last_Signal_Received return Interfaces.C.int;
   pragma Import (C, Last_Signal_Received, "last_signal_received");

   --  _exit() bypasses atexit handlers — prevents Metal assertion failure
   procedure C_Exit (Status : Interfaces.C.int);
   pragma Import (C, C_Exit, "_exit");

   function Is_Another_Watchdog_Running return Boolean;
   procedure Write_Watchdog_PID;
   procedure Write_Watchdog_Heartbeat;
   function Read_PID return Integer;
   function Is_Process_Alive (Pid : Integer) return Boolean;
   function Get_Heartbeat_Age_S return Duration;
   function Read_Args return String;
   procedure Restart_Server (Old_Pid : Integer);
   procedure Check_Server;
   function Get_Port return String;
   function Get_Host return String;
   procedure Check_All_APIs;

   Shutdown_Requested : exception;

   Run_Dir    : constant String := "run";
   PID_File   : constant String := Run_Dir & "/adelaide_server.pid";
   HB_File    : constant String := Run_Dir & "/adelaide_server.heartbeat";
   Args_File  : constant String := Run_Dir & "/adelaide_server.args";
   Server_Bin : constant String := "bin/adelaide_server";
   WD_PID_File : constant String := Run_Dir & "/adelaide_watchdog.pid";
   Shutdown_Flag : constant String := Run_Dir & "/.shutdown_requested";
   WD_HB_File    : constant String :=
     Run_Dir & "/adelaide_watchdog.heartbeat";

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

   function Get_PID return Integer;
   pragma Import (C, Get_PID, "getpid");

   --  Check if another watchdog is already running.
   --  Uses PID file + heartbeat freshness (same logic as server).
   function Is_Another_Watchdog_Running return Boolean is
      F       : File_Type;
      S       : String (1 .. 16);
      L       : Natural;
      Old_PID : Integer;
   begin
      if not Exists (WD_PID_File) then
         return False;
      end if;

      --  Read PID
      begin
         Open (F, In_File, WD_PID_File);
         Get_Line (F, S, L);
         Close (F);
         Old_PID := Integer'Value (S (1 .. L));
      exception
         when others => return False;
      end;

      --  Can't check our own PID
      if Old_PID = Get_PID then
         return False;
      end if;

      --  Check if process is alive
      if Sys_Kill (Old_PID, 0) /= 0 then
         return False;  --  Dead => stale
      end if;

      --  Process alive -- verify it's actually a watchdog by checking
      --  if it wrote a heartbeat recently (within 30s).
      --  If the heartbeat is stale, the PID was recycled.
      declare
         HB_F       : File_Type;
         HB_S       : String (1 .. 32);
         HB_L       : Natural;
         HB_Time    : Duration;
         Now_Time   : constant Time := Clock;
         Now_Dur    : constant Duration :=
           To_Duration (Now_Time - Time_Of (0, Time_Span_Zero));
      begin
         if not Exists (WD_HB_File) then
            return False;
         end if;
         Open (HB_F, In_File, WD_HB_File);
         Get_Line (HB_F, HB_S, HB_L);
         Close (HB_F);
         HB_Time := Duration'Value (HB_S (1 .. HB_L));
         if Now_Dur - HB_Time > 2.0 then
            return False;  --  Stale heartbeat (>5s) => recycled PID
         end if;
      exception
         when others => return False;
      end;

      return True;  --  Another watchdog is alive with fresh heartbeat
   end Is_Another_Watchdog_Running;

   --  Write our own PID file and heartbeat for other instances to detect.
   procedure Write_Watchdog_PID is
      F : File_Type;
   begin
      if not Exists (Run_Dir) then
         Create_Path (Run_Dir);
      end if;
      Create (F, Out_File, WD_PID_File);
      Put_Line (F, Integer'Image (Get_PID));
      Close (F);
   end Write_Watchdog_PID;

   procedure Write_Watchdog_Heartbeat is
      F : File_Type;
      Tmp_File : constant String := WD_HB_File & ".tmp";
      T : constant Duration :=
        To_Duration (Clock - Time_Of (0, Time_Span_Zero));
   begin
      --  [ATOMIC-WRITE] Write to tmp, then rename. Same pattern as
      --  the server's Write_Heartbeat to prevent race conditions.
      Create (F, Out_File, Tmp_File);
      Put_Line (F, Duration'Image (T));
      Close (F);
      Ada.Directories.Rename (Tmp_File, WD_HB_File);
   exception
      when others =>
         begin
            if Ada.Directories.Exists (Tmp_File) then
               Ada.Directories.Delete_File (Tmp_File);
            end if;
         exception
            when others => null;
         end;
   end Write_Watchdog_Heartbeat;

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

   -----------------
   -- Read_Args --
   -----------------

   --  Reads server launch arguments from run/adelaide_server.args.
   --  Returns the arguments as a single string (may be empty).
   --  The file is written by run.py before launching the server.

   function Read_Args return String is
      F : File_Type;
      S : String (1 .. 256);
      L : Natural;
   begin
      if not Exists (Args_File) then
         return "";
      end if;
      Open (F, In_File, Args_File);
      if End_Of_File (F) then
         Close (F);
         return "";
      end if;
      Get_Line (F, S, L);
      Close (F);
      return S (1 .. L);
   exception
      when others =>
         return "";
   end Read_Args;

   ----------------------
   -- Restart_Server --
   ----------------------

   procedure Restart_Server (Old_Pid : Integer) is
      Alr       : String_Access;
      Cmd       : String_Access;
      Args      : Argument_List (1 .. 8);
      New_Pid   : Process_Id;
      N_Args    : Natural := 0;
      Raw_Args  : constant String := Read_Args;
      Arg_Start : Natural := Raw_Args'First;
      Arg_End   : Natural;
   begin
      Put_Line (Standard_Error,
        "[Watchdog] Server (PID" & Integer'Image (Old_Pid) &
        ") is dead or frozen. Restarting...");

       --  Kill old process if still hanging around
       if Old_Pid > 0 and then Is_Process_Alive (Old_Pid) then
          Put_Line (Standard_Error,
            "[Watchdog] Sending SIGTERM to old PID" &
            Integer'Image (Old_Pid));
          declare
             Unused_Result : Integer;
             Wait_Loops    : Integer := 0;
          begin
             Unused_Result := Sys_Kill (Old_Pid, 15);  --  SIGTERM
             --  Wait up to 5 seconds for it to exit
             while Wait_Loops < 5 and then Is_Process_Alive (Old_Pid) loop
                delay 1.0;
                Wait_Loops := Wait_Loops + 1;
             end loop;
             
             --  If still alive, force kill
             if Is_Process_Alive (Old_Pid) then
                Put_Line (Standard_Error,
                  "[Watchdog] Process ignored SIGTERM. Sending SIGKILL to PID" &
                  Integer'Image (Old_Pid));
                Unused_Result := Sys_Kill (Old_Pid, 9);  --  SIGKILL
                delay 1.0;
             end if;
          end;
       end if;

      --  Delete stale heartbeat so run.py's new server doesn't bypass Check_Single_Instance
      if Exists (HB_File) then
         Delete_File (HB_File);
      end if;

      --  We rely on run.py to restart the server (prevents double spawning)
      Put_Line (Standard_Error, "[Watchdog] Server killed. run.py will handle restart.");
      
      --  Note: run.py's wait() will unblock, see the exit code,
      --  dump the panic logs, and spawn the new process itself.

   exception
      when E : others =>
         Put_Line (Standard_Error,
           "[Watchdog] Failed to kill stale server: " &
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

      --  Check if run.py wrote the shutdown flag (intentional Ctrl+C).
      --  If so, the server was stopped on purpose — do NOT restart.
      if Exists (Shutdown_Flag) then
         Put_Line (Standard_Error,
           "[Watchdog] Shutdown flag detected. Server was stopped" &
           " intentionally. Not restarting.");
         --  Clean up the flag so a fresh run.py launch starts clean.
         begin
            Delete_File (Shutdown_Flag);
         exception
            when others => null;
         end;
         --  Clean up server IPC files too so single-instance check
         --  doesn't block the next launch.
         begin
            if Exists (PID_File) then
               Delete_File (PID_File);
            end if;
            if Exists (HB_File) then
               Delete_File (HB_File);
            end if;
         exception
            when others => null;
         end;
         --  Exit the watchdog cleanly — no restart.
         raise Shutdown_Requested;
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

   ----------------------
   -- Check_All_APIs --
   ----------------------

   --  Tests each API endpoint via HTTP GET with ?ping=true.
   --  Prints status of each endpoint.  Uses curl for HTTP.
   --  Port is read from environment variable ADLAIDE_SERVER_PORT.

   Endpoints : constant array (1 .. 10) of access constant String :=
     [new String'("/api/version"),
      new String'("/api/tags"),
      new String'("/api/power"),
      new String'("/v1/models"),
      new String'("/v1/embeddings"),
      new String'("/api/chat"),
      new String'("/v1/audio/speech"),
      new String'("/v1/audio/transcriptions"),
      new String'("/api/telemetry"),
      new String'("/api/ps")];

   --  Port/Host resolution: args > env vars > defaults
   function Get_Port return String is
   begin
      for I in 1 .. Ada.Command_Line.Argument_Count loop
         if Ada.Command_Line.Argument (I) = "--port"
           and then I < Ada.Command_Line.Argument_Count
         then
            return Ada.Command_Line.Argument (I + 1);
         end if;
      end loop;
      if Ada.Environment_Variables.Exists ("ADLAIDE_SERVER_PORT") then
         return Ada.Environment_Variables.Value ("ADLAIDE_SERVER_PORT");
      end if;
      return "11420";
   end Get_Port;

   function Get_Host return String is
   begin
      for I in 1 .. Ada.Command_Line.Argument_Count loop
         if Ada.Command_Line.Argument (I) = "--host"
           and then I < Ada.Command_Line.Argument_Count
         then
            return Ada.Command_Line.Argument (I + 1);
         end if;
      end loop;
      if Ada.Environment_Variables.Exists ("ADLAIDE_SERVER_HOST") then
         return Ada.Environment_Variables.Value ("ADLAIDE_SERVER_HOST");
      end if;
      return "127.0.0.1";
   end Get_Host;

   procedure Check_All_APIs is
      Port     : constant String := Get_Port;
      Host     : constant String := Get_Host;
      Base_URL : constant String := "http://" & Host & ":" & Port;
      Success  : Boolean;
      Ret_Code : Integer;
   begin
      Put_Line (Standard_Error,
        "[Watchdog] === API Health Check (port " & Port & ") ===");
      for Ep of Endpoints loop
         declare
            Ep_Name : constant String := Ep.all;
            Cmd   : constant String :=
              "curl -s -o /dev/null -w '%{http_code}' " &
              "--max-time 2 " &
              Base_URL & Ep_Name & "?ping=true";
            Args  : Argument_List (1 .. 2);
         begin
            Args (1) := new String'("-c");
            Args (2) := new String'(Cmd);
            Spawn
              (Program_Name => "/bin/sh",
               Args         => Args,
               Output_File  => "",
               Success      => Success,
               Return_Code  => Ret_Code);
            if Success and then Ret_Code = 0 then
               Put_Line (Standard_Error,
                 "[Watchdog]   " & Ep_Name & " : UP");
            else
               Put_Line (Standard_Error,
                 "[Watchdog]   " & Ep_Name & " : DOWN (code" &
                 Integer'Image (Ret_Code) & ")");
            end if;
            Free (Args (1));
            Free (Args (2));
         end;
      end loop;
      Put_Line (Standard_Error,
        "[Watchdog] === End Health Check ===");
   end Check_All_APIs;

--  Program start
begin
   --  [DO NOT REMOVE THIS] LAUNCH GUARD
   --  Refuse to run if not launched through run.py orchestration.
   --  This prevents orphaned processes, broken health monitoring, and
   --  resource leaks from direct binary execution.
   if not Ada.Environment_Variables.Exists
     ("ADLAIDE_WATCHDOG_ORCHESTRATED")
   then
      Put_Line (Standard_Error,
        "[Watchdog] FATAL: Cannot run adelaide_watchdog directly.");
      Put_Line (Standard_Error,
        "[Watchdog] This binary MUST be launched through " &
        "run.py (or run.sh).");
      Put_Line (Standard_Error,
        "[Watchdog] Direct execution bypasses orchestration and " &
        "causes resource leaks.");
      Put_Line (Standard_Error,
        "[Watchdog] Use: ./run.py --no-gui   OR   python3 run.py");
      Ada.Command_Line.Set_Exit_Status (Ada.Command_Line.Failure);
      return;
   end if;

   --  SINGLE-INSTANCE LOCK: Refuse to start if another watchdog is running
   if Is_Another_Watchdog_Running then
      Put_Line (Standard_Error,
        "[Watchdog] FATAL: Another adelaide_watchdog instance " &
        "is already running!");
      Put_Line (Standard_Error,
        "[Watchdog] Kill the existing one first: " &
        "kill $(cat run/adelaide_watchdog.pid)");
      Ada.Command_Line.Set_Exit_Status (Ada.Command_Line.Failure);
      return;
   end if;

   --  Write our own PID file so future instances can detect us
   Write_Watchdog_PID;

   --  [DO NOT REMOVE] Install SIGINT/SIGTERM handlers for graceful shutdown.
   Install_Shutdown_Handlers;

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

   declare
      API_Check_Count : Natural := 0;
   begin
      loop
         --  [DO NOT REMOVE] Graceful shutdown check (SIGINT/SIGTERM/SIGQUIT).
         if Is_Shutdown_Requested /= 0 then
            declare
               Sig_Num : constant Integer := Integer (Last_Signal_Received);
               Sig_Name : constant String :=
                 (case Sig_Num is
                  when 2  => "SIGINT",
                  when 3  => "SIGQUIT",
                  when 15 => "SIGTERM",
                  when others => "Signal" & Integer'Image (Sig_Num));
            begin
               Put_Line (Standard_Error,
                 "[Watchdog] " & Sig_Name & " received. Shutting down gracefully...");
            end;
            --  Clean up our own PID file
            if Exists (WD_PID_File) then
               Delete_File (WD_PID_File);
            end if;
            if Exists (Run_Dir & "/adelaide_watchdog.heartbeat") then
               Delete_File (Run_Dir & "/adelaide_watchdog.heartbeat");
            end if;
            Put_Line (Standard_Error,
              "[Watchdog] Clean shutdown complete.");
            C_Exit (0);
         end if;

         --  Update our heartbeat so future instances can verify we're alive
         Write_Watchdog_Heartbeat;

         Check_Server;

         --  Check all APIs every 30 seconds
         API_Check_Count := API_Check_Count + 1;
         if API_Check_Count >= 30 then
            API_Check_Count := 0;
            Check_All_APIs;
         end if;

         exit when Oneshot;

         delay Check_Interval;
      end loop;
   end;
exception
   when Shutdown_Requested =>
      --  Clean exit: server was stopped intentionally via run.py Ctrl+C.
      --  No error message, no non-zero exit code.
      null;
   when E : others =>
      Put_Line (Standard_Error,
        "[Watchdog] Fatal error: " & Exception_Message (E));
      Ada.Command_Line.Set_Exit_Status (Ada.Command_Line.Failure);
end Adelaide_Watchdog;
