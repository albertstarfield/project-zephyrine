pragma SPARK_Mode (Off);
--  Watchdog IPC — file-based cross-process state export
--
--  The main server writes its PID, heartbeat, and inference
--  state into a well-known directory so the external watchdog
--  process (adelaide_watchdog) can monitor it across process
--  boundaries without shared memory or socket dependencies.
--
--  Directory: run/  (relative to CWD at server startup)

package Watchdog_IPC is

   function Check_Single_Instance return Boolean;
   --  Checks if another adelaide_server instance is already running.
   --  Returns True if another instance is running (should exit).
   --  Returns False if safe to proceed (no other instance or stale PID).

   procedure Init;
   --  Creates the run/ directory (if absent) and writes PID + initial heartbeat.

   procedure Write_Heartbeat;
   --  Overwrites run/adelaide_server.heartbeat with the current monotonic time.
   --  Called from the server main loop every ~1 s.

   procedure Write_Exit_Reason (Reason : String; Signal_Or_Code : Integer);
   --  Writes an explicit exit reason and exit code/signal to run/adelaide_server.exit_reason
   --  before the server terminates.

end Watchdog_IPC;
