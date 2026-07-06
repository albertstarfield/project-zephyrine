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
   --  Also starts the background heartbeat task.

   procedure Update_Heartbeat;
   --  Updates the shared heartbeat timestamp (fast, non-blocking).
   --  The background task writes the actual file independently.
   --  Called from the server main loop every ~1 s.

   procedure Write_Heartbeat;
   --  DIRECT file write — used only during Init and shutdown.
   --  For normal operation, use Update_Heartbeat instead.

   procedure Write_Exit_Reason (Reason : String; Signal_Or_Code : Integer);
   --  Writes an explicit exit reason and exit code/signal to run/adelaide_server.exit_reason
   --  before the server terminates.

   procedure Shutdown_Heartbeat_Task;
   --  Signals the background heartbeat task to stop.
   --  Called during clean shutdown.

end Watchdog_IPC;
