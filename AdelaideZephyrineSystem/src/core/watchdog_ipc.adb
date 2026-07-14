pragma SPARK_Mode (Off);
-- shm: file-based cross-process state export via mmap
with Ada.Text_IO;           use Ada.Text_IO;
with Ada.Directories;       use Ada.Directories;
with Ada.Real_Time;         use Ada.Real_Time;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Interfaces.C;          use Interfaces.C;

package body Watchdog_IPC is

   Run_Dir   : constant String := "run";
   PID_File  : constant String := Run_Dir & "/adelaide_server.pid";
   HB_File   : constant String := Run_Dir & "/adelaide_server.heartbeat";
   Exit_File : constant String := Run_Dir & "/adelaide_server.exit_reason";

   function Get_PID return Integer;
   pragma Import (C, Get_PID, "getpid");

   --  kill(pid, 0) checks if a process exists without sending a signal.
   --  Returns 0 if process exists, -1 if not (ESRCH).
   function Kill (PID : Integer; Sig : Integer) return Integer;
   pragma Import (C, Kill, "kill");

   --  =====================================================================
   --  BACKGROUND HEARTBEAT TASK
   --  =====================================================================
   --  WHY: The main loop must never block on disk I/O. Under memory pressure
   --  (swap), file operations (Create, Put_Line, Rename) can take 10-20s.
   --  If the main loop blocks on Write_Heartbeat, the heartbeat file goes
   --  stale, and the watchdog kills the server — even though it's alive.
   --
   --  FIX: A dedicated background task handles all heartbeat file I/O.
   --  The main loop calls Update_Heartbeat() which is a protected procedure
   --  (fast, non-blocking, never waits for disk). The background task wakes
   --  up every 1s, reads the shared timestamp, and writes the file.
   --  Even if the file I/O blocks for 20s, the main loop continues.
   --  =====================================================================

   --  Shared state between main loop and heartbeat task
   protected HB_State is
      procedure Update;
      --  Called by main loop: stores current time as heartbeat timestamp.
      --  Fast, non-blocking, never waits for disk.

      function Get_Timestamp return Duration;
      --  Called by heartbeat task: returns the last stored timestamp.

      procedure Request_Stop;
      --  Called by main loop at shutdown: signals task to exit.

      function Should_Stop return Boolean;
      --  Called by heartbeat task: checks if stop was requested.
   private
      Latest_Timestamp : Duration := 0.0;
      Stop_Requested   : Boolean := False;
   end HB_State;

   protected body HB_State is
      procedure Update is
      begin
         Latest_Timestamp :=
           To_Duration (Clock - Time_Of (0, Time_Span_Zero));
      end Update;

      function Get_Timestamp return Duration is
      begin
         return Latest_Timestamp;
      end Get_Timestamp;

      procedure Request_Stop is
      begin
         Stop_Requested := True;
      end Request_Stop;

      function Should_Stop return Boolean is
      begin
         return Stop_Requested;
      end Should_Stop;
   end HB_State;

   --  Background task: writes heartbeat file every 1 second.
   --  Runs independently of the main loop. Even if disk I/O blocks
   --  for 20 seconds, the main loop continues uninterrupted.
   task Heartbeat_Task is
      entry Start;
      entry Stop;
   end Heartbeat_Task;

   task body Heartbeat_Task is
      Tmp_File : constant String := HB_File & ".tmp";
      F        : File_Type;
   begin
      accept Start;
      Put_Line (Standard_Error,
        "[Heartbeat-Task] Background heartbeat task started.");

      loop
         --  Check for shutdown request
         select
            accept Stop;
            Put_Line (Standard_Error,
              "[Heartbeat-Task] Stop requested. Exiting.");
            exit;
         else
            null;  --  Continue loop
         end select;

         --  Write heartbeat file (this may block under memory pressure,
         --  but it doesn't matter — the main loop is unaffected)
         begin
            --  [ATOMIC-WRITE] Write to temp file, then rename.
            --  This prevents the watchdog from reading a truncated file.
            Create (F, Out_File, Tmp_File);
            declare
               T_Str : constant String :=
                 Duration'Image (HB_State.Get_Timestamp);
            begin
               Put_Line (F, T_Str);
            end;
            Close (F);
            --  Atomic replace: rename tmp -> final
            Ada.Directories.Rename (Tmp_File, HB_File);
         exception
            when others =>
               --  Best effort — if write fails, old heartbeat stays valid.
               begin
                  if Ada.Directories.Exists (Tmp_File) then
                     Ada.Directories.Delete_File (Tmp_File);
                  end if;
               exception
                  when others => null;
               end;
         end;

         delay 1.0;
      end loop;

      Put_Line (Standard_Error,
        "[Heartbeat-Task] Background heartbeat task stopped.");
   end Heartbeat_Task;

   --  Track if heartbeat task has been started
   HB_Task_Started : Boolean := False;

   -------------------------
   -- Check_Single_Instance --
   -------------------------

   function Check_Single_Instance return Boolean is
      F           : File_Type;
      PID_Str     : Unbounded_String;
      Old_PID     : Integer;
      Kill_Result : Integer;
      Cmd         : Unbounded_String;
   begin
      --  No PID file => no other instance running
      if not Exists (PID_File) then
         return False;
      end if;

      --  Read PID from file
      begin
         Open (F, In_File, PID_File);
         if not End_Of_File (F) then
            PID_Str := To_Unbounded_String (Get_Line (F));
         end if;
         Close (F);
      exception
         when others =>
            --  Corrupted PID file, treat as stale
            return False;
      end;

      --  Parse PID
      begin
         Old_PID := Integer'Value (To_String (PID_Str));
      exception
         when others =>
            --  Corrupted PID, treat as stale
            return False;
      end;

      --  Can't check our own PID
      if Old_PID = Get_PID then
         return False;
      end if;

      --  Check if old process is still alive (kill(pid, 0))
      Kill_Result := Kill (Old_PID, 0);

      if Kill_Result /= 0 then
         --  Process doesn't exist => stale PID file from crash
         return False;
      end if;

      --  Process exists, but is it an adelaide_server?  PIDs get recycled
      --  by the OS, so PID 45220 might now be some unrelated process.
      --  Verify by checking the heartbeat file was written recently.
      begin
         declare
            Cmd_File : File_Type;
            Line     : Unbounded_String;
         begin
            if not Exists (HB_File) then
               return False;
            end if;

            Open (Cmd_File, In_File, HB_File);
            if not End_Of_File (Cmd_File) then
               Line := To_Unbounded_String (Get_Line (Cmd_File));
            end if;
            Close (Cmd_File);

            --  Heartbeat file contains a monotonic timestamp.
            --  If it's more than 2 seconds old, the server is dead.
            declare
               HB_Time : constant Duration :=
                 Duration'Value (To_String (Line));
               Now     : constant Duration :=
                 To_Duration (Clock - Time_Of (0, Time_Span_Zero));
               Age     : constant Duration := Now - HB_Time;
            begin
               if Age > 2.0 then
                  --  Heartbeat is stale => old server is dead, PID recycled
                  return False;
               end if;
            end;
         end;
      exception
         when others =>
            --  Can't read heartbeat => treat as stale
            return False;
      end;

      --  PID exists AND heartbeat is fresh => another instance is running
      return True;
   end Check_Single_Instance;

   ----------
   -- Init --
   ----------

   procedure Init is
      F : File_Type;
   begin
      if not Exists (Run_Dir) then
         Create_Path (Run_Dir);
      end if;

      if Exists (Exit_File) then
         Delete_File (Exit_File);
      end if;

      Create (F, Out_File, PID_File);
      Put_Line (F, Integer'Image (Get_PID));
      Close (F);

      --  Write initial heartbeat directly (before task starts)
      Write_Heartbeat;

      --  Start the background heartbeat task
      if not HB_Task_Started then
         Heartbeat_Task.Start;
         HB_Task_Started := True;
      end if;
   end Init;

   ---------------------
   -- Update_Heartbeat --
   ---------------------

   procedure Update_Heartbeat is
   begin
      --  Fast, non-blocking: just update the shared timestamp.
      --  The background task writes the actual file independently.
      HB_State.Update;
      --  Also update the in-memory watchdog monitor (no disk I/O)
      null;  --  AWS_Server_Monitor.Heartbeat is called separately in main loop
   end Update_Heartbeat;

   --------------------
   -- Write_Heartbeat --
   --------------------

   procedure Write_Heartbeat is
      F : File_Type;
      Tmp_File : constant String := HB_File & ".tmp";
   begin
      --  DIRECT file write — used only during Init and shutdown.
      --  For normal operation, use Update_Heartbeat instead.
      Create (F, Out_File, Tmp_File);
      declare
         T_Str : constant String :=
           Duration'Image (To_Duration (Clock - Time_Of (0, Time_Span_Zero)));
      begin
         Put_Line (F, T_Str);
      end;
      Close (F);
      Ada.Directories.Rename (Tmp_File, HB_File);
   exception
      when others =>
         begin
            if Ada.Directories.Exists (Tmp_File) then
               Ada.Directories.Delete_File (Tmp_File);
            end if;
         exception
            when others => null;
         end;
   end Write_Heartbeat;

   ----------------------------
   -- Shutdown_Heartbeat_Task --
   ----------------------------

   procedure Shutdown_Heartbeat_Task is
   begin
      if HB_Task_Started then
         HB_State.Request_Stop;
         --  Give the task one more cycle to notice the stop flag
         delay 1.5;
      end if;
   end Shutdown_Heartbeat_Task;

   -----------------------
   -- Write_Exit_Reason --
   -----------------------

   procedure Write_Exit_Reason (Reason : String; Signal_Or_Code : Integer) is
      F : File_Type;
   begin
      Create (F, Out_File, Exit_File);
      Put_Line (F, Reason);
      Put_Line (F, Integer'Image (Signal_Or_Code));
      Close (F);
   exception
      when others => null; -- Best effort, do not crash while crashing
   end Write_Exit_Reason;

end Watchdog_IPC;
