pragma SPARK_Mode (Off);
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
      --  Verify by checking the command name via ps.
      begin
         declare
            Cmd_File : File_Type;
            Line     : Unbounded_String;
         begin
            --  Run: ps -p PID -o comm=
            --  Returns the command name (truncated to 15 chars on macOS)
            --  We use GNAT.OS_Lib or just Ada.Text_IO to read from a pipe.
            --  Since we can't do popen in pure Ada, use a temp file approach:
            --  Actually, simpler: just check if the heartbeat file was written
            --  recently by THIS same binary.  If the heartbeat is stale
            --  (>30s old), the server is dead even if the PID was recycled.
            if not Exists (HB_File) then
               return False;
            end if;

            Open (Cmd_File, In_File, HB_File);
            if not End_Of_File (Cmd_File) then
               Line := To_Unbounded_String (Get_Line (Cmd_File));
            end if;
            Close (Cmd_File);

            --  Heartbeat file contains a monotonic timestamp.
            --  If it's more than 5 seconds old, the server is dead.
            declare
               HB_Time : constant Duration :=
                 Duration'Value (To_String (Line));
               Now     : constant Duration :=
                 To_Duration (Clock - Time_Of (0, Time_Span_Zero));
               Age     : constant Duration := Now - HB_Time;
            begin
               if Age > 5.0 then
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

      Write_Heartbeat;
   end Init;

   --------------------
   -- Write_Heartbeat --
   --------------------

   procedure Write_Heartbeat is
      F : File_Type;
   begin
      Create (F, Out_File, HB_File);
      declare
         T_Str : constant String :=
           Duration'Image (To_Duration (Clock - Time_Of (0, Time_Span_Zero)));
      begin
         Put_Line (F, T_Str);
      end;
      Close (F);
   end Write_Heartbeat;

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
