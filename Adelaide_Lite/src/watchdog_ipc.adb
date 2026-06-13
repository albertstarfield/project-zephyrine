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

      if Kill_Result = 0 then
         --  Process exists => another instance is running
         return True;
      end if;

      --  Process doesn't exist => stale PID file from crash
      return False;
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
