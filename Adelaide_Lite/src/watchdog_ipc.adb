pragma SPARK_Mode (Off);
with Ada.Text_IO;           use Ada.Text_IO;
with Ada.Directories;       use Ada.Directories;
with Ada.Real_Time;         use Ada.Real_Time;

package body Watchdog_IPC is

   Run_Dir   : constant String := "run";
   PID_File  : constant String := Run_Dir & "/adelaide_server.pid";
   HB_File   : constant String := Run_Dir & "/adelaide_server.heartbeat";
   Exit_File : constant String := Run_Dir & "/adelaide_server.exit_reason";

   function Get_PID return Integer;
   pragma Import (C, Get_PID, "getpid");

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
