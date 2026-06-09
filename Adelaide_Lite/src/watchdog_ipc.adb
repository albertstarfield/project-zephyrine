pragma SPARK_Mode (Off);
with Ada.Text_IO;           use Ada.Text_IO;
with Ada.Directories;       use Ada.Directories;
with Ada.Real_Time;         use Ada.Real_Time;

package body Watchdog_IPC is

   Run_Dir   : constant String := "run";
   PID_File  : constant String := Run_Dir & "/adelaide_server.pid";
   HB_File   : constant String := Run_Dir & "/adelaide_server.heartbeat";

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

end Watchdog_IPC;
