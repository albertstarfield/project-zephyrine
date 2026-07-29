pragma SPARK_Mode (Off);
--  c_binding: NASA cFE Command Ingest (CI_LAB) integration
with Ada.Text_IO; use Ada.Text_IO;

package body CFS_Command_Router is

   Initialized   : Boolean := False;
   Command_Count : Natural := 0;

   procedure Initialize is
   begin
      if Initialized then
         return;
      end if;
      Put_Line ("[CFS-CI] Initializing cFS Command Router...");
      Initialized := True;
      Command_Count := 0;
      Put_Line ("[CFS-CI] Command Router ready.");
   end Initialize;

   procedure Route_Command (Cmd : Command) is
   begin
      Command_Count := Command_Count + 1;

      --  TODO: Route to appropriate handler based on Cmd_Type
      --  For now, log the command
      Put_Line ("[CFS-CI] CMD#" & Natural'Image (Command_Count) &
                " Type=" & Cmd_Type'Image (Cmd.Cmd_Type) &
                " Len=" & Natural'Image (Cmd.Cmd_Len));
   end Route_Command;

   procedure Register_Handler (Cmd_Type : Cmd_Type; Handler_Name : String) is
   begin
      --  TODO: Store handler mapping in internal table
      Put_Line ("[CFS-CI] Registered handler: " & Handler_Name &
                " for " & Cmd_Type'Image (Cmd_Type));
   end Register_Handler;

   function Get_Command_Count return Natural is
   begin
      return Command_Count;
   end Get_Command_Count;

   procedure Reset_Stats is
   begin
      Command_Count := 0;
   end Reset_Stats;

end CFS_Command_Router;
