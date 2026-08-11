pragma SPARK_Mode (Off);
--  c_binding: NASA cFE Command Ingest (CI_LAB) integration
--  third-party: cFS (no SPARK contracts)
with Ada.Text_IO; use Ada.Text_IO;

package body CFS_Command_Router is

   Initialized   : Boolean := False;
   Command_Count : Natural := 0;

   -- @test: Initialize covered by sabotage_verifier
   -- Procedure Initialize: TODO document purpose and behavior
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

   -- @test: Route_Command covered by sabotage_verifier
   -- Procedure Route_Command: TODO document purpose and behavior
   procedure Route_Command (Cmd : Command) is
   begin
      Command_Count := Command_Count + 1;

      --  TODO: Route to appropriate handler based on Cmd_Type
      --  For now, log the command
      Put_Line ("[CFS-CI] CMD#" & Natural'Image (Command_Count) &
                " Type=" & Cmd_Type'Image (Cmd.Cmd_Kind) &
                " Len=" & Natural'Image (Cmd.Cmd_Len));
   end Route_Command;

   -- @test: Register_Handler covered by sabotage_verifier
   -- Procedure Register_Handler: TODO document purpose and behavior
   procedure Register_Handler (Cmd_Kind : Cmd_Type; Handler_Name : String) is
   begin
      --  TODO: Store handler mapping in internal table
      Put_Line ("[CFS-CI] Registered handler: " & Handler_Name &
                 " for " & Cmd_Type'Image (Cmd_Kind));
   end Register_Handler;

   -- @test: Get_Command_Count covered by sabotage_verifier
   function Get_Command_Count return Natural is
   begin
      return Command_Count;
   end Get_Command_Count;

   -- @test: Reset_Stats covered by sabotage_verifier
   -- Procedure Reset_Stats: TODO document purpose and behavior
   procedure Reset_Stats is
   begin
      Command_Count := 0;
   end Reset_Stats;

end CFS_Command_Router;
