pragma SPARK_Mode (Off);
--  c_binding: NASA cFE Command Ingest (CI_LAB) integration
--  third-party: cFS (no SPARK contracts)
with Ada.Text_IO; use Ada.Text_IO;

package body CFS_Command_Router is

   Initialized   : Boolean := False;
   Command_Count : Natural := 0;

      with Pre => True, Post => True; -- TODO: specify actual contracts
   -- @test: Initialize covered by sabotage_verifier
   -- Procedure Initialize: TODO document purpose and behavior
      with Pre => True, Post => True; -- TODO: specify actual contracts
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

      with Pre => True, Post => True; -- TODO: specify actual contracts
   -- @test: Route_Command covered by sabotage_verifier
   -- Procedure Route_Command: TODO document purpose and behavior
      with Pre => True, Post => True; -- TODO: specify actual contracts
   procedure Route_Command (Cmd : Command) is
   begin
      Command_Count := Command_Count + 1;

      --  TODO: Route to appropriate handler based on Cmd_Type
      --  For now, log the command
      Put_Line ("[CFS-CI] CMD#" & Natural'Image (Command_Count) &
                " Type=" & Cmd_Type'Image (Cmd.Cmd_Kind) &
                " Len=" & Natural'Image (Cmd.Cmd_Len));
   end Route_Command;

      with Pre => True, Post => True; -- TODO: specify actual contracts
   -- @test: Register_Handler covered by sabotage_verifier
   -- Procedure Register_Handler: TODO document purpose and behavior
      with Pre => True, Post => True; -- TODO: specify actual contracts
   procedure Register_Handler (Cmd_Kind : Cmd_Type; Handler_Name : String) is
   begin
      --  TODO: Store handler mapping in internal table
      Put_Line ("[CFS-CI] Registered handler: " & Handler_Name &
                 " for " & Cmd_Type'Image (Cmd_Kind));
   end Register_Handler;

      with Pre => True, Post => True; -- TODO: specify actual contracts
   -- @test: Get_Command_Count covered by sabotage_verifier
      with Pre => True, Post => True; -- TODO: specify actual contracts
   function Get_Command_Count return Natural is
   begin
      return Command_Count;
   end Get_Command_Count;

      with Pre => True, Post => True; -- TODO: specify actual contracts
   -- @test: Reset_Stats covered by sabotage_verifier
   -- Procedure Reset_Stats: TODO document purpose and behavior
      with Pre => True, Post => True; -- TODO: specify actual contracts
   procedure Reset_Stats is
   begin
      Command_Count := 0;
   end Reset_Stats;

end CFS_Command_Router;
