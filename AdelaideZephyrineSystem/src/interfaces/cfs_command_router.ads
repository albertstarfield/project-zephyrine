pragma SPARK_Mode (Off);
--  c_binding: NASA cFE Command Ingest (CI_LAB) integration
--  Wraps cFS CI_LAB for Adelaide command routing
package CFS_Command_Router is

   --  Command types
   type Cmd_Type is (GNC, Telemetry, Health, Configuration, Custom);

   --  Command record
   type Command is record
      Cmd_Type    : Cmd_Type := Custom;
      Cmd_Data    : String (1 .. 256);
      Cmd_Len     : Natural := 0;
      Source      : String (1 .. 32);
      Source_Len  : Natural := 0;
   end record;

   --  Initialize the cFS Command Router
   procedure Initialize with Pre => True, Post => True;

   --  Route a command to the appropriate handler
   procedure Route_Command (Cmd : Command)
     with Pre => Cmd.Cmd_Len > 0;

   --  Register a command handler for a specific command type
   procedure Register_Handler (Cmd_Type : Cmd_Type; Handler_Name : String)
     with Pre => Handler_Name'Length > 0;

   --  Get command statistics
   function Get_Command_Count return Natural
     with Pre => True;

   --  Reset command statistics
   procedure Reset_Stats with Pre => True, Post => True;

end CFS_Command_Router;
