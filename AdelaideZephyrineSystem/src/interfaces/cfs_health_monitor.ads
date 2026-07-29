pragma SPARK_Mode (Off);
--  c_binding: NASA cFE Health & Safety (HS) app integration
--  Wraps cFS HS app for Adelaide health monitoring
package CFS_Health_Monitor is

   --  Health check result
   type Health_Status is (Healthy, Degraded, Critical, Failed);

   --  Application health record
   type App_Health is record
      App_Name   : String (1 .. 64);
      App_Len    : Natural := 0;
      Status     : Health_Status := Healthy;
      CPU_Pct    : Float := 0.0;
      Mem_Pct    : Float := 0.0;
      Watchdog   : Boolean := True;
   end record;

   --  Initialize the cFS Health Monitor
   procedure Initialize with Pre => True, Post => True;

   --  Check health of a named application
   function Check_App_Health (App_Name : String) return Health_Status
     with Pre => App_Name'Length > 0;

   --  Get overall system health
   function Get_System_Health return Health_Status
     with Pre => True;

   --  Enable/disable watchdog for an app
   procedure Set_Watchdog (App_Name : String; Enabled : Boolean)
     with Pre => App_Name'Length > 0;

   --  Reset health counters
   procedure Reset_Counters with Pre => True, Post => True;

end CFS_Health_Monitor;
