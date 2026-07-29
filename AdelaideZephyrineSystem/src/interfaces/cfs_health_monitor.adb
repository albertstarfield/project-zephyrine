pragma SPARK_Mode (Off);
--  c_binding: NASA cFE Health & Safety (HS) app integration
with Ada.Text_IO; use Ada.Text_IO;

package body CFS_Health_Monitor is

   Initialized : Boolean := False;
   System_Stat : Health_Status := Healthy;

   procedure Initialize is
   begin
      if Initialized then
         return;
      end if;
      Put_Line ("[CFS-HS] Initializing cFS Health Monitor...");
      Initialized := True;
      System_Stat := Healthy;
      Put_Line ("[CFS-HS] Health Monitor ready.");
   end Initialize;

   function Check_App_Health (App_Name : String) return Health_Status is
   begin
      --  TODO: Query cFS HS app for real health data via Software Bus
      --  For now, return Healthy (all apps assumed OK)
      return Healthy;
   end Check_App_Health;

   function Get_System_Health return Health_Status is
   begin
      return System_Stat;
   end Get_System_Health;

   procedure Set_Watchdog (App_Name : String; Enabled : Boolean) is
   begin
      --  TODO: Send HS command to enable/disable watchdog
      null;
   end Set_Watchdog;

   procedure Reset_Counters is
   begin
      System_Stat := Healthy;
   end Reset_Counters;

end CFS_Health_Monitor;
