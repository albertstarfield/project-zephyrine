pragma SPARK_Mode (Off);
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;

package Scheduler_Manager is
   procedure Initialize;
   procedure Schedule (Delay_Seconds : Integer; Prompt : String);
end Scheduler_Manager;
