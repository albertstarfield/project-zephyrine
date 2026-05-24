with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;

package Scheduler_Manager is
   pragma Spark_Mode (Off);
   procedure Initialize;
   procedure Schedule (Delay_Seconds : Integer; Prompt : String);
end Scheduler_Manager;
