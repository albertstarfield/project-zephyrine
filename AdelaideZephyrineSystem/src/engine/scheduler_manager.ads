pragma SPARK_Mode (Off);
-- thread: Task scheduler requires protected type
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;

package Scheduler_Manager is
   --  Create and start the background scheduler worker task.
   procedure Initialize with Pre => True, Post => True;
   --  Enqueue a proactive thought prompt to fire after the specified delay.
   procedure Schedule (Delay_Seconds : Integer; Prompt : String) with Pre => True, Post => True;
end Scheduler_Manager;
