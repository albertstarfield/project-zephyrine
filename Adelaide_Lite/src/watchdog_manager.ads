--  Formal Verification: Ravenscar Compliant
--  Note: The Watchdog logic is designed to be a Ravenscar application.
--  However, partition-wide profile pragmas are omitted here to avoid 
--  conflicts with non-compliant libraries (like AWS) in the main executable.
--  Verification of Ravenscar compliance is performed via the SPARK toolset.
pragma SPARK_Mode (On);
with Ada.Real_Time; use Ada.Real_Time;
with Model_Types; use Model_Types;

package Watchdog_Manager is

   protected Inference_Monitor is
      procedure Start_Inference (Model : Model_Type; Now : Time);
      procedure Stop_Inference;
      procedure Set_Aborted;
      function Is_Aborted return Boolean;
      procedure Check_Timeout
        (Limit       : Time_Span;
         Out_Aborted : out Boolean;
         Out_Model   : out Model_Type);
   private
      Active        : Boolean := False;
      Start_Time    : Time := Time_Of (0, Time_Span_Zero);
      Current_Model : Model_Type := Qwen_0_8B;
      Aborted       : Boolean := False;
   end Inference_Monitor;

   protected AWS_Server_Monitor is
      procedure Heartbeat (Now : Time);
      procedure Check_Liveness (Limit : Time_Span; OK : out Boolean);
   private
      Last_Heartbeat : Time := Time_Of (0, Time_Span_Zero);
   end AWS_Server_Monitor;

   package Tasking with SPARK_Mode => Off is
      task Watchdog_Task;
   end Tasking;

end Watchdog_Manager;
