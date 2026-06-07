--  Formal Verification Profile: Jorvik
--  Note: Ravenscar was found to be too restrictive due to dependencies on AWS 
--  and the need for multiple protected entries/procedures. Jorvik provides 
--  the necessary flexibility while still allowing formal verification of 
--  tasking properties in SPARK.
--
--  Architectural Rationale:
--  "If an application cannot be reasonably expressed within the Ravenscar 
--  subset, it isn’t a Ravenscar application... That maxim is true for the 
--  Jorvik profile as well. If an application 'genuinely requires' requeue 
--  statements, for example, maybe a larger subset is appropriate" 
--  (Rogers, 2021).
--
--  Citation (APA 7):
--  Rogers, P. (2021, May 26). An introduction to Jorvik, the new tasking 
--  profile in Ada 2022. AdaCore. 
--  https://www.adacore.com/blog/introduction-to-jorvik
pragma Profile (Jorvik);
pragma Partition_Elaboration_Policy (Sequential);
pragma SPARK_Mode (On);
with Ada.Real_Time; use Ada.Real_Time;
with Model_Types; use Model_Types;
with Model_Manager;

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
