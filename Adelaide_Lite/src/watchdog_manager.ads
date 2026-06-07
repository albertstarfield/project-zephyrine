--  Formal Verification Profile: Ravenscar
--  The Watchdog is isolated as a Ravenscar application to ensure deterministic 
--  and formally verifiable tasking behavior. 
--
--  Note on Jorvik: While the larger project uses Jorvik (or standard Ada) 
--  to accommodate AWS and streaming complexities, this core monitor stays 
--  within the stricter Ravenscar subset.
--
--  Architectural Rationale:
--  "If an application cannot be reasonably expressed within the Ravenscar 
--  subset, it isn’t a Ravenscar application... That maxim is true for the 
--  Jorvik profile as well." (Rogers, 2021).
--
--  Citation (APA 7):
--  Rogers, P. (2021, May 26). An introduction to Jorvik, the new tasking 
--  profile in Ada 2022. AdaCore. 
--  https://www.adacore.com/blog/introduction-to-jorvik
pragma Profile (Ravenscar);
pragma Partition_Elaboration_Policy (Sequential);
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
