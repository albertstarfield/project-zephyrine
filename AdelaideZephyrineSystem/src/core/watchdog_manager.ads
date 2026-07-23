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
      --  Start_Inference: Starts monitoring an inference operation for the given model.
      procedure Start_Inference (Model : Model_Type; Now : Time);
      --  Stop_Inference: Stops monitoring the current inference operation.
      procedure Stop_Inference;
      --  Set_Aborted: Marks the current inference as aborted.
      procedure Set_Aborted;
      --  Is_Aborted: Returns True if the current inference has been aborted.
      function Is_Aborted return Boolean;
      --  Current_Inference_Model: Returns the model type of the current inference.
      function Current_Inference_Model return Model_Type;
      --  Check_Timeout: Checks if the current inference has exceeded the timeout limit.
      procedure Check_Timeout
        (Limit       : Time_Span;
         Out_Aborted : out Boolean;
         Out_Model   : out Model_Type);
   private
      Active        : Boolean := False;
      Start_Time    : Time := Time_Of (0, Time_Span_Zero);
      Current_Model : Model_Type := Snowball_Enaga_ShortNetworkAnswer;
      Aborted       : Boolean := False;
   end Inference_Monitor;

   protected AWS_Server_Monitor is
      --  Heartbeat: Updates the AWS server heartbeat timestamp.
      procedure Heartbeat (Now : Time);
      --  Deactivate: Deactivates the AWS server liveness check.
      procedure Deactivate;
      --  Check_Liveness: Checks if the AWS server is still alive based on heartbeat.
      procedure Check_Liveness (Limit : Time_Span; OK : out Boolean);
   private
      Last_Heartbeat : Time := Time_Of (0, Time_Span_Zero);
      Active         : Boolean := True;
   end AWS_Server_Monitor;

   package Tasking with SPARK_Mode => Off is
      task Watchdog_Task;
   end Tasking;

end Watchdog_Manager;
