with Ada.Real_Time; use Ada.Real_Time;
with Model_Manager; use Model_Manager;

package Watchdog_Manager is
   pragma Spark_Mode (On);

   protected Inference_Monitor is
      procedure Start_Inference (Model : Model_Type; Now : Time);
      procedure Stop_Inference;
      procedure Set_Aborted;
      function Is_Aborted return Boolean;
      procedure Check_Timeout
        (Now         : Time;
         Limit       : Time_Span;
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
      procedure Check_Liveness (Now : Time; Limit : Time_Span; OK : out Boolean);
   private
      Last_Heartbeat : Time := Time_Of (0, Time_Span_Zero);
   end AWS_Server_Monitor;

   task Watchdog_Task is
      pragma Storage_Size (1024 * 1024);
   end Watchdog_Task;

end Watchdog_Manager;
