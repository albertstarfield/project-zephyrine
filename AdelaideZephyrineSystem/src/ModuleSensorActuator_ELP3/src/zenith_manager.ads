pragma SPARK_Mode (Off);
-- c_binding: Zenith hardware FFI
with Zenith_Orion;

package Zenith_Manager is

   --  Protected object to store telemetry safely for non-SPARK consumers
   protected Telemetry_Store is
      procedure Update (Timing : Duration; Jitter_Max : Duration; Jitter_Avg : Duration);
      function Get_Timing return Duration;
      function Get_Jitter_Max return Duration;
      function Get_Jitter_Avg return Duration;
   private
      Current_Timing : Duration := 0.0;
      Current_J_Max  : Duration := 0.0;
      Current_J_Avg  : Duration := 0.0;
   end Telemetry_Store;

   task Zenith_Orion_Task is
      entry Start;
   end Zenith_Orion_Task;

end Zenith_Manager;
