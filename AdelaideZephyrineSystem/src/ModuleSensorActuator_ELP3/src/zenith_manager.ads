pragma SPARK_Mode (Off);
-- c_binding: Zenith hardware FFI
with Zenith_Orion;

package Zenith_Manager is

   --  Protected object to store telemetry safely for non-SPARK consumers
   protected Telemetry_Store is
      procedure Update (Timing : Duration; Jitter_Max : Duration; Jitter_Avg : Duration) with Pre => True, Post => True;
      function Get_Timing return Duration with Pre => True, Post => True;
      --  Get_Jitter_Max: Returns the maximum observed jitter.
      function Get_Jitter_Max return Duration with Pre => True, Post => True;
      --  Get_Jitter_Avg: Returns the average observed jitter.
      function Get_Jitter_Avg return Duration with Pre => True, Post => True;
   private
      Current_Timing : Duration := 0.0;
      Current_J_Max  : Duration := 0.0;
      Current_J_Avg  : Duration := 0.0;
   end Telemetry_Store;

   task Zenith_Orion_Task is
      entry Start;
   end Zenith_Orion_Task;

end Zenith_Manager;
