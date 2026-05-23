with Ada.Calendar;
with Ada.Real_Time;

package body Zenith_Manager with SPARK_Mode => On is

   protected body Telemetry_Store is
      procedure Update (Timing : Duration; Jitter_Max : Duration; Jitter_Avg : Duration) is
      begin
         Current_Timing := Timing;
         Current_J_Max  := Jitter_Max;
         Current_J_Avg  := Jitter_Avg;
      end Update;

      function Get_Timing return Duration is (Current_Timing);
      function Get_Jitter_Max return Duration is (Current_J_Max);
      function Get_Jitter_Avg return Duration is (Current_J_Avg);
   end Telemetry_Store;

   task body Zenith_Orion_Task is
      use type Ada.Calendar.Time;
      Last_Jitter_Reset : Ada.Calendar.Time := Ada.Calendar.Clock;
   begin
      accept Start;
      Zenith_Orion.Initialize;
      loop
         Zenith_Orion.Paced_Loop;
         
         --  Update local telemetry store (SPARK-friendly)
         declare
            J : constant Zenith_Orion.Jitter_Data := Zenith_Orion.Get_Jitter_Profile;
         begin
            Telemetry_Store.Update (Zenith_Orion.Get_Current_Timing, J.Max_Jitter, J.Avg_Jitter);
         end;
         
         --  The interval for jitter reset logic can be managed here if needed
         --  but we'll just update every loop for simplicity in the manager.
      end loop;
   end Zenith_Orion_Task;

end Zenith_Manager;
