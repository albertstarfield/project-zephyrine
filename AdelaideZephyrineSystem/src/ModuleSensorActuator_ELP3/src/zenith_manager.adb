pragma SPARK_Mode (Off);
-- c_binding: Zenith hardware FFI
with Ada.Calendar;
with Ada.Real_Time;
with Zenith_Orion;

package body Zenith_Manager is

   protected body Telemetry_Store is
      --  Update: Updates the telemetry store with new timing and jitter values.
      procedure Update
        (Timing : Duration; Jitter_Max : Duration; Jitter_Avg : Duration)
      is
      begin
         Current_Timing := Timing;
         Current_J_Max  := Jitter_Max;
         Current_J_Avg  := Jitter_Avg;
      end Update;

      --  Get_Timing: Returns the current loop timing duration.
      function Get_Timing return Duration is (Current_Timing);
      --  Get_Jitter_Max: Returns the maximum observed jitter.
      function Get_Jitter_Max return Duration is (Current_J_Max);
      --  Get_Jitter_Avg: Returns the average observed jitter.
      function Get_Jitter_Avg return Duration is (Current_J_Avg);
   end Telemetry_Store;

   task body Zenith_Orion_Task is
   begin
      accept Start;
      Zenith_Orion.Initialize;
      loop  --  Intentional: ELP3 paced loop runs until task termination by supervisor
         Zenith_Orion.Paced_Loop;
         
         declare
            J : constant Zenith_Orion.Jitter_Data :=
              Zenith_Orion.Get_Jitter_Profile;
         begin
            Telemetry_Store.Update
              (Zenith_Orion.Get_Current_Timing, J.Max_Jitter, J.Avg_Jitter);
         end;
      end loop;
   end Zenith_Orion_Task;

end Zenith_Manager;
