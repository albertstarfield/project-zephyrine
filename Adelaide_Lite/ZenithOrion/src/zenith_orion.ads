pragma SPARK_Mode (Off);
package Zenith_Orion is

   type Jitter_Data is record
      Max_Jitter : Duration;
      Min_Jitter : Duration;
      Avg_Jitter : Duration;
   end record;

   --  Initialize ZenithOrion ELP3 core
   procedure Initialize;

   --  The 1ms Deterministic Loop
   --  ELP3: ZenithOrion - 1ms Pacing Lock (Deterministic)
   procedure Paced_Loop;

   function Get_Current_Timing return Duration;
   function Get_Jitter_Profile return Jitter_Data;

end Zenith_Orion;
