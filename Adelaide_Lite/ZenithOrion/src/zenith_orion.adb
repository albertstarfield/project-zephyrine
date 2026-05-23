with Ada.Real_Time; use Ada.Real_Time;

package body Zenith_Orion with SPARK_Mode => On is

   Target_Interval : constant Time_Span := Milliseconds (1);
   
   Last_Execution_Time : Duration := 0.0;
   
   --  Jitter Tracking
   Max_J : Duration := 0.0;
   Min_J : Duration := 3600.0;
   Sum_J : Duration := 0.0;
   J_Count : Natural := 0;

   procedure Initialize is
   begin
      null;
   end Initialize;

   procedure Paced_Loop is
      Start_Time : Time;
      End_Time   : Time;
      Elapsed    : Time_Span;
      Delay_Time : Time_Span;
      Now        : Time;
   begin
      Start_Time := Clock;
      
      --  Critical Deterministic Routine (ELP3)
      --  Place SPARK-verified logic here
      
      End_Time := Clock;
      Elapsed := End_Time - Start_Time;
      Last_Execution_Time := To_Duration (Elapsed);
      
      --  Dynamic Pacing Delay
      --  If we finish in < 1ms, wait the difference to maintain 1ms cadence
      if Elapsed < Target_Interval then
         Delay_Time := Target_Interval - Elapsed;
         delay until (End_Time + Delay_Time);
      end if;
      
      --  Update Jitter Profile
      Now := Clock;
      declare
         Actual_Interval : constant Time_Span := Now - Start_Time;
         Jitter : constant Duration := (if Actual_Interval > Target_Interval 
                                        then To_Duration (Actual_Interval - Target_Interval)
                                        else To_Duration (Target_Interval - Actual_Interval));
      begin
         if Jitter > Max_J then Max_J := Jitter; end if;
         if Jitter < Min_J then Min_J := Jitter; end if;
         Sum_J := Sum_J + Jitter;
         if J_Count < Natural'Last then
            J_Count := J_Count + 1;
         end if;
      end;
   end Paced_Loop;

   function Get_Current_Timing return Duration is
   begin
      return Last_Execution_Time;
   end Get_Current_Timing;

   function Get_Jitter_Profile return Jitter_Data is
   begin
      if J_Count = 0 then
         return (0.0, 0.0, 0.0);
      end if;
      
      return (Max_J, Min_J, Sum_J / Duration (J_Count));
   end Get_Jitter_Profile;

end Zenith_Orion;
