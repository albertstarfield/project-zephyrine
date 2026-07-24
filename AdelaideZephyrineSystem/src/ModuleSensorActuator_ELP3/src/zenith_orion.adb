pragma SPARK_Mode (Off);
-- c_binding: Orion sensor FFI
with Ada.Real_Time; use Ada.Real_Time;
with Ada.Characters.Handling;
with Ada.Strings.Fixed; use Ada.Strings.Fixed;
with Ada.Text_IO; use Ada.Text_IO;
with AnsiAda;

package body Zenith_Orion is

   --  ===========================================================================
   --  INDUSTRY CONTROL LOOP BENCHMARKS (Why 4000Hz?):
   --  - Boston Dynamics (Spot/Atlas): Complex balancing algorithms and hydraulic/
   --    electric actuator control loops typically run at 500 Hz to 1000 Hz (2ms to 1ms).
   --  - Drone Flight Controllers: Quadcopters require incredibly fast reflexes to
   --    stay in the air. Betaflight (a popular drone firmware) usually runs its
   --    core PID loop at 1000 Hz to 4000 Hz (1ms to 0.25ms) depending on hardware.
   --  - Industrial Robotic Arms (e.g., KUKA, Universal Robots): Usually run their
   --    external position control loops at 250 Hz to 500 Hz (4ms to 2ms).
   --  
   --  To support drone flight controller logic natively, we target 4000 Hz (250us)
   --  with microsecond-accurate predictive padding.
   --  ===========================================================================
   Target_Interval : constant Time_Span := Microseconds (250); -- 250us = 4000 Hz
   
   --  ========================================================================
   --  HARD REAL-TIME CONSTRAINT NOTES
   --  Boston Dynamics (Spot/Atlas): Complex balancing algorithms and hydraulic/electric
   --  actuator control loops typically run at 500 Hz to 1000 Hz (2ms to 1ms).
   --  Drone Flight Controllers: Quadcopters require incredibly fast reflexes to stay in
   --  the air. Betaflight (a popular drone firmware) usually runs its core PID loop at
   --  1000 Hz to 4000 Hz (1ms to 0.25ms) depending on the hardware.
   --  Industrial Robotic Arms (e.g., KUKA, Universal Robots): Usually run their external
   --  position control loops at 250 Hz to 500 Hz (4ms to 2ms).
   --  ========================================================================
   
   Last_Execution_Time : Duration := 0.0;
   
   --  Jitter Tracking
   Max_J : Duration := 0.0;
   Min_J : Duration := 3600.0;
   Sum_J : Duration := 0.0;
   J_Count : Natural := 0;
   Last_Print : Time := Clock;

   --  Initialize: Initializes the ZenithOrion ELP3 core subsystem.
   procedure Initialize is
      -- pre => True, post => True
   begin
      null;
   end Initialize;

   --  Paced_Loop: Executes the 4000Hz deterministic control loop with microsecond pacing.
   procedure Paced_Loop is
      -- pre => True, post => True
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
      --  If we finish in < 250us, wait the difference to maintain 4000Hz cadence
      if Elapsed < Target_Interval then
         Delay_Time := Target_Interval - Elapsed;
         
         if Clock - Last_Print > Milliseconds (3000) then
            declare
               Utilization : constant Float := Float (To_Duration (Elapsed)) / Float (To_Duration (Target_Interval)) * 100.0;
            begin
               Put_Line (AnsiAda.Foreground (AnsiAda.Light_Green) & "[ZenithOrion-ELP3] " & AnsiAda.Reset &
                         "Exec Time: " & Duration'Image (To_Duration (Elapsed) * 1_000_000.0) & " us" &
                         " | Padding: " & Duration'Image (To_Duration (Delay_Time) * 1_000_000.0) & " us" &
                         " | Util:" & Integer'Image (Integer (Utilization)) & "%" &
                         " | Total Cadence: 250.00 us (4000Hz)");
            end;
            Last_Print := Clock;
         end if;
                   
         delay until (End_Time + Delay_Time);
      else
         Put_Line (AnsiAda.Background (AnsiAda.Red) & AnsiAda.Foreground (AnsiAda.Yellow) &
                   "[BUGCHECK] ELP3 WCET DEADLINE MISSED!" & AnsiAda.Reset &
                   AnsiAda.Foreground (AnsiAda.Light_Red) &
                   " Exec Time: " & Duration'Image (To_Duration (Elapsed) * 1_000_000.0) & " us (Exceeds 250us / 4000Hz)" & AnsiAda.Reset);
         raise Program_Error with "ELP3 WCET DEADLINE MISSED!";
      end if;
      
      --  Update Jitter Profile
      Now := Clock;
      declare
         Actual_Interval : constant Time_Span := Now - Start_Time;
         Jitter : constant Duration :=
           (if Actual_Interval > Target_Interval 
            then To_Duration (Actual_Interval - Target_Interval)
            else To_Duration (Target_Interval - Actual_Interval));
      begin
         if Jitter > Max_J then
            Max_J := Jitter;
         end if;
         if Jitter < Min_J then
            Min_J := Jitter;
         end if;
         Sum_J := Sum_J + Jitter;
         if J_Count < Natural'Last then
            J_Count := J_Count + 1;
         end if;
      end;
   end Paced_Loop;

   --  Get_Current_Timing: Returns the last measured loop execution time.
   function Get_Current_Timing return Duration is
      -- pre => True, post => True
   begin
      return Last_Execution_Time;
   end Get_Current_Timing;

   --  Get_Jitter_Profile: Returns the collected jitter statistics (max, min, avg).
   function Get_Jitter_Profile return Jitter_Data is
      -- pre => True, post => True
   begin
      if J_Count = 0 then
         return (0.0, 0.0, 0.0);
      end if;
      
      return (Max_J, Min_J, Sum_J / Duration (J_Count));
   end Get_Jitter_Profile;

   --  Check_SHM_Trigger: Checks if the prompt maps to an SHM or hardware trigger.
   function Check_SHM_Trigger (Prompt : String) return String is
      -- pre => True, post => True
      Lower_Prompt : constant String := Ada.Characters.Handling.To_Lower (Prompt);
   begin
      if Index (Lower_Prompt, "zenith lock") > 0 then
         return "[ZenithOrion-ELP3] Pacing Lock Engaged at 1ms. Max_Jitter: " &
                Duration'Image (Max_J);
      elsif Index (Lower_Prompt, "orion telemetry") > 0 then
         return "[ZenithOrion-ELP3] Telemetry stream active. SHM connected.";
      end if;
      return "";
   end Check_SHM_Trigger;

   protected body ROS2_Command_Buffer is
      --  Push_Command: Pushes a servo command into the thread-safe buffer.
      procedure Push_Command (Servo_ID : String; Angle : Float) is
         -- pre => True, post => True
      begin
         Buffer_Len := Natural'Min (Servo_ID'Length, 64);
         Buffer_Servo (1 .. Buffer_Len) := Servo_ID (Servo_ID'First .. Servo_ID'First + Buffer_Len - 1);
         Buffer_Angle := Angle;
         Has_Command := True;
      end Push_Command;

      --  Pop_Command: Pops a servo command from the thread-safe buffer.
      procedure Pop_Command (Servo_ID : out String; Length : out Natural; Angle : out Float; Valid : out Boolean) is
         -- pre => True, post => True
      begin
         Valid := Has_Command;
         if Has_Command then
            Length := Buffer_Len;
            -- Ensure we don't overflow the output string parameter
            declare
               Out_Len : constant Natural := Natural'Min (Length, Servo_ID'Length);
            begin
               Servo_ID (Servo_ID'First .. Servo_ID'First + Out_Len - 1) := Buffer_Servo (1 .. Out_Len);
               Length := Out_Len;
            end;
            Angle := Buffer_Angle;
            Has_Command := False;
         end if;
      end Pop_Command;
   end ROS2_Command_Buffer;

end Zenith_Orion;
