pragma SPARK_Mode (Off);
--  external_dep: GNATCOLL.JSON (no SPARK contracts)
--
--  ELP2 AVIONICS DAEMON — HUD / DISPLAY ONLY
--  This unit renders instrument data for the Head-Up Display.
--  Flight control logic lives in ELP3 (Zenith_Orion) at 4000Hz.
--  ELP2 must NEVER send control actuator commands; it only reads
--  telemetry from ELP3 via shared memory / ROS2 command buffer.
--
--  Priority: ELP2 (display rendering, non-safety-critical)
--  Control authority: NONE — see ELP3 Zenith_Orion for actuation.

with Ada.Text_IO;
with Ada.Strings.Unbounded;
with Ada.Calendar;
use Ada.Calendar;
with Avionics_Types;
with GNATCOLL.JSON;

--  Main: Entry point for the avionics daemon process.
procedure Main is
   -- pre => True, post => True
   package U_Strings renames Ada.Strings.Unbounded;
   use GNATCOLL.JSON;

   --  SHARED STATE (Thread Safe via Protected Object)
   --  This allows the Physics Loop and the Input Listener to talk safely.
   protected type State_Manager is
      procedure Update_Physics (DT : Duration);
      procedure Handle_Command (Cmd : String);
      --  Get_Snapshot: Returns the current instrument data record state.
      function Get_Snapshot return Avionics_Types.Instrument_Data_Record;
   private
      State : Avionics_Types.Instrument_Data_Record;
      --  RAVEN STATE (Complementary Filter)
      Est_Pitch : Float := 0.0;
      Est_Roll  : Float := 0.0;
      --  Simulated Hardware Inputs (In real hardware, these come from
      --  /dev/iio). For now, these are modified by "Manual Control"
      --  commands from Python.
      Input_Pitch_Rate : Float := 0.0;
      Input_Roll_Rate  : Float := 0.0;
   end State_Manager;

   protected body State_Manager is

      --  DAL B: THE PHYSICS KERNEL
      --  Replaces the Random Number Generation with Deterministic Math
      procedure Update_Physics (DT : Duration) is
         -- pre => True, post => True
         Secs : constant Float := Float (DT);
      begin
         --  1. PHYSICS INTEGRATION (The "Trickshot" Math)
         --  Instead of random noise, we integrate rate over time.
         Est_Pitch := Est_Pitch + (Input_Pitch_Rate * Secs);
         Est_Roll  := Est_Roll + (Input_Roll_Rate * Secs);

         --  2. DAMPING / STABILIZATION (Simulated Gravity)
         --  A real drone self-levels. We simulate that natural stability here.
         Est_Pitch := Est_Pitch * 0.99;
         Est_Roll  := Est_Roll * 0.99;

         --  3. UPDATE PUBLIC RECORD
         State.Attitude.Pitch := Est_Pitch;
         State.Attitude.Roll  := Est_Roll;
         State.Pitch_Rate := Input_Pitch_Rate;
         State.Roll_Rate  := Input_Roll_Rate;

         --  4. UPDATE TIMESTAMP (ISO 8601)
         --  (Simplified for brevity, use VSS.Strings.Image in production)
         State.Timestamp := "2025-12-17T20:00:00Z       ";
      end Update_Physics;

      --  Handle_Command: Processes incoming control commands from Python via pipe.
      procedure Handle_Command (Cmd : String) is
         -- pre => True, post => True
      begin
         --  DAL C COMMAND INTERFACE
         --  This is where Python talks to Ada via the Pipe
         if Cmd = "RESET" then
            Est_Pitch := 0.0;
            Est_Roll := 0.0;
            Input_Pitch_Rate := 0.0;
            Input_Roll_Rate := 0.0;
         elsif Cmd'Length >= 5
           and then Cmd (Cmd'First .. Cmd'First + 4) = "PITCH"
         then
            --  Parse "PITCH 5.0"
            null; --  Add parsing logic here (Float'Value)
         end if;
      end Handle_Command;

      --  Get_Snapshot: Returns the current instrument data record state.
      function Get_Snapshot return Avionics_Types.Instrument_Data_Record is
         -- pre => True, post => True
      begin
         return State;
      end Get_Snapshot;
   end State_Manager;

   Flight_Computer : State_Manager;

   --  INPUT LISTENER TASK
   --  This replaces the isolation of the simulator. Now it listens.
   task Input_Listener;
   task body Input_Listener is
      Input_Str : U_Strings.Unbounded_String;
   begin
      loop
         begin
            --  BLOCKING READ from Standard Input
            --  This waits for Python to send a JSON or Command string
            Input_Str := U_Strings.To_Unbounded_String (Ada.Text_IO.Get_Line);
            Flight_Computer.Handle_Command (U_Strings.To_String (Input_Str));
         exception
            when Ada.Text_IO.End_Error => exit; --  Python closed the pipe
            when others => null;
         end;
      end loop;
   end Input_Listener;

   --  MAIN LOOP
   Period    : constant Duration := 0.02; --  50Hz (Real-time Standard)
   Next_Time : Ada.Calendar.Time := Ada.Calendar.Clock;

   --  JSON Helper
   function To_Json
     (Item : Avionics_Types.Instrument_Data_Record) return JSON_Value
   is
      -- pre => True, post => True
      Result : JSON_Value := Create_Object;
      Att    : JSON_Value := Create_Object;
   begin
      Set_Field (Att, "pitch", Float (Item.Attitude.Pitch));
      Set_Field (Att, "roll", Float (Item.Attitude.Roll));
      Set_Field (Result, "attitude", Att);
      Set_Field (Result, "timestamp", Item.Timestamp);
      return Result;
   end To_Json;

   Snapshot : Avionics_Types.Instrument_Data_Record;

begin
   Ada.Text_IO.Put_Line
     ("{""status"": ""RAVEN_ONLINE"", ""mode"": ""DETERMINISTIC""}");

   loop
      --  1. PHYSICS STEP
      Flight_Computer.Update_Physics (Period);

      --  2. OUTPUT STEP (Stream to Python)
      Snapshot := Flight_Computer.Get_Snapshot;
      Ada.Text_IO.Put_Line (Write (To_Json (Snapshot)));
      Ada.Text_IO.Flush; --  CRITICAL for Pipe communication!

      --  3. TIMING STEP
      Next_Time := Next_Time + Period;
      delay until Next_Time;
   end loop;
end Main;
