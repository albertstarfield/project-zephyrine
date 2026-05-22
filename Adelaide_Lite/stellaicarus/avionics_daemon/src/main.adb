with Ada.Text_IO;
with Ada.Command_Line;
with Ada.Strings.Unbounded;
with Ada.Calendar;
with Avionics_Types;

-- JSON Libraries (VSS)
with VSS.JSON;
with VSS.Strings;
with VSS.JSON.Serialization;

procedure Main is
   package U_Strings renames Ada.Strings.Unbounded;
   
   -- SHARED STATE (Thread Safe via Protected Object)
   -- This allows the Physics Loop and the Input Listener to talk safely.
   protected type State_Manager is
      procedure Update_Physics(DT : Duration);
      procedure Handle_Command(Cmd : String);
      function Get_Snapshot return Avionics_Types.Instrument_Data_Record;
   private
      State : Avionics_Types.Instrument_Data_Record;
      -- RAVEN STATE (Complementary Filter)
      Est_Pitch : Float := 0.0;
      Est_Roll  : Float := 0.0;
      -- Simulated Hardware Inputs (In real hardware, these come from /dev/iio)
      -- For now, these are modified by "Manual Control" commands from Python
      Input_Pitch_Rate : Float := 0.0; 
      Input_Roll_Rate  : Float := 0.0;
   end State_Manager;

   protected body State_Manager is
      
      -- DAL B: THE PHYSICS KERNEL
      -- Replaces the Random Number Generation  with Deterministic Math
      procedure Update_Physics(DT : Duration) is
         Alpha : constant Float := 0.98; 
         Secs : constant Float := Float(DT);
      begin
         -- 1. PHYSICS INTEGRATION (The "Trickshot" Math)
         -- Instead of random noise, we integrate rate over time.
         Est_Pitch := Est_Pitch + (Input_Pitch_Rate * Secs);
         Est_Roll  := Est_Roll  + (Input_Roll_Rate * Secs);

         -- 2. DAMPING / STABILIZATION (Simulated Gravity)
         -- A real drone self-levels. We simulate that natural stability here.
         Est_Pitch := Est_Pitch * 0.99; 
         Est_Roll  := Est_Roll * 0.99;

         -- 3. UPDATE PUBLIC RECORD
         State.Attitude.Pitch := Est_Pitch;
         State.Attitude.Roll  := Est_Roll;
         State.Pitch_Rate := Input_Pitch_Rate;
         State.Roll_Rate  := Input_Roll_Rate;
         
         -- 4. UPDATE TIMESTAMP (ISO 8601)
         -- (Simplified for brevity, use VSS.Strings.Image in production)
         State.Timestamp := "2025-12-17T20:00:00Z       "; 
      end Update_Physics;

      procedure Handle_Command(Cmd : String) is
      begin
         -- DAL C COMMAND INTERFACE
         -- This is where Python talks to Ada via the Pipe
         if Cmd = "RESET" then
            Est_Pitch := 0.0;
            Est_Roll := 0.0;
            Input_Pitch_Rate := 0.0;
            Input_Roll_Rate := 0.0;
         elsif Cmd'Length > 4 and then Cmd(1..4) = "PITCH" then
            -- Parse "PITCH 5.0"
            null; -- Add parsing logic here (Float'Value)
         end if;
      end Handle_Command;

      function Get_Snapshot return Avionics_Types.Instrument_Data_Record is
      begin
         return State;
      end Get_Snapshot;
   end State_Manager;

   Flight_Computer : State_Manager;

   -- INPUT LISTENER TASK
   -- This replaces the isolation of the simulator. Now it listens.
   task Input_Listener;
   task body Input_Listener is
      Input_Str : U_Strings.Unbounded_String;
   begin
      loop
         begin
            -- BLOCKING READ from Standard Input
            -- This waits for Python to send a JSON or Command string
            Input_Str := U_Strings.To_Unbounded_String(Ada.Text_IO.Get_Line);
            Flight_Computer.Handle_Command(U_Strings.To_String(Input_Str));
         exception
            when Ada.Text_IO.End_Error => exit; -- Python closed the pipe
            when others => null;
         end;
      end loop;
   end Input_Listener;

   -- MAIN LOOP
   Period : constant Duration := 0.02; -- 50Hz (Real-time Standard)
   Next_Time : Ada.Calendar.Time := Ada.Calendar.Clock;
   
   -- JSON Helper (Keep your existing To_Json function here)
   function To_Json (Item : Avionics_Types.Instrument_Data_Record) return VSS.JSON.JSON_Object_SPtr is
     -- ... [Reuse code from source: 56-71] ...
     Result : VSS.JSON.JSON_Object_SPtr := VSS.JSON.Create_Object;
   begin
      -- Minimal stub for compilation check
      return Result; 
   end To_Json;
   
   JSON_String : U_Strings.Unbounded_String;

begin
   Ada.Text_IO.Put_Line("{""status"": ""RAVEN_ONLINE"", ""mode"": ""DETERMINISTIC""}");
   
   loop
      -- 1. PHYSICS STEP
      Flight_Computer.Update_Physics(Period);
      
      -- 2. OUTPUT STEP (Stream to Python)
      -- JSON_String := VSS.JSON.Serialization.To_String(To_Json(Flight_Computer.Get_Snapshot));
      -- Ada.Text_IO.Put_Line(U_Strings.To_String(JSON_String));
      -- Ada.Text_IO.Flush; -- CRITICAL for Pipe communication!
      
      -- 3. TIMING STEP
      Next_Time := Next_Time + Period;
      delay until Next_Time;
   end loop;
end Main;