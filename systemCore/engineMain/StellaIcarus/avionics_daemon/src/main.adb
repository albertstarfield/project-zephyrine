with Ada.Text_IO;
with Ada.Command_Line;
with Ada.Numerics.Float_Random;
with Ada.Strings.Unbounded;
with Ada.Calendar;

with VSS.JSON;
with VSS.Strings;
with VSS.JSON.Serialization;

with Avionics_Types;

procedure Main is
   -- This daemon simulates avionics data and streams it as JSON to stdout.
   -- The Python orchestrator reads this stream. This approach, using standard
   -- streams for Inter-Process Communication (IPC), is highly portable and
   -- avoids the complexity of true shared memory, which is platform-specific.

   -- Simulation State Variables
   package U_Strings renames Ada.Strings.Unbounded;

   Gen    : Ada.Numerics.Float_Random.Generator;
   State  : Avionics_Types.Instrument_Data_Record;
   Mode_Str : U_Strings.Unbounded_String;
   Nav_Ref_Str : U_Strings.Unbounded_String;

   -- Simulation constants
   Period_Sec : constant Duration := 1.0 / 20.0; -- 20 Hz stream rate
   PI         : constant Float := 3.14159_26535;

   -- Helper function to convert a record to a JSON object using VSS
   function To_Json (Item : Avionics_Types.Instrument_Data_Record) return VSS.JSON.JSON_Object_SPtr
   is
      Result    : VSS.JSON.JSON_Object_SPtr := VSS.JSON.Create_Object;
      Attitude  : VSS.JSON.JSON_Object_SPtr := VSS.JSON.Create_Object;
      Autopilot : VSS.JSON.JSON_Object_SPtr := VSS.JSON.Create_Object;
      Flight_Dir: VSS.JSON.JSON_Object_SPtr := VSS.JSON.Create_Object;
      Current_Mode : constant String := U_Strings.To_String(Mode_Str);
   begin
      -- Common instruments for all modes
      Result.Set ("timestamp", VSS.Strings.To_VSS_String (Item.Timestamp));
      Result.Set ("mode", U_Strings.To_VSS_String(Mode_Str));
      Result.Set ("altimeter", Item.Altimeter);
      Result.Set ("vertical_speed_indicator", Item.Vertical_Speed_Indicator);
      Result.Set ("heading_indicator", Item.Heading_Indicator);
      Result.Set ("g_force", Item.G_Force);
      
      -- Attitude Indicator
      Attitude.Set ("pitch", Item.Attitude.Pitch);
      Attitude.Set ("roll", Item.Attitude.Roll);
      Result.Set ("attitude_indicator", Attitude);

      -- Flight Director
      Flight_Dir.Set("command_pitch", Item.Flight_Director.Command_Pitch);
      Flight_Dir.Set("command_roll", Item.Flight_Director.Command_Roll);
      Result.Set("flight_director", Flight_Dir);
      
      -- Autopilot Status
      Autopilot.Set ("AP", Item.Autopilot_Status.AP);
      Autopilot.Set ("HDG", Item.Autopilot_Status.HDG);
      Autopilot.Set ("NAV", Item.Autopilot_Status.NAV);
      Result.Set ("autopilot_status", Autopilot);

      -- Mode-specific instruments
      if Current_Mode = "Atmospheric Flight" or Current_Mode = "Planetary Reconnaissance" then
         -- These instruments are relevant within an atmosphere
         Result.Set ("airspeed_indicator", Item.Airspeed_Indicator);
         Result.Set ("mach_number", Item.Mach_Number);
         Result.Set ("angle_of_attack", Item.Angle_Of_Attack);
         Result.Set ("pitch_rate", Item.Pitch_Rate);
         Result.Set ("roll_rate", Item.Roll_Rate);
         Result.Set ("yaw_damper_indicator", Item.Yaw_Damper_Indicator);
         Result.Set ("air_pressure", Item.Air_Pressure);
         Result.Set ("gps_speed", Item.GPS_Speed);

         -- Flight Controls
         declare
            Controls : VSS.JSON.JSON_Object_SPtr := VSS.JSON.Create_Object;
         begin
            Controls.Set("elevon_deflection", Item.Flight_Controls.Elevon_Deflection);
            Controls.Set("body_flap_position", Item.Flight_Controls.Body_Flap_Position);
            Controls.Set("rudder_deflection", Item.Flight_Controls.Rudder_Deflection);
            Controls.Set("speedbrake_position", Item.Flight_Controls.Speedbrake_Position);
            Result.Set("flight_controls", Controls);
         end;

         -- HSI Data
         declare
            HSI : VSS.JSON.JSON_Object_SPtr := VSS.JSON.Create_Object;
         begin
            HSI.Set("selected_course", Item.HSI.Selected_Course);
            HSI.Set("course_deviation", Item.HSI.Course_Deviation);
            HSI.Set("waypoint_bearing", Item.HSI.Waypoint_Bearing);
            HSI.Set("waypoint_range_nm", Item.HSI.Waypoint_Range_NM);
            HSI.Set("hac_turn_angle", Item.HSI.HAC_Turn_Angle);
            Result.Set("hsi_data", HSI);
         end;
         
      elsif Current_Mode = "Interstellar Flight" then
         -- These instruments are relevant for space travel
         Result.Set ("relative_velocity_c", Item.Relative_Velocity_C); -- Velocity as fraction of light speed
         Result.Set ("navigation_reference", U_Strings.To_VSS_String(Nav_Ref_Str));
      end if;
      
      return Result;
   end To_Json;

begin
   Ada.Numerics.Float_Random.Reset (Gen);
   -- Set the initial mode using the new terminology
   U_Strings.Set(Mode_Str, "Atmospheric Flight");
   U_Strings.Set(Nav_Ref_Str, "Sol System");
   
   loop
      -- 1. SIMULATE DATA UPDATE
      declare
         Last_Pitch : constant Float := State.Attitude.Pitch;
         Last_Roll  : constant Float := State.Attitude.Roll;
         Current_Mode : constant String := U_Strings.To_String(Mode_Str);
      begin
         -- Simulate common values
         State.Attitude.Roll  := Float'Max (-60.0, Float'Min (60.0, Last_Roll + Ada.Numerics.Float_Random.Random (Gen) * 4.0 - 2.0));
         State.Attitude.Pitch := Float'Max (-30.0, Float'Min (30.0, Last_Pitch + Ada.Numerics.Float_Random.Random (Gen) * 2.0 - 1.0));
         State.Roll_Rate  := (State.Attitude.Roll - Last_Roll) / Float(Period_Sec);
         State.Pitch_Rate := (State.Attitude.Pitch - Last_Pitch) / Float(Period_Sec);
         State.G_Force := 1.0 + (abs(State.Attitude.Roll) / 60.0)**2 * 2.0; -- Simplified G-force in turns
         State.Flight_Director.Command_Pitch := State.Attitude.Pitch + 2.0;
         State.Flight_Director.Command_Roll  := State.Attitude.Roll - 1.0;

         -- Simulate mode-specific values
         if Current_Mode = "Atmospheric Flight" or Current_Mode = "Planetary Reconnaissance" then
            State.Heading_Indicator := (State.Heading_Indicator + (State.Roll_Rate / 10.0)) mod 360.0;
            State.Vertical_Speed_Indicator := Float'Max (-4000.0, Float'Min (4000.0, State.Vertical_Speed_Indicator + Ada.Numerics.Float_Random.Random(Gen) * 20.0 - 10.0));
            State.Altimeter := State.Altimeter + (State.Vertical_Speed_Indicator * Float (Period_Sec) / 60.0);
            State.Airspeed_Indicator := Float'Max (60.0, Float'Min (700.0, State.Airspeed_Indicator + Ada.Numerics.Float_Random.Random(Gen) * 4.0 - 2.0));
            State.Mach_Number := State.Airspeed_Indicator / (661.47 * (1.0 - State.Altimeter/145442.0)**2.5); -- Simplified Mach calc
            State.Angle_Of_Attack := 2.5 + State.Attitude.Pitch / 5.0 - (State.Airspeed_Indicator - 120.0) / 50.0;

            -- Simulate Flight Controls
            State.Flight_Controls.Elevon_Deflection := State.Attitude.Pitch * -0.5;
            State.Flight_Controls.Rudder_Deflection := State.Roll_Rate * -0.2;
            State.Flight_Controls.Speedbrake_Position := 0.0;
            
         elsif Current_Mode = "Interstellar Flight" then
            State.Relative_Velocity_C := State.Relative_Velocity_C + (Ada.Numerics.Float_Random.Random(Gen) - 0.49) * 0.0001;
            State.Relative_Velocity_C := Float'Max(0.0, Float'Min(0.99, State.Relative_Velocity_C));
            State.Altimeter := State.Altimeter + State.Relative_Velocity_C * 2.998E8 * Float(Period_Sec) / 1.496E11; -- Altitude in AU
         end if;


         -- Randomly change mode occasionally using the new terminology
         if Ada.Numerics.Float_Random.Random (Gen) < 0.001 then
            declare
               Mode_Choice : constant Integer := Integer'Val(Integer'Floor(Ada.Numerics.Float_Random.Random (Gen) * 3.0));
            begin
               case Mode_Choice is
                  when 0 => U_Strings.Set(Mode_Str, "Atmospheric Flight");
                  when 1 => U_Strings.Set(Mode_Str, "Planetary Reconnaissance");
                  when 2 => 
                     U_Strings.Set(Mode_Str, "Interstellar Flight");
                     U_Strings.Set(Nav_Ref_Str, "Sol -> Proxima Centauri");
               end case;
            end;
         end if;

         -- Update timestamp
         declare
             Now : constant Ada.Calendar.Time := Ada.Calendar.Clock;
         begin
             State.Timestamp := VSS.Strings.To_Std_String(VSS.Strings.Image(Now, Milliseconds => 3));
         end;

      end;
      
      -- 2. CONVERT TO JSON
      declare
         JSON_Data    : VSS.JSON.JSON_Object_SPtr;
         JSON_String  : U_Strings.Unbounded_String;
      begin
         JSON_Data := To_Json (State);
         VSS.JSON.Serialization.To_String (JSON_Data.To_JSON_Value, JSON_String);
         
         -- 3. PRINT TO STDOUT AND FLUSH
         Ada.Text_IO.Put_Line (U_Strings.To_String (JSON_String));
         Ada.Text_IO.Flush;
      exception
         when others =>
            Ada.Text_IO.Put_Line ("{""error"":""Failed to serialize avionics data to JSON""}");
            Ada.Text_IO.Flush;
      end;
      
      -- 4. DELAY FOR THE SPECIFIED PERIOD
      delay Period_Sec;
      
   end loop;

exception
   when others =>
      Ada.Text_IO.Put_Line (Ada.Command_Line.Command_Name & ": Unhandled exception. Exiting.");
end Main;