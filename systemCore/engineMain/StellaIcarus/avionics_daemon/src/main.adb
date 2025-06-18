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
      Turn_Coord: VSS.JSON.JSON_Object_SPtr := VSS.JSON.Create_Object;
      CDI_GS    : VSS.JSON.JSON_Object_SPtr := VSS.JSON.Create_Object;
      Current_Mode : constant String := U_Strings.To_String(Mode_Str);
   begin
      -- Common instruments for all modes
      Result.Set ("timestamp", VSS.Strings.To_VSS_String (Item.Timestamp));
      Result.Set ("mode", U_Strings.To_VSS_String(Mode_Str));
      Result.Set ("altimeter", Item.Altimeter);
      Result.Set ("vertical_speed_indicator", Item.Vertical_Speed_Indicator);
      Result.Set ("heading_indicator", Item.Heading_Indicator);

      -- Attitude Indicator
      Attitude.Set ("pitch", Item.Attitude.Pitch);
      Attitude.Set ("roll", Item.Attitude.Roll);
      Result.Set ("attitude_indicator", Attitude);
      
      -- Autopilot Status
      Autopilot.Set ("AP", Item.Autopilot_Status.AP);
      Autopilot.Set ("HDG", Item.Autopilot_Status.HDG);
      Autopilot.Set ("NAV", Item.Autopilot_Status.NAV);
      Result.Set ("autopilot_status", Autopilot);

      -- Mode-specific instruments
      if Current_Mode = "Atmospheric Flight" or Current_Mode = "Planetary Reconnaissance" then
         -- These instruments are relevant within an atmosphere
         Result.Set ("gps_speed", Item.GPS_Speed);
         Result.Set ("air_pressure", Item.Air_Pressure);
         Result.Set ("airspeed_indicator", Item.Airspeed_Indicator);
         Result.Set ("selected_airspeed", Item.Selected_Airspeed);
         Result.Set ("selected_altitude", Item.Selected_Altitude);
         Result.Set ("selected_heading", Item.Selected_Heading);
         Result.Set ("yaw_damper_indicator", Item.Yaw_Damper_Indicator);

         -- Turn Coordinator (more relevant for aerodynamic flight)
         Turn_Coord.Set ("rate", Item.Turn_Coordinator.Roll);
         Turn_Coord.Set ("slip_skid", Item.Turn_Coordinator.Pitch);
         Result.Set ("turn_coordinator", Turn_Coord);

         -- CDI/GS Indicator (for approaches)
         CDI_GS.Set ("course_deviation", Item.CDI_GS_Indicator.Course_Deviation);
         CDI_GS.Set ("glideslope_deviation", Item.CDI_GS_Indicator.Glideslope_Deviation);
         Result.Set ("cdi_gs_indicator", CDI_GS);
         
      elsif Current_Mode = "Interstellar Flight" then
         -- These instruments are relevant for space travel
         Result.Set ("relative_velocity_c", Item.Relative_Velocity_C); -- Velocity as fraction of light speed
         Result.Set ("navigation_reference", U_Strings.To_VSS_String(Nav_Ref_Str)); -- e.g., "Sol -> Proxima Centauri"
         -- "Altimeter" in this mode could represent distance from a reference point (e.g., star, station) in AU or km.
         -- "Heading" would be relative to the galactic plane.
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
      -- This logic mimics the Python simulation for consistency.
      declare
         Last_Pitch : constant Float := State.Attitude.Pitch;
         Last_Roll  : constant Float := State.Attitude.Roll;
         Last_Hdg   : constant Float := State.Heading_Indicator;
         Last_Alt   : constant Float := State.Altimeter;
         Last_VS    : constant Float := State.Vertical_Speed_Indicator;
         Last_IAS   : constant Float := State.Airspeed_Indicator;
         Current_Mode : constant String := U_Strings.To_String(Mode_Str);
      begin
         -- Simulate common values
         State.Attitude.Roll  := Float'Max (-60.0, Float'Min (60.0, Last_Roll + Ada.Numerics.Float_Random.Random (Gen) * 4.0 - 2.0));
         State.Attitude.Pitch := Float'Max (-30.0, Float'Min (30.0, Last_Pitch + Ada.Numerics.Float_Random.Random (Gen) * 2.0 - 1.0));
         State.Turn_Coordinator.Roll := State.Attitude.Roll / 15.0; -- Turn rate
         State.Heading_Indicator := (Last_Hdg + State.Turn_Coordinator.Roll * Float (Period_Sec)) mod 360.0;
         if State.Heading_Indicator < 0.0 then
            State.Heading_Indicator := State.Heading_Indicator + 360.0;
         end if;
         
         -- Simulate mode-specific values
         if Current_Mode = "Atmospheric Flight" or Current_Mode = "Planetary Reconnaissance" then
            State.Vertical_Speed_Indicator := Float'Max (-3000.0, Float'Min (3000.0, Last_VS + Ada.Numerics.Float_Random.Random(Gen) * 10.0 - 5.0));
            State.Altimeter := Last_Alt + (State.Vertical_Speed_Indicator * Float (Period_Sec) / 60.0);
            State.Airspeed_Indicator := Float'Max (60.0, Float'Min (400.0, Last_IAS + Ada.Numerics.Float_Random.Random(Gen) * 2.0 - 1.0));
            State.GPS_Speed := State.Airspeed_Indicator + VSS.JSON.Sin(Last_Hdg * PI / 180.0) * 5.0;
            State.Relative_Velocity_C := State.Airspeed_Indicator / (2.998E8 * 1.944); -- Convert knots to fraction of c (rough)
         elsif Current_Mode = "Interstellar Flight" then
            State.Relative_Velocity_C := State.Relative_Velocity_C + (Ada.Numerics.Float_Random.Random(Gen) - 0.49) * 0.0001;
            State.Relative_Velocity_C := Float'Max(0.0, Float'Min(0.99, State.Relative_Velocity_C));
            State.Altimeter := Last_Alt + State.Relative_Velocity_C * 2.998E8 * Float(Period_Sec) / 1.496E11; -- Altitude in AU
            State.Vertical_Speed_Indicator := State.Relative_Velocity_C * 2.998E8; -- VS in m/s
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