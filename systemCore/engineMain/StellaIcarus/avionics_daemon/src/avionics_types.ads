package Avionics_Types is

   -- A record to hold the state of the autopilot system.
   type Autopilot_Status_Record is record
      AP  : Boolean := True;
      HDG : Boolean := True;
      NAV : Boolean := False;
      IAS : Boolean := True;
      ALT : Boolean := True;
      APR : Boolean := False;
      REV : Boolean := False;
   end record;

   -- A record for indicators with deviations, like CDI or Glideslope.
   type Deviation_Indicator_Record is record
      Course_Deviation    : Float := 0.0;
      Glideslope_Deviation: Float := 0.0;
   end record;

   -- A record for the attitude indicator (pitch and roll).
   type Attitude_Record is record
      Pitch : Float := 0.0;
      Roll  : Float := 0.0;
   end record;

   -- The main record that aggregates all instrument data into a single structure.
   type Instrument_Data_Record is record
      Timestamp                  : String(1 .. 27); -- ISO 8601 Format
      Mode                       : String(1 .. 32) := (others => ' ');
      Mode_Len                   : Natural; -- To handle variable length string
      GPS_Speed                  : Float := 115.0;
      Air_Pressure               : Float := 29.92;
      Airspeed_Indicator         : Float := 120.0;
      Vertical_Speed_Indicator   : Float := 0.0;
      Altimeter                  : Float := 5000.0;
      Heading_Indicator          : Float := 180.0;
      Selected_VS                : Integer := 500;
      Selected_Airspeed          : Integer := 130;
      Selected_Altitude          : Integer := 10_000;
      Selected_Heading           : Float := 190.0;
      Yaw_Damper_Indicator       : Boolean := True;
      -- New fields for Interstellar mode
      Relative_Velocity_C        : Float := 0.0;
      Navigation_Reference       : String(1 .. 32) := (others => ' ');
      Nav_Ref_Len                : Natural;
      -- Common records
      Attitude                   : Attitude_Record;
      Turn_Coordinator           : Attitude_Record; -- Using Attitude for rate/slip
      CDI_GS_Indicator           : Deviation_Indicator_Record;
      Autopilot_Status           : Autopilot_Status_Record;
   end record;

end Avionics_Types;