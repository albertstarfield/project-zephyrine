with Ada.Streams;
with Interfaces.C;  -- Required for unsigned_char type

-- This package defines the data format for communication with the MCU
-- It includes error detection using a checksum and parity bit
package MCU_Protocol is
   -- Control values (sent to MCU)
   type Control_Values is record
      Servo_1     : Natural range 0 .. 100;  -- First servo motor
      Servo_2     : Natural range 0 .. 100;  -- Second servo motor
      Servo_3     : Natural range 0 .. 100;  -- Third servo motor
      Propeller   : Natural range 0 .. 100;  -- Main propeller engine
   end record;
   
   -- Sensor values (received from MCU)
   type Sensor_Values is record
      Gyroscope    : Integer;  -- Angular velocity in degrees/second
      Accelerometer: Integer;  -- Acceleration in mg
      Magnetometer : Integer;  -- Magnetic field in uT
      Barometer    : Integer;  -- Pressure in hPa
   end record;
   
   -- Message types
   type Message_Type is (Control_Message, Sensor_Message);
   
   -- Error codes
   type Error_Code is (No_Error, Checksum_Error, Parity_Error, Invalid_Message_Type);
   
   -- Validation result type (to return multiple values)
   type Validation_Result is record
      Msg_Type : Message_Type;
      Error    : Error_Code;
   end record;
   
   -- Maximum message size (including headers and error detection)
   Max_Message_Size : constant := 12;
   
   -- Encode control values into a message buffer
   -- Returns the encoded message
   function Encode_Control (Values : Control_Values) 
                          return Ada.Streams.Stream_Element_Array;
   
   -- Decode a message buffer into sensor values
   -- Returns the decoded values and sets Error parameter
   function Decode_Sensor (Buffer : Ada.Streams.Stream_Element_Array;
                           Error  : out Error_Code) 
                          return Sensor_Values;
   
   -- Validate a message buffer for errors
   -- Returns a Validation_Result record containing message type and error code
   function Validate_Message (Buffer : Ada.Streams.Stream_Element_Array) 
                            return Validation_Result;
   
   function Create_Mixed_Control return Control_Values;

private
   -- Message header structure
   type Message_Header is record
      Msg_Type : Message_Type;
      Length   : Natural range 0 .. Max_Message_Size;
   end record;
   
   -- Error detection fields
   type Error_Detection is record
      Checksum : Interfaces.C.unsigned_char;  -- Sum of all data bytes
      Parity   : Interfaces.C.unsigned_char;  -- XOR of all data bits
   end record;
   
   -- Why two error detection mechanisms?
   -- 1. Checksum: Detects most common errors like single-bit errors, burst errors
   -- 2. Parity: Provides additional protection against specific error patterns
   -- Together they provide robust error detection for critical avionics systems
   -- This is essential because corrupted control signals could cause catastrophic failures
end MCU_Protocol;