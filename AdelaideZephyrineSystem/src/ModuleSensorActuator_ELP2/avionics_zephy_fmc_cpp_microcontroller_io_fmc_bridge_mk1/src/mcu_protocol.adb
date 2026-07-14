with Ada.Streams;
with Interfaces;
with Ada.Unchecked_Conversion; -- justified: FFI type conversion required for C interop
use all type Ada.Streams.Stream_Element;

package body MCU_Protocol is
   use type Ada.Streams.Stream_Element_Offset;
   use type Interfaces.Integer_16;

   subtype Two_Bytes is Ada.Streams.Stream_Element_Array (1 .. 2);
   function To_Int16 is new Ada.Unchecked_Conversion ( -- justified: FFI type conversion required for C interop
      Source => Two_Bytes,
      Target => Interfaces.Integer_16);

   -- Convert Message_Type to byte representation
   function To_Byte (MT : Message_Type) return Ada.Streams.Stream_Element is
   begin
      case MT is
         when Control_Message => return 0;
         when Sensor_Message  => return 1;
      end case;
   end To_Byte;
   
   -- Convert byte to Message_Type
   function To_Message_Type (B : Ada.Streams.Stream_Element) return Message_Type is
   begin
      case B is
         when 0 => return Control_Message;
         when 1 => return Sensor_Message;
         when others => raise Constraint_Error;
      end case;
   end To_Message_Type;
   
   -- Calculate checksum for a data buffer
   -- Checksum is the sum of all bytes modulo 256
   function Calculate_Checksum (Data : Ada.Streams.Stream_Element_Array) 
                              return Ada.Streams.Stream_Element is
      Sum : Ada.Streams.Stream_Element := 0;
   begin
      for I in Data'Range loop
         Sum := Sum + Data(I);
      end loop;
      return Sum;
   end Calculate_Checksum;
   
   -- Calculate parity for a data buffer
   -- Parity is the XOR of all bits in the data
   function Calculate_Parity (Data : Ada.Streams.Stream_Element_Array) 
                            return Ada.Streams.Stream_Element is
      Parity : Ada.Streams.Stream_Element := 0;
   begin
      for I in Data'Range loop
         Parity := Parity xor Data(I);
      end loop;
      return Parity;
   end Calculate_Parity;
   
   ---------------------
   -- Encode_Control --
   ---------------------
   function Encode_Control (Values : Control_Values) 
                          return Ada.Streams.Stream_Element_Array is
      Buffer : Ada.Streams.Stream_Element_Array (0 .. 6);
   begin
      -- Message type
      Buffer (0) := To_Byte (Control_Message);
      
      -- Control values
      Buffer (1) := Ada.Streams.Stream_Element (Values.Servo_1);
      Buffer (2) := Ada.Streams.Stream_Element (Values.Servo_2);
      Buffer (3) := Ada.Streams.Stream_Element (Values.Servo_3);
      Buffer (4) := Ada.Streams.Stream_Element (Values.Propeller);
      
      -- Calculate checksum
      Buffer (5) := Calculate_Checksum (Buffer (0 .. 4));
      
      return Buffer (0 .. 5);
   end Encode_Control;
   
   --------------------
   -- Decode_Sensor --
   --------------------
   function Decode_Sensor (Buffer : Ada.Streams.Stream_Element_Array;
                           Error  : out Error_Code) 
                          return Sensor_Values is
      Result : Sensor_Values;
      Calculated_Checksum : Ada.Streams.Stream_Element;
      F : constant Ada.Streams.Stream_Element_Offset := Buffer'First;
   begin
      Error := No_Error;
      
      -- Validate message type
      if To_Message_Type (Buffer (F)) /= Sensor_Message then
         Error := Invalid_Message_Type;
         return (0, 0, 0, 0);
      end if;
      
      -- Validate checksum
      Calculated_Checksum := Calculate_Checksum (Buffer (F .. F + 8));
      if Buffer (F + 9) /= Calculated_Checksum then
         Error := Checksum_Error;
         return (0, 0, 0, 0);
      end if;
      
      -- Extract sensor values (16-bit values stored in big-endian format)
      -- Gyroscope (bytes 1-2)
      declare
         Gyro_Bytes : Two_Bytes := (Buffer (F + 1), Buffer (F + 2));
      begin
         Result.Gyroscope := Integer (To_Int16 (Gyro_Bytes));
      end;
      
      -- Accelerometer (bytes 3-4)
      declare
         Accel_Bytes : Two_Bytes := (Buffer (F + 3), Buffer (F + 4));
      begin
         Result.Accelerometer := Integer (To_Int16 (Accel_Bytes));
      end;
      
      -- Magnetometer (bytes 5-6)
      declare
         Mag_Bytes : Two_Bytes := (Buffer (F + 5), Buffer (F + 6));
      begin
         Result.Magnetometer := Integer (To_Int16 (Mag_Bytes));
      end;
      
      -- Barometer (bytes 7-8)
      declare
         Baro_Bytes : Two_Bytes := (Buffer (F + 7), Buffer (F + 8));
      begin
         Result.Barometer := Integer (To_Int16 (Baro_Bytes));
      end;
      
      return Result;
   end Decode_Sensor;
   
   ------------------------
   -- Validate_Message --
   ------------------------
   function Validate_Message (Buffer : Ada.Streams.Stream_Element_Array) 
                            return Validation_Result is
      Msg_Type : Message_Type;
      Calculated_Checksum : Ada.Streams.Stream_Element;
      Length : Ada.Streams.Stream_Element_Offset;
      Result : Validation_Result;
      F : constant Ada.Streams.Stream_Element_Offset := Buffer'First;
   begin
      -- Check minimum message length
      if Buffer'Length < 2 then
         Result.Msg_Type := Control_Message;
         Result.Error := Invalid_Message_Type;
         return Result;
      end if;
      
      -- Determine message type
      Msg_Type := To_Message_Type (Buffer (F));
      
      -- Determine expected length based on message type
      case Msg_Type is
         when Control_Message => Length := 6;  -- 1 type + 4 values + 1 checksum
         when Sensor_Message  => Length := 10; -- 1 type + 8 sensor + 1 checksum
      end case;
      
      -- Check message length
      if Buffer'Length < Length then
         Result.Msg_Type := Msg_Type;
         Result.Error := Invalid_Message_Type;
         return Result;
      end if;
      
      -- Validate checksum
      Calculated_Checksum := Calculate_Checksum (Buffer (F .. F + Length - 2));
      if Buffer (F + Length - 1) /= Calculated_Checksum then
         Result.Msg_Type := Msg_Type;
         Result.Error := Checksum_Error;
         return Result;
      end if;
      
      -- All checks passed
      Result.Msg_Type := Msg_Type;
      Result.Error := No_Error;
      return Result;
   end Validate_Message;
   
   -- Additional helper functions for testing
   function Create_Test_Control (Index : Natural) return Control_Values is
      Result : Control_Values;
   begin
      -- Create different test patterns based on index
      case Index mod 4 is
         when 0 => 
            -- All values at 50 (midpoint)
            Result := (50, 50, 50, 50);
         when 1 => 
            -- Max values for servos, min for propeller
            Result := (100, 100, 100, 0);
         when 2 => 
            -- Min values for servos, max for propeller
            Result := (0, 0, 0, 100);
         when others => 
            -- Mixed values
            Result := (25, 75, 50, 30);
      end case;
      return Result;
   end Create_Test_Control;
   
   function Create_Mixed_Control return Control_Values is
      Result : Control_Values;
   begin
      -- Create mixed values with special handling for propeller
      Result := (35, 65, 45, 20);  -- Propeller intentionally lower for safety
      return Result;
   end Create_Mixed_Control;
   
end MCU_Protocol;