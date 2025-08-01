with Ada.Streams;
with Interfaces;
with Interfaces.C;
with Ada.Unchecked_Conversion;

package body MCU_Protocol is
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
                              return Interfaces.C.unsigned_char is
      Sum : Interfaces.C.unsigned := 0;
   begin
      for I in Data'Range loop
         Sum := Sum + Interfaces.C.unsigned (Data (I));
      end loop;
      return Interfaces.C.unsigned_char (Sum mod 256);
   end Calculate_Checksum;
   
   -- Calculate parity for a data buffer
   -- Parity is the XOR of all bits in the data
   function Calculate_Parity (Data : Ada.Streams.Stream_Element_Array) 
                            return Interfaces.C.unsigned_char is
      Parity : Interfaces.C.unsigned_char := 0;
   begin
      for I in Data'Range loop
         Parity := Interfaces."xor" (Parity, Data (I));
      end loop;
      return Parity;
   end Calculate_Parity;
   
   ---------------------
   -- Encode_Control --
   ---------------------
   function Encode_Control (Values : Control_Values) 
                          return Ada.Streams.Stream_Element_Array is
      Buffer : Ada.Streams.Stream_Element_Array (0 .. 6);
      -- Why 6 bytes?
      -- 1 byte for message type
      -- 4 bytes for control values (1 byte each, 0-100 fits in 8 bits)
      -- 1 byte for checksum
      -- (Parity is included in checksum calculation)
   begin
      -- Message type
      Buffer (0) := To_Byte (Control_Message);
      
      -- Control values (each scaled to 0-100, which fits in a byte)
      Buffer (1) := Ada.Streams.Stream_Element (Values.Servo_1);
      Buffer (2) := Ada.Streams.Stream_Element (Values.Servo_2);
      Buffer (3) := Ada.Streams.Stream_Element (Values.Servo_3);
      Buffer (4) := Ada.Streams.Stream_Element (Values.Propeller);
      
      -- Calculate error detection fields
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
      Calculated_Checksum : Interfaces.C.unsigned_char;
      Calculated_Parity   : Interfaces.C.unsigned_char;
   begin
      Error := No_Error;
      
      -- Validate message type
      if To_Message_Type (Buffer (0)) /= Sensor_Message then
         Error := Invalid_Message_Type;
         return (0, 0, 0, 0);
      end if;
      
      -- Validate checksum
      Calculated_Checksum := Calculate_Checksum (Buffer (0 .. 8));
      if Buffer (9) /= Calculated_Checksum then
         Error := Checksum_Error;
         return (0, 0, 0, 0);
      end if;
      
      -- Extract sensor values (16-bit values stored in big-endian format)
      -- Gyroscope (bytes 1-2)
      declare
         function To_Integer is new Ada.Unchecked_Conversion (
            Source => Ada.Streams.Stream_Element_Array (1 .. 2),
            Target => Interfaces.Integer_16);
         Gyro_Bytes : Ada.Streams.Stream_Element_Array (1 .. 2) := 
            (Buffer (1), Buffer (2));
      begin
         Result.Gyroscope := Integer (To_Integer (Gyro_Bytes));
      end;
      
      -- Accelerometer (bytes 3-4)
      declare
         function To_Integer is new Ada.Unchecked_Conversion (
            Source => Ada.Streams.Stream_Element_Array (1 .. 2),
            Target => Interfaces.Integer_16);
         Accel_Bytes : Ada.Streams.Stream_Element_Array (1 .. 2) := 
            (Buffer (3), Buffer (4));
      begin
         Result.Accelerometer := Integer (To_Integer (Accel_Bytes));
      end;
      
      -- Magnetometer (bytes 5-6)
      declare
         function To_Integer is new Ada.Unchecked_Conversion (
            Source => Ada.Streams.Stream_Element_Array (1 .. 2),
            Target => Interfaces.Integer_16);
         Mag_Bytes : Ada.Streams.Stream_Element_Array (1 .. 2) := 
            (Buffer (5), Buffer (6));
      begin
         Result.Magnetometer := Integer (To_Integer (Mag_Bytes));
      end;
      
      -- Barometer (bytes 7-8)
      declare
         function To_Integer is new Ada.Unchecked_Conversion (
            Source => Ada.Streams.Stream_Element_Array (1 .. 2),
            Target => Interfaces.Integer_16);
         Baro_Bytes : Ada.Streams.Stream_Element_Array (1 .. 2) := 
            (Buffer (7), Buffer (8));
      begin
         Result.Barometer := Integer (To_Integer (Baro_Bytes));
      end;
      
      return Result;
   end Decode_Sensor;
   
   ------------------------
   -- Validate_Message --
   ------------------------
   function Validate_Message (Buffer : Ada.Streams.Stream_Element_Array) 
                            return Validation_Result is
      Msg_Type : Message_Type;
      Calculated_Checksum : Interfaces.C.unsigned_char;
      Length : Natural;
      Result : Validation_Result;
   begin
      -- Check minimum message length
      if Buffer'Length < 2 then
         Result.Msg_Type := Control_Message;
         Result.Error := Invalid_Message_Type;
         return Result;
      end if;
      
      -- Determine message type
      Msg_Type := To_Message_Type (Buffer (0));
      
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
      Calculated_Checksum := Calculate_Checksum (Buffer (0 .. Length-2));
      if Buffer (Length-1) /= Calculated_Checksum then
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