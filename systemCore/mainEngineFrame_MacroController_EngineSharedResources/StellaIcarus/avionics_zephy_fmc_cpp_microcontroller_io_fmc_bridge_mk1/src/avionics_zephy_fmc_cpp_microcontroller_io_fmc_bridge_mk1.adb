with Ada.Text_IO;
with Ada.Command_Line;
with Ada.Real_Time;
with Socket_IO;
with MCU_Protocol;
with Ada.Streams;
-- Make operators for Socket_FD directly visible

use all type Socket_IO.Socket_FD;

-- Make operators for MCU_Protocol types directly visible
use all type MCU_Protocol.Error_Code;
use all type MCU_Protocol.Message_Type;

use Ada.Real_Time;

procedure Avionics_Zephy_FMC_CPP_Microcontroller_IO_FMC_Bridge_MK1 is
   Socket_Path : constant String := "./mcuIO";
   Arg_Count   : constant Natural := Ada.Command_Line.Argument_Count;

   -- Connection state
   Socket : Socket_IO.Socket_FD := Socket_IO.Null_Socket;

   -- Buffer for reading messages
   Read_Buffer : Ada.Streams.Stream_Element_Array 
  (0 .. Ada.Streams.Stream_Element_Offset(Socket_IO.Max_Message_Size - 1));
   Bytes_Read  : Integer;

   -- Current control values
   Current_Controls : MCU_Protocol.Control_Values := (0, 0, 0, 0);

   -- Test mode state
   Test_Mode : Boolean := False;
   Test_Index : Natural := 0;
   Last_Test_Time : Ada.Real_Time.Time := Ada.Real_Time.Clock;

   -- Test interval (2 seconds)
   Test_Interval : constant Ada.Real_Time.Time_Span :=
      Ada.Real_Time.To_Time_Span (2.0);

   -- Function to handle received sensor data
   procedure Process_Sensor_Data (Data : MCU_Protocol.Sensor_Values) is
   begin
      Ada.Text_IO.Put_Line ("--> Received sensor data");
      Ada.Text_IO.Put_Line ("    Gyroscope: " & Data.Gyroscope'Image & " deg/s");
      Ada.Text_IO.Put_Line ("    Accelerometer: " & Data.Accelerometer'Image & " mg");
      Ada.Text_IO.Put_Line ("    Magnetometer: " & Data.Magnetometer'Image & " uT");
      Ada.Text_IO.Put_Line ("    Barometer: " & Data.Barometer'Image & " hPa");

      -- Here you would process the sensor data for flight control
      -- For example, adjust control values based on sensor feedback
   end Process_Sensor_Data;

   -- Function to send control values
   procedure Send_Control_Values is
      Encoded : Ada.Streams.Stream_Element_Array :=
         MCU_Protocol.Encode_Control (Current_Controls);
      Bytes_Sent : Integer;
   begin
      Bytes_Sent := Socket_IO.Write (Socket, Encoded);
      if Bytes_Sent = Encoded'Length then
         Ada.Text_IO.Put_Line ("--> Sent control values: " &
            Current_Controls.Servo_1'Image &
            Current_Controls.Servo_2'Image &
            Current_Controls.Servo_3'Image &
            Current_Controls.Propeller'Image);
      else
         Ada.Text_IO.Put_Line ("Error: Failed to send full control message");
      end if;
   end Send_Control_Values;

begin
   -- Check for the test mode flag
   if Arg_Count >= 1 and then Ada.Command_Line.Argument (1) = "--test-mode" then
      Test_Mode := True;
      Ada.Text_IO.Put_Line ("--> Starting in test mode");
   end if;

   -- Create the socket connection
   Socket := Socket_IO.Connect_Socket (Socket_Path);
   if Socket = Socket_IO.Null_Socket then
      Ada.Text_IO.Put_Line ("Error: Failed to connect to socket at " & Socket_Path);
      Ada.Text_IO.Put_Line ("Make sure the MCU interface is running");
      Ada.Command_Line.Set_Exit_Status (Ada.Command_Line.Failure);
      return;
   end if;

   Ada.Text_IO.Put_Line ("--> Connected to MCU socket at " & Socket_Path);

   -- Main communication loop
   Main_Loop : loop
      -- Read any available sensor data
      Bytes_Read := Socket_IO.Read (Socket, Read_Buffer);
      if Bytes_Read > 0 then
         declare
            Validation_Result : MCU_Protocol.Validation_Result;
            Sensor_Data : MCU_Protocol.Sensor_Values;
         begin
            -- Validate the message - assign to a single record variable
            Validation_Result := MCU_Protocol.Validate_Message (Read_Buffer (0 .. Ada.Streams.Stream_Element_Offset (Bytes_Read - 1)));
            
            if Validation_Result.Error = MCU_Protocol.No_Error and 
               Validation_Result.Msg_Type = MCU_Protocol.Sensor_Message then
               -- Decode and process sensor data
               Sensor_Data := MCU_Protocol.Decode_Sensor (
                  Read_Buffer (0 .. Ada.Streams.Stream_Element_Offset (Bytes_Read - 1)), Validation_Result.Error);

               if Validation_Result.Error = MCU_Protocol.No_Error then
                  Process_Sensor_Data (Sensor_Data);
               else
                  Ada.Text_IO.Put_Line ("Error: Invalid sensor data received");
               end if;
            end if;
         end;
      end if;

      -- In test mode, periodically send test control values
      if Test_Mode then
         if Ada.Real_Time.Clock - Last_Test_Time > Test_Interval then
            -- Create test control values
            if Test_Index < 10 then
               -- First 10 cycles: individual servo tests
               case Test_Index mod 4 is
                  when 0 => Current_Controls := (100, 0, 0, 0);  -- Test servo 1
                  when 1 => Current_Controls := (0, 100, 0, 0);  -- Test servo 2
                  when 2 => Current_Controls := (0, 0, 100, 0);  -- Test servo 3
                  when others => Current_Controls := (0, 0, 0, 100);  -- Test propeller
               end case;
            elsif Test_Index < 15 then
               -- Next 5 cycles: mixed values
               Current_Controls := MCU_Protocol.Create_Mixed_Control;
            else
               -- After 15 cycles, reset and start over
               Test_Index := 0;
               Current_Controls := (0, 0, 0, 0);
               Send_Control_Values;  -- Send zero values to stop everything
               delay Ada.Real_Time.To_Duration (Ada.Real_Time.To_Time_Span (1.0));  -- Give time to stop
               Test_Index := 0;
               goto Continue_Test;
            end if;

            Send_Control_Values;
            Test_Index := Test_Index + 1;
            Last_Test_Time := Ada.Real_Time.Clock;

            <<Continue_Test>>
            null;
         end if;
      end if;

      -- Small delay to prevent CPU hogging
      delay 0.01;
   end loop Main_Loop;

exception
   when others =>
      Ada.Text_IO.Put_Line ("An unexpected error occurred.");
      Socket_IO.Close_Socket (Socket);
      Ada.Command_Line.Set_Exit_Status (Ada.Command_Line.Failure);
end Avionics_Zephy_FMC_CPP_Microcontroller_IO_FMC_Bridge_MK1;