pragma SPARK_Mode (Off);
--  c_binding: NASA cFE Telemetry Output (TO_LAB) integration
--  Wraps cFS TO_LAB for Adelaide telemetry aggregation
package CFS_Telemetry is

   --  Telemetry message types
   type TLM_Type is (Housekeeping, Sensor, Attitude, Event, Custom);

   --  Telemetry message record
   type TLM_Message is record
      Msg_Type   : TLM_Type := Housekeeping;
      Msg_Data   : String (1 .. 512);
      Msg_Len    : Natural := 0;
      Timestamp  : Duration := 0.0;
   end record;

   --  Initialize the cFS Telemetry subsystem
   procedure Initialize with Pre => True, Post => True;

   --  Send a telemetry packet
   procedure Send_Telemetry (Msg : TLM_Message)
     with Pre => Msg.Msg_Len > 0;

   --  Send housekeeping telemetry (CPU, memory, etc.)
   procedure Send_Housekeeping (CPU_Pct : Float; Mem_Pct : Float; Uptime : Duration)
     with Pre => CPU_Pct >= 0.0 and CPU_Pct <= 100.0;

   --  Send a sensor reading
   procedure Send_Sensor_Telemetry (Sensor_Name : String; Value : Float)
     with Pre => Sensor_Name'Length > 0;

   --  Send an attitude report
   procedure Send_Attitude_Telemetry (Roll, Pitch, Yaw : Float);

   --  Flush pending telemetry
   procedure Flush with Pre => True, Post => True;

end CFS_Telemetry;
