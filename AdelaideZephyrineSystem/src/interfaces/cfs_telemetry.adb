pragma SPARK_Mode (Off);
--  c_binding: NASA cFE Telemetry Output (TO_LAB) integration
with Ada.Text_IO; use Ada.Text_IO;

package body CFS_Telemetry is

   Initialized : Boolean := False;

   procedure Initialize is
   begin
      if Initialized then
         return;
      end if;
      Put_Line ("[CFS-TLM] Initializing cFS Telemetry subsystem...");
      Initialized := True;
      Put_Line ("[CFS-TLM] Telemetry subsystem ready.");
   end Initialize;

   procedure Send_Telemetry (Msg : TLM_Message) is
   begin
      --  TODO: Build CFE_MSG_Message_t and transmit via Software Bus
      Put_Line ("[CFS-TLM] Sending " & TLM_Type'Image (Msg.Msg_Type) &
                " (" & Natural'Image (Msg.Msg_Len) & " bytes)");
   end Send_Telemetry;

   procedure Send_Housekeeping (CPU_Pct : Float; Mem_Pct : Float; Uptime : Duration) is
   begin
      Put_Line ("[CFS-TLM] HK: CPU=" & Float'Image (CPU_Pct) & "%" &
                " MEM=" & Float'Image (Mem_Pct) & "%" &
                " UPTIME=" & Duration'Image (Uptime));
   end Send_Housekeeping;

   procedure Send_Sensor_Telemetry (Sensor_Name : String; Value : Float) is
   begin
      Put_Line ("[CFS-TLM] SENSOR: " & Sensor_Name & " = " & Float'Image (Value));
   end Send_Sensor_Telemetry;

   procedure Send_Attitude_Telemetry (Roll, Pitch, Yaw : Float) is
   begin
      Put_Line ("[CFS-TLM] ATT: R=" & Float'Image (Roll) &
                " P=" & Float'Image (Pitch) &
                " Y=" & Float'Image (Yaw));
   end Send_Attitude_Telemetry;

   procedure Flush is
   begin
      --  TODO: Flush Software Bus buffers
      null;
   end Flush;

end CFS_Telemetry;
