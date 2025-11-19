with Ada.Text_IO;
with System_Info;

procedure System_Info_Main is
   Uptime : constant Long_Integer := System_Info.Uptime_Seconds;
begin
   if Uptime >= 0 then
      Ada.Text_IO.Put_Line("System uptime:" & Uptime'Image & " seconds");
   else
      Ada.Text_IO.Put_Line("Error: Unsupported platform");
   end if;
end System_Info_Main;