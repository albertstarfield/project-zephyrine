with Ada.Text_IO;
procedure Child_Process is
begin
   delay 5.0; -- Stay alive for 5 seconds
   Ada.Text_IO.Put_Line ("  CHILD: My time is up. Goodbye!");
end Child_Process;
