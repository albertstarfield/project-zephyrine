with Ada.Text_IO;
with Ada.Command_Line;
procedure Child_Process is
begin
   Ada.Text_IO.Put_Line ("CHILD: Hello! I am the child process, PID: " & Ada.Command_Line.Command_Name);
   Ada.Text_IO.Put_Line ("CHILD: I will run for 30 seconds and then exit.");
   delay 30.0;
   Ada.Text_IO.Put_Line ("CHILD: My time is up. Goodbye!");
end Child_Process;