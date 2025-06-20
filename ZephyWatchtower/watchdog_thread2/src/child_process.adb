with Ada.Text_IO;
with Ada.Command_Line;
procedure Child_Process is
begin
   Ada.Text_IO.Put_Line ("CHILD: Hello! I am the test child process.");
   Ada.Text_IO.Put_Line ("CHILD: My arguments were: ");
   for I in 1 .. Ada.Command_Line.Argument_Count loop
      Ada.Text_IO.Put_Line ("  Arg " & Integer'Image(I) & ": " & Ada.Command_Line.Argument(I));
   end loop;
   Ada.Text_IO.Put_Line ("CHILD: I will run for 30 seconds and then exit.");
   delay 30.0;
   Ada.Text_IO.Put_Line ("CHILD: My time is up. Goodbye!");
end Child_Process;