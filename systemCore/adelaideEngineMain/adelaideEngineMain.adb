with Ada.Text_IO; use Ada.Text_IO;



procedure Hello_World is
   procedure Print(Message : String) is
      -- The body of the procedure that prints out the message to standard output.
   begin
      Put_Line (Message);
   end Print;
begin
   -- Calling the Print procedure with "Hello, World!" as parameter.
   Print("Hello, World!");
end Hello_World;