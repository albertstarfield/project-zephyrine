with Ada.Text_IO;
with Message_Generator; -- Import our new, provable package

--  This is the main procedure, our "I/O Shell".
--  We explicitly turn SPARK checking OFF for this unit. Its job is to
--  interact with the non-provable outside world.
procedure Hello_Sekai_Spark
  with SPARK_Mode => Off
is
   --  Declare a constant string, initialized by calling our PROVEN function.
   The_Message : constant String := Message_Generator.Get_Message;
begin
   --  Perform the unprovable I/O operation.
   Ada.Text_IO.Put_Line (The_Message);
end Hello_Sekai_Spark;