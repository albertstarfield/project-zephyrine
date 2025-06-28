package body Message_Generator
  with SPARK_Mode
is

   function Get_Message return String is
   begin
      --  This is the implementation that gnatprove will check against the
      --  postcondition in the .ads file.
      return "Hello, Sekai! (Proven and Separated)";
   end Get_Message;

end Message_Generator;