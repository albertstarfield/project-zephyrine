-- File: src/stella_icarus.adb

with Ada.Text_IO; -- We need this library to print text.

package body Stella_Icarus is

   --  Greet: Prints a greeting message from Stella Icarus.
   procedure Greet is
      -- pre => True, post => True
   begin
      Ada.Text_IO.Put_Line ("Hello from Stella Icarus! The Ada skies are clear.");
   end Greet;

end Stella_Icarus;
