-- File: src/stella_icarus.adb

with Ada.Text_IO; -- We need this library to print text.

package body Stella_Icarus is

   procedure Greet is
   begin
      Ada.Text_IO.Put_Line ("Hello from Stella Icarus! The Ada skies are clear.");
   end Greet;

end Stella_Icarus;
