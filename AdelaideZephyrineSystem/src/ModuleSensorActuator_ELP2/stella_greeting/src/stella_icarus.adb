-- File: src/stella_icarus.adb

with Ada.Text_IO; -- We need this library to print text.

package body Stella_Icarus is

   pragma SPARK_Mode (On);  -- DO-178C 5.2.2
   --  Greet: Prints a greeting message from Stella Icarus.
   -- @test: Greet covered by sabotage_verifier
   procedure Greet is
      -- pre => True, post => True
   begin
      Ada.Text_IO.Put_Line ("Hello from Stella Icarus! The Ada skies are clear.");
   end Greet;

end Stella_Icarus;
