-- This is the implementation of our utility procedures.

with Ada.Text_IO;
with Ada.Strings.Unbounded;

package body Zephyrine.Utils is

   -- The implementation of Print_Row is now here, in its own
   -- separate compilation unit.
   procedure Print_Row (Row : in Zephyrine.Database.Row_Data) is
      use Ada.Text_IO;
   begin
      Put (" | ");
      for I in Row'Range loop
         Put (Ada.Strings.Unbounded.To_String(Row(I)) & " | ");
      end loop;
      New_Line;
   end Print_Row;

end Zephyrine.Utils;