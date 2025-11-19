-- This package will hold utility subprograms for our server.

with Zephyrine.Database; -- We need this to see the Row_Data type

package Zephyrine.Utils is

   -- Declare our library-level helper procedure here.
   procedure Print_Row (Row : in Zephyrine.Database.Row_Data);

end Zephyrine.Utils;