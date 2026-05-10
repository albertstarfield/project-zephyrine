pragma SPARK_Mode (On);
with Ada.Directories;

package body Watchdog_Core is

   function File_Exists (Path : String) return Boolean is
      pragma SPARK_Mode (Off);
   begin
      return Ada.Directories.Exists (Path);
   end File_Exists;

end Watchdog_Core;
