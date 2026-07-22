pragma SPARK_Mode (On);

package Watchdog_Core is

   -- File_Exists: pure Ada, no FFI, no access types.
   -- SPARK_Mode is On to formally verify it.
   -- CONTRACT INTENT: Pre => Path'Length > 0
   --                  Post => result in Boolean (trivially true)
   -- WARRANTY: No aliasing, no side effects, deterministic exception path.
   function File_Exists (Path : String) return Boolean
     with Pre    => Path'Length > 0;

end Watchdog_Core;
