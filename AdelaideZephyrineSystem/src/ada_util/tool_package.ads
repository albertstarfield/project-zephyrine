pragma SPARK_Mode (Off);
-- justification: subprocess import via GNAT.Expect — impure I/O cannot be expressed in SPARK
--  Tool_Package: System package management (apt, brew, pacman, pip).
--  Native Ada replacement for src/python/package.py
package Tool_Package is
   --  Execute_Package: Manage system packages.
   --  Params: "install <pkg>" or "remove <pkg>" or "update" or "search <pkg>"
   function Execute_Package (Params : String) return String with Pre => True, Post => True;
end Tool_Package;
