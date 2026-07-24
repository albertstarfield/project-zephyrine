pragma SPARK_Mode (Off);
-- justification: subprocess import via GNAT.Expect — impure I/O cannot be expressed in SPARK
--  Tool_Grep: Search file contents using pattern matching.
--  Native Ada replacement for src/python/grep.py
package Tool_Grep is
   --  Execute_Grep: Search for pattern in files.
   --  Params: "<pattern> [path] [--include *.ext]"
   function Execute_Grep (Params : String) return String with Pre => True, Post => True;
end Tool_Grep;
