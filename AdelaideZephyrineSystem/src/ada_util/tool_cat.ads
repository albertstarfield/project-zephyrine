pragma SPARK_Mode (Off);
-- justification: subprocess import via GNAT.Expect — impure I/O cannot be expressed in SPARK
--  Tool_Cat: Read and display file contents.
--  Native Ada replacement for src/python/cat_tool.py
package Tool_Cat is
   --  Execute_Cat: Display file contents.
   --  Params: "<filepath>" or "<filepath> --line-numbers"
   function Execute_Cat (Params : String) return String with Pre => True, Post => True;
end Tool_Cat;
