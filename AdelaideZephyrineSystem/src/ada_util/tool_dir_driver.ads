pragma SPARK_Mode (Off);
-- justification: subprocess import via GNAT.Expect — impure I/O cannot be expressed in SPARK
--  Tool_Dir_Driver: Directory operations (ls, find, tree, pwd, mkdir, rm).
--  Native Ada replacement for src/python/directory.py
package Tool_Dir_Driver is
   --  Execute_Dir: Perform directory operations.
   --  Params: "ls [path]" or "find <path> <pattern>" or "tree [path] [depth]"
   --          or "pwd" or "mkdir <path>" or "rm <path>"
   function Execute_Dir (Params : String) return String with Pre => True, Post => True;
end Tool_Dir_Driver;
