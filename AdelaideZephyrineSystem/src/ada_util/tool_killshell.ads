pragma SPARK_Mode (Off);
-- justification: subprocess import via GNAT.Expect — impure I/O cannot be expressed in SPARK
--  Tool_Killshell: Process management (kill, list, find).
--  Native Ada replacement for src/python/killshell.py
package Tool_Killshell is
   --  Execute_Killshell: Manage processes.
   --  Params: "kill <pid>" or "list" or "find <name>"
   function Execute_Killshell (Params : String) return String with Pre => True, Post => True;
end Tool_Killshell;
