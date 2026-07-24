pragma SPARK_Mode (Off);
-- justification: subprocess import via GNAT.Expect — impure I/O cannot be expressed in SPARK
--  Tool_Code: Execute code snippets (Python/Ada safe eval).
--  Native Ada replacement for src/python/code_tool.py
package Tool_Code is
   --  Execute_Code: Run a code snippet.
   --  Params: "<language> <code>" e.g. "python print('hello')"
   function Execute_Code (Params : String) return String with Pre => True, Post => True;
end Tool_Code;
