pragma SPARK_Mode (Off);
-- justification: subprocess import via GNAT.Expect — impure I/O cannot be expressed in SPARK
--  Tool_Math: Evaluate mathematical expressions.
--  Native Ada replacement for src/python/math_tool.py
package Tool_Math is
   --  Execute_Math: Evaluate a math expression.
   --  Params: "<expression>" e.g. "2 + 3 * 4"
   function Execute_Math (Params : String) return String with Pre => True, Post => True;
end Tool_Math;
