pragma SPARK_Mode (Off);
-- justification: subprocess import via GNAT.Expect — impure I/O cannot be expressed in SPARK
--  Tool_Test: Run tests (pytest, gnatprove, cargo test, etc.).
--  Native Ada replacement for src/python/test.py
package Tool_Test is
   --  Execute_Test: Run test suite.
   --  Params: "pytest [args]" or "gnatprove" or "cargo" or "lint"
   function Execute_Test (Params : String) return String with Pre => True, Post => True;
end Tool_Test;
