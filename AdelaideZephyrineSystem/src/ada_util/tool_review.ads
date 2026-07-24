pragma SPARK_Mode (Off);
-- justification: subprocess import via GNAT.Expect — impure I/O cannot be expressed in SPARK
--  Tool_Review: Code review (diff analysis, lint suggestions).
--  Native Ada replacement for src/python/review.py
package Tool_Review is
   --  Execute_Review: Review code changes.
   --  Params: "diff" or "file <filepath>" or "pr <number>"
   function Execute_Review (Params : String) return String with Pre => True, Post => True;
end Tool_Review;
