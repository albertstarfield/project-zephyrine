pragma SPARK_Mode (Off);
-- justification: subprocess import via GNAT.Expect — impure I/O cannot be expressed in SPARK
--  Tool_Issue: GitHub issue management (list, create, close, comment).
--  Native Ada replacement for src/python/issue.py
package Tool_Issue is
   --  Execute_Issue: Manage GitHub issues.
   --  Params: "list" or "create <title> <body>" or "close <number>"
   --          or "comment <number> <text>"
   function Execute_Issue (Params : String) return String with Pre => True, Post => True;
end Tool_Issue;
