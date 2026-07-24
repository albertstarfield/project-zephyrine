pragma SPARK_Mode (Off);
-- justification: subprocess import via GNAT.Expect — impure I/O cannot be expressed in SPARK
--  Tool_Git: Git operations (status, log, diff, commit, branch, checkout).
--  Native Ada replacement for src/python/git.py
package Tool_Git is
   --  Execute_Git: Run a git command.
   --  Params: "<command> [args...]"
   function Execute_Git (Params : String) return String with Pre => True, Post => True;
end Tool_Git;
