pragma SPARK_Mode (Off);
-- justification: subprocess import via GNAT.Expect — impure I/O cannot be expressed in SPARK
--  Tool_Hook: Git hooks management (pre-commit, post-commit, etc.).
--  Native Ada replacement for src/python/hook.py
package Tool_Hook is
   --  Execute_Hook: Manage git hooks.
   --  Params: "list" or "install <hook>" or "remove <hook>" or "run <hook>"
   function Execute_Hook (Params : String) return String with Pre => True, Post => True;
end Tool_Hook;
