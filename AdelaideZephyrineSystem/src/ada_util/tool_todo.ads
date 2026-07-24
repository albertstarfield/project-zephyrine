pragma SPARK_Mode (Off);
-- justification: subprocess import via GNAT.Expect — impure I/O cannot be expressed in SPARK
--  Tool_Todo: Task management (add, list, done, remove, clear, search).
--  Native Ada replacement for src/python/todo.py
package Tool_Todo is
   --  Execute_Todo: Manage tasks.
   --  Params: "add <task>" or "list" or "done <id>" or "remove <id>"
   --          or "clear" or "search <query>"
   function Execute_Todo (Params : String) return String with Pre => True, Post => True;
end Tool_Todo;
