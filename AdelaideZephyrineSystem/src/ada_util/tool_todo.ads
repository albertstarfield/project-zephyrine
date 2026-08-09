-- tool_todo.ads
-- Task management (add, list, done, remove, clear, search).
-- Native Ada replacement for src/python/todo.py.
-- Uses GNATCOLL.JSON for .todos.json persistence.
--
-- Axioms:
--   - ISO/IEC 8652:2012 RM A.4.3 (Unbounded Strings)
--   - GNATCOLL.JSON for JSON read/write
--   - Ada.Directories for file existence checks

pragma SPARK_Mode (Off);
-- Justification: File I/O via GNATCOLL.JSON and Ada.Text_IO — impure operations.

with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;

package Tool_Todo is

   --  Execute_Todo: Manage tasks.
   --  Params: "add <task>" or "list" or "done <id>" or "remove <id>"
   --          or "clear" or "search <query>"
   function Execute_Todo (Params : String) return String
     with Pre => True, Post => True;

end Tool_Todo;
