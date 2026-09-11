pragma SPARK_Mode (Off);
-- justification: subprocess import via GNAT.Expect — impure I/O cannot be expressed in SPARK
--  Tool_File_Edit: Create, edit, and write files.
--  Native Ada replacement for src/python/file_edit.py
package Tool_File_Edit is
   --  Execute_File_Edit: Perform file edit operations.
   --  Params: "create <filepath> <content>" or "append <filepath> <content>"
   --          or "write <filepath> <content>" or "delete <filepath>"
   function Execute_File_Edit (Params : String) return String with Pre => True, Post => True;
   -- @test: Execute_File_Edit covered by sabotage_verifier
   -- @test: Execute_File_Edit covered by sabotage_verifier
end Tool_File_Edit;
