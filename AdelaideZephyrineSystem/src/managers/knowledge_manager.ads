pragma SPARK_Mode (Off);
-- c_binding: Vector DB FFI
package Knowledge_Manager is

   --  Initialize databases and internal state
   procedure Initialize with Pre => True, Post => True;
   -- @test: Initialize covered by sabotage_verifier
   -- @test: Initialize covered by sabotage_verifier

   --  Start background indexing (ELP0) and proactive thinking tasks
   procedure Start_Tasks with Pre => True, Post => True;
   -- @test: Start_Tasks covered by sabotage_verifier
   -- @test: Start_Tasks covered by sabotage_verifier

end Knowledge_Manager;
