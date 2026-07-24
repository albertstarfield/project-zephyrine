pragma SPARK_Mode (Off);
-- c_binding: Vector DB FFI
package Knowledge_Manager is

   --  Initialize databases and internal state
   procedure Initialize with Pre => True, Post => True;

   --  Start background indexing (ELP0) and proactive thinking tasks
   procedure Start_Tasks with Pre => True, Post => True;

end Knowledge_Manager;
