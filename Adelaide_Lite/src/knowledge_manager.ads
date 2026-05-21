package Knowledge_Manager is
   pragma Spark_Mode (Off);

   --  Initialize databases and internal state
   procedure Initialize;

   --  Start background indexing (ELP0) and proactive thinking tasks
   procedure Start_Tasks;

end Knowledge_Manager;
