with Math_Utils;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;

package Database_Manager is
   pragma Spark_Mode (Off);

   procedure Initialize;

   --  Scaling parameter for Salience (S = HitFrequency / (1 + Alpha * DeltaT))
   Alpha : constant Float := 0.0001;

   procedure Remember (User_Input : String; Assistant_Response : String);

   --  Native Response Cache storage
   procedure Add_To_Cache (Prompt : String; Embedding : Math_Utils.Vector; Response : String);

   --  Semantic Retrieval from Cache
   function Get_Cached_Response (Embedding : Math_Utils.Vector; WCET : Duration) return String;

   --  Simple keyword recall (Existing logic)
   function Recall (Query : String) return String;

   --  Literature/Reference Index storage (ELP0)
   procedure Add_Literature_Chunk
     (File_Path : String; 
      Content   : String; 
      Embedding : Math_Utils.Vector;
      Doc_Hash  : String);

   --  Semantic Retrieval for RAG (ELP1)
   type Chunk_Result is record
      File_Path : Unbounded_String;
      Content   : Unbounded_String;
      Score     : Float;
   end record;
   type Chunk_Array is array (Positive range <>) of Chunk_Result;

   procedure Search_Literature
     (Embedding : Math_Utils.Vector;
      Results   : out Chunk_Array;
      Count     : out Natural);

   --  Retrieve a random literature chunk for background thinking
   procedure Get_Random_Literature_Chunk
     (Content : out Unbounded_String;
      Success : out Boolean);

   --  Knowledge Graph (GraphML style)
   procedure Add_Graph_Relation
     (Source   : String;
      Relation : String;
      Target   : String;
      Weight   : Float := 1.0);

   procedure Export_GraphML (Filename : String);

   procedure Close;

end Database_Manager;
