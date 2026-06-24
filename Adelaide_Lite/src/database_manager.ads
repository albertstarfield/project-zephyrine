pragma SPARK_Mode (Off);
with Math_Utils;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;

package Database_Manager is

   procedure Initialize;

   --  Scaling parameter for Salience (S = HitFrequency / (1 + Alpha * DeltaT))
   Alpha : constant Float := 0.0001;

   procedure Remember
     (Prompt   : String;
      Response : String;
      Image_B64 : String := "");

   --  Prune memory based on Least Salience Mathematical Framework
   procedure Evict_Low_Salience (Chunk_Size : Positive);

   --  Native Response Cache storage
   procedure Add_To_Cache (Prompt : String;
                           Embedding : Math_Utils.Vector;
                           Response : String);

   --  Semantic Retrieval from Cache
   function Get_Cached_Response (Embedding : Math_Utils.Vector;
                                 WCET : Duration) return String;

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

   --  Semantic Retrieval for Interaction (ELP1)
   procedure Search_Interaction
     (Embedding : Math_Utils.Vector;
      Results   : out Chunk_Array;
      Count     : out Natural);

   --  LSH-based retrieval for Interaction (speculation context, ELP0)
   --  Finds entries whose 10-bit LSH is within Tolerance Hamming distance.
   procedure Search_Interaction_By_LSH
     (Hash      : Integer;
      Tolerance : Integer;
      Results   : out Chunk_Array;
      Count     : out Natural);

   --  LSH-based retrieval for Literature (speculation context, ELP0)
   procedure Search_Literature_By_LSH
     (Hash      : Integer;
      Tolerance : Integer;
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
      Weight   : Float := 1.0;
      Context  : String := "");

   procedure Export_GraphML (Filename : String);

   --  [VITAL-DO-NOT-REMOVE] Seed blacklist for think-only responses.
   --  When a seed produces only <think> with no visible content,
   --  it is blacklisted permanently. Generate skips blacklisted seeds.
   procedure Blacklist_Seed (Seed : Natural);
   function Is_Seed_Blacklisted (Seed : Natural) return Boolean;
   function Get_Blacklist_Size return Natural;

   procedure Close;

end Database_Manager;
