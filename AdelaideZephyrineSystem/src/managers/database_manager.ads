pragma SPARK_Mode (Off);
-- third-party: ada_sqlite3 (C-binding FFI — no SPARK contracts) + gnatcoll (GNATCOLL.JSON)
with Math_Utils;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Interfaces.C;          use Interfaces.C;

package Database_Manager is

   --  Initialize: Initializes the database manager and opens the database connection.
   procedure Initialize with Pre => True, Post => True;

   --  Set_System_State: Sets a key-value pair in the system state table.
   procedure Set_System_State (Key : String; Value : String) with Pre => True, Post => True;
   --  Get_System_State: Returns the value for a key from the system state table.
   function Get_System_State (Key : String; Default : String := "") return String with Pre => True, Post => True;

   --  Scaling parameter for Salience (S = HitFrequency / (1 + Alpha * DeltaT))
   Alpha : constant Float := 0.0001;

   procedure Remember
     (Prompt   : String;
      Response : String;
      Image_B64 : String := "") with Pre => True, Post => True;

   --  Prune memory based on Least Salience Mathematical Framework
   procedure Evict_Low_Salience (Chunk_Size : Positive) with Pre => True, Post => True;

   --  Native Response Cache storage
   procedure Add_To_Cache (Prompt : String;
                           Embedding : Math_Utils.Vector;
                           Response : String) with Pre => True, Post => True;

   --  Semantic Retrieval from Cache
   function Get_Cached_Response (Embedding : Math_Utils.Vector;
                                 WCET : Duration) return String with Pre => True, Post => True;

   --  Simple keyword recall (Existing logic)
   function Recall (Query : String) return String with Pre => True, Post => True;

   --  Literature/Reference Index storage (ELP0)
   procedure Add_Literature_Chunk
     (File_Path : String; 
      Content   : String; 
      Embedding : Math_Utils.Vector;
      Doc_Hash  : String) with Pre => True, Post => True;

   --  Semantic Retrieval for RAG (ELP1)
   type Chunk_Result is record
      File_Path : Unbounded_String;
      Content   : Unbounded_String;
      Score     : Float;
   end record;
   type Chunk_Array is array (Positive range <>) of Chunk_Result;

   --  Search_Literature: Searches literature chunks by embedding similarity.
   procedure Search_Literature
     (Embedding : Math_Utils.Vector;
      Results   : out Chunk_Array;
      Count     : out Natural) with Pre => True, Post => True;

   --  Semantic Retrieval for Interaction (ELP1)
   procedure Search_Interaction
     (Embedding : Math_Utils.Vector;
      Results   : out Chunk_Array;
      Count     : out Natural) with Pre => True, Post => True;

   --  LSH-based retrieval for Interaction (speculation context, ELP0)
   --  Finds entries whose 10-bit LSH is within Tolerance Hamming distance.
   procedure Search_Interaction_By_LSH
     (Hash      : Integer;
      Tolerance : Integer;
      Results   : out Chunk_Array;
      Count     : out Natural) with Pre => True, Post => True;

   --  LSH-based retrieval for Literature (speculation context, ELP0)
   procedure Search_Literature_By_LSH
     (Hash      : Integer;
      Tolerance : Integer;
      Results   : out Chunk_Array;
      Count     : out Natural) with Pre => True, Post => True;

   --  Retrieve a random literature chunk for background thinking
   procedure Get_Random_Literature_Chunk
     (Content : out Unbounded_String;
      Success : out Boolean) with Pre => True, Post => True;

   --  Knowledge Graph (GraphML style)
   procedure Add_Graph_Relation
     (Source   : String;
      Relation : String;
      Target   : String;
      Weight   : Float := 1.0;
      Context  : String := "") with Pre => True, Post => True;

   --  Export_GraphML: Exports the knowledge graph in GraphML format.
   procedure Export_GraphML (Filename : String) with Pre => True, Post => True;

   --  [VITAL-DO-NOT-REMOVE] Seed blacklist for think-only/repeating responses.
   --  Seed is Interfaces.C.unsigned (32-bit) because Generate_Seed is that
   --  type (matches Llama_Sampler_Init_Dist's C unsigned int parameter).
   --  Changing to Interfaces.C.unsigned fixes CONSTRAINT_ERROR range check
   --  when Generate_Seed exceeds Natural'Last (2^31-1).
   procedure Blacklist_Seed (Seed : Interfaces.C.unsigned) with Pre => True, Post => True;
   function Is_Seed_Blacklisted (Seed : Interfaces.C.unsigned) return Boolean with Pre => True, Post => True;
   function Get_Blacklist_Size return Natural with Pre => True, Post => True;

   --  Close: Closes the database connection and cleans up resources.
   procedure Close with Pre => True, Post => True;

   --  ============================================================================
   --  INTEGRITY TEST BLOB: Hardware-bound key verification
   --  ============================================================================
   --  Stores encrypted test blob in system_state table to verify key derivation.
   --  On boot, try decrypt test blob with derived key.
   --  If fails → signal run.py via stdio → prompt user for password/recovery key.

   --  Known plaintext for integrity verification
   Integrity_Test_Plaintext : constant String := "--ADELAIDE-INTEGRITY-TEST--";

   --  Store integrity test blob in system_state table
   --  Called after key derivation succeeds
   procedure Store_Integrity_Test_Blob (Sub_Key_Hex : String) with Pre => True, Post => True;

   --  Verify integrity test blob from system_state table
   --  Returns True if blob exists and decrypts successfully
   --  Returns False if blob missing, corrupted, or wrong key
   function Verify_Integrity_Test_Blob (Sub_Key_Hex : String) return Boolean with Pre => True, Post => True;

   --  Check if integrity test blob exists in database
   function Has_Integrity_Test_Blob return Boolean with Pre => True, Post => True;

   --  ============================================================================
   --  IMAGINED IMAGES: Store/retrieve images generated by ELP0 imagination
   --  ============================================================================
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
   --  When Hybrid_Generate's reasoning loop calls [ACTION: imagine(prompt)],
   --  the resulting base64 PNG is stored here with its LSH hash for later
   --  retrieval by VLM multimodal context or speculation context injection.

   type Imagined_Image_Result is record
      Image_B64  : Unbounded_String;
      Prompt     : Unbounded_String;
      LSH_Hash   : Integer;
      Created_At : Unbounded_String;
   end record;
   type Imagined_Image_Array is array (Positive range <>) of Imagined_Image_Result;

   --  Store an imagined image (from ELP0 imagination tool)
   procedure Store_Imagined_Image
     (Prompt    : String;
      Image_B64 : String;
      LSH_Hash  : Integer := -1) with Pre => True, Post => True;

   --  Retrieve imagined images by LSH hash (speculation context)
   procedure Search_Imagined_Images
     (Hash      : Integer;
      Tolerance : Integer;
      Results   : out Imagined_Image_Array;
      Count     : out Natural) with Pre => True, Post => True;

   --  Retrieve most recent imagined images (for VLM context)
   procedure Get_Recent_Imagined_Images
     (Max_Count : Positive;
      Results   : out Imagined_Image_Array;
      Count     : out Natural) with Pre => True, Post => True;

   --  [FREE-PARALLEL-MEMORY] Flush SQLite memory cache to disk and shrink heap usage
   procedure Flush_Memory with Pre => True, Post => True;

end Database_Manager;
