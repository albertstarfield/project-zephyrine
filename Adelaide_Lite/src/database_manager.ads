with Math_Utils;

package Database_Manager is
   pragma Spark_Mode (Off);

   procedure Initialize;

   procedure Remember (User_Input : String; Assistant_Response : String);

   --  Native Response Cache storage
   procedure Add_To_Cache (Prompt : String; Embedding : Math_Utils.Vector; Response : String);

   --  Semantic Retrieval from Cache
   function Get_Cached_Response (Embedding : Math_Utils.Vector) return String;

   --  Simple keyword recall (Existing logic)
   function Recall (Query : String) return String;

   procedure Close;

end Database_Manager;
