with Llama_Interface;
with Math_Utils;
with Streaming_Queue;
with System;

package Model_Manager is
   pragma Spark_Mode (Off);

   type Model_Type is (Qwen_0_8B, Qwen_4B, Qwen_Embedding, MMProj);

   procedure Initialize;

   procedure Load_Model
     (Kind          : Model_Type;
      Success       : out Boolean;
      Requested_Ctx : Positive := 4096);

   procedure Unload_Model (Kind : Model_Type);

   procedure Force_Unload_And_Reload (Kind : Model_Type);

   function Llama_Abort_Callback (Data : System.Address) return Boolean;
   pragma Convention (C, Llama_Abort_Callback);

   function Get_Context
     (Kind : Model_Type) return Llama_Interface.Llama_Context;

   function Get_Model
     (Kind : Model_Type) return Llama_Interface.Llama_Model;

   --  Perform inference (simplified for now)
   function Generate
     (Kind          : Model_Type;
      Prompt        : String;
      Session_ID    : String := "";
      Requested_Ctx : Positive := 4096;
      Stream        : Streaming_Queue.Queue_Access := null) return String;

   --  Perform multi-hop reasoning (0.8b thinking -> 4b final)
   function Hybrid_Generate
     (Prompt     : String;
      Session_ID : String := "";
      Stream     : Streaming_Queue.Queue_Access := null) return String;

   function Get_Embedding (Prompt : String) return Math_Utils.Vector;

   function Get_Kind_For_Model_Name (Name : String) return Model_Type;

   function Is_Loaded (Kind : Model_Type) return Boolean;

   function Count_Tokens (Text : String) return Positive;

   function Get_Request_Category
     (Msg        : String;
      Session_ID : String := "") return String;

   function Grade_Response_Quality
     (Response_Text : String;
      Prompt        : String;
      Search_Used   : Boolean;
      Has_Citations : Boolean;
      Session_ID    : String := "") return Natural;

   function Generator_Callback (Prompt : String) return String;

end Model_Manager;
