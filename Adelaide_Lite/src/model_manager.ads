pragma SPARK_Mode (Off);
with Llama_Interface;
with Math_Utils;
with Streaming_Queue;
with System;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with GNATCOLL.JSON;
with Model_Types; use Model_Types;

package Model_Manager is

   --  REMOVED: Generate_Speculative (ggml draft-model token speculation).
   --  Was disabled via `Enable_Speculative => False` due to ggml-metal GPU
   --  buffer races during QWEN_0_8B unload causing Abort trap: 6.
   --  Replaced by Speculative_Cache (query-level semantic cache populated
   --  proactively by ELP0 background tasks — see Knowledge_Manager).

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

   --  Perform inference
   procedure Generate
     (Kind            : Model_Type;
      Prompt          : String;
      Result          : out Unbounded_String;
      Images          : GNATCOLL.JSON.JSON_Array := GNATCOLL.JSON.Empty_Array;
      Session_ID      : String := "";
      Requested_Ctx   : Positive := 4096;
      Stream          : Streaming_Queue.Queue_Access := null;
      Orch_Think_Open : Boolean := False;
      Level           : ELP_Level := ELP1);

   --  Perform multi-hop reasoning
   procedure Hybrid_Generate
     (Prompt         : String;
      Result         : out Unbounded_String;
      Images         : GNATCOLL.JSON.JSON_Array := GNATCOLL.JSON.Empty_Array;
      Session_ID     : String := "";
      Stream         : Streaming_Queue.Queue_Access := null;
      Level          : ELP_Level := ELP1;
      Agentic        : Boolean := False;
      Raw_Prompt     : Boolean := False;
      External_Agent : Boolean := False);

   procedure Get_Embedding
     (Prompt : String;
      Result : out Math_Utils.Vector;
      Length : out Natural;
      Level  : ELP_Level := ELP1);

   function Should_Abort_ELP0 return Boolean;

   function Get_Kind_For_Model_Name (Name : String) return Model_Type;

   function Is_Loaded (Kind : Model_Type) return Boolean;

   function Count_Tokens (Text : String) return Positive;

   function Get_Request_Category
     (Msg        : String;
      Session_ID : String := "";
      Level      : ELP_Level := ELP1) return String;

   function Grade_Response_Quality
     (Response_Text : String;
      Prompt        : String;
      Search_Used   : Boolean;
      Has_Citations : Boolean;
      Session_ID    : String := "";
      Level         : ELP_Level := ELP1) return Natural;

   procedure Push_Chunk
     (Stream     : Streaming_Queue.Queue_Access;
      Session_ID : String;
      Str_Piece  : String);

   function Generator_Callback (Prompt : String) return String;

   function Sanitize_Think_Tags (Text : String) return String;

   Current_WCET : Duration := 0.0;
   Current_WCET_ELP0 : Duration := 0.0;
   Current_WCET_ELP1 : Duration := 0.0;
   Current_WCET_ELP2 : Duration := 0.0;
   Current_WCET_ELP3 : Duration := 0.0;

   --  ELP3 Timing Correction / Jitter Profile
   Current_Jitter_Max : Duration := 0.0;
   Current_Jitter_Avg : Duration := 0.0;

   --  Last user prompt (set by Hybrid_Generate at ELP1 level).
   --  Read by Proactive_Cache_Task in Knowledge_Manager to predict
   --  follow-up questions and pre-populate Speculative_Cache.
   Last_User_Prompt : Unbounded_String := Null_Unbounded_String;

end Model_Manager;
