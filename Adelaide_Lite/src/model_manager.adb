with Llama_Interface;
with Math_Utils;
with Streaming_Queue;
with System;
with Ada.Strings.Unbounded;
with GNATCOLL.JSON;
with Ada.Text_IO; use Ada.Text_IO;

package body Model_Manager is

   use Ada.Strings.Unbounded;
   use GNATCOLL.JSON;
   use Streaming_Queue;

   procedure Initialize is
   begin
      Put_Line ("[Model] Initializing Llama Backend...");
      Llama_Interface.Llama_Backend_Init;
   end Initialize;

   procedure Load_Model
     (Kind          : Model_Type;
      Success       : out Boolean;
      Requested_Ctx : Positive := 4096) is
      pragma Unreferenced (Requested_Ctx);
   begin
      Put_Line ("[Model] Loading model " & Kind'Image);
      Success := True;
   end Load_Model;

   procedure Unload_Model (Kind : Model_Type) is
   begin
      Put_Line ("[Model] Unloading model " & Kind'Image);
   end Unload_Model;

   procedure Force_Unload_And_Reload (Kind : Model_Type) is
   begin
      Put_Line ("[Model] Force reload " & Kind'Image);
   end Force_Unload_And_Reload;

   function Llama_Abort_Callback (Data : System.Address) return Boolean is
      pragma Unreferenced (Data);
   begin
      return False;
   end Llama_Abort_Callback;

   function Get_Context
     (Kind : Model_Type) return Llama_Interface.Llama_Context is
      pragma Unreferenced (Kind);
   begin
      return Llama_Interface.Null_Context;
   end Get_Context;

   function Get_Model
     (Kind : Model_Type) return Llama_Interface.Llama_Model is
      pragma Unreferenced (Kind);
   begin
      return Llama_Interface.Null_Model;
   end Get_Model;

   procedure Generate
     (Kind            : Model_Type;
      Prompt          : String;
      Result          : out Unbounded_String;
      Images          : GNATCOLL.JSON.JSON_Array := GNATCOLL.JSON.Empty_Array;
      Session_ID      : String := "";
      Requested_Ctx   : Positive := 4096;
      Stream          : Streaming_Queue.Queue_Access := null;
      Orch_Think_Open : Boolean := False;
      Level           : ELP_Level := ELP1) is
      pragma Unreferenced (Requested_Ctx, Orch_Think_Open, Level, Session_ID);
      use type Streaming_Queue.Queue_Access;
   begin
      if Images.Length > 0 then
         Put_Line ("[Model] Multimodal request with images.");
      end if;

      Result := To_Unbounded_String ("Response to: " & Prompt & " [" & Kind'Image & "]");

      if Stream /= null then
         Stream.Push ("Piece 1 of " & Prompt);
      end if;
   end Generate;

   procedure Hybrid_Generate
     (Prompt     : String;
      Result     : out Unbounded_String;
      Images     : GNATCOLL.JSON.JSON_Array := GNATCOLL.JSON.Empty_Array;
      Session_ID : String := "";
      Stream     : Streaming_Queue.Queue_Access := null;
      Level      : ELP_Level := ELP1) is
   begin
      Generate
        (Qwen_4B, Prompt, Result, Images, Session_ID, 4096, Stream,
         False, Level);
   end Hybrid_Generate;

   procedure Get_Embedding
     (Prompt : String;
      Result : out Math_Utils.Vector;
      Length : out Natural) is
      pragma Unreferenced (Prompt, Result);
   begin
      Length := 0;
   end Get_Embedding;

   function Get_Kind_For_Model_Name (Name : String) return Model_Type is
   begin
      if Name = "adelaide-hybrid" then
         return Qwen_4B;
      elsif Name = "adelaide-embedding" then
         return Qwen_Embedding;
      else
         return Qwen_0_8B;
      end if;
   end Get_Kind_For_Model_Name;

   function Is_Loaded (Kind : Model_Type) return Boolean is
      pragma Unreferenced (Kind);
   begin
      return True;
   end Is_Loaded;

   function Count_Tokens (Text : String) return Positive is
   begin
      return Text'Length / 4 + 1;
   end Count_Tokens;

   function Get_Request_Category
     (Msg        : String;
      Session_ID : String := "";
      Level      : ELP_Level := ELP1) return String is
      pragma Unreferenced (Msg, Session_ID, Level);
   begin
      return "General";
   end Get_Request_Category;

   function Grade_Response_Quality
     (Response_Text : String;
      Prompt        : String;
      Search_Used   : Boolean;
      Has_Citations : Boolean;
      Session_ID    : String := "";
      Level         : ELP_Level := ELP1) return Natural is
      pragma Unreferenced
        (Response_Text, Prompt, Search_Used, Has_Citations, Session_ID, Level);
   begin
      return 5;
   end Grade_Response_Quality;

   procedure Push_Chunk
     (Stream     : Streaming_Queue.Queue_Access;
      Session_ID : String;
      Str_Piece  : String) is
      pragma Unreferenced (Session_ID);
   begin
      if Stream /= null then
         Stream.Push (Str_Piece);
      end if;
   end Push_Chunk;

   function Generator_Callback (Prompt : String) return String is
   begin
      return "Callback response to " & Prompt;
   end Generator_Callback;

   function Should_Abort_ELP0 return Boolean is
   begin
      return False;
   end Should_Abort_ELP0;

end Model_Manager;
