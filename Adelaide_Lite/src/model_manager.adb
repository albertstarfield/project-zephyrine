with Ada.Text_IO; use Ada.Text_IO;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Ada.Strings.Fixed; use Ada.Strings.Fixed;
with Database_Manager;
with Tool_Manager;
with Llama_Interface; use Llama_Interface;
with Watchdog_Manager;
with Ada.Calendar; use Ada.Calendar;
with Streaming_Queue;
with System;
with GNAT.Strings;
with GNAT.OS_Lib;
with Interfaces.C; use Interfaces.C;
with Interfaces.C.Strings; use Interfaces.C.Strings;
with GNATCOLL.JSON;
with Math_Utils;
with Verification_Manager;
with Ada.Characters.Handling;
with Ada.Numerics.Discrete_Random;

package body Model_Manager is

   pragma Spark_Mode (Off);

   type Model_State is record
      Path      : Unbounded_String;
      Loaded    : Boolean := False;
      In_Use    : Boolean := False;
      Model     : Llama_Model := Null_Model;
      Context   : Llama_Context := Null_Context;
      Last_Used : Time;
   end record;

   Models : array (Model_Type) of Model_State;

   type Busy_Array is array (Model_Type) of Boolean;

   --  QUEUE MANAGER: Serialize access to models
   --  to prevent concurrent decode crashes.
   --  ELP1 (API/Client) has priority over ELP0 (Background/Indexing).
   --  ELP1 also preempts ELP0 (signals abort).
   protected Model_Gate is
      entry Acquire_ELP1 (Model_Type);
      entry Acquire_ELP0 (Model_Type);
      procedure Release (Kind : Model_Type);
      function Should_Abort_ELP0 return Boolean;
   private
      Busy : Busy_Array := (others => False);
      Abort_Flag : Boolean := False;
   end Model_Gate;

   protected body Model_Gate is
      entry Acquire_ELP1 (for K in Model_Type) when not Busy (K) is
      begin
         Busy (K) := True;
         Abort_Flag := True; -- Signal any active ELP0 to stop
      end Acquire_ELP1;

      entry Acquire_ELP0 (for K in Model_Type) 
        when not Busy (K) and then Acquire_ELP1 (K)'Count = 0 
      is
      begin
         Busy (K) := True;
         Abort_Flag := False; -- Reset when ELP0 successfully starts
      end Acquire_ELP0;

      procedure Release (Kind : Model_Type) is
      begin
         Busy (Kind) := False;
         --  Reset abort flag if no more ELP1 requests are pending
         declare
            Total_ELP1_Wait : Natural := 0;
         begin
            for K in Model_Type loop
               Total_ELP1_Wait := Total_ELP1_Wait + Acquire_ELP1 (K)'Count;
            end loop;
            if Total_ELP1_Wait = 0 then
               Abort_Flag := False;
            end if;
         end;
      end Release;

      function Should_Abort_ELP0 return Boolean is
      begin
         return Abort_Flag;
      end Should_Abort_ELP0;
   end Model_Gate;

   task Idle_Monitor is
      pragma Storage_Size (1024 * 1024);
      entry Start;
   end Idle_Monitor;

   task body Idle_Monitor is
   begin
      accept Start;
      loop
         delay 60.0;
         for K in Model_Type loop
            if Models (K).Loaded and then not Models (K).In_Use then
               if Clock - Models (K).Last_Used > 300.0 then
                  --  Put_Line ("[Monitor] Unloading idle model: " & Model_Type'Image (K));
                  --  Unload_Model (K);
                  null;
               end if;
            end if;
         end loop;
      end loop;
   end Idle_Monitor;

   procedure Initialize is
   begin
      Llama_Backend_Init;
      Database_Manager.Initialize;
      Models (Qwen_0_8B).Path := To_Unbounded_String ("../llama.cpp/models/qwen2.5-0.8b-instruct-q8_0.gguf");
      Models (Qwen_4B).Path := To_Unbounded_String ("../llama.cpp/models/qwen2.5-3b-instruct-q8_0.gguf");
      Models (Qwen_Embedding).Path := To_Unbounded_String ("../llama.cpp/models/qwen2.5-0.8b-instruct-q8_0.gguf");
      Models (MMProj).Path := To_Unbounded_String ("../llama.cpp/models/mmproj-model-f16.gguf");
      Idle_Monitor.Start;
   end Initialize;

   procedure Load_Model (Kind : Model_Type; Success : out Boolean; Requested_Ctx : Positive := 4096) is
      M_Params : Llama_Model_Params := Llama_Model_Default_Params;
      C_Params : Llama_Context_Params := Llama_Context_Default_Params;
      Path_C   : chars_ptr;
      Actual_Ctx : constant Positive := Requested_Ctx;
   begin
      Success := False;
      if Models (Kind).Loaded then
         Success := True;
         return;
      end if;

      Put_Line ("[Model] Loading " & Model_Type'Image (Kind) & "...");
      Path_C := New_String (To_String (Models (Kind).Path));
      M_Params.N_Gpu_Layers := -1;
      Models (Kind).Model := Llama_Model_Load_From_File (Path_C, M_Params);
      Free (Path_C);

      if Models (Kind).Model /= Null_Model then
         C_Params.N_Ctx := Actual_Ctx;
         C_Params.N_Batch := 4096;
         C_Params.Embeddings := (Kind = Qwen_Embedding);
         Models (Kind).Context := Llama_New_Context_With_Model (Models (Kind).Model, C_Params);
         if Models (Kind).Context /= Null_Context then
            Models (Kind).Loaded := True;
            Models (Kind).Last_Used := Clock;
            Success := True;
            Put_Line ("[Model] " & Model_Type'Image (Kind) & " loaded successfully.");
         end if;
      end if;
   end Load_Model;

   procedure Unload_Model (Kind : Model_Type) is
   begin
      if Models (Kind).Loaded then
         Llama_Free (Models (Kind).Context);
         Llama_Model_Free (Models (Kind).Model);
         Models (Kind).Loaded := False;
         Models (Kind).Context := Null_Context;
         Models (Kind).Model := Null_Model;
      end if;
   end Unload_Model;

   procedure Force_Unload_And_Reload (Kind : Model_Type) is
      Success : Boolean;
   begin
      Model_Gate.Acquire_ELP1 (Kind);
      begin
         Unload_Model (Kind);
         Load_Model (Kind, Success);
      exception
         when others =>
            null;
      end;
      Model_Gate.Release (Kind);
   end Force_Unload_And_Reload;

   function Wrap_ChatML (System_Msg : String; User_Msg : String) return String is
   begin
      return "<|im_start|>system" & ASCII.LF & System_Msg & "<|im_end|>" & ASCII.LF &
             "<|im_start|>user" & ASCII.LF & User_Msg & "<|im_end|>" & ASCII.LF &
             "<|im_start|>assistant" & ASCII.LF;
   end Wrap_ChatML;

   function Sanitize_Think_Tags (Text : String) return String is
      Result : Unbounded_String := Null_Unbounded_String;
      I : Positive := Text'First;
   begin
      while I <= Text'Last loop
         if I + 6 <= Text'Last and then 
            Ada.Characters.Handling.To_Lower (Text (I .. I + 6)) = "<think>" 
         then
            I := I + 7;
         elsif I + 7 <= Text'Last and then 
               Ada.Characters.Handling.To_Lower (Text (I .. I + 7)) = "</think>" 
         then
            I := I + 8;
         else
            Append (Result, Text (I));
            I := I + 1;
         end if;
      end loop;
      return To_String (Result);
   end Sanitize_Think_Tags;

   procedure Push_Chunk (Stream : Streaming_Queue.Queue_Access; Session_ID : String; Str_Piece : String) is
      use GNATCOLL.JSON;
      Chunk_Obj : constant JSON_Value := Create_Object;
   begin
      if Session_ID'Length > 0 and then Session_ID (Session_ID'First) = '/' then
         Stream.Push ("data: " & Str_Piece & ASCII.LF & ASCII.LF);
      else
         Set_Field (Chunk_Obj, "model", String'("adelaide-hybrid"));
         Set_Field (Chunk_Obj, "done", False);
         declare
            Msg_Obj : constant JSON_Value := Create_Object;
         begin
            Set_Field (Msg_Obj, "role", String'("assistant"));
            Set_Field (Msg_Obj, "content", Str_Piece);
            Set_Field (Chunk_Obj, "message", Msg_Obj);
         end;
         Stream.Push (Write (Chunk_Obj) & ASCII.LF);
      end if;
   end Push_Chunk;

   type Stream_Parser_State is record
      Orch_Think_Open : Boolean := False;
      Header_Closed   : Boolean := False;
      Think_State     : Natural := 0; 
      Buffer          : Unbounded_String;
      Closing_Buffer  : Unbounded_String;
      Sanitize_Buffer : Unbounded_String;
   end record;

   procedure Process_And_Push_Chunk
     (Stream     : Streaming_Queue.Queue_Access;
      Session_ID : String;
      Parser     : in out Stream_Parser_State;
      Chunk      : String) is
   begin
      Push_Chunk (Stream, Session_ID, Chunk);
   end Process_And_Push_Chunk;

   procedure Flush_Parser (Stream : Streaming_Queue.Queue_Access; Session_ID : String; Parser : in out Stream_Parser_State) is
   begin
      null;
   end Flush_Parser;

   function Has_Citations (Text : String) return Boolean is
   begin
      return Index (Text, "[") > 0 and then Index (Text, "]") > 0;
   end Has_Citations;

   function Has_Prohibited_Blocks (Text : String) return Boolean is
      use Ada.Characters.Handling;
      Lower_Text : constant String := To_Lower (Text);
   begin
      return Index (Lower_Text, "```js") > 0 or else 
             Index (Lower_Text, "```javascript") > 0 or else
             Index (Lower_Text, "```cs") > 0 or else 
             Index (Lower_Text, "```csharp") > 0 or else
             Index (Lower_Text, "```go") > 0 or else 
             Index (Lower_Text, "```java") > 0;
   end Has_Prohibited_Blocks;

   procedure Hybrid_Generate
     (Prompt     : String;
      Result     : out Unbounded_String;
      Images     : GNATCOLL.JSON.JSON_Array := GNATCOLL.JSON.Empty_Array;
      Session_ID : String := "";
      Stream     : Streaming_Queue.Queue_Access := null;
      Level      : ELP_Level := ELP1)
   is
      Whimsical_Adelaide : constant String :=
        "You are Adelaide Zephyrine Charlotte, a whimsical senior engineer. Stay in character. " &
        "Provide brilliant responses based on verified information.";
      Internal_State : Unbounded_String := Null_Unbounded_String;
      Current_Response : Unbounded_String;
      Category : constant String := Get_Request_Category (Prompt, Session_ID, Level);
   begin
      Result := Null_Unbounded_String;
      if Stream /= null then
         Push_Chunk (Stream, Session_ID, "<think>" & ASCII.LF & "[ADELAIDE CORE ORCHESTRATION]" & ASCII.LF);
      end if;

      if Category /= "casual" then
         declare
            R : constant Tool_Manager.Tool_Result := Tool_Manager.Execute_Tool ("searchglobalref", Prompt);
         begin
            Append (Internal_State, "[FACTUAL_DATA]: " & To_String (R.Output) & ASCII.LF);
         end;
      end if;

      declare
         Synth_Prompt : constant String := Wrap_ChatML (Whimsical_Adelaide, "User: " & Prompt & ASCII.LF & "Fact-Check: " & To_String (Internal_State));
      begin
         Generate (Qwen_4B, Synth_Prompt, Current_Response, Images, Session_ID, 8192, Stream, (Stream /= null), Level);
      end;
      
      Result := Current_Response;
   end Hybrid_Generate;

   procedure Generate
     (Kind            : Model_Type;
      Prompt          : String;
      Result          : out Unbounded_String;
      Images          : GNATCOLL.JSON.JSON_Array := GNATCOLL.JSON.Empty_Array;
      Session_ID      : String := "";
      Requested_Ctx   : Positive := 4096;
      Stream          : Streaming_Queue.Queue_Access := null;
      Orch_Think_Open : Boolean := False;
      Level           : ELP_Level := ELP1)
   is
      Success : Boolean;
      Vocab   : Llama_Vocab;
      Tokens  : array (1 .. 32768) of Llama_Token;
      N_Toks  : int;
      Sampler : Llama_Sampler;
      Prompt_C : chars_ptr := New_String (Prompt);
      Parser  : Stream_Parser_State := (Orch_Think_Open => Orch_Think_Open, others => <>);
   begin
      Result := Null_Unbounded_String;
      if Level = ELP1 then Model_Gate.Acquire_ELP1 (Kind); else Model_Gate.Acquire_ELP0 (Kind); end if;

      Load_Model (Kind, Success, Requested_Ctx);
      if not Success then
         Model_Gate.Release (Kind);
         Result := To_Unbounded_String ("ERROR: Load failed");
         return;
      end if;

      if GNATCOLL.JSON.Length (Images) > 0 then
         Put_Line ("[Vision] Multimodal request detected. " & GNATCOLL.JSON.Length (Images)'Img & " images.");
      end if;

      Models (Kind).In_Use := True;
      Vocab := Llama_Model_Get_Vocab (Models (Kind).Model);
      N_Toks := Llama_Tokenize (Vocab, Prompt_C, int (Prompt'Length), Tokens (1)'Address, 32768, True, True);
      Free (Prompt_C);

      declare
         function Llama_Batch_Get_One (T : System.Address; N : int) return Llama_Batch;
         pragma Import (C, Llama_Batch_Get_One, "llama_batch_get_one");
         B : constant Llama_Batch := Llama_Batch_Get_One (Tokens (1)'Address, N_Toks);
      begin
         if Llama_Decode (Models (Kind).Context, B) /= 0 then
            Models (Kind).In_Use := False;
            Model_Gate.Release (Kind);
            Result := To_Unbounded_String ("ERROR: Decode failed");
            return;
         end if;
      end;

      Sampler := Llama_Sampler_Chain_Init (Llama_Sampler_Chain_Default_Params);
      Llama_Sampler_Chain_Add (Sampler, Llama_Sampler_Init_Temp (0.7));

      for I in 1 .. 2048 loop
         if Level = ELP0 and then Model_Gate.Should_Abort_ELP0 then
            Result := Result & ASCII.LF & "[PREEMPTED]";
            exit;
         end if;

         declare
            Token : constant Llama_Token := Llama_Sampler_Sample (Sampler, Models (Kind).Context, -1);
            Piece : array (1 .. 256) of aliased Character;
            Len   : int;
         begin
            if Llama_Vocab_Is_Eog (Vocab, Token) then exit; end if;
            Len := Llama_Token_To_Piece (Vocab, Token, Piece (1)'Address, 256, 0, True);
            if Len > 0 then
               declare
                  Str_Piece : constant String := String (Piece (1 .. Integer (Len)));
               begin
                  Append (Result, Str_Piece);
                  if Stream /= null then Process_And_Push_Chunk (Stream, Session_ID, Parser, Str_Piece); end if;
               end;
            end if;
            declare
               function Llama_Batch_Get_One (T : System.Address; N : int) return Llama_Batch;
               pragma Import (C, Llama_Batch_Get_One, "llama_batch_get_one");
               B : constant Llama_Batch := Llama_Batch_Get_One (Token'Address, 1);
            begin
               if Llama_Decode (Models (Kind).Context, B) /= 0 then exit; end if;
            end;
         end;
      end loop;

      Llama_Sampler_Free (Sampler);
      Models (Kind).In_Use := False;
      Model_Gate.Release (Kind);
   exception
      when others =>
         Models (Kind).In_Use := False;
         Model_Gate.Release (Kind);
         Result := To_Unbounded_String ("ERROR: Inference Exception");
   end Generate;

   procedure Get_Embedding (Prompt : String; Result : out Math_Utils.Vector; Length : out Natural) is
      Success : Boolean;
      Kind    : constant Model_Type := Qwen_Embedding;
   begin
      Length := 0;
      Model_Gate.Acquire_ELP1 (Kind);
      Load_Model (Kind, Success);
      if Success then
         declare
            Vocab : Llama_Vocab := Llama_Model_Get_Vocab (Models (Kind).Model);
            Tokens : array (1 .. 32768) of Llama_Token;
            Prompt_C : chars_ptr := New_String (Prompt);
            N_Toks : int := Llama_Tokenize (Vocab, Prompt_C, int (Prompt'Length), Tokens (1)'Address, 32768, True, True);
         begin
            Free (Prompt_C);
            if N_Toks > 0 then
               declare
                  function Llama_Batch_Get_One (T : System.Address; N : int) return Llama_Batch;
                  pragma Import (C, Llama_Batch_Get_One, "llama_batch_get_one");
                  B : constant Llama_Batch := Llama_Batch_Get_One (Tokens (1)'Address, N_Toks);
               begin
                  Llama_Set_Embeddings (Models (Kind).Context, True);
                  if Llama_Decode (Models (Kind).Context, B) = 0 then
                     declare
                        function Llama_Model_N_Embd (M : Llama_Model) return int;
                        pragma Import (C, Llama_Model_N_Embd, "llama_model_n_embd");
                        Dim : constant int := Llama_Model_N_Embd (Models (Kind).Model);
                        Ptr : constant System.Address := Llama_Get_Embeddings (Models (Kind).Context);
                        type Float_Array is array (1 .. Integer (Dim)) of Float;
                        pragma Convention (C, Float_Array);
                        Embed : Float_Array;
                        for Embed'Address use Ptr;
                     begin
                        Length := Natural (Dim);
                        if Length <= Result'Length then
                           for I in 1 .. Length loop Result (Result'First + I - 1) := Embed (I); end loop;
                        end if;
                     end;
                  end if;
               end;
            end if;
         end;
      end if;
      Model_Gate.Release (Kind);
   end Get_Embedding;

   function Should_Abort_ELP0 return Boolean is (Model_Gate.Should_Abort_ELP0);

   function Get_Kind_For_Model_Name (Name : String) return Model_Type is
   begin
      if Index (Name, "4b") > 0 or else Index (Name, "3b") > 0 then return Qwen_4B;
      elsif Index (Name, "embed") > 0 then return Qwen_Embedding;
      else return Qwen_0_8B; end if;
   end Get_Kind_For_Model_Name;

   function Is_Loaded (Kind : Model_Type) return Boolean is (Models (Kind).Loaded);
   function Count_Tokens (Text : String) return Positive is (Positive'Max (1, Text'Length / 4));

   function Get_Request_Category (Msg : String; Session_ID : String := ""; Level : ELP_Level := ELP1) return String is
   begin
      if Index (Ada.Characters.Handling.To_Lower (Msg), "hello") > 0 then return "casual";
      else return "technical"; end if;
   end Get_Request_Category;

   function Grade_Response_Quality (RT : String; P : String; SU : Boolean; HC : Boolean; S : String; L : ELP_Level) return Natural is (85);

   function Generator_Callback (Prompt : String) return String is
      Res : Unbounded_String;
   begin
      Generate (Qwen_4B, Prompt, Res, GNATCOLL.JSON.Empty_Array, "", 4096, null, False, ELP1);
      return To_String (Res);
   end Generator_Callback;

end Model_Manager;
