with Ada.Text_IO; use Ada.Text_IO;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Ada.Strings.Fixed; use Ada.Strings.Fixed;
with Database_Manager;
with Tool_Manager;
with Llama_Interface; use Llama_Interface;
with Ada.Calendar; use Ada.Calendar;
with Streaming_Queue; use Streaming_Queue;
with System;
with Interfaces.C; use Interfaces.C;
with Interfaces.C.Strings; use Interfaces.C.Strings;
with GNATCOLL.JSON;
with Math_Utils;
with Ada.Characters.Handling;

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
         Abort_Flag := True;
      end Acquire_ELP1;

      entry Acquire_ELP0 (for K in Model_Type) 
        when not Busy (K) and then Acquire_ELP1 (K)'Count = 0 
      is
      begin
         Busy (K) := True;
         Abort_Flag := False;
      end Acquire_ELP0;

      procedure Release (Kind : Model_Type) is
         T1 : Natural := 0;
      begin
         Busy (Kind) := False;
         for K in Model_Type loop
            T1 := T1 + Acquire_ELP1 (K)'Count;
         end loop;
         if T1 = 0 then Abort_Flag := False; end if;
      end Release;

      function Should_Abort_ELP0 return Boolean is (Abort_Flag);
   end Model_Gate;

   function Llama_Abort_Callback (Data : System.Address) return Boolean is
   begin
      return Model_Gate.Should_Abort_ELP0;
   end Llama_Abort_Callback;

   function Get_Context (Kind : Model_Type) return Llama_Interface.Llama_Context is (Models (Kind).Context);
   function Get_Model (Kind : Model_Type) return Llama_Interface.Llama_Model is (Models (Kind).Model);

   procedure Initialize is
   begin
      Llama_Backend_Init;
      Database_Manager.Initialize;
      Models (Qwen_0_8B).Path := To_Unbounded_String ("../llama.cpp/models/qwen2.5-0.8b-instruct-q8_0.gguf");
      Models (Qwen_4B).Path := To_Unbounded_String ("../llama.cpp/models/qwen2.5-3b-instruct-q8_0.gguf");
      Models (Qwen_Embedding).Path := To_Unbounded_String ("../llama.cpp/models/qwen2.5-0.8b-instruct-q8_0.gguf");
      Models (MMProj).Path := To_Unbounded_String ("../llama.cpp/models/mmproj-model-f16.gguf");
   end Initialize;

   procedure Load_Model (Kind : Model_Type; Success : out Boolean; Requested_Ctx : Positive := 4096) is
      M_P : Llama_Model_Params := Llama_Model_Default_Params;
      C_P : Llama_Context_Params := Llama_Context_Default_Params;
      P_C : chars_ptr;
   begin
      Success := False;
      if Models (Kind).Loaded then Success := True; return; end if;

      Put_Line ("[Model] Loading " & Model_Type'Image (Kind) & "...");
      P_C := New_String (To_String (Models (Kind).Path));
      M_P.N_Gpu_Layers := -1;
      Models (Kind).Model := Llama_Model_Load_From_File (P_C, M_P);
      Free (P_C);

      if Models (Kind).Model /= Null_Model then
         C_P.N_Ctx := Interfaces.C.unsigned (Requested_Ctx);
         C_P.N_Batch := 4096;
         C_P.Embeddings := (Kind = Qwen_Embedding);
         Models (Kind).Context := Llama_Init_From_Model (Models (Kind).Model, C_P);
         if Models (Kind).Context /= Null_Context then
            Models (Kind).Loaded := True;
            Models (Kind).Last_Used := Clock;
            Success := True;
         end if;
      end if;
   end Load_Model;

   procedure Unload_Model (Kind : Model_Type) is
   begin
      if Models (Kind).Loaded then
         Llama_Free (Models (Kind).Context);
         Llama_Model_Free (Models (Kind).Model);
         Models (Kind).Loaded := False;
      end if;
   end Unload_Model;

   procedure Force_Unload_And_Reload (Kind : Model_Type) is
      S : Boolean;
   begin
      Model_Gate.Acquire_ELP1 (Kind);
      Unload_Model (Kind);
      Load_Model (Kind, S);
      Model_Gate.Release (Kind);
   end Force_Unload_And_Reload;

   function Wrap_ChatML (S_M, U_M : String) return String is
   begin
      return "<|im_start|>system" & ASCII.LF & S_M & "<|im_end|>" & ASCII.LF &
             "<|im_start|>user" & ASCII.LF & U_M & "<|im_end|>" & ASCII.LF &
             "<|im_start|>assistant" & ASCII.LF;
   end Wrap_ChatML;

   function Sanitize_Think_Tags (Text : String) return String is
      Res : Unbounded_String := Null_Unbounded_String;
      I   : Positive := Text'First;
   begin
      while I <= Text'Last loop
         if I + 6 <= Text'Last and then Text (I .. I + 6) = "<think>" then I := I + 7;
         elsif I + 7 <= Text'Last and then Text (I .. I + 7) = "</think>" then I := I + 8;
         else Append (Res, Text (I)); I := I + 1; end if;
      end loop;
      return To_String (Res);
   end Sanitize_Think_Tags;

   procedure Push_Chunk (Stream : Streaming_Queue.Queue_Access; Session_ID : String; Str_Piece : String) is
      use GNATCOLL.JSON;
      C_O : constant JSON_Value := Create_Object;
   begin
      if Session_ID'Length > 0 and then Session_ID (Session_ID'First) = '/' then
         Stream.Push ("data: " & Str_Piece & ASCII.LF & ASCII.LF);
      else
         Set_Field (C_O, "model", String'("adelaide-hybrid"));
         Set_Field (C_O, "done", False);
         declare
            M_O : constant JSON_Value := Create_Object;
         begin
            Set_Field (M_O, "role", String'("assistant"));
            Set_Field (M_O, "content", Str_Piece);
            Set_Field (C_O, "message", M_O);
         end;
         Stream.Push (Write (C_O) & ASCII.LF);
      end if;
   end Push_Chunk;

   procedure Hybrid_Generate
     (Prompt     : String;
      Result     : out Unbounded_String;
      Images     : GNATCOLL.JSON.JSON_Array := GNATCOLL.JSON.Empty_Array;
      Session_ID : String := "";
      Stream     : Streaming_Queue.Queue_Access := null;
      Level      : ELP_Level := ELP1)
   is
      Persona : constant String := "You are Adelaide, a whimsical senior engineer.";
      Int_Res : Unbounded_String;
   begin
      Generate (Qwen_4B, Wrap_ChatML (Persona, Prompt), Int_Res, Images, Session_ID, 8192, Stream, (Stream /= null), Level);
      Result := Int_Res;
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
      S : Boolean;
      V : Llama_Vocab;
      T : array (1 .. 32768) of Llama_Token;
      N : int;
      Samp : Llama_Sampler;
      P_C  : chars_ptr := New_String (Prompt);
   begin
      Result := Null_Unbounded_String;
      if Level = ELP1 then Model_Gate.Acquire_ELP1 (Kind); else Model_Gate.Acquire_ELP0 (Kind); end if;
      Load_Model (Kind, S, Requested_Ctx);
      if not S then Model_Gate.Release (Kind); Result := To_Unbounded_String ("ERROR"); return; end if;

      if GNATCOLL.JSON.Length (Images) > 0 then
         Put_Line ("[Vision] Multimodal request detected.");
      end if;

      Models (Kind).In_Use := True;
      V := Llama_Model_Get_Vocab (Models (Kind).Model);
      N := Llama_Tokenize (V, P_C, int (Prompt'Length), T (1)'Address, 32768, True, True);
      Free (P_C);

      declare
         function Batch_Get_One (T : System.Address; N : int) return Llama_Batch;
         pragma Import (C, Batch_Get_One, "llama_batch_get_one");
         B : constant Llama_Batch := Batch_Get_One (T (1)'Address, N);
      begin
         if Llama_Decode (Models (Kind).Context, B) /= 0 then
            Models (Kind).In_Use := False; Model_Gate.Release (Kind); return;
         end if;
      end;

      Samp := Llama_Sampler_Chain_Init (Llama_Sampler_Chain_Default_Params);
      Llama_Sampler_Chain_Add (Samp, Llama_Sampler_Init_Temp (0.7));

      for I in 1 .. 2048 loop
         if Level = ELP0 and then Model_Gate.Should_Abort_ELP0 then exit; end if;
         declare
            Tok : constant Llama_Token := Llama_Sampler_Sample (Samp, Models (Kind).Context, -1);
            Pie : array (1 .. 256) of aliased Character;
            Len : int;
         begin
            if Llama_Vocab_Is_Eog (V, Tok) then exit; end if;
            Len := Llama_Token_To_Piece (V, Tok, Pie (1)'Address, 256, 0, True);
            if Len > 0 then
               declare
                  S_P : constant String := String (Pie (1 .. Integer (Len)));
               begin
                  Append (Result, S_P);
                  if Stream /= null then Push_Chunk (Stream, Session_ID, S_P); end if;
               end;
            end if;
            declare
               function Batch_Get_One (T : System.Address; N : int) return Llama_Batch;
               pragma Import (C, Batch_Get_One, "llama_batch_get_one");
               B : constant Llama_Batch := Batch_Get_One (Tok'Address, 1);
            begin
               if Llama_Decode (Models (Kind).Context, B) /= 0 then exit; end if;
            end;
         end;
      end loop;
      Llama_Sampler_Free (Samp);
      Models (Kind).In_Use := False;
      Model_Gate.Release (Kind);
   end Generate;

   procedure Get_Embedding (Prompt : String; Result : out Math_Utils.Vector; Length : out Natural) is
      S : Boolean;
   begin
      Length := 0;
      Model_Gate.Acquire_ELP1 (Qwen_Embedding);
      Load_Model (Qwen_Embedding, S);
      if S then
         declare
            V : Llama_Vocab := Llama_Model_Get_Vocab (Models (Qwen_Embedding).Model);
            T : array (1 .. 32768) of Llama_Token;
            P_C : chars_ptr := New_String (Prompt);
            N : int := Llama_Tokenize (V, P_C, int (Prompt'Length), T (1)'Address, 32768, True, True);
         begin
            Free (P_C);
            if N > 0 then
               declare
                  function Batch_Get_One (T : System.Address; N : int) return Llama_Batch;
                  pragma Import (C, Batch_Get_One, "llama_batch_get_one");
                  B : constant Llama_Batch := Batch_Get_One (T (1)'Address, N);
               begin
                  Llama_Set_Embeddings (Models (Qwen_Embedding).Context, True);
                  if Llama_Decode (Models (Qwen_Embedding).Context, B) = 0 then
                     declare
                        function Model_N_Embd (M : Llama_Model) return int;
                        pragma Import (C, Model_N_Embd, "llama_model_n_embd");
                        D : constant int := Model_N_Embd (Models (Qwen_Embedding).Model);
                        Ptr : constant System.Address := Llama_Get_Embeddings (Models (Qwen_Embedding).Context);
                        type F_Arr is array (1 .. Integer (D)) of Float;
                        pragma Convention (C, F_Arr);
                        Embed : F_Arr; for Embed'Address use Ptr;
                     begin
                        Length := Natural (D);
                        if Length <= Result'Length then
                           for I in 1 .. Length loop Result (Result'First + I - 1) := Embed (I); end loop;
                        end if;
                     end;
                  end if;
               end;
            end if;
         end;
      end if;
      Model_Gate.Release (Qwen_Embedding);
   end Get_Embedding;

   function Should_Abort_ELP0 return Boolean is (Model_Gate.Should_Abort_ELP0);
   function Get_Kind_For_Model_Name (N : String) return Model_Type is (if Index (N, "4b") > 0 then Qwen_4B else Qwen_0_8B);
   function Is_Loaded (K : Model_Type) return Boolean is (Models (K).Loaded);
   function Count_Tokens (T : String) return Positive is (Positive'Max (1, T'Length / 4));
   function Get_Request_Category (M : String; S : String := ""; L : ELP_Level := ELP1) return String is (if Index (Ada.Characters.Handling.To_Lower (M), "hello") > 0 then "casual" else "technical");
   function Grade_Response_Quality (RT, P, S : String; SU, HC : Boolean; L : ELP_Level) return Natural is (85);

   function Generator_Callback (P : String) return String is
      R : Unbounded_String;
   begin
      Generate (Qwen_4B, P, R, GNATCOLL.JSON.Empty_Array, "", 4096, null, False, ELP1);
      return To_String (R);
   end Generator_Callback;

end Model_Manager;
