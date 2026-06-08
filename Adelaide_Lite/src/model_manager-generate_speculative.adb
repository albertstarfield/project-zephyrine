with Llama_Interface; use Llama_Interface;
with Ada.Calendar;
with Interfaces.C; use Interfaces.C;
with Interfaces.C.Strings; use Interfaces.C.Strings;
with Ada.Exceptions; use Ada.Exceptions;
with Kratos;

separate (Model_Manager)
procedure Generate_Speculative
  (Target_Kind     : Model_Type;
   Draft_Kind      : Model_Type;
   Prompt          : String;
   Result          : out Unbounded_String;
   Images          : GNATCOLL.JSON.JSON_Array := GNATCOLL.JSON.Empty_Array;
   Session_ID      : String := "";
   Requested_Ctx   : Positive := 4096;
   Stream          : Streaming_Queue.Queue_Access := null;
   Orch_Think_Open : Boolean := False;
   Level           : ELP_Level := ELP1)
is
   pragma SPARK_Mode (Off);
   Success       : Boolean;
   Vocab         : Llama_Vocab;
   Draft_Vocab   : Llama_Vocab;
   Tokens        : array (1 .. 32768) of Llama_Token;
   N_Toks        : int;
   Target_Sampler : Llama_Sampler;
   Draft_Sampler  : Llama_Sampler;
   S_Params       : Llama_Sampler_Chain_Params;

   Clean_P  : constant String := Sanitize_UTF8 (Prompt);
   Prompt_C : chars_ptr := New_String (Clean_P);
   Parser   : Stream_Parser_State;
   T0, T1   : Ada.Calendar.Time;

   --  Speculative decoding parameters
   Draft_Batch_K : constant Integer := 4;  -- Number of draft tokens to propose per step
   Max_Tokens    : Integer := 2048;  -- Computed as Actual_Ctx / 2 after Load_Model

   --  Draft token buffer for speculative verification
   type Draft_Token_Array is array (1 .. Draft_Batch_K) of Llama_Token;
   Draft_Tokens  : Draft_Token_Array;
   N_Draft       : Integer;

   procedure Emit_Token (Tok : Llama_Token; V : Llama_Vocab) is
      Piece : array (1 .. 256) of aliased Character;
      Len   : int;
   begin
      if Llama_Vocab_Is_Eog (V, Tok) then
         return;
      end if;
      Len := Llama_Token_To_Piece (V, Tok, Piece (1)'Address, 256, 0, True);
      if Len > 0 then
         declare
            Str_Piece : String (1 .. Integer (Len));
         begin
            for J in 1 .. Integer (Len) loop
               Str_Piece (J) := Piece (J);
               Append (Result, Piece (J));
            end loop;
            if Stream /= null then
               Process_And_Push_Chunk (Stream, Session_ID, Parser, Str_Piece);
            end if;
         end;
      end if;
   end Emit_Token;

   procedure Decode_Context (Ctx : Llama_Context; Tok : Llama_Token) is
      B   : constant Llama_Batch := Llama_Batch_Get_One (Tok'Address, 1);
      Ret : int;
   begin
      if Kratos.Guard_Enter = 0 then
         Ret := Llama_Decode (Ctx, B);
         Kratos.Guard_Exit;
      else
         Kratos.Log_Crash;
         Ret := -1;
      end if;
      if Ret /= 0 then
         raise Program_Error with "Decode failed (" & Ret'Img & ")";
      end if;
   end Decode_Context;

   procedure Batch_Decode (Ctx : Llama_Context; Tks : Draft_Token_Array;
                           Count : Integer; Start_Pos : int) is
      B   : Llama_Batch;
      Ret : int;
   begin
      B := Llama_Batch_Init (int (Count), 0, 1);
      for I in 1 .. Count loop
         Llama_Batch_Add_Safe (B'Address, Tks (I), Start_Pos + int (I - 1), 0, I = Count);
      end loop;
      if Kratos.Guard_Enter = 0 then
         Ret := Llama_Decode (Ctx, B);
         Kratos.Guard_Exit;
      else
         Kratos.Log_Crash;
         Ret := -1;
      end if;
      Llama_Batch_Free (B);
      if Ret /= 0 then
         raise Program_Error with "Batch decode failed (" & Ret'Img & ")";
      end if;
   end Batch_Decode;

begin
   T0 := Ada.Calendar.Clock;
   pragma Unreferenced (Images);
   Result := Null_Unbounded_String;
   Parser.Orch_Think_Open := Orch_Think_Open;

   --  Acquire both models for speculative decoding
   Priority_Model_Gate.Request_ELP1;
   Priority_Model_Gate.Acquire_ELP1 (Target_Kind);

   --  Try to acquire draft model; fall back to single-model if unavailable
   Priority_Model_Gate.Request_ELP1;
   Priority_Model_Gate.Acquire_ELP1 (Draft_Kind);

   --  Load target model
   Load_Model (Target_Kind, Success, Requested_Ctx);
   if not Success then
      Priority_Model_Gate.Release_ELP1 (Draft_Kind);
      Priority_Model_Gate.Release_ELP1 (Target_Kind);
      Result := To_Unbounded_String ("ERROR: Target load failed");
      Ada.Text_IO.Put_Line ("ERROR: Target load failed");
      Free (Prompt_C);
      return;
   end if;

   --  Max tokens = half the actual context window
   Max_Tokens := Integer (Models (Target_Kind).Current_Ctx) / 2;
   Ada.Text_IO.Put_Line ("[Speculative] Max_Tokens:" & Max_Tokens'Img &
                         " (Ctx:" & Models (Target_Kind).Current_Ctx'Img & ")");

   --  Load draft model (smaller context is fine for draft)
   Load_Model (Draft_Kind, Success, 4096);
   if not Success then
      Ada.Text_IO.Put_Line ("[!] Draft model load failed, falling back to target-only generation");
      --  Continue without draft model (degraded to standard generation)
   end if;

   Models (Target_Kind).In_Use := True;
   Models (Target_Kind).Last_Used := Clock;

   Vocab := Llama_Model_Get_Vocab (Models (Target_Kind).Model);
   N_Toks := Llama_Tokenize
     (Vocab, Prompt_C, int (Clean_P'Length), Tokens (1)'Address,
      32768, True, True);
   Free (Prompt_C);

   if N_Toks < 0 then
      Models (Target_Kind).In_Use := False;
      if Models (Draft_Kind).Loaded then
         Models (Draft_Kind).In_Use := False;
         Priority_Model_Gate.Release_ELP1 (Draft_Kind);
      end if;
      Priority_Model_Gate.Release_ELP1 (Target_Kind);
      Result := To_Unbounded_String ("ERROR: Tokenization failed");
      Ada.Text_IO.Put_Line ("ERROR: Tokenization failed");
      return;
   end if;

   --  Clear both model memories
   Llama_Interface.Llama_Memory_Clear
     (Llama_Interface.Llama_Get_Memory (Models (Target_Kind).Context), False);
   if Models (Draft_Kind).Loaded then
      Draft_Vocab := Llama_Model_Get_Vocab (Models (Draft_Kind).Model);
      Llama_Interface.Llama_Memory_Clear
        (Llama_Interface.Llama_Get_Memory (Models (Draft_Kind).Context), False);
   end if;

   --  Prefill: feed prompt tokens into both models
   declare
      Batch_Size  : constant int := 512;
      Current_Pos : int := 0;
      Tokens_Left : int := N_Toks;
   begin
      while Tokens_Left > 0 loop
         declare
            To_Decode : constant int :=
              (if Tokens_Left > Batch_Size
               then Batch_Size
               else Tokens_Left);
         begin
            --  Decode on target
            declare
               B   : constant Llama_Batch :=
                 Llama_Batch_Get_One (Tokens (Integer (Current_Pos) + 1)'Address, To_Decode);
               Ret : int;
            begin
               if Kratos.Guard_Enter = 0 then
                  Ret := Llama_Decode (Models (Target_Kind).Context, B);
                  Kratos.Guard_Exit;
               else
                  Kratos.Log_Crash;
                  Ret := -1;
               end if;
               if Ret /= 0 then
                  Models (Target_Kind).In_Use := False;
                  if Models (Draft_Kind).Loaded then
                     Models (Draft_Kind).In_Use := False;
                     Priority_Model_Gate.Release_ELP1 (Draft_Kind);
                  end if;
                  Priority_Model_Gate.Release_ELP1 (Target_Kind);
                  Ada.Text_IO.Put_Line ("ERROR: Target decode failed");
                  Result := To_Unbounded_String ("ERROR: Target decode failed (" & Ret'Img & ")");
                  return;
               end if;
            end;
            --  Decode on draft
            if Models (Draft_Kind).Loaded then
               declare
                  B   : constant Llama_Batch :=
                    Llama_Batch_Get_One (Tokens (Integer (Current_Pos) + 1)'Address, To_Decode);
                  Ret : int;
               begin
                  if Kratos.Guard_Enter = 0 then
                     Ret := Llama_Decode (Models (Draft_Kind).Context, B);
                     Kratos.Guard_Exit;
                  else
                     Kratos.Log_Crash;
                     Ret := -1;
                  end if;
                  if Ret /= 0 then
                     Ada.Text_IO.Put_Line ("[!] Draft decode error during prefill, continuing target-only");
                  end if;
               end;
            end if;
            Tokens_Left := Tokens_Left - To_Decode;
            Current_Pos := Current_Pos + To_Decode;
         end;
      end loop;
   end;

   --  Verbose: prefill complete
   if Stream /= null then
      Push_Chunk (Stream, Session_ID,
        "[Speculative] Prefill complete. Tokens:" & N_Toks'Img &
        " Draft loaded:" & Boolean'Image (Models (Draft_Kind).Loaded) & ASCII.LF);
   end if;

   --  Create samplers
   S_Params := Llama_Sampler_Chain_Default_Params;
   Target_Sampler := Llama_Sampler_Chain_Init (S_Params);
   Llama_Sampler_Chain_Add (Target_Sampler, Llama_Sampler_Init_Penalties (64, 1.1, 0.1, 0.1));
   Llama_Sampler_Chain_Add (Target_Sampler, Llama_Sampler_Init_Top_K (40));
   Llama_Sampler_Chain_Add (Target_Sampler, Llama_Sampler_Init_Top_P (0.9, 1));
   Llama_Sampler_Chain_Add (Target_Sampler, Llama_Sampler_Init_Temp (0.7));
   Llama_Sampler_Chain_Add (Target_Sampler, Llama_Sampler_Init_Dist (1234));

   if Models (Draft_Kind).Loaded then
      Draft_Sampler := Llama_Sampler_Chain_Init (S_Params);
      Llama_Sampler_Chain_Add (Draft_Sampler, Llama_Sampler_Init_Penalties (128, 1.2, 0.5, 0.5));
      Llama_Sampler_Chain_Add (Draft_Sampler, Llama_Sampler_Init_Top_K (40));
      Llama_Sampler_Chain_Add (Draft_Sampler, Llama_Sampler_Init_Top_P (0.9, 1));
      Llama_Sampler_Chain_Add (Draft_Sampler, Llama_Sampler_Init_Temp (0.8));
      Llama_Sampler_Chain_Add (Draft_Sampler, Llama_Sampler_Init_Dist (5678));
   end if;

   Parser.Orch_Think_Open := Orch_Think_Open;

   --  Verbose: push status into thinking block
   if Stream /= null then
      Push_Chunk (Stream, Session_ID,
        "[Speculative] Models loaded. Target:" & Target_Kind'Img &
        " Draft:" & Draft_Kind'Img &
        " Ctx:" & Models (Target_Kind).Current_Ctx'Img &
        " MaxTok:" & Max_Tokens'Img & ASCII.LF);
   end if;

     --  STANDARD AUTOREGRESSIVE GENERATION (target model only)
    --  Draft model (0.8B) is too small to provide useful speculative tokens
    for I in 1 .. Max_Tokens loop
       declare
          Token : constant Llama_Token :=
            Llama_Sampler_Sample (Target_Sampler, Models (Target_Kind).Context, -1);
       begin
          if Llama_Vocab_Is_Eog (Vocab, Token) then
             exit;
          end if;
          Emit_Token (Token, Vocab);
          Decode_Context (Models (Target_Kind).Context, Token);
       end;
    end loop;

   --  Generation complete (server-side only)
   Ada.Text_IO.Put_Line ("[Speculative] Generation complete. Tokens generated:" & Max_Tokens'Img);

    if Stream /= null then
      Flush_Parser (Stream, Session_ID, Parser);
   end if;

   --  Cleanup
   Llama_Sampler_Free (Target_Sampler);
   if Models (Draft_Kind).Loaded then
      Llama_Sampler_Free (Draft_Sampler);
      Models (Draft_Kind).In_Use := False;
      Priority_Model_Gate.Release_ELP1 (Draft_Kind);
   end if;
   Models (Target_Kind).In_Use := False;
   Priority_Model_Gate.Release_ELP1 (Target_Kind);

   T1 := Ada.Calendar.Clock;
   declare
      Dur : constant Duration := T1 - T0;
   begin
      if Dur > Current_WCET then
         Current_WCET := Dur;
      end if;
      case Level is
         when ELP0 =>
            if Dur > Current_WCET_ELP0 then
               Current_WCET_ELP0 := Dur;
            end if;
         when ELP1 =>
            if Dur > Current_WCET_ELP1 then
               Current_WCET_ELP1 := Dur;
            end if;
         when ELP2 =>
            if Dur > Current_WCET_ELP2 then
               Current_WCET_ELP2 := Dur;
            end if;
         when ELP3 =>
            if Dur > Current_WCET_ELP3 then
               Current_WCET_ELP3 := Dur;
            end if;
      end case;
   end;
exception
   when E : others =>
      Ada.Text_IO.Put_Line ("Generate_Speculative Error: " &
        Ada.Exceptions.Exception_Information (E));
      Models (Target_Kind).In_Use := False;
      if Models (Draft_Kind).Loaded then
         Models (Draft_Kind).In_Use := False;
         Priority_Model_Gate.Release_ELP1 (Draft_Kind);
      end if;
      Priority_Model_Gate.Release_ELP1 (Target_Kind);
      Result := To_Unbounded_String ("ERROR: Speculative Decode failed");
end Generate_Speculative;
