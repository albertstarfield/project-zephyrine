with Llama_Interface; use Llama_Interface;
with Ada.Calendar;
with Interfaces.C; use Interfaces.C;
with Interfaces.C.Strings; use Interfaces.C.Strings;
with Ada.Exceptions; use Ada.Exceptions;

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
   Success  : Boolean;
   Vocab    : Llama_Vocab;
   Tokens   : array (1 .. 32768) of Llama_Token;
   N_Toks   : int;
   Sampler  : Llama_Sampler;
   S_Params : Llama_Sampler_Chain_Params;
   
   Clean_P  : constant String := Sanitize_UTF8 (Prompt);
   Prompt_C : chars_ptr := New_String (Clean_P);
   Parser   : Stream_Parser_State;
   T0, T1   : Ada.Calendar.Time;
begin
   T0 := Ada.Calendar.Clock;
   pragma Unreferenced (Images);
   Result := Null_Unbounded_String;
   Parser.Orch_Think_Open := Orch_Think_Open;

   -- We acquire target only for now as a fallback implementation
   -- Proper speculative requires dual acquisition
   Priority_Model_Gate.Request_ELP1;
   Priority_Model_Gate.Acquire_ELP1 (Target_Kind);

   Load_Model (Target_Kind, Success, Requested_Ctx);
   if not Success then
      Priority_Model_Gate.Release_ELP1 (Target_Kind);
      Result := To_Unbounded_String ("ERROR: Load failed"); Ada.Text_IO.Put_Line("ERROR: Load failed");
      Free (Prompt_C);
      return;
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
      Priority_Model_Gate.Release_ELP1 (Target_Kind);
      Result := To_Unbounded_String ("ERROR: Tokenization failed"); Ada.Text_IO.Put_Line("ERROR: Tokenization failed");
      return;
   end if;

   Llama_Interface.Llama_Memory_Clear
     (Llama_Interface.Llama_Get_Memory (Models (Target_Kind).Context), False);

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
            B : constant Llama_Batch :=
              Llama_Batch_Get_One
                (Tokens (Integer (Current_Pos) + 1)'Address, To_Decode);
         begin
            declare
               Ret : constant int := Llama_Decode (Models (Target_Kind).Context, B);
            begin
               if Ret /= 0 then
                  Models (Target_Kind).In_Use := False;
                  Priority_Model_Gate.Release_ELP1 (Target_Kind);
                  Ada.Text_IO.Put_Line("ERROR: Decode failed"); Result := To_Unbounded_String ("ERROR: Decode failed (" & Ret'Img & ")");
                  return;
               end if;
            end;
            Tokens_Left := Tokens_Left - To_Decode;
            Current_Pos := Current_Pos + To_Decode;
         end;
      end loop;
   end;

   S_Params := Llama_Sampler_Chain_Default_Params;
   Sampler := Llama_Sampler_Chain_Init (S_Params);
   Llama_Sampler_Chain_Add
     (Sampler, Llama_Sampler_Init_Penalties (64, 1.1, 0.1, 0.1));
   Llama_Sampler_Chain_Add (Sampler, Llama_Sampler_Init_Top_K (40));
   Llama_Sampler_Chain_Add (Sampler, Llama_Sampler_Init_Top_P (0.9, 1));
   Llama_Sampler_Chain_Add (Sampler, Llama_Sampler_Init_Temp (0.7));
   Llama_Sampler_Chain_Add (Sampler, Llama_Sampler_Init_Dist (1234));

   Parser.Orch_Think_Open := Orch_Think_Open;

   for I in 1 .. 2048 loop
      declare
         -- TODO: Insert draft model speculative loop here
         Token : constant Llama_Token :=
           Llama_Sampler_Sample (Sampler, Models (Target_Kind).Context, -1);
         Piece : array (1 .. 256) of aliased Character;
         Len   : int;
      begin
         if Llama_Vocab_Is_Eog (Vocab, Token) then
            exit;
         end if;
         Len := Llama_Token_To_Piece
           (Vocab, Token, Piece (1)'Address, 256, 0, True);
         if Len > 0 then
            declare
               Str_Piece : String (1 .. Integer (Len));
            begin
               for J in 1 .. Integer (Len) loop
                  Str_Piece (J) := Piece (J);
                  Append (Result, Piece (J));
               end loop;

               if Stream /= null then
                  -- Artificial streaming delays to prevent Javascript client overload
                  Process_And_Push_Chunk
                    (Stream, Session_ID, Parser, Str_Piece);
               end if;
            end;
         end if;

         declare
            B : constant Llama_Batch :=
              Llama_Batch_Get_One (Token'Address, 1);
            Ret : constant int := Llama_Decode (Models (Target_Kind).Context, B);
         begin
            if Ret /= 0 then
               Result := To_Unbounded_String (To_String (Result) & " [ABORTED]");
               exit;
            end if;
         end;
      end;
   end loop;

   if Stream /= null then
      Flush_Parser (Stream, Session_ID, Parser);
   end if;

   Llama_Sampler_Free (Sampler);
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
      Priority_Model_Gate.Release_ELP1 (Target_Kind);
      Result := To_Unbounded_String ("ERROR: Speculative Decode failed");
end Generate_Speculative;
