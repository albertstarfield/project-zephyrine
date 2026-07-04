pragma SPARK_Mode (Off);
--  ============================================================================
--  RERANKER — Native llama.cpp reranking via Qwen3-Reranker-0.6B
--  ============================================================================

with Ada.Text_IO; use Ada.Text_IO;
with Ada.Exceptions;
with Ada.Real_Time; use Ada.Real_Time;
with Interfaces.C; use Interfaces.C;
with Interfaces.C.Strings;
with System; use System;
with System.Storage_Elements;
with Ada.Unchecked_Conversion;
with AnsiAda;

package body Reranker is

   --  Reranker model path
   Model_Path : constant String :=
     "model/Qwen3-Reranker-0.6B-Q8_0.gguf";

   --  Internal state
   Reranker_Model    : Llama_Interface.Llama_Model := Llama_Interface.Null_Model;
   Reranker_Context  : Llama_Interface.Llama_Context := Llama_Interface.Null_Context;
   Ready             : Boolean := False;
   N_Cls_Out         : Natural := 0;
   Ctx_Size          : constant unsigned := 512;

   --  Constants
   GGML_TYPE_Q4_0 : constant int := 2;

   function Is_Ready return Boolean is
   begin
      return Ready;
   end Is_Ready;

   procedure Initialize (Success : out Boolean) is
      use type Llama_Interface.Llama_Model;
      use type Llama_Interface.Llama_Context;

      M_Params   : Llama_Interface.Llama_Model_Params;
      C_Params   : Llama_Interface.Llama_Context_Params;
      Path_C     : Interfaces.C.Strings.chars_ptr;
      Init_Start : Time;
   begin
      if Ready then
         Success := True;
         return;
      end if;

      Put_Line ("[Reranker] Initializing Qwen3-Reranker-0.6B...");
      Init_Start := Clock;

      --  Load model
      M_Params := Llama_Interface.Llama_Model_Default_Params;
      M_Params.N_Gpu_Layers := -1;  -- All GPU (only ~609MB, fits easily)
      Path_C := Interfaces.C.Strings.New_String (Model_Path);
      Reranker_Model := Llama_Interface.Llama_Model_Load_From_File (Path_C, M_Params);
      Interfaces.C.Strings.Free (Path_C);

      if Reranker_Model = Llama_Interface.Null_Model then
         Put_Line (AnsiAda.Background (AnsiAda.Red)
            & "[BUGCHECK] [Reranker] FAILED to load model from " & Model_Path
            & AnsiAda.Reset);
         Success := False;
         return;
      end if;

      Put_Line ("[Reranker] Model loaded.");

      --  Create context with RANK pooling type
      C_Params := Llama_Interface.Llama_Context_Default_Params;
      C_Params.N_Ctx := Ctx_Size;
      C_Params.N_Batch := Ctx_Size;
      C_Params.N_Ubatch := Ctx_Size;
      C_Params.N_Threads := 4;
      C_Params.N_Threads_Batch := 4;
      C_Params.Type_K := GGML_TYPE_Q4_0;
      C_Params.Type_V := GGML_TYPE_Q4_0;
      C_Params.Flash_Attn_Type := 1;
      C_Params.Pooling_Type := 4;  -- LLAMA_POOLING_TYPE_RANK
      C_Params.Attention_Type := 1;  -- NON-CAUSAL

      Reranker_Context := Llama_Interface.Llama_Init_From_Model (Reranker_Model, C_Params);

      if Reranker_Context = Llama_Interface.Null_Context then
         Put_Line (AnsiAda.Background (AnsiAda.Red)
            & "[BUGCHECK] [Reranker] FAILED to create context"
            & AnsiAda.Reset);
         Llama_Interface.Llama_Model_Free (Reranker_Model);
         Reranker_Model := Llama_Interface.Null_Model;
         Success := False;
         return;
      end if;

      --  Get number of classifier outputs
      N_Cls_Out := Natural (Llama_Interface.Llama_Model_N_Cls_Out (Reranker_Model));
      if N_Cls_Out = 0 then
         N_Cls_Out := 1;
      end if;

      Ready := True;
      Success := True;

      declare
         Duration_ms : constant Natural :=
           Natural (To_Duration (Clock - Init_Start) * 1000.0);
      begin
         Put_Line ("[Reranker] Initialized in" & Natural'Image (Duration_ms) & "ms"
                   & " (cls_out=" & Natural'Image (N_Cls_Out) & ")");
      end;
   exception
      when E : others =>
         Put_Line (AnsiAda.Background (AnsiAda.Red)
            & "[BUGCHECK] [Reranker] Exception during init: "
            & Ada.Exceptions.Exception_Message (E)
            & AnsiAda.Reset);
         Ready := False;
         Success := False;
   end Initialize;

   procedure Free_Reranker is
      use type Llama_Interface.Llama_Context;
      use type Llama_Interface.Llama_Model;
   begin
      if Reranker_Context /= Llama_Interface.Null_Context then
         Llama_Interface.Llama_Free (Reranker_Context);
         Reranker_Context := Llama_Interface.Null_Context;
      end if;
      if Reranker_Model /= Llama_Interface.Null_Model then
         Llama_Interface.Llama_Model_Free (Reranker_Model);
         Reranker_Model := Llama_Interface.Null_Model;
      end if;
      Ready := False;
      Put_Line ("[Reranker] Freed.");
   end Free_Reranker;

   --  Score a single (query, document) pair through the reranker.
   --
   --  RETRY LOGIC:
   --    1. Model not loaded → attempt re-initialization once
   --    2. Decode fail → flush KV cache, delay 10ms, retry once
   --    3. OOM (Storage_Error) → halve token count, retry once
   --
   --  Why retries matter:
   --    The reranker scores 10 entrySlices sequentially. Without retry, a single
   --    transient failure (Metal warm-up, OOM spike) would give that entrySlice
   --    score -1.0e9 and potentially lose the best result.
   --  ========================================================================
   function Score_Pair (Query : String; EntrySlice : String) return Float is
      Pair_Text : constant String := Query & Character'Val (9) & EntrySlice;

      type Token_Array is array (Natural range <>) of Llama_Interface.Llama_Token;
      Tokens : Token_Array (0 .. 511);
      N_Toks : int;

      Pair_C   : Interfaces.C.Strings.chars_ptr;
      Pair_Len : int;

      Batch   : Llama_Interface.Llama_Batch;
      Ret     : int;
      Emb_Ptr : System.Address;

      --  Read float from raw C address (C wrapper in llama_safe.cpp)
      function Read_Float_At (Addr : System.Address) return Interfaces.C.C_float;
      pragma Import (C, Read_Float_At, "read_float_at_address");

      --  ====================================================================
      --  RETRY 1: Model not loaded → re-initialize
      --  ====================================================================
      --  If the reranker model was freed (e.g., by FreeParallelMemory or
      --  OOM cleanup), attempt to reload it. Only try once to avoid infinite
      --  retry loops if the model file is missing or corrupt.
      --  ====================================================================
      procedure Try_Reinitialize is
         Init_Success : Boolean;
      begin
         Put_Line ("[Reranker] Model not loaded, attempting re-initialization...");
         Initialize (Init_Success);
         if not Init_Success then
            Put_Line ("[Reranker] Re-initialization FAILED.");
         end if;
      end Try_Reinitialize;

   begin
      --  ====================================================================
      --  RETRY 1 CHECK: If model not loaded, try to re-initialize
      --  ====================================================================
      if not Ready then
         Try_Reinitialize;
         if not Ready then
            return -1.0e9;  -- Re-init failed, can't score
         end if;
      end if;

      Pair_C := Interfaces.C.Strings.New_String (Pair_Text);
      Pair_Len := int (Pair_Text'Length);

      --  Tokenize
      N_Toks := Llama_Interface.Llama_Tokenize
        (Vocab          => Llama_Interface.Llama_Model_Get_Vocab (Reranker_Model),
         Text           => Pair_C,
         Text_Len       => Pair_Len,
         Tokens         => Tokens (0)'Address,
         N_Tokens_Max   => int (Tokens'Length),
         Add_Special    => True,
         Parse_Special  => True);

      Interfaces.C.Strings.Free (Pair_C);

      if N_Toks <= 0 then
         return -1.0e9;
      end if;

      --  Build batch
      Batch.N_Tokens := N_Toks;
      Batch.Token    := Tokens (0)'Address;
      Batch.Embd     := System.Null_Address;
      Batch.Pos      := System.Null_Address;
      Batch.N_Seq_Id := System.Null_Address;
      Batch.Seq_Id   := System.Null_Address;
      Batch.Logits   := System.Null_Address;

      --  Clear KV cache for fresh scoring
      Llama_Interface.Llama_Memory_Clear (Llama_Interface.Llama_Get_Memory (Reranker_Context), True);

      --  Decode
      Ret := Llama_Interface.Llama_Decode (Reranker_Context, Batch);

      --  ====================================================================
      --  RETRY 2: Decode fail → flush KV + delay + retry once
      --  ====================================================================
      --  Why this happens: Metal command buffer timeout, transient GPU state
      --  corruption after previous OOM, or KV cache stale from prior scoring.
      --  The flush + delay gives the GPU driver time to reset.
      --  ====================================================================
      if Ret /= 0 then
         Put_Line ("[Reranker] Decode failed (ret=" & int'Image (Ret)
                   & "), flushing KV and retrying...");
         Llama_Interface.Llama_Memory_Clear (Llama_Interface.Llama_Get_Memory (Reranker_Context), True);
         delay 0.01;  --  10ms for GPU driver to settle

         Ret := Llama_Interface.Llama_Decode (Reranker_Context, Batch);
         if Ret /= 0 then
            Put_Line ("[Reranker] Retry also failed (ret=" & int'Image (Ret) & ").");
            return -1.0e9;
         end if;
      end if;

      --  Get embeddings (reranking scores via POOLING_TYPE_RANK)
      Emb_Ptr := Llama_Interface.Llama_Get_Embeddings_Seq (Reranker_Context, 0);

      if Emb_Ptr = System.Null_Address then
         return -1.0e9;
      end if;

      return Float (Read_Float_At (Emb_Ptr));

   --  ========================================================================
   --  RETRY 3: OOM (Storage_Error) → halve tokens, retry once
   --  ========================================================================
   --  Why this happens: Reranker context is only 512 tokens. On systems with
   --  very tight memory (e.g., 16GB with 98% used), even 512 tokens can OOM
   --  if the GPU layer allocation competes with the main model's KV cache.
   --  Halving to 256 tokens reduces memory pressure while still allowing
   --  meaningful scoring (most reranker pairs are well under 256 tokens).
   --  ========================================================================
   exception
      when Storage_Error =>
         Put_Line ("[Reranker] OOM during scoring, halving tokens and retrying...");
         delay 0.01;  --  10ms for memory to settle

         --  Re-tokenize with halved token limit
         Pair_C := Interfaces.C.Strings.New_String (Pair_Text);
         Pair_Len := int (Pair_Text'Length);

         N_Toks := Llama_Interface.Llama_Tokenize
           (Vocab          => Llama_Interface.Llama_Model_Get_Vocab (Reranker_Model),
            Text           => Pair_C,
            Text_Len       => Pair_Len,
            Tokens         => Tokens (0)'Address,
            N_Tokens_Max   => int (Natural'Min (Natural (N_Toks), 256)),
            Add_Special    => True,
            Parse_Special  => True);

         Interfaces.C.Strings.Free (Pair_C);

         if N_Toks <= 0 then
            return -1.0e9;
         end if;

         --  Rebuild batch with fewer tokens
         Batch.N_Tokens := N_Toks;
         Batch.Token    := Tokens (0)'Address;
         Batch.Embd     := System.Null_Address;
         Batch.Pos      := System.Null_Address;
         Batch.N_Seq_Id := System.Null_Address;
         Batch.Seq_Id   := System.Null_Address;
         Batch.Logits   := System.Null_Address;

         --  Flush KV and retry decode
         Llama_Interface.Llama_Memory_Clear (Llama_Interface.Llama_Get_Memory (Reranker_Context), True);
         Ret := Llama_Interface.Llama_Decode (Reranker_Context, Batch);

         if Ret /= 0 then
            Put_Line ("[Reranker] OOM retry also failed (ret=" & int'Image (Ret) & ").");
            return -1.0e9;
         end if;

         --  Get embeddings from reduced decode
         Emb_Ptr := Llama_Interface.Llama_Get_Embeddings_Seq (Reranker_Context, 0);

         if Emb_Ptr = System.Null_Address then
            return -1.0e9;
         end if;

         Put_Line ("[Reranker] OOM retry succeeded with halved tokens.");
         return Float (Read_Float_At (Emb_Ptr));

      when others =>
         return -1.0e9;
   end Score_Pair;

   procedure Rerank_Scores
     (Query         : String;
       EntrySlice_Contents  : access function (Idx : Natural) return String;
       N_EntrySlices        : Natural;
      Top_K         : Natural;
      Best_Idx      : out Natural;
      Best_Score    : out Float)
   is
      pragma Unreferenced (Top_K);
   begin
      Best_Idx := 1;
      Best_Score := -1.0e9;

      if not Ready or N_EntrySlices = 0 then
         return;
      end if;

      --  Score each document against the query (1-indexed to match Chunk_Array)
      for I in 1 .. N_EntrySlices loop
         declare
            Score : constant Float := Score_Pair (Query, EntrySlice_Contents (I));
         begin
            if Score > Best_Score then
               Best_Score := Score;
               Best_Idx := I;  -- Already 1-indexed
            end if;
         end;
      end loop;

      Put_Line ("[Reranker] Scored" & Natural'Image (N_EntrySlices) & " entrySlices."
                & " Best: #" & Natural'Image (Best_Idx)
                & " score=" & Float'Image (Best_Score));
   end Rerank_Scores;

   procedure Rerank
     (Query     : String;
      Documents : Rerank_Result_Array;
      Top_K     : Natural;
      Results   : out Unbounded_String;
      Count     : out Natural)
   is
      pragma Unreferenced (Query, Documents, Top_K);
   begin
      Results := Null_Unbounded_String;
      Count := 0;
   end Rerank;

end Reranker;
