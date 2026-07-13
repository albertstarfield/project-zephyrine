pragma SPARK_Mode (Off);
--  ============================================================================
--  RERANKER — Native llama.cpp reranking via Qwen3-Reranker-0.6B
--  ============================================================================
--  Uses LLAMA_POOLING_TYPE_RANK to get relevance scores from a reranking model.
--  Input: "query\tdocument" pairs
--  Output: float score (higher = more relevant)
--
--  WHY: LSH/QRNN hash search returns candidates by Hamming distance (recall).
--  Reranker scores them by actual semantic relevance (precision).
--  This is the key to high-quality memory injection for Context streaming
--  and Knowledge literature streaming.
--
--  FLOW:
--    1. Load reranker model (once, on first use via FreeParallelMemory)
--    2. For each candidate: tokenize "query\tdocument" -> decode -> get score
--    3. Sort by score descending -> top-K injection into context window
--    4. Unload reranker (FreeParallelMemory) before loading main model
--
--  MODEL: Qwen3-Reranker-0.6B-Q8_0.gguf (~609MB)
--  POOLING: LLAMA_POOLING_TYPE_RANK (4) — attaches classification head
--  ============================================================================

with Interfaces.C;
with Interfaces.C.Strings;
with System;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;

with Llama_Interface;

package Reranker is

   --  A single reranking result: entrySlice ID + relevance score
   type Rerank_Result is record
      EntrySlice_ID : Natural;
      Score         : Float;
   end record;

   type Rerank_Result_Array is array (Positive range <>) of Rerank_Result;

   --  Rerank a list of document texts against a query.
   --  Documents are passed as an unbounded string array (Ada-friendly).
   --  Returns results sorted by score (highest first).
   procedure Rerank
     (Query     : String;
      Documents : Rerank_Result_Array;
      Top_K     : Natural;
      Results   : out Unbounded_String;
      Count     : out Natural);

   --  Rerank entrySlices using pre-computed search results.
   --  Scores each (query, entrySlice) pair through the reranker model.
   --  Returns Top_K best entrySlice IDs (1-indexed).
   procedure Rerank_Scores
     (Query              : String;
      EntrySlice_Contents  : access function (Idx : Natural) return String;
      N_EntrySlices        : Natural;
      Top_K         : Natural;
      Best_Idx      : out Natural;
      Best_Score    : out Float);

   --  Initialize the reranker (load model, create context).
   --  Called on first use. Model stays loaded until Free_Reranker.
   procedure Initialize (Success : out Boolean);

   --  Free the reranker model (FreeParallelMemory).
   --  Must be called before loading the main chat model.
   procedure Free_Reranker;

   --  Check if reranker is loaded
   function Is_Ready return Boolean;

end Reranker;
