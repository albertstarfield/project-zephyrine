pragma SPARK_Mode (Off);
--  ============================================================================
--  SPECULATIVE DECODING — DEAD CODE
--  ============================================================================
--  [DEAD-CODE] Draft-model speculative decoding is DISABLED.
--
--  WHY DISABLED:
--  Draft-model speculative decoding (Qwen3.5-0.8B) causes output quality
--  downgrade and buffer corruption. The ggml flash attention kernel crashes
--  with SIGABRT (corrupt output buffer j=0xFFFFFFFF, n_outputs=0) during
--  the verify phase. The draft model's KV cache interferes with the target
--  model's Metal kernel state, producing corrupted token buffers.
--
--  REPLACEMENT — THREE FASTER, CRASH-FREE SYSTEMS:
--  1. SPECULATION CONTEXT (ELP0): LSH-based embedding similarity lookup
--     injects <SpeculationContextGuidance_Interaction> and
--     <SpeculationContextGuidance_Literature> into the system prompt.
--     Runs on ELP0 (embedding model), no draft model needed.
--  2. RESPONSE CACHE: Fuzzy string matching cache (O(1) hash lookup).
--     Normalizes prompts (lowercase, collapse whitespace) for matching.
--     Stores model responses after first inference.
--  3. PROACTIVE ENGINE: Handless Mode — assistant initiates conversations,
--     asks questions, shares observations proactively. Curiosity Engine
--     generates questions from accumulated knowledge.
--
--  These systems are faster (no draft model overhead), crash-free, and
--  provide better output quality than draft-model speculative decoding.
--
--  This file is retained for reference only. All code within is dead.
--  ============================================================================

with Interfaces.C;
with System;
with Llama_Interface;
with Model_Types; use Model_Types;

package Speculative_Decode is

   --  Maximum tokens to draft before verification
   --  WHY 5: Balances speedup with verification overhead.
   --  Drafting too many tokens wastes compute if early ones mismatch.
   Max_Draft_Tokens : constant := 5;

   --  Draft model type (must be loaded before speculative decoding)
   Draft_Model_Type : constant Model_Type := Snowball_Enaga_ShortNetworkAnswer;

   --  ============================================================================
   --  SPECULATIVE DECODING API
   --  ============================================================================

   --  Generate tokens using speculative decoding
   --  WHY: Accelerates generation by using draft model for candidates.
   --  PARAMS:
   --    Prompt: input text to generate from
   --    Max_Tokens: maximum tokens to generate
   --    Target_Context: main model context (Qwen3-4B/9B)
   --    Draft_Context: draft model context (Qwen3.5-0.8B)
   --    Release_Target: if True, release target model after generation
   --    Release_Draft: if True, release draft model after generation
   --  RETURNS: generated text as String
   --  ALGORITHM:
   --    1. Tokenize prompt
   --    2. Loop until max_tokens or end-of-sequence:
   --       a. Draft: Generate N tokens with draft model
   --       b. Verify: Process all N tokens with target model
   --       c. Accept: Keep matching prefix, resample rest
   --    3. Detokenize and return
   function Generate_Speculative
     (Prompt          : String;
      Max_Tokens      : Positive;
      Target_Context  : Llama_Interface.Llama_Context;
      Draft_Context   : Llama_Interface.Llama_Context;
      Release_Target  : Boolean := False;
      Release_Draft   : Boolean := False) return String;

   --  ============================================================================
   --  DRAFT MODEL MANAGEMENT
   --  ============================================================================

   --  Initialize draft model for speculative decoding
   --  WHY: Loads Qwen3.5-0.8B into memory for fast candidate generation.
   --  NOTE: Must be called before Generate_Speculative.
   procedure Init_Draft_Model;

   --  Release draft model from memory
   --  WHY: Frees GPU memory when not doing speculative decoding.
   procedure Release_Draft_Model;

   --  Check if draft model is loaded
   function Is_Draft_Model_Loaded return Boolean;

   --  Get the draft context
   function Get_Draft_Context return Llama_Interface.Llama_Context;

   --  ============================================================================
   --  VERIFICATION
   --  ============================================================================

   --  Verify draft tokens against target model
   --  WHY: Ensures output quality matches target model.
   --  PARAMS:
   --    Draft_Tokens: candidate tokens from draft model
   --    N_Draft: number of draft tokens
   --    Target_Context: main model context
   --  RETURNS: number of accepted tokens (0 = all rejected)
   function Verify_Draft_Tokens
     (Draft_Tokens    : System.Address;
      N_Draft         : Interfaces.C.size_t;
      Target_Context  : Llama_Interface.Llama_Context)
      return Interfaces.C.size_t;

end Speculative_Decode;
