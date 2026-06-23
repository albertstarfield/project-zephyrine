pragma SPARK_Mode (Off);
--  ============================================================================
--  Speculative Decoding for Adelaide
--  ============================================================================
--  WHY THIS EXISTS:
--  Speculative decoding accelerates LLM inference by using a smaller, faster
--  "draft" model to generate candidate tokens, then verifying them in parallel
--  with the larger "target" model. This provides:
--    - 2-3x speedup for text generation (more tokens per second)
--    - Same output quality as the target model (verification step)
--    - Better GPU utilization (parallel draft + verify)
--
--  HOW IT WORKS:
--    1. Draft Model (Qwen3.5-0.8B) generates N candidate tokens quickly
--    2. Target Model (Qwen3-4B or 9B) verifies all N tokens in parallel
--    3. If all N match: accept all, continue with next N candidates
--    4. If some mismatch: accept prefix, resample from target distribution
--    5. Repeat until generation complete
--
--  DRAFT MODEL:
--  - Qwen3.5-0.8B (not 0.5B from oMLX)
--  - Faster inference, lower quality, used only for candidates
--  - Must be compatible with target model's tokenizer
--
--  INTEGRATION WITH ADELAIDE:
--  - model_manager.adb: Generate_Speculative procedure
--  - llama_interface.ads: FFI bindings for llama.cpp speculative APIs
--  - speculative_cache.ads: Semantic cache for prefix matching
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
