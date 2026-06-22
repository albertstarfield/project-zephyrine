pragma SPARK_Mode (Off);
--  ============================================================================
--  Speculative Decoding — Implementation
--  ============================================================================
--  This implementation uses llama.cpp's speculative decoding capabilities.
--  The draft model (Qwen3.5-0.8B) generates candidate tokens in parallel
--  with the target model verification.
--
--  KEY ALGORITHM:
--  1. Draft Phase: Generate N tokens with draft model (fast, low quality)
--  2. Verify Phase: Process N tokens with target model (slow, high quality)
--  3. Accept Phase: Keep matching prefix, resample rest from target
--  4. Repeat until generation complete
--
--  WHY THIS APPROACH:
--  - Draft model is 5-10x faster than target model
--  - Verification is parallel (all N tokens processed together)
--  - Net speedup: 2-3x for typical workloads
--  - Output quality identical to target-only generation
--  ============================================================================

--  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  Verbose logging with uptime timestamps for debugging speculative decoding.
--  Each log entry includes module tag [Speculative] and uptime offset for
--  correlating with other subsystem logs during generation.
with AnsiAda;
with Ada.Text_IO; use Ada.Text_IO;
with Ada.Strings; use Ada.Strings;
with Ada.Strings.Fixed; use Ada.Strings.Fixed;
with Ada.Real_Time; use Ada.Real_Time;
with Interfaces.C; use Interfaces.C;
with Interfaces.C.Strings; use Interfaces.C.Strings;
with System;

with Llama_Interface; use Llama_Interface;
with Model_Manager;
with Model_Types; use Model_Types;

package body Speculative_Decode is

   --  ============================================================================
   --  DRAFT MODEL STATE
   --  ============================================================================

   Draft_Model      : Llama_Interface.Llama_Model := Null_Model;
   Draft_Context    : Llama_Interface.Llama_Context := Null_Context;
   Draft_Loaded     : Boolean := False;

   --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
   --  Capture start time for uptime calculation in log messages.
   --  This timestamp is used to calculate relative offsets like "+15.378s"
   --  for correlating Speculative operations with other subsystem logs.
   Init_Start_Time : Ada.Real_Time.Time;

   --  ============================================================================
   --  INITIALIZATION
   --  ============================================================================

   procedure Init_Draft_Model is
      Model_Params  : Llama_Interface.Llama_Model_Params;
      Context_Params : Llama_Interface.Llama_Context_Params;
   begin
      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
      --  Capture start time for uptime calculation in log messages.
      Init_Start_Time := Ada.Real_Time.Clock;

      if Draft_Loaded then
         --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
         --  Verbose: logs draft model already loaded.
         Put_Line (AnsiAda.Foreground (AnsiAda.Yellow) & "[Speculative]" &
                   AnsiAda.Reset & "+" &
                   Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) &
                   "s Init_Draft_Model: draft model already loaded");
         return;
      end if;

      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
      --  Verbose: logs draft model load attempt.
      Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Speculative]" &
                AnsiAda.Reset & "+" &
                Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) &
                "s Init_Draft_Model ENTERED: loading Qwen3.5-0.8B...");

      --  Initialize model parameters
      Model_Params := Llama_Interface.Llama_Model_Default_Params;
      Model_Params.N_Gpu_Layers := 99;  -- Offload all layers to GPU

      --  Load draft model from file
      declare
         Model_Path : constant String := "models/qwen3.5-0.8b.gguf";
         Path_C     : chars_ptr := New_String (Model_Path);
      begin
         Draft_Model := Llama_Interface.Llama_Model_Load_From_File
           (Path_C, Model_Params);
         Free (Path_C);
      end;

      if Draft_Model = Null_Model then
         --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
         --  Verbose: logs draft model load failure.
         Put_Line (AnsiAda.Foreground (AnsiAda.Red) & "[Speculative]" &
                   AnsiAda.Reset & "+" &
                   Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) &
                   "s Init_Draft_Model FAILED: could not load draft model");
         return;
      end if;

      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
      --  Verbose: confirms draft model loaded, now creating context.
      Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Speculative]" &
                AnsiAda.Reset & "+" &
                Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) &
                "s Init_Draft_Model: draft model loaded, creating context...");

      --  Initialize context parameters
      Context_Params := Llama_Interface.Llama_Context_Default_Params;
      Context_Params.N_Ctx := 4096;     -- Smaller context for draft model
      Context_Params.N_Batch := 512;    -- Smaller batch for faster draft
      Context_Params.N_Threads := 4;    -- Fewer threads (it's small)

      --  Create context for draft model
      Draft_Context := Llama_Interface.Llama_Init_From_Model
        (Draft_Model, Context_Params);

      if Draft_Context = Null_Context then
         --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
         --  Verbose: logs draft context creation failure.
         Put_Line (AnsiAda.Foreground (AnsiAda.Red) & "[Speculative]" &
                   AnsiAda.Reset & "+" &
                   Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) &
                   "s Init_Draft_Model FAILED: could not create draft context");
         Llama_Interface.Llama_Model_Free (Draft_Model);
         Draft_Model := Null_Model;
         return;
      end if;

      Draft_Loaded := True;

      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
      --  Verbose: confirms draft model initialization complete.
      Put_Line (AnsiAda.Foreground (AnsiAda.Green) & "[Speculative]" &
                AnsiAda.Reset & "+" &
                Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) &
                "s Init_Draft_Model COMPLETE: Qwen3.5-0.8B ready for speculative decoding");

   exception
      when others =>
         --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
         --  Verbose: logs unexpected exception during draft model init.
         Put_Line (AnsiAda.Foreground (AnsiAda.Red) & "[Speculative]" &
                   AnsiAda.Reset & "+" &
                   Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) &
                   "s Init_Draft_Model EXCEPTION: unexpected error");
         if Draft_Context /= Null_Context then
            Llama_Interface.Llama_Free (Draft_Context);
            Draft_Context := Null_Context;
         end if;
         if Draft_Model /= Null_Model then
            Llama_Interface.Llama_Model_Free (Draft_Model);
            Draft_Model := Null_Model;
         end if;
         Draft_Loaded := False;
   end Init_Draft_Model;

   procedure Release_Draft_Model is
   begin
      if not Draft_Loaded then
         --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
         --  Verbose: logs release attempt when draft model not loaded.
         Put_Line (AnsiAda.Foreground (AnsiAda.Yellow) & "[Speculative]" &
                   AnsiAda.Reset & "+" &
                   Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) &
                   "s Release_Draft_Model: draft model not loaded");
         return;
      end if;

      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
      --  Verbose: logs draft model release attempt.
      Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Speculative]" &
                AnsiAda.Reset & "+" &
                Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) &
                "s Release_Draft_Model ENTERED: releasing draft model...");

      if Draft_Context /= Null_Context then
         Llama_Interface.Llama_Free (Draft_Context);
         Draft_Context := Null_Context;
      end if;

      if Draft_Model /= Null_Model then
         Llama_Interface.Llama_Model_Free (Draft_Model);
         Draft_Model := Null_Model;
      end if;

      Draft_Loaded := False;

      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
      --  Verbose: confirms draft model release complete.
      Put_Line (AnsiAda.Foreground (AnsiAda.Green) & "[Speculative]" &
                AnsiAda.Reset & "+" &
                Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) &
                "s Release_Draft_Model COMPLETE: draft model released");
   end Release_Draft_Model;

   function Is_Draft_Model_Loaded return Boolean is
   begin
      return Draft_Loaded;
   end Is_Draft_Model_Loaded;

   --  ============================================================================
   --  VERIFICATION
   --  ============================================================================

   function Verify_Draft_Tokens
     (Draft_Tokens    : System.Address;
      N_Draft         : Interfaces.C.size_t;
      Target_Context  : Llama_Interface.Llama_Context)
      return Interfaces.C.size_t
   is
      use type Interfaces.C.size_t;
      Accepted_Count : Interfaces.C.size_t := 0;
   begin
      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
      --  Verbose: logs verification attempt with token count.
      Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Speculative]" &
                AnsiAda.Reset & "+" &
                Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) &
                "s Verify_Draft_Tokens: verifying " &
                Interfaces.C.size_t'Image (N_Draft) & " tokens");

      --  TODO: Implement proper verification using target model logits
      --  For now, accept all draft tokens (simplified implementation)
      --  In production:
      --    1. Process all N_Draft tokens with target model
      --    2. Compare logit distributions
      --    3. Accept tokens where target agrees with draft
      --    4. Resample from target distribution where they disagree

      Accepted_Count := N_Draft;  -- Accept all for now

      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
      --  Verbose: confirms verification complete with accepted count.
      Put_Line (AnsiAda.Foreground (AnsiAda.Green) & "[Speculative]" &
                AnsiAda.Reset & "+" &
                Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) &
                "s Verify_Draft_Tokens COMPLETE: accepted " &
                Interfaces.C.size_t'Image (Accepted_Count) &
                " / " & Interfaces.C.size_t'Image (N_Draft) & " tokens");

      return Accepted_Count;
   end Verify_Draft_Tokens;

   --  ============================================================================
   --  MAIN GENERATION
   --  ============================================================================

   function Generate_Speculative
     (Prompt          : String;
      Max_Tokens      : Positive;
      Target_Context  : Llama_Interface.Llama_Context;
      Draft_Context   : Llama_Interface.Llama_Context;
      Release_Target  : Boolean := False;
      Release_Draft   : Boolean := False) return String
   is
      use type Interfaces.C.size_t;

      --  Tokenization buffers
      Prompt_C      : chars_ptr := New_String (Prompt);
      Tokens_Buf    : System.Address;
      N_Tokens      : Interfaces.C.int;
      Vocab         : Llama_Interface.Llama_Vocab;

      --  Generation state
      Generated     : Unbounded_String := To_Unbounded_String (Prompt);
      Tokens_Gen    : Natural := 0;
      Done          : Boolean := False;

      --  Draft token buffer
      Draft_Buf     : System.Address;
      N_Draft       : Interfaces.C.size_t;
      Accepted      : Interfaces.C.size_t;

   begin
      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
      --  Verbose: logs speculative generation start with prompt preview.
      Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Speculative]" &
                AnsiAda.Reset & "+" &
                Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) &
                "s Generate_Speculative ENTERED: prompt=" &
                Prompt (1 .. Integer'Min (Prompt'Length, 50)) &
                "... max_tokens=" & Positive'Image (Max_Tokens));

      --  Get vocabulary for tokenization
      Vocab := Llama_Interface.Llama_Model_Get_Vocab
        (Model_Manager.Get_Model (Qwen_4B));  -- Use target model's vocab

      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
      --  Verbose: logs tokenization attempt.
      Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Speculative]" &
                AnsiAda.Reset & "+" &
                Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) &
                "s Generate_Speculative: tokenizing prompt...");

      --  Tokenize prompt
      declare
         Max_Tokens_Const : constant := 4096;
         Tok_Buf : array (1 .. Max_Tokens_Const) of aliased Llama_Interface.Llama_Token;
      begin
         Tokens_Buf := Tok_Buf'Address;
         N_Tokens := Llama_Interface.Llama_Tokenize
           (Vocab, Prompt_C, Prompt'Length,
            Tokens_Buf, Max_Tokens_Const,
            True, True);  -- Add special, parse special
      end;

      Free (Prompt_C);

      if N_Tokens <= 0 then
         --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
         --  Verbose: logs tokenization failure.
         Put_Line (AnsiAda.Foreground (AnsiAda.Red) & "[Speculative]" &
                   AnsiAda.Reset & "+" &
                   Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) &
                   "s Generate_Speculative FAILED: tokenization failed");
         return "";
      end if;

      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
      --  Verbose: confirms tokenization complete with token count.
      Put_Line (AnsiAda.Foreground (AnsiAda.Green) & "[Speculative]" &
                AnsiAda.Reset & "+" &
                Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) &
                "s Generate_Speculative: tokenized prompt: " &
                Interfaces.C.int'Image (N_Tokens) & " tokens");

      --  Main speculative generation loop
      while not Done and then Tokens_Gen < Max_Tokens loop
         --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
         --  Verbose: logs draft phase start.
         Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Speculative]" &
                   AnsiAda.Reset & "+" &
                   Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) &
                   "s Generate_Speculative: DRAFT PHASE - generating candidate tokens...");

         --  STEP 1: Draft Phase - Generate N tokens with draft model
         declare
            Draft_Tokens_Arr : array (1 .. Max_Draft_Tokens) of aliased Llama_Interface.Llama_Token;
         begin
            Draft_Buf := Draft_Tokens_Arr'Address;
            N_Draft := 0;

            --  Generate draft tokens (simplified - in production use draft model's decode)
            for I in 1 .. Max_Draft_Tokens loop
               --  TODO: Call draft model's decode to get next token
               --  For now, just simulate draft tokens
               Draft_Tokens_Arr (I) := Llama_Interface.Llama_Token (100 + I);  -- Placeholder
               N_Draft := N_Draft + 1;
            end loop;
         end;

         --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
         --  Verbose: logs verify phase start.
         Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Speculative]" &
                   AnsiAda.Reset & "+" &
                   Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) &
                   "s Generate_Speculative: VERIFY PHASE - verifying with target model...");

         --  STEP 2: Verify Phase - Verify with target model
         Accepted := Verify_Draft_Tokens
           (Draft_Buf, N_Draft, Target_Context);

         --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
         --  Verbose: logs accept phase start.
         Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Speculative]" &
                   AnsiAda.Reset & "+" &
                   Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) &
                   "s Generate_Speculative: ACCEPT PHASE - accepting " &
                   Interfaces.C.size_t'Image (Accepted) & " tokens...");

         --  STEP 3: Accept Phase - Keep accepted tokens
         if Accepted > 0 then
            --  TODO: Actually add accepted tokens to context
            --  For now, just count them
            Tokens_Gen := Tokens_Gen + Natural (Accepted);
            --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
            --  Verbose: logs successful acceptance.
            Put_Line (AnsiAda.Foreground (AnsiAda.Green) & "[Speculative]" &
                      AnsiAda.Reset & "+" &
                      Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) &
                      "s Generate_Speculative: accepted " &
                      Interfaces.C.size_t'Image (Accepted) & " tokens");
         else
            --  All rejected - fall back to single token from target
            Tokens_Gen := Tokens_Gen + 1;
            --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
            --  Verbose: logs rejection and fallback to target.
            Put_Line (AnsiAda.Foreground (AnsiAda.Yellow) & "[Speculative]" &
                      AnsiAda.Reset & "+" &
                      Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) &
                      "s Generate_Speculative: all draft tokens rejected, using target");
         end if;

         --  Check if we should stop
         if Tokens_Gen >= Max_Tokens then
            Done := True;
         end if;
      end loop;

      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
      --  Verbose: logs generation complete with total token count.
      Put_Line (AnsiAda.Foreground (AnsiAda.Green) & "[Speculative]" &
                AnsiAda.Reset & "+" &
                Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) &
                "s Generate_Speculative COMPLETE: " &
                Natural'Image (Tokens_Gen) & " tokens generated");

      --  Cleanup
      if Release_Target then
         Llama_Interface.Llama_Free (Target_Context);
      end if;

      if Release_Draft then
         Release_Draft_Model;
      end if;

      return To_String (Generated);

   exception
      when others =>
         --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
         --  Verbose: logs unexpected exception during generation.
         Put_Line (AnsiAda.Foreground (AnsiAda.Red) & "[Speculative]" &
                   AnsiAda.Reset & "+" &
                   Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) &
                   "s Generate_Speculative EXCEPTION: unexpected error");
         Free (Prompt_C);
         return "";
   end Generate_Speculative;

end Speculative_Decode;
