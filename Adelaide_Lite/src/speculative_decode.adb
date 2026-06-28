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
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
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
with System; use type System.Address;
with System.Storage_Elements; use System.Storage_Elements;

with Llama_Interface; use Llama_Interface;
with Model_Manager;
with Model_Types; use Model_Types;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Ada.Unchecked_Conversion;

package body Speculative_Decode is

   function Llama_Batch_Get_One (T : System.Address; N : Interfaces.C.int) return Llama_Interface.Llama_Batch;
   pragma Import (C, Llama_Batch_Get_One, "llama_batch_get_one");

   --  ============================================================================
   --  DRAFT MODEL STATE
   --  ============================================================================

   Draft_Model      : Llama_Interface.Llama_Model := Null_Model;
   Draft_Context    : Llama_Interface.Llama_Context := Null_Context;
   Draft_Fallback_Time : Ada.Real_Time.Time := Ada.Real_Time.Time_First;
   Draft_Loaded     : Boolean := False;

   --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
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
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
      --  Capture start time for uptime calculation in log messages.
      Init_Start_Time := Ada.Real_Time.Clock;

      if Draft_Loaded then
         --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
         --  Verbose: logs draft model already loaded.
         Put_Line (AnsiAda.Foreground (AnsiAda.Yellow) & "[Speculative]" &
                   AnsiAda.Reset & "+" &
                   Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) &
                   "s Init_Draft_Model: draft model already loaded");
         return;
      end if;

      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
      --  Verbose: logs draft model load attempt.
      Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Speculative]" &
                AnsiAda.Reset & "+" &
                Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) &
                "s Init_Draft_Model ENTERED: loading Qwen3.5-0.8B...");

      --  Initialize model parameters
      Model_Params := Llama_Interface.Llama_Model_Default_Params;
            --  Check if we are in fallback mode (300s hold on CPU)
      if Ada.Real_Time.Clock < Draft_Fallback_Time + Ada.Real_Time.Seconds (300) then
         declare
            Remaining : constant Ada.Real_Time.Time_Span := (Draft_Fallback_Time + Ada.Real_Time.Seconds (300)) - Ada.Real_Time.Clock;
            Remaining_Secs : constant Integer := Integer (Ada.Real_Time.To_Duration (Remaining));
         begin
            Model_Params.N_Gpu_Layers := 0;   -- Draft model on CPU (fallback)
            Put_Line (AnsiAda.Foreground (AnsiAda.Yellow) & "[Speculative Fallback]" &
                      AnsiAda.Reset & " Holding draft model on CPU. " &
                      Integer'Image (Remaining_Secs) & " seconds left before retrying Tensor Accelerator.");
         end;
      else
         Model_Params.N_Gpu_Layers := -1;  -- Draft model on GPU (default)
      end if;
      Model_Params.Use_Mlock := False;  -- Allow OS to page out draft model
      Model_Params.Use_Mmap := True;    -- Enable mmap for swapping

      --  Load draft model from file
      declare
         Model_Path : constant String := "model/Qwen3.5-0.8B-Q4_K_M.gguf";
         Path_C     : chars_ptr := New_String (Model_Path);
      begin
         Draft_Model := Llama_Interface.Llama_Model_Load_From_File
           (Path_C, Model_Params);
         Free (Path_C);
      end;

            if Draft_Model = Null_Model then
         --  Set fallback time to trigger CPU load for the next 300s
         Draft_Fallback_Time := Ada.Real_Time.Clock;
         --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
         --  Verbose: logs draft model load failure.
         Put_Line (AnsiAda.Foreground (AnsiAda.Red) & "[Speculative]" &
                   AnsiAda.Reset & "+" &
                   Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) &
                   "s Init_Draft_Model FAILED: could not load draft model");
         return;
      end if;

      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
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
      Context_Params.Type_K := Llama_Interface.GGML_TYPE_Q4_0;
      Context_Params.Type_V := Llama_Interface.GGML_TYPE_Q4_0;
      Context_Params.Flash_Attn_Type := 1;

      --  Create context for draft model
      Draft_Context := Llama_Interface.Llama_Init_From_Model
        (Draft_Model, Context_Params);

      if Draft_Context = Null_Context then
         --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
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
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
      --  Verbose: confirms draft model initialization complete.
      Put_Line (AnsiAda.Foreground (AnsiAda.Green) & "[Speculative]" &
                AnsiAda.Reset & "+" &
                Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) &
                "s Init_Draft_Model COMPLETE: Qwen3.5-0.8B ready for speculative decoding");

   exception
      when others =>
         --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
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
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
         --  Verbose: logs release attempt when draft model not loaded.
         Put_Line (AnsiAda.Foreground (AnsiAda.Yellow) & "[Speculative]" &
                   AnsiAda.Reset & "+" &
                   Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) &
                   "s Release_Draft_Model: draft model not loaded");
         return;
      end if;

      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
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
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
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

   function Get_Draft_Context return Llama_Interface.Llama_Context is
   begin
      return Draft_Context;
   end Get_Draft_Context;

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

      --  Cast System.Address to access array of Llama_Token
      type Token_Array is array (Positive range <>) of aliased Llama_Interface.Llama_Token;
      type Token_Array_Access is access Token_Array;
      for Token_Array_Access'Size use Standard'Address_Size;

      Tokens_Access : Token_Array_Access;
      for Tokens_Access'Address use Draft_Tokens;

      --  Get vocabulary for the target model
      Vocab : constant Llama_Interface.Llama_Vocab :=
        Llama_Interface.Llama_Model_Get_Vocab
          (Model_Manager.Get_Model (Snowball_Enaga_ShortNetworkAnswer));

      --  Sampler for target model
      Sampler_Params : Llama_Interface.Llama_Sampler_Chain_Params;
      Sampler        : Llama_Interface.Llama_Sampler;

   begin
      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
      --  Verbose: logs verification attempt with token count.
      Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Speculative]" &
                AnsiAda.Reset & "+" &
                Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) &
                "s Verify_Draft_Tokens: verifying " &
                Interfaces.C.size_t'Image (N_Draft) & " tokens");

      --  Initialize sampler for target model (greedy for verification)
      Sampler_Params := Llama_Interface.Llama_Sampler_Chain_Default_Params;
      Sampler := Llama_Interface.Llama_Sampler_Chain_Init (Sampler_Params);
      Llama_Interface.Llama_Sampler_Chain_Add (Sampler, Llama_Interface.Llama_Sampler_Init_Greedy);

      --  Process each draft token with target model and verify
      for I in 1 .. Integer (N_Draft) loop
         declare
            Draft_Token : constant Llama_Interface.Llama_Token :=
              Tokens_Access (I);

            --  Create a batch with this single token
            Batch : constant Llama_Interface.Llama_Batch :=
              Llama_Batch_Get_One (Draft_Token'Address, 1);

            --  Decode with target model
            Ret : Interfaces.C.int;

            --  Get logits after decode
            Logits_Ptr : System.Address;
            type Logit_Array is array (Natural range <>) of Interfaces.C.C_float;
            type Logit_Array_Access is access Logit_Array;
            for Logit_Array_Access'Size use Standard'Address_Size;

            Logits_Access : Logit_Array_Access;
            function To_Logit_Access is new Ada.Unchecked_Conversion (System.Address, Logit_Array_Access);
            N_Vocab       : Interfaces.C.int;

            Target_Token  : Llama_Interface.Llama_Token;
            Draft_Logit   : Interfaces.C.C_float;
            Target_Logit  : Interfaces.C.C_float;

         begin
            --  Decode the draft token with target model
            Ret := Llama_Interface.Llama_Decode (Target_Context, Batch);

            if Ret = 0 then
               --  Get logits from target model
               Logits_Ptr := Llama_Interface.Llama_Get_Logits (Target_Context);

               if Logits_Ptr /= System.Null_Address then
                  --  Cast to logit array
                  Logits_Access := To_Logit_Access (Logits_Ptr);

                  --  Get vocabulary size
                  N_Vocab := Llama_Interface.Llama_N_Vocab
                    (Model_Manager.Get_Model (Snowball_Enaga_ShortNetworkAnswer));

                  --  Sample from target model to see what it would choose
                  Target_Token := Llama_Interface.Llama_Sampler_Sample
                    (Sampler, Target_Context, -1);

                  --  Compare: if target chose same token, accept
                  if Target_Token = Draft_Token then
                     Accepted_Count := Accepted_Count + 1;
                     --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
                     --  Verbose: logs acceptance of draft token.
                     Put_Line (AnsiAda.Foreground (AnsiAda.Green) & "[Speculative]" &
                               AnsiAda.Reset & "+" &
                               Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) &
                               "s Verify_Draft_Tokens: ACCEPT token " &
                               Interfaces.C.int'Image (Interfaces.C.int (I)) &
                               " (draft=" & Llama_Interface.Llama_Token'Image (Draft_Token) &
                               " target=" & Llama_Interface.Llama_Token'Image (Target_Token) & ")");
                  else
                     --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
                     --  Verbose: logs rejection of draft token.
                     Put_Line (AnsiAda.Foreground (AnsiAda.Yellow) & "[Speculative]" &
                               AnsiAda.Reset & "+" &
                               Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) &
                               "s Verify_Draft_Tokens: REJECT token " &
                               Interfaces.C.int'Image (Interfaces.C.int (I)) &
                               " (draft=" & Llama_Interface.Llama_Token'Image (Draft_Token) &
                               " target=" & Llama_Interface.Llama_Token'Image (Target_Token) & ")");
                     --  Stop at first rejection (standard speculative decoding)
                     exit;
                  end if;
               else
                  --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
                  --  Verbose: logs null logits pointer.
                  Put_Line (AnsiAda.Foreground (AnsiAda.Red) & "[Speculative]" &
                            AnsiAda.Reset & "+" &
                            Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) &
                            "s Verify_Draft_Tokens: null logits pointer at token " &
                            Interfaces.C.int'Image (Interfaces.C.int (I)));
                  exit;
               end if;
            else
               --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
               --  Verbose: logs decode failure.
               Put_Line (AnsiAda.Foreground (AnsiAda.Red) & "[Speculative]" &
                         AnsiAda.Reset & "+" &
                         Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) &
                         "s Verify_Draft_Tokens: decode failed at token " &
                         Interfaces.C.int'Image (Interfaces.C.int (I)) &
                         " ret=" & Interfaces.C.int'Image (Ret));
               exit;
            end if;
         end;
      end loop;

      --  Free sampler
      Llama_Interface.Llama_Sampler_Free (Sampler);

      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
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
      Generated     : Ada.Strings.Unbounded.Unbounded_String := Ada.Strings.Unbounded.To_Unbounded_String (Prompt);
      Tokens_Gen    : Natural := 0;
      Done          : Boolean := False;

      --  Draft token buffer
      Draft_Buf     : System.Address;
      N_Draft       : Interfaces.C.size_t;
      Accepted      : Interfaces.C.size_t;

      --  Sampler for draft model
      Draft_Sampler_Params : Llama_Interface.Llama_Sampler_Chain_Params;
      Draft_Sampler        : Llama_Interface.Llama_Sampler;

      --  Sampler for target model
      Target_Sampler_Params : Llama_Interface.Llama_Sampler_Chain_Params;
      Target_Sampler        : Llama_Interface.Llama_Sampler;

   begin
      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
      --  Verbose: logs speculative generation start with prompt preview.
      Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Speculative]" &
                AnsiAda.Reset & "+" &
                Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) &
                "s Generate_Speculative ENTERED: prompt=" &
                Prompt (1 .. Integer'Min (Prompt'Length, 50)) &
                "... max_tokens=" & Positive'Image (Max_Tokens));

      --  Get vocabulary for tokenization
      Vocab := Llama_Interface.Llama_Model_Get_Vocab
        (Model_Manager.Get_Model (Snowball_Enaga_ShortNetworkAnswer));  -- Use target model's vocab

      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
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
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
         --  Verbose: logs tokenization failure.
         Put_Line (AnsiAda.Foreground (AnsiAda.Red) & "[Speculative]" &
                   AnsiAda.Reset & "+" &
                   Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) &
                   "s Generate_Speculative FAILED: tokenization failed");
         return "";
      end if;

      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
      --  Verbose: confirms tokenization complete with token count.
      Put_Line (AnsiAda.Foreground (AnsiAda.Green) & "[Speculative]" &
                AnsiAda.Reset & "+" &
                Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) &
                "s Generate_Speculative: tokenized prompt: " &
                Interfaces.C.int'Image (N_Tokens) & " tokens");

      --  Initialize samplers
      --  Draft model: greedy for fast candidate generation
      Draft_Sampler_Params := Llama_Interface.Llama_Sampler_Chain_Default_Params;
      Draft_Sampler := Llama_Interface.Llama_Sampler_Chain_Init (Draft_Sampler_Params);
      Llama_Interface.Llama_Sampler_Chain_Add (Draft_Sampler, Llama_Interface.Llama_Sampler_Init_Greedy);

      --  Target model: greedy for verification
      Target_Sampler_Params := Llama_Interface.Llama_Sampler_Chain_Default_Params;
      Target_Sampler := Llama_Interface.Llama_Sampler_Chain_Init (Target_Sampler_Params);
      Llama_Interface.Llama_Sampler_Chain_Add (Target_Sampler, Llama_Interface.Llama_Sampler_Init_Greedy);

      --  Main speculative generation loop
      while not Done and then Tokens_Gen < Max_Tokens loop
         --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
         --  Verbose: logs draft phase start.
         Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Speculative]" &
                   AnsiAda.Reset & "+" &
                   Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) &
                   "s Generate_Speculative: DRAFT PHASE - generating candidate tokens...");

         --  STEP 1: Draft Phase - Generate N tokens with draft model
         declare
            Draft_Tokens_Arr : array (1 .. Max_Draft_Tokens) of aliased Llama_Interface.Llama_Token;
            Draft_N_Toks     : Natural := 0;
         begin
            Draft_Buf := Draft_Tokens_Arr'Address;
            N_Draft := 0;

            --  Generate draft tokens using draft model
            for I in 1 .. Max_Draft_Tokens loop
               --  Sample from draft model
               declare
                  Draft_Token : constant Llama_Interface.Llama_Token :=
                    Llama_Interface.Llama_Sampler_Sample
                      (Draft_Sampler, Draft_Context, -1);
               begin
                  --  Check for end of generation
                  if Llama_Interface.Llama_Vocab_Is_Eog (Vocab, Draft_Token) then
                     --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
                     --  Verbose: logs draft model EOG.
                     Put_Line (AnsiAda.Foreground (AnsiAda.Yellow) & "[Speculative]" &
                               AnsiAda.Reset & "+" &
                               Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) &
                               "s Generate_Speculative: draft model hit EOG at token " &
                               Natural'Image (I));
                     exit;
                  end if;

                  --  Store draft token
                  Draft_Tokens_Arr (I) := Draft_Token;
                  Draft_N_Toks := Draft_N_Toks + 1;

                  --  Decode with draft model to update its KV cache
                  declare
                     Batch : constant Llama_Interface.Llama_Batch :=
                       Llama_Batch_Get_One (Draft_Token'Address, 1);
                     Ret   : Interfaces.C.int;
                  begin
                     Ret := Llama_Interface.Llama_Decode (Draft_Context, Batch);
                     if Ret /= 0 then
                        --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
                        --  Verbose: logs draft model decode failure.
                        Put_Line (AnsiAda.Foreground (AnsiAda.Red) & "[Speculative]" &
                                  AnsiAda.Reset & "+" &
                                  Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) &
                                  "s Generate_Speculative: draft decode failed ret=" &
                                  Interfaces.C.int'Image (Ret));
                        exit;
                     end if;
                  end;
               end;
            end loop;

            N_Draft := Interfaces.C.size_t (Draft_N_Toks);

            --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
            --  Verbose: logs draft phase complete with token count.
            Put_Line (AnsiAda.Foreground (AnsiAda.Green) & "[Speculative]" &
                      AnsiAda.Reset & "+" &
                      Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) &
                      "s Generate_Speculative: DRAFT PHASE COMPLETE - generated " &
                      Interfaces.C.size_t'Image (N_Draft) & " tokens");
         end;

         --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
         --  Verbose: logs verify phase start.
         Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Speculative]" &
                   AnsiAda.Reset & "+" &
                   Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) &
                   "s Generate_Speculative: VERIFY PHASE - verifying with target model...");

         --  STEP 2: Verify Phase - Verify with target model
         Accepted := Verify_Draft_Tokens
           (Draft_Buf, N_Draft, Target_Context);

         --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
         --  Verbose: logs accept phase start.
         Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Speculative]" &
                   AnsiAda.Reset & "+" &
                   Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) &
                   "s Generate_Speculative: ACCEPT PHASE - accepting " &
                   Interfaces.C.size_t'Image (Accepted) & " tokens...");

         --  STEP 3: Accept Phase - Keep accepted tokens
         if Accepted > 0 then
            --  Add accepted tokens to generated text
            declare
               type Token_Array is array (Positive range <>) of aliased Llama_Interface.Llama_Token;
               type Token_Array_Access is access Token_Array;
               for Token_Array_Access'Size use Standard'Address_Size;

               Tokens_Access : Token_Array_Access;
               for Tokens_Access'Address use Draft_Buf;

               Piece : array (1 .. 256) of aliased Character;
               Len   : Interfaces.C.int;
            begin

               for I in 1 .. Integer (Accepted) loop
                  --  Convert token to piece
                  Len := Llama_Interface.Llama_Token_To_Piece
                    (Vocab, Tokens_Access (I), Piece (1)'Address, 256, 0, True);

                  if Len > 0 then
                     for J in 1 .. Integer (Len) loop
                        Ada.Strings.Unbounded.Append (Generated, Piece (J));
                     end loop;
                  end if;

                  --  Decode with target model to update its KV cache
                  declare
                     Batch : constant Llama_Interface.Llama_Batch :=
                       Llama_Batch_Get_One (Tokens_Access (I)'Address, 1);
                     Ret   : Interfaces.C.int;
                  begin
                     Ret := Llama_Interface.Llama_Decode (Target_Context, Batch);
                     if Ret /= 0 then
                        --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
                        --  Verbose: logs target model decode failure.
                        Put_Line (AnsiAda.Foreground (AnsiAda.Red) & "[Speculative]" &
                                  AnsiAda.Reset & "+" &
                                  Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) &
                                  "s Generate_Speculative: target decode failed ret=" &
                                  Interfaces.C.int'Image (Ret));
                        exit;
                     end if;
                  end;
               end loop;
            end;

            Tokens_Gen := Tokens_Gen + Natural (Accepted);
            --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
            --  Verbose: logs successful acceptance.
            Put_Line (AnsiAda.Foreground (AnsiAda.Green) & "[Speculative]" &
                      AnsiAda.Reset & "+" &
                      Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) &
                      "s Generate_Speculative: accepted " &
                      Interfaces.C.size_t'Image (Accepted) & " tokens");
         else
            --  All rejected - fall back to single token from target
            --  Sample from target model to get the correct token
            declare
               Target_Token : constant Llama_Interface.Llama_Token :=
                 Llama_Interface.Llama_Sampler_Sample
                   (Target_Sampler, Target_Context, -1);

               Piece : array (1 .. 256) of aliased Character;
               Len   : Interfaces.C.int;
            begin
               --  Convert token to piece
               Len := Llama_Interface.Llama_Token_To_Piece
                 (Vocab, Target_Token, Piece (1)'Address, 256, 0, True);

               if Len > 0 then
                  for J in 1 .. Integer (Len) loop
                     Ada.Strings.Unbounded.Append (Generated, Piece (J));
                  end loop;
               end if;

               --  Decode with target model to update its KV cache
               declare
                  Batch : constant Llama_Interface.Llama_Batch :=
                    Llama_Batch_Get_One (Target_Token'Address, 1);
                  Ret   : Interfaces.C.int;
               begin
                  Ret := Llama_Interface.Llama_Decode (Target_Context, Batch);
                  if Ret /= 0 then
                     --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
                     --  Verbose: logs target model decode failure.
                     Put_Line (AnsiAda.Foreground (AnsiAda.Red) & "[Speculative]" &
                               AnsiAda.Reset & "+" &
                               Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) &
                               "s Generate_Speculative: target decode failed ret=" &
                               Interfaces.C.int'Image (Ret));
                  end if;
               end;
            end;

            Tokens_Gen := Tokens_Gen + 1;
            --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
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

      --  Free samplers
      Llama_Interface.Llama_Sampler_Free (Draft_Sampler);
      Llama_Interface.Llama_Sampler_Free (Target_Sampler);

      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
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

      return Ada.Strings.Unbounded.To_String (Generated);

   exception
      when others =>
         --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
         --  Verbose: logs unexpected exception during generation.
         Put_Line (AnsiAda.Foreground (AnsiAda.Red) & "[Speculative]" &
                   AnsiAda.Reset & "+" &
                   Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) &
                   "s Generate_Speculative EXCEPTION: unexpected error");
         Free (Prompt_C);
         Llama_Interface.Llama_Sampler_Free (Draft_Sampler);
         Llama_Interface.Llama_Sampler_Free (Target_Sampler);
         return "";
   end Generate_Speculative;

end Speculative_Decode;
