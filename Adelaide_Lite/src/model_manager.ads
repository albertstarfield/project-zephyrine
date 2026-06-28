pragma SPARK_Mode (Off);
with Llama_Interface;     use Llama_Interface;
with Mtmd_Interface;    use Mtmd_Interface;
with Math_Utils;         use Math_Utils;
with Streaming_Queue;
with KV_Cache_Manager;
with System;
with Interfaces.C;     use Interfaces.C;
with Ada.Unchecked_Deallocation;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Ada.Real_Time;    use Ada.Real_Time;
with GNATCOLL.JSON;
with Model_Types; use Model_Types;
with Speculative_Decode;

package Model_Manager is

   --  =========================================================================
   --  PARALLEL=1 CONSTRAINT (CRITICAL — READ BEFORE MODIFYING MODEL LIFECYCLE)
   --  =========================================================================
   --  This server runs with parallel=1. This means:
   --    - Only ONE model can be loaded in GPU memory at any time.
   --    - If 2 models are loaded simultaneously, Metal OOM occurs because:
   --      Model A weights + Model B weights + KV cache + compute buffers
   --      exceeds available GPU VRAM (typically 8-16GB on Apple Silicon).
   --    - The embedding model (~1GB) MUST be fully unloaded (model + context
   --      freed from GPU) BEFORE the chat model (5.8GB) loads.
   --    - The chat model MUST be fully unloaded BEFORE the embedding model
   --      loads for the next request.
   --
   --  Queue Processing Flow:
   --    1. User sends request
   --    2. Get_Embedding loads embedding model → computes vector → UNLOADS
   --    3. Hybrid_Generate loads chat model → generates response → UNLOADS
   --    4. Only then can the next request start loading its model
   --
   --  ELP0 (background) and ELP1 (foreground) share the same GPU.
   --  ELP0 tasks are preempted when ELP1 arrives (Priority_Model_Gate).
   --  Both must follow the one-model-at-a-time rule.
   --
   --  VIOLATION = Metal OOM = server crash = you get killed.
   --  =========================================================================

   --  Token array type for virtual context paging cache
   subtype Cached_Token is Interfaces.C.int;
   type Cached_Token_Array is array (Positive range <>) of Cached_Token;
   type Cached_Token_Access is access Cached_Token_Array;
   procedure Free_Cached_Tokens is new Ada.Unchecked_Deallocation
     (Cached_Token_Array, Cached_Token_Access);

   --  REMOVED: Generate_Speculative (ggml draft-model token speculation).
   --  Was disabled via `Enable_Speculative => False` due to ggml-metal GPU
   --  buffer races during QWEN_0_8B unload causing Abort trap: 6.
   --  Replaced by Speculative_Cache (query-level semantic cache populated
   --  proactively by ELP0 background tasks — see Knowledge_Manager).

   procedure Initialize;

   procedure Load_Model
     (Kind          : Model_Type;
      Success       : out Boolean;
      Requested_Ctx : Positive := 4096;
      Level         : ELP_Level := ELP1;
      Is_File_Index : Boolean := False);

   procedure Unload_Model (Kind : Model_Type);

   procedure Force_Unload_And_Reload (Kind : Model_Type);

   function Llama_Abort_Callback (Data : System.Address) return Boolean;
   pragma Convention (C, Llama_Abort_Callback);

   function Get_Context
     (Kind : Model_Type) return Llama_Interface.Llama_Context;

   function Get_Model
     (Kind : Model_Type) return Llama_Interface.Llama_Model;

   --  Get the mtmd (multimodal) context for vision processing
   --  Returns Null_Mtmd_Context if MMProj is not loaded
   function Get_Mtmd_Context
     (Kind : Model_Type) return Mtmd_Interface.Mtmd_Context;

    --  KV CACHE SSD SPILLOVER
    --  Save KV cache to SSD after generation
    procedure Save_KV_Cache_To_SSD
      (Kind     : Model_Type;
       Tokens   : System.Address;
       N_Tokens : Interfaces.C.size_t);

    --  Load KV cache from SSD if available
    function Load_KV_Cache_From_SSD
      (Kind     : Model_Type;
       Tokens   : out System.Address;
       N_Tokens : out Interfaces.C.size_t) return Boolean;

   --  Perform inference
   procedure Generate
     (Kind            : Model_Type;
       Prompt          : String;
       Result          : out Unbounded_String;
       Images          : GNATCOLL.JSON.JSON_Array := GNATCOLL.JSON.Empty_Array;
       Session_ID      : String := "";
       Requested_Ctx   : Positive := 4096;
       Stream          : Streaming_Queue.Queue_Access := null;
       Orch_Think_Open : Boolean := False;
       Level           : ELP_Level := ELP1;
       --  VIRTUAL CTX PAGING: pre-tokenized Internal_State tokens.
       --  When provided, these are written to the token array FIRST,
       --  then Prompt is tokenized into the remaining slots.  This
       --  avoids re-tokenizing the same facts on every context fault hop.
       Virtual_Tokens  : Cached_Token_Access := null;
       Virtual_Tok_Len : Natural := 0;
       --  When False, Generate does NOT release In_Use or the ELP lock.
       --  The caller (Hybrid_Generate) is responsible for releasing the
       --  model after all post-processing is complete.  This prevents the
       --  Idle_Monitor from unloading the model while Hybrid_Generate is
       --  still executing tool calls, streaming, etc.
       --  [FREE-PARALLEL-MEMORY] When True, free ALL GPU memory for this model
       --  after generation completes. This is NOT just for LLM models — it
       --  applies to any heavy GPU component: Stable Diffusion Flux, LSH/QRNN
       --  hash workers, database memory, embedding models, etc. The idea is
       --  LM Studio-style: one component at a time, clean load/use/unload.
       --  When False, keep the component resident for the next hop (Hybrid_Generate).
       FreeParallelMemory : Boolean := True;
       --  When True, bypasses ELP gate acquire/release entirely.
       --  Used by Hybrid_Generate for its internal sub-calls (router,
       --  factual check) which run while the gate is already held.
       --  Without this, each hop re-acquires an already-locked gate
       --  causing a deadlock (ELP1 never runs after the first hop).
       Skip_Gate       : Boolean := False;
       --  When True, enables draft-model speculative decoding (Qwen3.5-0.8B)
       Use_Speculative : Boolean := False);

   --  ============================================================================
   --  SPECULATIVE DECODING
   --  ============================================================================
   --  WHY THIS EXISTS:
   --  Speculative decoding accelerates LLM inference by using a smaller,
   --  faster "draft" model (Qwen3.5-0.8B) to generate candidate tokens,
   --  then verifying them in parallel with the larger "target" model.
   --  This provides 2-3x speedup for text generation.
   --
   --  HOW IT WORKS:
   --    1. Draft Model generates N candidate tokens quickly
   --    2. Target Model verifies all N tokens in parallel
   --    3. Accept matching prefix, resample rest from target distribution
   --    4. Repeat until generation complete
   --
   --  DRAFT MODEL:
   --  - Qwen3.5-0.8B (not 0.5B from oMLX)
   --  - Faster inference, lower quality, used only for candidates
   --  - Must be compatible with target model's tokenizer
   --  ============================================================================



   --  Perform multi-hop reasoning
   procedure Hybrid_Generate
     (Prompt         : String;
      Result         : out Unbounded_String;
      Images         : GNATCOLL.JSON.JSON_Array := GNATCOLL.JSON.Empty_Array;
      Session_ID     : String := "";
      Stream         : Streaming_Queue.Queue_Access := null;
      Level          : ELP_Level := ELP1;
      Agentic        : Boolean := False;
      Raw_Prompt     : Boolean := False;
      External_Agent : Boolean := False);

    procedure Get_Embedding
      (Prompt : String;
       Result : out Math_Utils.Vector;
       Length : out Natural;
       Level  : ELP_Level := ELP1);

     --  [BATCHING] Process multiple prompts in a single Metal command buffer
     --  Returns an array of vectors. Lengths are handled via a separate array.
     procedure Get_Embeddings_Batch
       (Prompts : Math_Utils.Prompt_List;
        Results : out Math_Utils.Embedding_Vector_List;
        Lengths : out Natural_List;
        Level   : ELP_Level := ELP1);

     --  [FREE-PARALLEL-MEMORY] Global universal call procedure
     --  Releases VRAM and CPU RAM for all models, Stable Diffusion, and Moonshine.
     procedure FreeParallelMemory;


   --  POWER AWARE SCHEDULING
   --  [VITAL-DO-NOT-REMOVE]
   --  Allows external bridge (Python) to signal power state.
   --  When On_Battery is True and Level < 80, ELP0 execution is suspended.
   procedure Set_Power_Condition (On_Battery : Boolean; Level : Natural);

   function Should_Abort_ELP0 return Boolean;

   --  BLOCK UNTIL ELP1 COMPLETES (used by ELP0 background tasks).
   --
   --  WHY THIS EXISTS — the 1-second polling deadlock:
   --  Previously, ELP0 tasks (Indexing_Task, Proactive_Cache_Task) polled
   --  Should_Abort_ELP0 every 1 second.  When an ELP1 (user) request arrived,
   --  the abort callback fired inside Llama_Decode, but decode did not return
   --  quickly.  So ELP0 kept looping: Should_Abort=True → delay 1.0 → re-enter
   --  loop → Should_Abort=True → delay 1.0 → …  During this loop ELP0 held
   --  the model (Busy=True), so Acquire_ELP1 blocked forever.  The abort
   --  callback printed "[ELP0-ABORT-CHECK]" every second but never reached the
   --  code that calls Release_ELP0, because Llama_Decode had not returned.
   --
   --  HOW THIS FIXES IT:
   --  Instead of polling, ELP0 tasks call this entry which blocks on an Ada
   --  protected entry barrier (when ELP1_Pending = 0 and ELP1_Active_Count = 0).
   --  When ELP1 arrives (Request_ELP1 increments Pending), the barrier is False
   --  and the ELP0 task suspends immediately — no polling, no delay loop.
   --  Once the current Llama_Decode returns and ELP1 finishes (Release_ELP1
   --  decrements Active_Count), the barrier opens and the ELP0 task resumes.
   --  This eliminates the deadlock entirely.
   procedure Wait_For_ELP1_Idle;

   function Get_Kind_For_Model_Name (Name : String) return Model_Type;

   function Is_Loaded (Kind : Model_Type) return Boolean;

   function Count_Tokens (Text : String) return Positive;

   function Get_Request_Category
     (Msg        : String;
      Session_ID : String := "";
      Level      : ELP_Level := ELP1) return String;

   function Grade_Response_Quality
     (Response_Text : String;
      Prompt        : String;
      Search_Used   : Boolean;
      Has_Citations : Boolean;
      Session_ID    : String := "";
      Level         : ELP_Level := ELP1) return Natural;

   procedure Push_Chunk
     (Stream     : Streaming_Queue.Queue_Access;
      Session_ID : String;
      Str_Piece  : String);

   function Generator_Callback (Prompt : String) return String;

   function Sanitize_Think_Tags (Text : String) return String;

   --  GLOBAL METAL SERIALIZATION LOCK
   --  [QUIRK-M04] ggml-Metal GPU buffer corruption when multiple models
   --  (QWEN_EMBEDDING + QWEN_0_8B) decode concurrently on the same MTL device.
   --  Root cause: Metal command buffers from different llama.cpp contexts
   --  interleave, corrupting buffer metadata (malloc error: "pointer being
   --  freed was not allocated" at address 0x1). Fix: serialize all
   --  llama_decode + llama_get_embeddings calls through this entry barrier.
   procedure Acquire_Accel_Lock;
   procedure Release_Accel_Lock;

   --  [VITAL-DO-NOT-REMOVE] Metal backend health flag — OPPORTUNISTIC.
   --  Set to True when llama_decode returns -3 (OOM) or any Metal error
   --  that poisons the backend. KV Cache save task checks this flag before
   --  calling llama_state_save_file — calling it on a poisoned Metal backend
   --  causes SIGBUS → GNAT exception → exit() → ggml_metal_device_free →
   --  GGML_ASSERT([rsets->data count] == 0) → SIGABRT kills the server.
   --
   --  OPPORTUNISTIC: Flag auto-resets after Metal_OOM_Cooldown_Secs (30s).
   --  The save task retries every Metal_OOM_Retry_Secs (5s) until the
   --  cooldown expires, then tries again. This way:
   --    1. Immediate save is skipped (prevents SIGABRT)
   --    2. After 30s, GPU driver has time to recover
   --    3. Save retries automatically — no data loss
   Metal_Backend_Broken    : Boolean := False;
   Metal_OOM_Trigger_Time  : Duration := 0.0;  -- Time when OOM was detected
   Metal_OOM_Cooldown_Secs : constant Duration := 30.0;  -- Reset after 30s
   Metal_OOM_Retry_Secs    : constant Duration := 5.0;   -- Retry every 5s

   --  Check if Metal backend is still broken or has recovered.
   --  Auto-resets Metal_Backend_Broken after cooldown expires.
   function Is_Metal_Broken return Boolean;

   --  Mark Metal backend as broken (called on OOM decode failure).
   procedure Mark_Metal_Broken;

   --  [TENSOR-ACCEL-INOP] Graceful fallback mode for persistent compute errors.
   --  When 10+ consecutive ggml compute errors occur (ret=-2 from llama_decode),
   --  GPU acceleration is disabled (N_Gpu_Layers := 0) and a countdown timer
   --  starts. The system falls back to CPU-only processing for 10 minutes,
   --  then re-enables GPU. This prevents the server from thrashing on a broken
   --  Metal backend while still providing service via CPU.
   --
   --  TRIGGER: 10 consecutive ggml compute errors (any decode loop)
   --  EFFECT: N_Gpu_Layers forced to 0, red countdown printed every 1s
   --  RECOVERY: After 600s (10 min), GPU layers re-enabled
   Tensor_Accel_INOP       : Boolean := False;
   INOP_Consecutive_Errors : Natural := 0;   -- Consecutive ggml compute errors
   INOP_Error_Threshold    : constant Natural := 420;  -- Blaze it threshold
   INOP_Retry_Countdown    : Natural := 0;   -- Seconds remaining in cooldown
   INOP_Cooldown_Secs      : constant Natural := 600;  -- 10 minutes
   INOP_Trigger_Time       : Duration := 0.0;  -- Time when INOP was triggered

   --  Record a ggml compute error for the INOP counter.
   --  Increments counter; triggers INOP if threshold reached.
   procedure Record_INOP_Error;

   --  Clear the INOP error counter (called on successful decode).
   procedure Clear_INOP_Error;

   --  Check if Tensor_Accel_INOP is active.
   function Is_Tensor_INOP return Boolean;

   Current_WCET : Duration := 0.0;
   Current_WCET_ELP0 : Duration := 0.0;
   Current_WCET_ELP1 : Duration := 0.0;
   Current_WCET_ELP2 : Duration := 0.0;
   Current_WCET_ELP3 : Duration := 0.0;

   --  ELP3 Timing Correction / Jitter Profile
   Current_Jitter_Max : Duration := 0.0;
   Current_Jitter_Avg : Duration := 0.0;

   --  Last user prompt (set by Hybrid_Generate at ELP1 level).
   --  Read by Proactive_Cache_Task in Knowledge_Manager to predict
   --  follow-up questions and pre-populate Speculative_Cache.
   Last_User_Prompt : Unbounded_String := Null_Unbounded_String;

   --  CONTEXT FAULT MONITORING (printed every 5s by Context_Monitor task)
   --  Tracks the virtual context space and context fault paging state.
   --  Virtual Context: accumulated factual data from tool results, measured
   --  in bytes (Internal_State) then approximated to tokens.
   --  LLM Context: the actual llama.cpp context window (N_Ctx) that holds
   --  the tokenized prompt + KV cache for attention.
   --  Context Fault: Model requests additional context mid-generation via
   --  [CONTEXT_FAULT: query=... category=...]. Each fault adds a "hop".
   Current_Context_Fault_JMPs : Natural := 0;
   Current_Internal_State_Len : Natural := 0;
   Current_JMP_Count          : Natural := 0;
   --  Token tracking for context window utilization
   Current_Prompt_Tokens      : Natural := 0;   -- actual tokens in prompt
   Current_Ctx_Capacity       : Natural := 8192; -- llama context window size

   --  PREFILL SPEED & TIME BUDGET TRACKING
   --  ============================================================================
   --  WHY: Dynamic ctx expansion must respect a time budget. If prefill takes
   --  too long, the user perceives lag. We measure actual prefill tok/s
   --  (NOT cached/virtualized tokens — those are free) and weight it against
   --  free LLM context % to decide whether to expand.
   --
   --  TIME BUDGET: 30 seconds max for prefill. If prefill exceeds this,
   --  the system should NOT expand context further — it's already too slow.
   --
   --  THRESHOLD FORMULA:
   --    threshold_pct = 30 / prefill_elapsed * (free_ctx_pct / 100)
   --  This weights the time budget against available context headroom.
   --  If free_ctx is low (<25%), threshold drops → don't expand.
   --  If free_ctx is high (>75%), threshold rises → safe to expand.
   --
   --  PRINTED IN CTXMONITOR: Virtual prefill speed (tok/s) so we can
   --  observe degradation and tune the threshold dynamically.
   --  ============================================================================
   Prefill_Start_Time        : Ada.Real_Time.Time := Ada.Real_Time.Time_First;
   Prefill_Token_Count       : Natural := 0;    -- tokens actually decoded (not cached)
   Virtual_Prefill_Speed     : Duration := 0.0;  -- computed tok/s after prefill
   Prefill_Elapsed           : Duration := 0.0;  -- wall-clock seconds for prefill
   Free_Ctx_Pct              : Natural := 0;     -- free % of LLM context window
   Ctx_Expand_Threshold_Pct  : Natural := 75;    -- dynamic threshold for expansion

   --  CACHED VIRTUAL CTX TOKENS
   --  When Internal_State grows, we re-tokenize ONLY the new portion.
   --  The cached tokens are prepended to the prompt on each generation,
   --  skipping re-tokenization of already-known facts.  This makes
   --  context faulting faster: the LLM sees pre-tokenized facts without
   --  paying the tokenization cost again.
   Cached_Virtual_Tokens : Cached_Token_Access := null;
   Cached_Virtual_Len    : Natural := 0;
   Internal_State        : Unbounded_String := Null_Unbounded_String;

   --  [VITAL-DO-NOT-REMOVE] Randomized seed for Generate sampler.
   --  Initialized with Ada.Calendar.Seconds to get different output on
   --  each retry. Incremented on think-only retries to avoid identical
   --  degenerate responses.
   Generate_Seed : Interfaces.C.unsigned := 0;

   --  =========================================================================
   --  GPU MEMORY MONITOR (printed every 100ms by Print_GPU_Memory_Sync)
   --  =========================================================================
   --  Tracks free/total GPU VRAM across ALL backends (Metal, CUDA, OneAPI,
   --  SYCL, Vulkan, ROCm). If GPU memory query is inapplicable (CPU-only
   --  or Vulkan without memory query), reports "stable" or "UNSTABLE".
   --
   --  ADAPTIVE GPU LAYER STRATEGY:
   --  1. Start aggressive: Acceleration_Silicon_Layer = -1 (ALL layers on GPU)
   --  2. If OOM → fallback: remove 25% of current layers
   --     e.g. 32 → 24 → 18 → 14 → 10 → 8 → ...
   --  3. Record OOM timestamp
   --  4. After GPU_Retry_Interval (3 min) → reset to -1 (all on GPU)
   --  5. If OOM again → progressive fallback again → repeat
   --  This auto-probes whether the GPU can handle full offload, and
   --  backs off progressively when it can't, recovering automatically.

   Total_Model_Layers   : constant Natural := 32;  -- Qwen3.5HybridMythos
   GPU_Layer_Fallback   : constant Integer := 24;  -- Initial fallback (75%)
   GPU_Layer_Min        : constant Integer := 8;   -- Minimum layers on GPU
   GPU_Layer_Step       : constant Integer := 1;   -- 25% reduction each OOM
   GPU_Retry_Interval   : constant Duration := 180.0;  -- 3 minutes

    GPU_Free_MB          : Natural := 0;    -- Free GPU memory in megabytes
    GPU_Total_MB         : Natural := 0;    -- Total GPU memory in megabytes
    GPU_Layer_Percent    : Natural := 0;    -- Free/Total * 100, for display
    GPU_Is_Stable        : Boolean := True;  -- False if OOM/crash detected
    Acceleration_Silicon_Layer      : Integer := -1;   -- ACTUAL layers on GPU (-1 = all)
    GPU_Last_OOM_Time    : Ada.Real_Time.Time := Ada.Real_Time.Time_First;
    --  Time of last OOM. After GPU_Retry_Interval, Load_Model retries -1.

    --  Model record type
    type Model_Record is record
        Model       : Llama_Model := Null_Model;
        Context     : Llama_Context := Null_Context;
        Mtmd_Ctx    : Mtmd_Interface.Mtmd_Context := Null_Mtmd_Context;
        Path        : Unbounded_String;
        Loaded      : Boolean := False;
        In_Use      : Boolean := False;
        Last_Used   : Time := Time_First;
        Current_Ctx : unsigned := 0;
        Warm_Cached : Boolean := False;  -- True if in warm cache (not fully unloaded)
        Warm_Cache_Time : Time := Time_First;  -- When it was cached
    end record;

    --  Array of model records for each model type
    Models : array (Model_Type) of Model_Record;

    --  [OPTIMIZATION-M02] Function to check if model is available (loaded or warm cached)
    function Is_Model_Available (Kind : Model_Type) return Boolean;

    --  [OPTIMIZATION-M02] Helper function to access model state
    function Get_Model_State (Kind : Model_Type) return Model_Record;

    --  [OPTIMIZATION-M02] Warm cache time-to-live: 30 seconds
    --  Models stay in warm cache for this duration after "unload"
    --  This reduces the frequency of cold starts for frequently used models
    --  This constant is used in model_manager.adb body
    Warm_Cache_TTL : constant Duration := 30.0;

end Model_Manager;
