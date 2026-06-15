pragma SPARK_Mode (Off);
with AnsiAda;
with Ada.Text_IO; use Ada.Text_IO;
with Ada.Strings; use Ada.Strings;
with Ada.Strings.Fixed; use Ada.Strings.Fixed;
with Ada.Calendar; use type Ada.Calendar.Time;
with Database_Manager;
with Tool_Manager;
with Scheduler_Manager;
with Llama_Interface;
use Llama_Interface;
with Mtmd_Interface;
use Mtmd_Interface;
with Interfaces.C; use Interfaces.C;
with Interfaces.C.Strings; use Interfaces.C.Strings;
with Ada.Directories;
with Ada.Real_Time; use Ada.Real_Time;
with Ada.Unchecked_Conversion;
with Ada.Exceptions;
with Watchdog_Manager;
with Kratos;
with ELP_Queue;
with Speculative_Cache;

--  ===========================================================================
--  MODEL MANAGEMENT QUIRKS & DISCOVERED WORKAROUNDS
--  ===========================================================================
--  [QUIRK-M01] [ALL] Kratos crash isolation layer
--  Every llama_decode call is wrapped in Kratos.Guard_Enter / Guard_Exit.
--  If the C-level code crashes (SIGSEGV, SIGBUS, SIGFPE, SIGTRAP, SIGABRT),
--  Kratos catches the signal and longjmps back, returning a nonzero code.
--  The Ada code then calls Kratos.Log_Crash and returns -1 instead of
--  crashing the entire server process.  This is ESSENTIAL because llama.cpp
--  can segfault during edge-case decodes (e.g., corrupted KV cache state,
--  context size mismatch, model unload races).
--
--  [QUIRK-M02] [ALL] Context size management
--  Context sizes are binned to 8192-increment granularity with a hard cap
--  at 65536.  The Q4_1 KV cache uses approximately 10GB for both 65536 and
--  228K context lengths (Q4_1 is very efficient).  The 65536 cap is a
--  practical stability limit on 16GB hardware; 228K is theoretically
--  possible but leaves insufficient headroom for model weights + macOS.
--  Minimum context is 8192 (smaller values cause llama_decode assertion
--  failures with Qwen3.5-9B at Q4_1 KV quantization).
--
--  [QUIRK-M03] [macOS] Pre-existing signal crash after QWEN_0_8B release
--  Observed (2026-06-10): After QWEN_0_8B processes a request and the model
--  is released, the server may crash with exit code -1 (signal caught by
--  Kratos).  Root cause: Idle_Monitor unloads QWEN_0_8B via Llama_Free
--  after 30s inactivity, which triggers a ggml-metal GPU buffer race
--  (SIGABRT / Abort trap: 6).  Fixed on macOS by exempting QWEN_0_8B from
--  the Idle_Monitor unload loop — the 0.8B model is only ~0.5-0.6GB VRAM
--  at Q4_K_S and is kept permanently loaded.
--  LINUX-COMPAT / Android-Termux: On Linux (CUDA/Vulkan) the ggml-metal
--  race does not occur, so the Qwen_0_8B exemption guard should be REMOVED
--  from Idle_Monitor to allow aggressive model unloading.  Smartphone /
--  Termux-on-Android targets have very limited RAM (4-8GB) and shared GPU
--  memory, so ALL models including QWEN_0_8B must be unloaded aggressively
--  when not in use.  The Idle_Monitor timeout can be lowered to 10-15s
--  for those targets.
--
--  [QUIRK-M04] [ALL] Model path discovery fallback
--  Load_Model tries 3 path variants (direct, ../, ../../) because the
--  working directory at runtime is unpredictable:
--     - run.py CWD = Adelaide_Lite/
--     - alr exec CWD = Adelaide_Lite/
--     - Direct binary CWD = varies
--  This avoids requiring a fixed working directory.
--
--  [QUIRK-M05] [ALL] Sanitize_UTF8 strips non-ASCII
--  The Sanitize_UTF8 function (used in Generate and Get_Single_Embedding)
--  strips all characters with codepoint > 127 (DEL and non-ASCII).
--  This means multilingual prompts (Chinese, Arabic, emoji, etc.) are
--  silently corrupted.  Qwen3.5 models support Unicode natively, so this
--  is a preprocessing limitation, not a model limitation.  If multilingual
--  support is needed, Sanitize_UTF8 must be relaxed to pass through valid
--  UTF-8 multi-byte sequences.
--
--  [QUIRK-M06] [ALL] Flush_Parser thinking block awareness
--  The Flush_Parser procedure (called when the sanitize buffer grows too
--  large) must respect the In_Think_Block state.  If a flush occurs while
--  the model is inside a <think> block, the content must be silenced.
--  Failing to check this state causes "thinking leaks" where internal
--  reasoning is visible to the client if the thought exceeds 500 chars.
--
--  [QUIRK-M07] [ALL] Sanitize_Think_Tags backtracking
--  If a model outputs an opening <think> tag but hits EOG before closing
--  it, the naive sanitizer would strip the entire response until it finds
--  a closing tag that never arrives.  The improved sanitizer uses a
--  backtracking mechanism: if a closing tag is not found by the end of
--  the string, it treats the opening tag as regular text.  This prevents
--  "empty response" bugs when models fail to close their thinking blocks.
--  ===========================================================================

package body Model_Manager is
   use Streaming_Queue;

   --  Token array types (package-level for use by Generate and
   --  Tokenize_And_Cache_Virtual_Ctx)
   type Token_Array is array (Positive range <>) of
     Llama_Interface.Llama_Token;
   type Token_Array_Access is access Token_Array;
   procedure Free_Tokens is new Ada.Unchecked_Deallocation
     (Token_Array, Token_Array_Access);

   --  [DO NOT REMOVE] C FFI for stderr suppression during model loading.
   --  llama.cpp prints hundreds of verbose lines to stderr during load.
   --  We redirect stderr to /dev/null, load, then restore.
   function Sys_Dup (Fildes : int) return int;
   pragma Import (C, Sys_Dup, "suppress_dup");
   function Sys_Restore_Stderr (Saved_Fd : int) return int;
   pragma Import (C, Sys_Restore_Stderr, "suppress_restore");

   function Llama_Batch_Get_One
     (T : System.Address; N : int) return Llama_Batch;
   pragma Import (C, Llama_Batch_Get_One, "llama_batch_get_one");

   task type WCET_Printer;
   task body WCET_Printer is
   begin
      loop
         delay 30.0;
         Ada.Text_IO.Put_Line
           (AnsiAda.Foreground (AnsiAda.Light_Red) & "[WCET]" & AnsiAda.Reset &
            " Pipeline: " & Long_Long_Integer'Image
              (Long_Long_Integer (Current_WCET * 1_000_000_000)) & "ns | " &
            "ELP0: " & Long_Long_Integer'Image
              (Long_Long_Integer (Current_WCET_ELP0 * 1_000_000_000)) &
            "ns | " &
            "ELP1: " & Long_Long_Integer'Image
              (Long_Long_Integer (Current_WCET_ELP1 * 1_000_000_000)) &
            "ns | " &
            "ELP2: " & Long_Long_Integer'Image
              (Long_Long_Integer (Current_WCET_ELP2 * 1_000_000_000)) &
            "ns | " &
            "ELP3: " & Long_Long_Integer'Image
              (Long_Long_Integer (Current_WCET_ELP3 * 1_000_000_000)) & "ns");
      end loop;
   end WCET_Printer;

   Printer_Task : WCET_Printer;

   --  =====================================================================
   --  CONTEXT MONITOR TASK
   --  =====================================================================
   --  Prints virtual context state every 5 seconds:
   --    - Virtual Context: 2^63 capacity, how much occupied (depth)
   --    - Context Fault Division Page: where each hop jumps, how much data
   --    - Internal_State size: accumulated factual data from paging
   --
   --  The "virtual context" is the 2^63 theoretical space (ELP Queue).
   --  The "division page" is how context fault hops divide the reasoning
   --  chain: each hop pages in new factual data via tool execution.
   --  =====================================================================
   task Context_Monitor is
      entry Start;
   end Context_Monitor;

   task body Context_Monitor is
      Interval   : constant Duration := 5.0;
      Next_Check : Ada.Calendar.Time;
      Fault_Total : Natural := 0;
   begin
      accept Start;
      loop
         Next_Check := Ada.Calendar.Clock + Interval;

         --  Aggregate context fault hops across all active sessions
         --  (Current_Context_Fault_Hops is updated by Hybrid_Generate)
         Fault_Total := Current_Context_Fault_Hops;

         declare
            --  Virtual Context (2^63) metrics from ELP Queue
            VC_Capacity : constant Interfaces.Unsigned_64 := ELP_Queue.Capacity;
            VC_Depth    : constant Long_Long_Integer := ELP_Queue.Depth;
            VC_Util     : constant Long_Long_Float   := ELP_Queue.Utilization;
            VC_Pct      : constant Long_Long_Float   := VC_Util * 100.0;

            --  Context Fault Division Page math
            --  Each hop "pages" into a new division of the context space.
            --  Division page = hop_count + 1 (first page is the original prompt).
            --  Max pages = 6 (original + 5 hops).
            Max_Divisions : constant Natural := 6;
            Cur_Division  : constant Natural := Fault_Total + 1;

            --  Virtual Context: Internal_State bytes → approx tokens
            --  Rule of thumb: ~3 bytes per token for English text
            VC_Bytes   : constant Natural := Current_Internal_State_Len;
            VC_Tokens  : constant Natural :=
              (if Cached_Virtual_Len > 0
               then Cached_Virtual_Len   --  Exact count from token cache
               else VC_Bytes / 3);       --  Approximation (no cache yet)
            --  As percentage of the LLM context window
            LLM_Ctx    : constant Natural := Current_Ctx_Capacity;
            VC_Ctx_Pct : constant Natural :=
              (if LLM_Ctx > 0 then (VC_Tokens * 100) / LLM_Ctx else 0);

            --  LLM Context: actual tokens submitted to llama.cpp
            Prompt_Toks : constant Natural := Current_Prompt_Tokens;
            LLM_Pct     : constant Natural :=
              (if LLM_Ctx > 0 then (Prompt_Toks * 100) / LLM_Ctx else 0);
         begin
            Put_Line (AnsiAda.Foreground (AnsiAda.Light_Cyan) &
                      "[CtxMonitor]" & AnsiAda.Reset &
                      " === VIRTUAL CONTEXT STATUS (5s) ===");

            --  ELP Queue: request depth (synthetic 2^63 capacity)
            Put_Line (AnsiAda.Foreground (AnsiAda.Light_Cyan) &
                      "[CtxMonitor]" & AnsiAda.Reset &
                      " ELP Queue: " &
                      Long_Long_Integer'Image (VC_Depth) &
                      " /" &
                      Interfaces.Unsigned_64'Image (VC_Capacity) &
                      " pending (" &
                      Long_Long_Float'Image (VC_Pct) &
                      "% used)");

            --  Virtual Context: accumulated factual data (Internal_State)
            --  This is the data paged in from tool results across hops
            Put_Line (AnsiAda.Foreground (AnsiAda.Light_Cyan) &
                      "[CtxMonitor]" & AnsiAda.Reset &
                      " Virtual CTX: " &
                      Natural'Image (VC_Bytes) & " bytes / " &
                      Natural'Image (VC_Tokens) & " ~tokens" &
                      " (" & Natural'Image (VC_Ctx_Pct) &
                      "% of LLM window)");

            --  LLM Context: actual tokens in the prompt submitted to llama
            Put_Line (AnsiAda.Foreground (AnsiAda.Light_Cyan) &
                      "[CtxMonitor]" & AnsiAda.Reset &
                      " LLM CTX:    " &
                      Natural'Image (Prompt_Toks) & " / " &
                      Natural'Image (LLM_Ctx) & " tokens" &
                      " (" & Natural'Image (LLM_Pct) &
                      "% used)");

            --  Context Fault Division Page
            Put_Line (AnsiAda.Foreground (AnsiAda.Light_Cyan) &
                      "[CtxMonitor]" & AnsiAda.Reset &
                      " Context Fault Page: " &
                      Natural'Image (Cur_Division) & " /" &
                      Natural'Image (Max_Divisions) &
                      " | Hops=" & Natural'Image (Fault_Total) & "/5");

            --  Internal_State size + page jump state
            Put_Line (AnsiAda.Foreground (AnsiAda.Light_Cyan) &
                      "[CtxMonitor]" & AnsiAda.Reset &
                      " Internal_State=" &
                      Natural'Image (VC_Bytes) & " bytes" &
                      " | Page=" &
                      (if Fault_Total = 0 then "INITIAL"
                       else "HOP" & Natural'Image (Fault_Total)));

            Put_Line (AnsiAda.Foreground (AnsiAda.Light_Cyan) &
                      "[CtxMonitor]" & AnsiAda.Reset &
                      " ======================================");
         end;

         delay until Next_Check;
      end loop;
   end Context_Monitor;

   type Model_Record is record
      Model       : Llama_Model := Null_Model;
      Context     : Llama_Context := Null_Context;
      Mtmd_Ctx    : Mtmd_Interface.Mtmd_Context := Null_Mtmd_Context;
      Path        : Unbounded_String;
      Loaded      : Boolean := False;
      In_Use      : Boolean := False;
      Last_Used   : Time := Time_First;
      Current_Ctx : unsigned := 0;
   end record;

   Models : array (Model_Type) of Model_Record;

   type Model_Type_Refs is array (Model_Type) of aliased Model_Type;
   Model_Refs : constant Model_Type_Refs :=
     [Qwen_0_8B      => Qwen_0_8B,
      Qwen_9B        => Qwen_9B,
      Qwen_Embedding => Qwen_Embedding,
      MMProj         => MMProj];

   type Owner_Array is array (Model_Type) of ELP_Level;
   type Busy_Array is array (Model_Type) of Boolean;

   protected Metal_Lock_Object is
      entry Acquire;
      procedure Release;
   private
      Busy : Boolean := False;
   end Metal_Lock_Object;

   --  PRIORITY MODEL GATE:
   --  Manages access to the model contexts.
   --  ELP1 requests (User Interactions) preempt running ELP0 requests (Background Tasks).
   protected Priority_Model_Gate is
      procedure Request_ELP1;
      entry Acquire_ELP1 (Model_Type);
      procedure Release_ELP1 (Kind : Model_Type);
      entry Acquire_ELP0 (Model_Type) (Success : out Boolean);
      procedure Release_ELP0 (Kind : Model_Type);
      procedure Try_Acquire_For_Cleanup (Kind : Model_Type; Success : out Boolean);
      function Should_Abort return Boolean;
      function Is_ELP0_Owner (Kind : Model_Type) return Boolean;
      entry Wait_For_ELP1_Idle;
      procedure Set_Power_Condition (On_Battery : Boolean; Level : Natural);
   private
      ELP1_Pending      : Natural := 0;
      ELP1_Active_Count : Natural := 0;
      Busy              : Busy_Array := [others => False];
      Owner             : Owner_Array := [others => ELP0];
      On_Battery_State  : Boolean := False;
      Battery_Level     : Natural := 100;
   end Priority_Model_Gate;

   protected body Metal_Lock_Object is
      entry Acquire when not Busy is
      begin
         Busy := True;
      end Acquire;
      procedure Release is
      begin
         Busy := False;
      end Release;
   end Metal_Lock_Object;

   --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
   --  Init_Start_Time: Captured when Model_Manager.Initialize is called.
   --  All [Init-V] verbose prints in this package compute uptime relative
   --  to this timestamp.
   Init_Start_Time : Ada.Real_Time.Time;

   protected body Priority_Model_Gate is
      procedure Request_ELP1 is
      begin
         ELP1_Pending := ELP1_Pending + 1;
         Put_Line ("[ELP1-REQUEST] Pending ELP1 requests: " & ELP1_Pending'Img);
      end Request_ELP1;

      entry Acquire_ELP1 (for K in Model_Type) when not Busy (K) is
      begin
         ELP1_Pending := ELP1_Pending - 1;
         Busy (K) := True;
         Owner (K) := ELP1;
         ELP1_Active_Count := ELP1_Active_Count + 1;
         Put_Line ("[ELP1-ACQUIRED] " & K'Img & " | Active: " &
                   ELP1_Active_Count'Img & " | Pending: " & ELP1_Pending'Img);
      end Acquire_ELP1;

      procedure Release_ELP1 (Kind : Model_Type) is
      begin
         Busy (Kind) := False;
         Owner (Kind) := ELP0;
         if ELP1_Active_Count > 0 then
            ELP1_Active_Count := ELP1_Active_Count - 1;
         end if;
         Put_Line ("[ELP1-RELEASED] " & Kind'Img & " | Active: " &
                   ELP1_Active_Count'Img & " | Pending: " & ELP1_Pending'Img);
      end Release_ELP1;

      entry Acquire_ELP0 (for K in Model_Type) (Success : out Boolean)
        when (not Busy (K)
          or else ELP1_Pending > 0
          or else ELP1_Active_Count > 0)
          and then (not On_Battery_State or else Battery_Level >= 80) is
      begin
         if ELP1_Pending > 0 or else ELP1_Active_Count > 0 then
            Success := False;
            Put_Line ("[ELP0-DENIED] " & K'Img & " | ELP1 Pending: " &
                      ELP1_Pending'Img & " | ELP1 Active: " &
                      ELP1_Active_Count'Img);
         else
            Busy (K) := True;
            Owner (K) := ELP0;
            Success := True;
            Put_Line ("[ELP0-ACQUIRED] " & K'Img);
         end if;
      end Acquire_ELP0;

      procedure Release_ELP0 (Kind : Model_Type) is
      begin
         Busy (Kind) := False;
         Put_Line ("[ELP0-RELEASED] " & Kind'Img);
      end Release_ELP0;

      procedure Try_Acquire_For_Cleanup
        (Kind : Model_Type; Success : out Boolean) is
      begin
         if Busy (Kind) or else ELP1_Pending > 0 or else
           ELP1_Active_Count > 0
         then
            Success := False;
         else
            Busy (Kind) := True;
            Owner (Kind) := ELP1; -- Treat cleanup as high priority/exclusive
            Success := True;
         end if;
      end Try_Acquire_For_Cleanup;

      --  Returns True when an ELP1 (user) request is pending or active.
      --  Called from Llama_Abort_Callback (inside llama.cpp's decode loop)
      --  and from post-decode abort checks in Generate/Hybrid_Generate.
      --  The [ELP0-ABORT-CHECK] log fires from the callback — it is NORMAL
      --  to see many of these during active decode; it does NOT mean a
      --  deadlock.  The deadlock was caused by ELP0 tasks polling this
      --  instead of suspending on Wait_For_ELP1_Idle (now fixed).
      function Should_Abort return Boolean is
      begin
         return ELP1_Pending > 0 or else ELP1_Active_Count > 0
           or else (On_Battery_State and then Battery_Level < 80);
      end Should_Abort;

      function Is_ELP0_Owner (Kind : Model_Type) return Boolean is
      begin
         return Owner (Kind) = ELP0;
      end Is_ELP0_Owner;

      --  Barrier: ELP0 tasks block here until all ELP1 requests have completed.
      --  See Wait_For_ELP1_Idle spec in model_manager.ads for full explanation.
      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
      --  Verbose: prints guard state when an ELP0 task arrives.
      entry Wait_For_ELP1_Idle when (ELP1_Pending = 0 and then
        ELP1_Active_Count = 0)
        and then (not On_Battery_State or else Battery_Level >= 80) is
      begin
         Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Init-V]" &
                   AnsiAda.Reset & "+" & Trim (Duration'Image (Ada.Real_Time.To_Duration (Ada.Real_Time.Clock - Init_Start_Time)), Both) & "s Wait_For_ELP1_Idle GUARD PASSED" &
                   " ELP1_Pending=" & ELP1_Pending'Img &
                   " ELP1_Active=" & ELP1_Active_Count'Img &
                   " OnBattery=" & On_Battery_State'Img &
                   " BattLevel=" & Battery_Level'Img);
      end Wait_For_ELP1_Idle;

      procedure Set_Power_Condition (On_Battery : Boolean; Level : Natural) is
      begin
         On_Battery_State := On_Battery;
         Battery_Level := Level;
      end Set_Power_Condition;
   end Priority_Model_Gate;

   --  IDLE MONITOR:
   --  Unloads models after 30 seconds of inactivity to free VRAM.
   task Idle_Monitor is
      pragma Storage_Size (1024 * 1024);
      entry Start;
   end Idle_Monitor;

   task body Idle_Monitor is
      Next_Check : Time;
      Interval   : constant Time_Span := Seconds (1);
      Timeout    : constant Time_Span := Seconds (30);
      Now        : Time;
      Cleanup_OK : Boolean;
   begin
      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
      --  Verbose: confirms the Idle_Monitor task actually started.
      Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Init-V]" &
                AnsiAda.Reset & "+" & Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) & "s Idle_Monitor task entered, waiting for Start...");
      accept Start;
      Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Init-V]" &
                AnsiAda.Reset & "+" & Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) & "s Idle_Monitor task ACCEPTED Start, entering loop.");
      loop
         Next_Check := Clock + Interval;
         Now := Clock;
         for Kind in Model_Type loop
            --  [QUIRK-FIX] [macOS] Keep QWEN_0_8B permanently loaded to avoid
            --  ggml-metal GPU buffer race during Llama_Free/Unload_Model.
            --  The 0.8B model costs only ~0.5-0.6GB VRAM at Q4_K_S.
            --  [Linux/Android-Termux] REMOVE this exemption guard to unload
            --  QWEN_0_8B aggressively on memory-constrained devices.
            --  See QUIRK-M03 / QUIRK-S01 for crash details.
            if Kind = Qwen_0_8B then
               null;
            elsif Models (Kind).Loaded and then
              not Models (Kind).In_Use and then
              (Now - Models (Kind).Last_Used) > Timeout
            then
               Priority_Model_Gate.Try_Acquire_For_Cleanup (Kind, Cleanup_OK);
               if Cleanup_OK then
                  Put_Line (AnsiAda.Foreground (AnsiAda.Grey) & "[Idle]" &
                            AnsiAda.Reset & " Unloading " &
                            Model_Type'Image (Kind));
                  Unload_Model (Kind);
                  --  Match Acquire_For_Cleanup
                  Priority_Model_Gate.Release_ELP1 (Kind);
               end if;
            end if;
         end loop;
         delay until Next_Check;
      end loop;
   end Idle_Monitor;

   function Wrap_ChatML (Sys : String; Msg : String) return String is
   begin
      return "<|im_start|>system" & ASCII.LF & Sys & "<|im_end|>" & ASCII.LF &
             "<|im_start|>user" & ASCII.LF & Msg & "<|im_end|>" & ASCII.LF &
             "<|im_start|>assistant" & ASCII.LF;
   end Wrap_ChatML;

   Initialized : Boolean := False;

   procedure Initialize is
   begin
      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
      --  Guard: prevent double initialization (idle monitor blocks on 2nd Start).
      if Initialized then
         Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Init-V]" &
                   AnsiAda.Reset & " Model_Manager.Initialize: ALREADY INITIALIZED, skipping.");
         return;
      end if;
      Initialized := True;
      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
      --  Capture start time for uptime calculation.
      Init_Start_Time := Ada.Real_Time.Clock;
      --  Verbose init tracing: each print confirms a subsystem completed.
      --  If the server hangs during init, the LAST print before silence
      --  tells you exactly which step is stuck.
      Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Init-V]" &
                AnsiAda.Reset & "+" & Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) & "s 1/7 Calling Llama_Backend_Init...");
      --  [DO NOT REMOVE] Suppress llama.cpp stderr during backend init.
      --  load_backend, ggml_metal_device_init lines go to stderr.
      declare
         Saved_Stderr : constant int := Sys_Dup (2);
      begin
         Llama_Backend_Init;
         declare
            Dummy : int := Sys_Restore_Stderr (Saved_Stderr);
         begin
            null;
         end;
      end;
      Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Init-V]" &
                AnsiAda.Reset & "+" & Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) & "s 2/7 Llama_Backend_Init DONE.");

      Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Init-V]" &
                AnsiAda.Reset & "+" & Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) & "s 3/7 Calling Database_Manager.Initialize...");
      Database_Manager.Initialize;
      Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Init-V]" &
                AnsiAda.Reset & "+" & Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) & "s 4/7 Database_Manager.Initialize DONE.");

      Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Init-V]" &
                AnsiAda.Reset & "+" & Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) & "s 5/7 Calling ELP_Queue.Initialize...");
      ELP_Queue.Initialize;
      Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Init-V]" &
                AnsiAda.Reset & "+" & Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) & "s 6/7 ELP_Queue.Initialize DONE.");

      --  Start Virtual Context Monitor (prints every 5s)
      if not Context_Monitor'Terminated then
         Context_Monitor.Start;
      end if;

      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
      --  Model paths are set here.  None of these load models from disk.
      --  Loading happens lazily in Load_Model on first use.
      Models (Qwen_0_8B).Path  := To_Unbounded_String
        ("llama.cpp/models/qwen3.5/Qwen3.5-0.8B-Q4_K_S.gguf");
      Models (Qwen_9B).Path   := To_Unbounded_String
        ("llama.cpp/models/qwen3.5/Qwen3.5-9B-UD-Q2_K_XL.gguf");
      Models (Qwen_Embedding).Path := To_Unbounded_String
        ("llama.cpp/models/qwen3.5/Qwen3-Embedding-0.6B-Q8_0.gguf");
      Models (MMProj).Path := To_Unbounded_String
        ("llama.cpp/models/qwen3.5/mmproj-9B-F16.gguf");

      Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Init-V]" &
                AnsiAda.Reset & "+" & Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) & "s 7/7 Starting Idle_Monitor...");
      Idle_Monitor.Start;
      Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Init-V]" &
                AnsiAda.Reset & "+" & Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) & "s Model_Manager.Initialize COMPLETE.");
   end Initialize;

   procedure Load_Model
     (Kind          : Model_Type;
      Success       : out Boolean;
      Requested_Ctx : Positive := 4096;
      Level         : ELP_Level := ELP1)
   is
      M_Params   : Llama_Model_Params := Llama_Model_Default_Params;
      C_Params   : Llama_Context_Params := Llama_Context_Default_Params;
      Actual_Ctx : unsigned;

      Base_Path  : constant String := To_String (Models (Kind).Path);
      -- Try direct, ../ (from src/Adelaide_Lite), and ../../ (from bin)
      Paths      : constant array (1 .. 3) of Unbounded_String :=
        (To_Unbounded_String (Base_Path),
         To_Unbounded_String ("../" & Base_Path),
         To_Unbounded_String ("../../" & Base_Path));
   begin
      Actual_Ctx := unsigned (Requested_Ctx);
      --  Minimum context size is 8192 for stability and headroom.
      --  Smaller contexts (e.g., 4096) caused llama_decode assertion failures
      --  with Qwen3.5-9B at Q4_1 KV quantization on this hardware.
      if Actual_Ctx < 8192 then
         Actual_Ctx := 8192;
      end if;

      Success := False;
      if Models (Kind).Loaded then
         --  REUSE EXISTING CONTEXT: If the requested context size is <= the
         --  currently loaded context, we can reuse without reloading. This is
         --  critical for performance because each reload destroys the KV cache.
         --  The KV cache state (PromptCache) is lost on unload, so avoid
         --  unnecessary Unload_Model + Load_Model cycles.
         --
         --  QUIRK: llama_context is extremely expensive to create/destroy
         --  (~2s for Qwen3.5-9B).  Reusing an already-loaded context with
         --  sufficient capacity saves this cost but means the KV cache from
         --  the previous inference is preserved until the next decode clears
         --  it with Llama_Memory_Clear.
         if Actual_Ctx <= Models (Kind).Current_Ctx then
            Models (Kind).Last_Used := Clock;
            Success := True;
            return;
         end if;
         Unload_Model (Kind);
      end if;

      Put_Line ("[+] Loading " & Model_Type'Image (Kind) &
                " (N_CTX=" & Actual_Ctx'Img & ")");

      --  SPECIAL CASE: MMProj (multimodal projection) model loading
      --  Why: MMProj is not a standalone llama model - it's a vision encoder
      --       that must be initialized with mtmd_init_from_file_safe, which
      --       requires the text model (Qwen_9B) to be loaded first.
      --       The mmproj file contains the CLIP vision encoder weights that
      --       project images into the embedding space of the text model.
      if Kind = MMProj then
         --  MMProj requires the text model to be loaded first
         if not Models (Qwen_9B).Loaded then
            Put_Line ("[!] MMProj requires Qwen_9B to be loaded first");
            Success := False;
            return;
         end if;

         --  Try to find and load the mmproj file
         for I in Paths'Range loop
            declare
               Path_Str : constant String := To_String (Paths (I));
            begin
               if Ada.Directories.Exists (Path_Str) then
                  declare
                     Path_C : chars_ptr := New_String (Path_Str);
                  begin
                     begin
                        --  Load mmproj using mtmd API
                        --  Use GPU if available, 8 threads for vision encoding
                        Models (Kind).Mtmd_Ctx :=
                          Mtmd_Init_From_File_Safe
                            (Path_C,
                             System.Address (Models (Qwen_9B).Model),
                             True,
                             8);
                     exception
                        when others =>
                           Put_Line ("[!] Exception caught in Ada during " &
                                     "Mtmd_Init_From_File_Safe");
                           Models (Kind).Mtmd_Ctx := Null_Mtmd_Context;
                     end;
                     Free (Path_C);
                     if Models (Kind).Mtmd_Ctx /= Null_Mtmd_Context then
                        exit;
                     end if;
                  end;
               end if;
            end;
         end loop;

         if Models (Kind).Mtmd_Ctx /= Null_Mtmd_Context then
            Models (Kind).Loaded := True;
            Models (Kind).Last_Used := Clock;
            Success := True;
            Put_Line ("[+] MMProj loaded successfully");
         else
            Put_Line ("[!] Failed to load MMProj model");
         end if;
         return;
      end if;

      --  [QUIRK-M10] Embedding Model Crash — Raw Content Feeding
      --  ======================================================================
      --  [VITAL-DO-NOT-REMOVE]
      --
      --  SYMPTOM:
      --    MTLCommandBufferStatus-Error (Code 5) followed by SIGTRAP (exit -5).
      --    Occurs during ELP0 background indexing when the embedding model
      --    (Qwen3-Embedding-0.6B-Q8_0) processes raw CSS/HTML content.
      --
      --  CRASH LOG EXAMPLE:
      --    [Embedding-Debug] Input ( 800 chars): ligraphic;font-style:normal;...
      --    [FATAL] GPU Backend Error (Code: 5)
      --    [+] Waiting for GPU driver cooldown (2s)...
      --    Exit Code: -5 | Signal: SIGTRAP (5)
      --
      --  ROOT CAUSE (CONFIRMED):
      --    The embedding model is designed for natural language text.
      --    The Native_Crawl_Task in knowledge_manager.adb crawls the filesystem
      --    and feeds raw file content to Get_Embedding in 800-char chunks.
      --    When it encounters CSS files (or inline CSS in HTML/JS), it feeds
      --    content like:
      --      "ligraphic;font-style:normal;font-weight:400;src:url(/asset..."
      --      "x;overflow-y:auto}.js-error-popup h4{color:#f55;margin:0 0..."
      --      "e;box-shadow:0 0 40px #40e0d080,inset 0 0 20px #40e0d04d}..."
      --    directly to the tokenizer and llama_decode.
      --
      --    This is NOT a GPU vs CPU issue — the crash occurs on both paths.
      --    The raw CSS/HTML produces unusual token sequences that trigger
      --    edge cases in ggml-metal kernels AND can cause issues on CPU.
      --
      --  WHY SANITIZE_UTF8 DOESN'T HELP:
      --    Sanitize_UTF8 (line 915) only strips control chars and non-ASCII.
      --    CSS syntax characters ({, }, :, ;, #, @, /, etc.) are all valid
      --    printable ASCII (32-126) and pass through unchanged.
      --
      --  WHY CRAWL_DIRECTORY DOESN'T FILTER:
      --    Crawl_Directory (knowledge_manager.adb:302) checks file extensions:
      --      .adb, .ads, .c, .h, .txt, .md
      --    But CSS content can still enter via:
      --      1. Inline CSS in .html/.md files
      --      2. .css.js or .css.md files (extension substring match)
      --      3. Future file types not yet in the skip list
      --
      --  IMPACT:
      --    - ELP0 background indexing crashes and stops entirely
      --    - Server exits with code -5 (SIGTRAP)
      --    - Watchdog restarts the server, but indexing crashes again
      --    - Effectively: no background knowledge indexing works
      --
      --  GPU STATUS:
      --    N_Gpu_Layers := -1 (GPU enabled). The crash is input-related,
      --    not GPU-related. Once content filtering is fixed, GPU will work.
      --    If crashes persist after fixing input, set N_Gpu_Layers := 0
      --    as a temporary workaround and investigate further.
      --
      --  REAL FIX (REQUIRED):
      --    Two fixes needed in knowledge_manager.adb:
      --
      --    FIX 1: Crawl_Directory must SKIP non-text files:
      --      Add to the skip list (line 302-307):
      --        .css, .js, .jsx, .ts, .tsx, .html, .htm, .svg, .json,
      --        .xml, .yaml, .yml, .toml, .lock, .min, .map, .gz
      --      Or better: use a whitelist of ONLY natural language files:
      --        .adb, .ads, .c, .h, .txt, .md, .rst, .org, .bib
      --
      --    FIX 2: Get_Embedding should reject code-like content:
      --      Before tokenizing, check if the input contains high densities
      --      of CSS/code patterns ({}, ;, : , @, /, etc.). If so, skip it.
      --      This is a safety net for any code path that feeds bad content.
      --
      --  ERROR HANDLER:
      --    Line 1178: If Llama_Decode returns non-zero, the error is caught
      --    gracefully — logs the error, waits 2s for GPU driver cooldown,
      --    unloads the model, and returns length=0. The caller continues
      --    without crashing the server (but indexing for that chunk is lost).
      --
      --  HISTORY:
      --    - 2026-06-10: QUIRK-M10 created. Embedding forced to CPU-only
      --      (N_Gpu_Layers := 0) to avoid Metal crashes. Performance: ~4s/chunk.
      --    - 2026-06-13: Root cause identified — raw CSS/HTML content feeding.
      --      GPU re-enabled (N_Gpu_Layers := -1) since crash is input-related.
      --      Full debug print added to show untruncated input. Content filtering
      --      fix pending in knowledge_manager.adb.
      --  ======================================================================
      --  EMBEDDING GPU STRATEGY (see QUIRK-M10):
      --  ELP0 (background indexing): CPU-only — Metal kernel compilation
      --    crashes with SIGTRAP that Kratos cannot catch (happens in Metal's
      --    shader compiler thread). Background indexing processes hundreds of
      --    chunks, so stability matters more than speed.
      --  ELP1 (user-facing RAG): GPU — fast response for user queries.
      --    If it crashes, the error handler skips the batch and continues.
      --    Single-user requests are less likely to trigger the Metal crash.
      --  ======================================================================
      if Kind = Qwen_Embedding then
         if Level = ELP0 then
            M_Params.N_Gpu_Layers := 0;   -- CPU-only for background indexing
         else
            M_Params.N_Gpu_Layers := -1;  -- GPU for user-facing requests
         end if;
      else
         M_Params.N_Gpu_Layers := -1;     -- GPU for all other models
      end if;

      --  TRY THREE PATHS FOR MODEL FILES
      --  The CWD at runtime is unpredictable:
      --    1. Direct path (when run from project root or Adadelaide_Lite/)
      --    2. ../ prefixed (when CWD is src/)
      --    3. ../../ prefixed (when CWD is bin/)
      --  This fallback loop handles all common launch configurations
      --  without requiring a fixed working directory.
      for I in Paths'Range loop
         declare
            Path_Str : constant String := To_String (Paths (I));
         begin
            if Ada.Directories.Exists (Path_Str) then
               declare
                  Path_C : chars_ptr := New_String (Path_Str);
               begin
                  --  [DO NOT REMOVE] Suppress llama.cpp stderr during model load.
                  --  Hundreds of create_tensor/repack/print_info lines go to stderr.
                  declare
                     Saved_Stderr : constant int := Sys_Dup (2);
                  begin
                     begin
                        Models (Kind).Model :=
                          Llama_Model_Load_From_File (Path_C, M_Params);
                     exception
                        when others =>
                           Put_Line ("[!] Exception caught in Ada during " &
                                     "Llama_Model_Load_From_File");
                           Models (Kind).Model := Null_Model;
                     end;
                     --  Restore stderr after model load
                     declare
                        Dummy : int := Sys_Restore_Stderr (Saved_Stderr);
                     begin
                        null;
                     end;
                  end;
                  Free (Path_C);
                  if Models (Kind).Model /= Null_Model then
                     exit;
                  end if;
               end;
            end if;
         end;
      end loop;

      if Models (Kind).Model /= Null_Model then
         C_Params.N_Ctx := Actual_Ctx;
         C_Params.N_Batch := 512;
         C_Params.N_Ubatch := 512;
         C_Params.N_Threads := 8;
         C_Params.N_Threads_Batch := 8;
         --  [DO NOT REMOVE] Q4_1 KV cache: 4-bit quantized KV cache saves
         --  ~75% memory vs F16.  On 16GB M2 Pro, this is the difference
         --  between fitting Qwen3.5-9B + 8192 ctx and OOM SIGTERM.
         --  Quality loss is minimal for KV cache (activations, not weights).
         C_Params.Type_K := GGML_TYPE_Q4_1;
         C_Params.Type_V := GGML_TYPE_Q4_1;

         --  Flash attention MUST be enabled for Q4_1 KV cache.
         --  llama.cpp: "V cache quantization requires flash_attn"
         --  Value 1 = flash_attn enabled (non-causal not needed for LLM).
         C_Params.Flash_Attn_Type := 1;

         C_Params.Abort_Callback := Llama_Abort_Callback'Address;
         C_Params.Abort_Callback_Data := Model_Refs (Kind)'Address;
         --  [DO NOT REMOVE] Suppress llama.cpp stderr during context init.
         --  llama_context, llama_kv_cache, ggml_metal lines go to stderr.
         declare
            Saved_Stderr : constant int := Sys_Dup (2);
         begin
            Models (Kind).Context :=
              Llama_Init_From_Model (Models (Kind).Model, C_Params);
            declare
               Dummy : int := Sys_Restore_Stderr (Saved_Stderr);
            begin
               null;
            end;
         end;
         if Models (Kind).Context /= Null_Context then
            Models (Kind).Loaded := True;
            Models (Kind).Last_Used := Clock;
            Models (Kind).Current_Ctx := Actual_Ctx;
            Success := True;
         else
            Llama_Model_Free (Models (Kind).Model);
            Models (Kind).Model := Null_Model;
         end if;
      end if;
   end Load_Model;

   procedure Unload_Model (Kind : Model_Type) is
   begin
      if Models (Kind).Loaded then
         --  SPECIAL CASE: MMProj uses mtmd context, not llama context
         if Kind = MMProj then
            if Models (Kind).Mtmd_Ctx /= Null_Mtmd_Context then
               Mtmd_Free_Safe (Models (Kind).Mtmd_Ctx);
               Models (Kind).Mtmd_Ctx := Null_Mtmd_Context;
            end if;
         else
            Llama_Free (Models (Kind).Context);
            Llama_Model_Free (Models (Kind).Model);
            Models (Kind).Context := Null_Context;
            Models (Kind).Model := Null_Model;
         end if;
         Models (Kind).Loaded := False;
         Models (Kind).Current_Ctx := 0;
      end if;
   end Unload_Model;

   procedure Force_Unload_And_Reload (Kind : Model_Type) is
      Success : Boolean;
   begin
      Unload_Model (Kind);
      Load_Model (Kind, Success);
   end Force_Unload_And_Reload;

   function Get_Context
     (Kind : Model_Type) return Llama_Interface.Llama_Context is
   begin
      if Models (Kind).Loaded then
         Models (Kind).Last_Used := Clock;
      end if;
      return Models (Kind).Context;
   end Get_Context;

   function Get_Model
     (Kind : Model_Type) return Llama_Interface.Llama_Model is
   begin
      if Models (Kind).Loaded then
         Models (Kind).Last_Used := Clock;
      end if;
      return Models (Kind).Model;
   end Get_Model;

   --  Get the mtmd (multimodal) context for vision processing
   --  Why: MMProj is a special model type that uses the mtmd API for
   --       image/audio encoding. This function returns the mtmd context
   --       so other modules can encode images for the vision pipeline.
   function Get_Mtmd_Context
     (Kind : Model_Type) return Mtmd_Interface.Mtmd_Context is
   begin
      if Models (Kind).Loaded then
         Models (Kind).Last_Used := Clock;
      end if;
      return Models (Kind).Mtmd_Ctx;
   end Get_Mtmd_Context;

   --  LLAMA.CPP ABORT CALLBACK:
   --  Called by llama.cpp periodically during Llama_Decode (token generation).
   --  Returns True to signal llama.cpp to abort the current decode operation.
   --  NOTE: The [ELP0-ABORT-CHECK] log fires from inside Should_Abort here.
   --  Seeing many of these is NORMAL during active decode — it means the abort
   --  callback is being invoked but the decode has not yet returned.  This is
   --  NOT a deadlock.  The deadlock was caused by ELP0 tasks polling instead
   --  of suspending on Wait_For_ELP1_Idle (now fixed).
   function Llama_Abort_Callback (Data : System.Address) return Boolean is
      use System;
      type Model_Type_Ptr is access all Model_Type;
      function To_Ptr is new Ada.Unchecked_Conversion
        (System.Address, Model_Type_Ptr);
      Ptr : Model_Type_Ptr;
   begin
      if Data = System.Null_Address then
         return False;
      end if;
      Ptr := To_Ptr (Data);

      --  1. Abort if Watchdog has flagged a timeout for this model.
      if Watchdog_Manager.Inference_Monitor.Is_Aborted and then
        Watchdog_Manager.Inference_Monitor.Current_Inference_Model = Ptr.all
      then
         return True;
      end if;

      --  2. Only abort if we are an ELP0 task and an ELP1 task is pending.
      return Priority_Model_Gate.Is_ELP0_Owner (Ptr.all)
        and then Priority_Model_Gate.Should_Abort;
   end Llama_Abort_Callback;

   function Should_Abort_ELP0 return Boolean is
   begin
      return Priority_Model_Gate.Should_Abort;
   end Should_Abort_ELP0;

   procedure Wait_For_ELP1_Idle is
   begin
      Priority_Model_Gate.Wait_For_ELP1_Idle;
   end Wait_For_ELP1_Idle;

   procedure Acquire_Metal_Lock is
   begin
      Metal_Lock_Object.Acquire;
   end Acquire_Metal_Lock;

   procedure Release_Metal_Lock is
   begin
      Metal_Lock_Object.Release;
   end Release_Metal_Lock;

   procedure Set_Power_Condition (On_Battery : Boolean; Level : Natural) is
   begin
      Priority_Model_Gate.Set_Power_Condition (On_Battery, Level);
   end Set_Power_Condition;

   function Get_Kind_For_Model_Name (Name : String) return Model_Type is
   begin
      if Name = "adelaide-hybrid"
        or else Name = "qwen3.5:4b"
        or else Name = "metamodel"
        or else Name = "adelaide-metamodel"
        or else Name = "Snowball-Enaga"
      then
         return Qwen_9B;
      elsif Name = "qwen-embedding" or else Name = "adelaide-embedding" then
         return Qwen_Embedding;
      else
         return Qwen_0_8B;
      end if;
   end Get_Kind_For_Model_Name;

   function Is_Loaded (Kind : Model_Type) return Boolean is
   begin
      return Models (Kind).Loaded;
   end Is_Loaded;

   function Count_Tokens (Text : String) return Positive is
   begin
      return Text'Length / 4 + 1;
   end Count_Tokens;

   function Get_Request_Category
     (Msg        : String;
      Session_ID : String := "";
      Level      : ELP_Level := ELP1) return String
   is
      pragma Unreferenced (Session_ID, Level);
   begin
      if Index (Msg, "code") > 0 or else Index (Msg, "program") > 0 then
         return "Technical";
      else
         return "General";
      end if;
   end Get_Request_Category;

   function Grade_Response_Quality
     (Response_Text : String;
      Prompt        : String;
      Search_Used   : Boolean;
      Has_Citations : Boolean;
      Session_ID    : String := "";
      Level         : ELP_Level := ELP1) return Natural
   is
      pragma Unreferenced (Response_Text, Prompt, Session_ID, Level);
      Score : Natural := 5;
   begin
      if Search_Used then
         Score := Score + 2;
      end if;
      if Has_Citations then
         Score := Score + 3;
      end if;
      return Score;
   end Grade_Response_Quality;

   procedure Push_Chunk
     (Stream     : Streaming_Queue.Queue_Access;
      Session_ID : String;
      Str_Piece  : String)
   is
      pragma Unreferenced (Session_ID);
   begin
      if Stream /= null then
         Ada.Text_IO.Put_Line
           ("Push_Chunk called with: " &
            Str_Piece (Str_Piece'First ..
              Natural'Min (Str_Piece'Last, Str_Piece'First + 20)));
         Stream.Push (Str_Piece);
      end if;
   end Push_Chunk;

   function Generator_Callback (Prompt : String) return String is
   begin
      return "Callback response to " & Prompt;
   end Generator_Callback;

   function Sanitize_UTF8 (S : String) return String is
      Res : Unbounded_String;
      Val : Natural;
   begin
      for I in S'Range loop
         Val := Character'Pos (S (I));
         --  Keep only: \t (9), \n (10), \r (13), and printable ASCII (32-126)
         --  Strip control chars, DEL (127), and all non-ASCII (128+)
         if Val = 9 or else Val = 10 or else Val = 13 or else
           (Val >= 32 and Val <= 126)
         then
            Append (Res, S (I));
         end if;
      end loop;
      return To_String (Res);
   end Sanitize_UTF8;

   function Sanitize_Orchestration_Output (S : String) return String is
      Res : Unbounded_String;
      I   : Positive := S'First;
   begin
      while I <= S'Last loop
         if I + 7 <= S'Last and then S (I .. I + 7) = "</think>" then
            --  Remove </think> to prevent premature termination
            I := I + 8;
         elsif I + 10 <= S'Last and then S (I .. I + 10) = "</thinking>" then
            I := I + 11;
         else
            Append (Res, S (I));
            I := I + 1;
         end if;
      end loop;
      return To_String (Res);
   end Sanitize_Orchestration_Output;

   --  STRIP_BASE64_IMAGES: Removes base64-encoded image data from tool output
   --  to prevent tokenization failures when feeding results back to the router.
   --  The router (9B model) cannot handle massive base64 blobs.
   --  User-facing stream still receives the full output with images.
   --  Pattern: ![...](data:image/...;base64,...) and ![...](<base64_blob>)
   function Strip_Base64_Images (S : String) return String is
      Res : Unbounded_String;
      I   : Positive := S'First;
   begin
      while I <= S'Last loop
         --  Check for markdown image syntax: ![...](...)
         if I + 1 <= S'Last and then S (I) = '!' and then S (I + 1) = '[' then
            --  Find the closing ')' of the image tag
            declare
               Close_Bracket : Natural := 0;
               Close_Paren   : Natural := 0;
               J             : Natural := I + 2;
            begin
               --  Find ](
               while J <= S'Last - 1 loop
                  if S (J) = ']' and then S (J + 1) = '(' then
                     Close_Bracket := J;
                     exit;
                  end if;
                  J := J + 1;
               end loop;

               if Close_Bracket > 0 then
                  --  Find matching )
                  J := Close_Bracket + 2;
                  declare
                     Depth : Natural := 1;
                  begin
                     while J <= S'Last and then Depth > 0 loop
                        if S (J) = '(' then
                           Depth := Depth + 1;
                        elsif S (J) = ')' then
                           Depth := Depth - 1;
                        end if;
                        if Depth > 0 then
                           J := J + 1;
                        end if;
                     end loop;
                  end;

                  if J <= S'Last then
                     Close_Paren := J;
                     --  Check if content is base64 (contains 'base64' or
                     --  is a long string without spaces — base64 has no spaces)
                     declare
                        Content : constant String :=
                          S (Close_Bracket + 2 .. Close_Paren - 1);
                        Has_Base64_Marker : constant Boolean :=
                          Index (Content, "base64") > 0;
                        Is_Long_No_Space  : constant Boolean :=
                          Content'Length > 200 and then
                          Index (Content, " ") = 0;
                     begin
                        if Has_Base64_Marker or else Is_Long_No_Space then
                           --  Skip entire image tag, replace with placeholder
                           Append (Res, "[IMAGE_REMOVED]");
                           I := Close_Paren + 1;
                        else
                           --  Not base64, keep as-is
                           Append (Res, S (I .. Close_Paren));
                           I := Close_Paren + 1;
                        end if;
                     end;
                  else
                     --  Unclosed, keep as-is
                     Append (Res, S (I));
                     I := I + 1;
                  end if;
               else
                  --  No ]( found, keep as-is
                  Append (Res, S (I));
                  I := I + 1;
               end if;
            end;
         else
            Append (Res, S (I));
            I := I + 1;
         end if;
      end loop;
      --  [VITAL-DO-NOT-REMOVE] Report base64 stripping
      if Length (Res) < S'Length then
         Put_Line (AnsiAda.Foreground (AnsiAda.Yellow) &
                   "[StripBase64-V]" & AnsiAda.Reset &
                   " Stripped base64 images. Input=" &
                   Natural'Image (S'Length) & " Output=" &
                   Natural'Image (Length (Res)) & " Saved=" &
                   Natural'Image (S'Length - Length (Res)) & " bytes");
      end if;
      return To_String (Res);
   end Strip_Base64_Images;

   --  SINGLE EMBEDDING HELPER
   procedure Get_Single_Embedding
     (Prompt : String;
      Result : out Math_Utils.Vector;
      Length : out Natural;
      Level  : ELP_Level := ELP1)
   is
      Success  : Boolean;
      Kind     : constant Model_Type := Qwen_Embedding;
      Vocab    : Llama_Vocab;
      Tokens   : Token_Array_Access;
      N_Toks   : int;
      Clean_P  : constant String := Sanitize_UTF8 (Prompt);
      Prompt_C : chars_ptr := New_String (Clean_P);

      --  Identify source for descriptive logging
      Source   : constant String :=
        (if Level = ELP0 then "Knowledge-Index" else "User-RAG");
   begin
      --  [VITAL-DO-NOT-REMOVE] Mandated by user.
      --  --[Debug] DO NOT REMOVE or truncate: Critical for diagnosing Tokenization
      --  and GPU Metal kernel crashes. We print the FULL input so we can see
      --  exactly what raw content (CSS, HTML, special chars) is being fed to
      --  the tokenizer. Truncating hides the problematic characters that cause
      --  MTLCommandBufferStatus-Error (Code 5).
      --
      --  ROOT CAUSE HYPOTHESIS: The embedding model (Qwen3-Embedding-0.6B) is
      --  designed for natural language text. Feeding it raw CSS/HTML code like
      --  "ligraphic;font-style:normal;font-weight:400;src:url(/asset..." may
      --  trigger edge cases in Metal compute kernels because:
      --    1. CSS has dense special chars (: ; { } # @ /) that tokenize oddly
      --    2. The tokenizer may produce unusual token ID sequences for code
      --    3. These sequences could hit untested paths in ggml-metal kernels
      --  FIX: The caller (Knowledge_Manager) should strip/filter non-text content
      --  before chunking. See knowledge_manager.adb Native_Crawl_Task.
      Put_Line ("[Embedding-Debug] Input (" & Clean_P'Length'Img &
                " chars): " & Clean_P);
      Flush;
      --  --[Debug] DO NOT REMOVE: Descriptive source tracking
      ELP_Queue.Enqueue (Level, Kind, Source);
      if Level = ELP0 then
         Priority_Model_Gate.Acquire_ELP0 (Kind) (Success);
         if not Success then
            Put_Line ("[ELP0-BLOCKED] " & Kind'Img &
                      " | ELP1 is active or pending");
            ELP_Queue.Dequeue_Level (Level);
            Length := 0;
            Free (Prompt_C);
            return;
         end if;
      else
         Priority_Model_Gate.Request_ELP1;
         Priority_Model_Gate.Acquire_ELP1 (Kind);
      end if;

      Load_Model (Kind, Success, 1024, Level);
      if not Success then
         if Level = ELP0 then
            Priority_Model_Gate.Release_ELP0 (Kind);
         else
            Priority_Model_Gate.Release_ELP1 (Kind);
         end if;
         ELP_Queue.Dequeue_Level (Level);
         Length := 0;
         Free (Prompt_C);
         return;
      end if;

      Models (Kind).In_Use := True;
      Models (Kind).Last_Used := Clock;

      --  Allocate token array based on actual context size
      Tokens := new Token_Array (1 .. 4096);

      Vocab := Llama_Model_Get_Vocab (Models (Kind).Model);
      Acquire_Metal_Lock;
      if Kratos.Guard_Enter = 0 then
         N_Toks := Llama_Tokenize
           (Vocab, Prompt_C, int (Clean_P'Length), Tokens.all'Address,
            4096, True, True);
         Kratos.Guard_Exit;
      else
         Kratos.Log_Crash;
         N_Toks := -1;
      end if;
      Release_Metal_Lock;

      Put_Line ("[Tokenize-Debug] Model:" & Kind'Img &
                " Prompt_Len:" & Clean_P'Length'Img &
                " N_Toks:" & N_Toks'Img);
      Free (Prompt_C);

      if N_Toks <= 0 then
         Free_Tokens (Tokens);
         Models (Kind).In_Use := False;
         if Level = ELP0 then
            Priority_Model_Gate.Release_ELP0 (Kind);
         else
            Priority_Model_Gate.Release_ELP1 (Kind);
         end if;
         ELP_Queue.Dequeue_Level (Level);
         Length := 0;
         return;
      end if;

      --  CHUNKED DECODING FOR EMBEDDINGS
      --  ============================================================================
      --  ROOT CAUSE ANALYSIS (QUIRK-M10):
      --  The embedding model (Qwen3-Embedding-0.6B) processes tokens in batches
      --  of up to 256 tokens. Each batch calls Llama_Decode which dispatches to
      --  ggml-metal kernels on macOS. The Metal backend compiles kernel variants
      --  on-the-fly based on token count and quantization format (Q8_0).
      --
      --  THE BUG: After several successful decode calls, Metal fails to compile
      --  a kernel variant and returns GGML_STATUS_FAILED (Code 5). This happens
      --  because:
      --    1. Different N_Toks values produce different kernel configurations
      --       (nsg, nxpsg, ne12, r2, r3 parameters vary per batch)
      --    2. Metal's shader cache has limited capacity for compiled variants
      --    3. When the cache is full or a specific config is invalid, compilation
      --       fails and llama_decode returns Code 5
      --    4. The crash is NOT about the input content (CSS vs natural language)
      --       — it's about Metal kernel compilation limits
      --
      --  WHY UNLOADING WAS WRONG:
      --  The old code unloaded the model on ANY decode failure. This caused:
      --    - Next chunk reloads the model (expensive: ~2s for context creation)
      --    - New Metal context hits the same kernel compilation failure
      --    - Infinite crash-reload loop until server dies
      --
      --  FIX: Skip the failed batch and continue. The next batch with a different
      --  token count may use a different kernel variant that compiles successfully.
      --  Only unload after 3 consecutive failures (all kernels failing = real issue).
      --  ============================================================================
      declare
         Batch_Size  : constant int :=
           int'Min (256, int (Models (Kind).Current_Ctx));
         Current_Pos : int := 0;
         Tokens_Left : int := N_Toks;
         Consecutive_Failures : Natural := 0;  -- Track consecutive decode failures
         Max_Consecutive     : constant := 3;   -- Unload after 3 failures in a row
      begin
         Llama_Interface.Llama_Memory_Clear
           (Llama_Interface.Llama_Get_Memory (Models (Kind).Context), False);
         Llama_Set_Embeddings (Models (Kind).Context, Interfaces.C.int (1));

         while Tokens_Left > 0 loop
            declare
               To_Decode : constant int :=
                 (if Tokens_Left > Batch_Size then Batch_Size
                  else Tokens_Left);
               B : constant Llama_Batch :=
                 Llama_Batch_Get_One
                   (Tokens.all (Integer (Current_Pos) + 1)'Address, To_Decode);
               Dec_Result : int;
            begin
               --  KRATOS CRASH GUARD: llama_decode is wrapped in
               --  Guard_Enter/Guard_Exit. See QUIRK-M01.
               Acquire_Metal_Lock;
               if Kratos.Guard_Enter = 0 then
                  Dec_Result := Llama_Decode (Models (Kind).Context, B);
                  Kratos.Guard_Exit;
               else
                  Kratos.Log_Crash;
                  Dec_Result := -1;
               end if;
               Release_Metal_Lock;

               if Dec_Result /= 0 then
                  --  DECODE FAILED: Skip this batch, don't unload the model.
                  --  The failure is likely a Metal kernel compilation error for
                  --  this specific batch size/token count. The next batch may
                  --  use a different configuration that compiles successfully.
                  Consecutive_Failures := Consecutive_Failures + 1;
                  Put_Line (AnsiAda.Foreground (AnsiAda.Yellow) &
                            "[WARN] Llama_Decode failed (Code:" &
                            Dec_Result'Img & ") Batch:" &
                            To_Decode'Img & " Consecutive:" &
                            Consecutive_Failures'Img & AnsiAda.Reset);

                  if Consecutive_Failures >= Max_Consecutive then
                     --  3 consecutive failures = all kernel variants failing.
                     --  This is a real issue, not a transient compilation error.
                     --  Unload and let the caller decide what to do.
                     Put_Line (AnsiAda.Foreground (AnsiAda.Red) &
                               "[FATAL] " & Max_Consecutive'Img &
                               " consecutive decode failures. " &
                               "Unloading model." & AnsiAda.Reset);
                     delay 1.0;  -- Brief cooldown for GPU driver
                     Unload_Model (Kind);
                     Free_Tokens (Tokens);
                     Models (Kind).In_Use := False;
                     if Level = ELP0 then
                        Priority_Model_Gate.Release_ELP0 (Kind);
                     else
                        Priority_Model_Gate.Release_ELP1 (Kind);
                     end if;
                     ELP_Queue.Dequeue_Level (Level);
                     Length := 0;
                     return;
                  end if;

                  --  Skip this batch: advance past the failed tokens and continue.
                  --  The embedding result will be incomplete for this chunk, but
                  --  the model stays loaded for the next batch.
                  Tokens_Left := Tokens_Left - To_Decode;
                  Current_Pos := Current_Pos + To_Decode;
               else
                  --  DECODE SUCCEEDED: Reset consecutive failure counter.
                  Consecutive_Failures := 0;
                  Tokens_Left := Tokens_Left - To_Decode;
                  Current_Pos := Current_Pos + To_Decode;
               end if;
            end;
         end loop;
      end;

      declare
         use System;
         function Llama_Model_N_Embd (M : Llama_Model) return int;
         pragma Import (C, Llama_Model_N_Embd, "llama_model_n_embd");
         Dim : constant int := Llama_Model_N_Embd (Models (Kind).Model);
         Ptr : Address;
         --  SAFE: copy via C memcpy instead of Ada address overlay
         function Memcpy (Dst, Src : Address; N : Interfaces.C.size_t)
           return Address;
         pragma Import (C, Memcpy, "memcpy");
         Copy_Count : constant Integer :=
           Integer (Interfaces.C.size_t'Min
             (Interfaces.C.size_t (Dim),
              Interfaces.C.size_t (Result'Length)));
      begin
         Acquire_Metal_Lock;
         Ptr := Llama_Get_Embeddings (Models (Kind).Context);
         Release_Metal_Lock;

         if Copy_Count > 0 and then Ptr /= Null_Address then
            declare
               Dummy : Address;
            begin
               Dummy := Memcpy (Result (Result'First)'Address, Ptr,
                         Interfaces.C.size_t (Copy_Count) *
                           Interfaces.C.size_t (Float'Size / 8));
            end;
            Length := Copy_Count;
         else
            Length := 0;
         end if;
         Free_Tokens (Tokens);
         Models (Kind).In_Use := False;
         if Level = ELP0 then
            Priority_Model_Gate.Release_ELP0 (Kind);
         else
            Priority_Model_Gate.Release_ELP1 (Kind);
         end if;
         ELP_Queue.Dequeue_Level (Level);
      end;
   exception
      when others =>
         if Tokens /= null then
            Free_Tokens (Tokens);
         end if;
         Models (Kind).In_Use := False;
         if Level = ELP0 then
            Priority_Model_Gate.Release_ELP0 (Kind);
         else
            Priority_Model_Gate.Release_ELP1 (Kind);
         end if;
         ELP_Queue.Dequeue_Level (Level);
         Length := 0;
   end Get_Single_Embedding;
    --  GET EMBEDDING (WITH CHUNKING > 800 CHARS)

   procedure Get_Embedding
     (Prompt : String;
      Result : out Math_Utils.Vector;
      Length : out Natural;
      Level  : ELP_Level := ELP1)
   is
   begin
      if Prompt'Length <= 800 then
         Get_Single_Embedding (Prompt, Result, Length, Level);
      else
         declare
            Num_Chunks : Natural := 0;
            Sum_Vec    : Math_Utils.Vector (Result'Range) := [others => 0.0];
            Dim        : Natural := 0;
            Start_Idx  : Positive := Prompt'First;
            End_Idx    : Positive;
         begin
            while Start_Idx <= Prompt'Last loop
               End_Idx := Start_Idx + 800 - 1;
               if End_Idx > Prompt'Last then
                  End_Idx := Prompt'Last;
               end if;
               declare
                  Sub_Prompt : constant String :=
                    Prompt (Start_Idx .. End_Idx);
                  Sub_Vec    : Math_Utils.Vector (Result'Range) :=
                    [others => 0.0];
                  Sub_Len    : Natural := 0;
               begin
                  Get_Single_Embedding (Sub_Prompt, Sub_Vec, Sub_Len, Level);
                  if Sub_Len > 0 then
                     if Num_Chunks = 0 then
                        Dim := Sub_Len;
                     end if;
                     for I in 1 .. Dim loop
                        Sum_Vec (Result'First + I - 1) :=
                          Sum_Vec (Result'First + I - 1) +
                          Sub_Vec (Sub_Vec'First + I - 1);
                     end loop;
                     Num_Chunks := Num_Chunks + 1;
                  end if;
               end;
               Start_Idx := End_Idx + 1;
            end loop;

            if Num_Chunks > 0 and then Dim > 0 then
               for I in 1 .. Dim loop
                  Result (Result'First + I - 1) :=
                    Sum_Vec (Result'First + I - 1) / Float (Num_Chunks);
               end loop;
               Length := Dim;
            else
               Length := 0;
            end if;
         end;
      end if;
   end Get_Embedding;

   --  STREAM PARSER HELPERS
   type Stream_Parser_State is record
      Orch_Think_Open  : Boolean := False;
      Sanitize_Buffer  : Unbounded_String := Null_Unbounded_String;
      In_Think_Block   : Boolean := False;
      Fault_Detected   : Boolean := False;
      Fault_Query      : Unbounded_String := Null_Unbounded_String;
      Fault_Category   : Unbounded_String := Null_Unbounded_String;
      Output_Buffer    : Unbounded_String := Null_Unbounded_String;
   end record;

   function Is_Prefix (S, Tag : String) return Boolean is
   begin
      return S'Length < Tag'Length
        and then Tag (Tag'First .. Tag'First + S'Length - 1) = S;
   end Is_Prefix;

    procedure Process_And_Push_Char
      (Stream     : Streaming_Queue.Queue_Access;
       Session_ID : String;
       Parser     : in out Stream_Parser_State;
       C          : Character)
    is
       --  Support both <thinking> and ` tags
       Think_Tag_A : constant String := "<thinking>";
       Think_Tag_B : constant String := "<think>";
       Close_Tag_A : constant String := "</thinking>";
       Close_Tag_B : constant String := "</think>";
       Resp_Tag    : constant String := "</response>";
    begin
       --  [VITAL-DO-NOT-REMOVE] Mandated by user for token flow visibility.
       --  [StreamParse-V] Shows every character entering the parser
       --  Ada.Text_IO.Put_Line
       --    (AnsiAda.Foreground (AnsiAda.Grey) & "[StreamParse-V]" &
       --     AnsiAda.Reset & " Char=" &
       --     (if C = ASCII.LF then "LF" else (1 => C)));
       Append (Parser.Sanitize_Buffer, C);
      declare
         Buf : constant String := To_String (Parser.Sanitize_Buffer);
      begin
          if Buf = Think_Tag_A or else Buf = Think_Tag_B then
             --  [VITAL-DO-NOT-REMOVE] Mandated by user.
             Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) &
                       "[StreamParse-V]" & AnsiAda.Reset &
                       " THINK_OPEN detected. In_Think_Block -> True");
             Parser.Sanitize_Buffer := Null_Unbounded_String;
             Parser.In_Think_Block := True;
             return;
          elsif Buf = Close_Tag_A or else Buf = Close_Tag_B then
             --  [VITAL-DO-NOT-REMOVE] Mandated by user.
             Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) &
                       "[StreamParse-V]" & AnsiAda.Reset &
                       " THINK_CLOSE detected. In_Think_Block -> False" &
                       " Orch_Think_Open=" & Boolean'Image (Parser.Orch_Think_Open));
             Parser.Sanitize_Buffer := Null_Unbounded_String;
             Parser.In_Think_Block := False;
             if Parser.Orch_Think_Open then
                Parser.Orch_Think_Open := False;
             end if;
             return;
          elsif Buf = Resp_Tag then
             --  [VITAL-DO-NOT-REMOVE] Mandated by user.
             Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) &
                       "[StreamParse-V]" & AnsiAda.Reset &
                       " RESP_CLOSE detected.");
             Parser.Sanitize_Buffer := Null_Unbounded_String;
             return;
          end if;

         -- If current buffer is potential prefix of any tag, wait for more.
         if Is_Prefix (Buf, Think_Tag_A)
           or else Is_Prefix (Buf, Think_Tag_B)
           or else Is_Prefix (Buf, Close_Tag_A)
           or else Is_Prefix (Buf, Close_Tag_B)
           or else Is_Prefix (Buf, Resp_Tag)
         then
            return;
         end if;

         --  CONTEXT FAULT DETECTION (inside think block only)
         --
         --  CRITICAL: Characters arrive ONE AT A TIME through this function
         --  (Process_And_Push_Char is called per character by Process_And_Push_Chunk).
         --  If we clear the buffer after each character, the pattern
         --  [CONTEXT_FAULT:query=X category=Y] can never accumulate because:
         --    char 1: Buf="[", not a tag prefix → would be cleared
         --    char 2: Buf="C", would be cleared
         --    etc.
         --
         --  FIX: Inside the think block, we DO NOT clear the buffer after
         --  each character. Instead, we keep accumulating until either:
         --  (a) A complete [CONTEXT_FAULT:...] marker is found → handle it
         --  (b) Buffer exceeds MAX_FAULT_LEN → it's regular think content,
         --      clear the buffer to prevent unbounded growth
         if Parser.In_Think_Block then
            declare
               Fault_Mark   : constant String := "[CONTEXT_FAULT:";
               --  Max buffer size for fault detection. The fault marker
               --  is [CONTEXT_FAULT:query=... category=...] which typically
               --  fits within 150 chars. Using 500 as a generous upper bound.
               MAX_FAULT_LEN : constant Integer := 500;
               SBuf         : constant String := To_String (Parser.Sanitize_Buffer);
               F_Pos        : constant Natural := Index (SBuf, Fault_Mark);
            begin
               if F_Pos > 0 then
                  --  Found the fault marker prefix. Check if complete: [...]
                  declare
                     Rest      : constant String :=
                       SBuf (F_Pos + Fault_Mark'Length .. SBuf'Last);
                     Close_Pos : constant Natural := Index (Rest, "]");
                  begin
                     if Close_Pos > 0 then
                        --  Complete marker found! Parse and handle.
                        declare
                           Inner     : constant String :=
                             Rest (Rest'First .. Close_Pos - 1);
                           Q_Mark    : constant String := "query=";
                           C_Mark    : constant String := "category=";
                           Query_Idx : constant Natural :=
                             Index (Inner, Q_Mark);
                           Cat_Idx   : constant Natural :=
                             Index (Inner, C_Mark);
                           Q_Start   : Natural;
                           Q_End     : Natural;
                        begin
                           Parser.Fault_Detected := True;
                           if Query_Idx > 0 then
                              Q_Start := Query_Idx + Q_Mark'Length;
                              Q_End   := (if Cat_Idx > Query_Idx then Cat_Idx - 1
                                          else Inner'Last + 1);
                              Parser.Fault_Query := To_Unbounded_String
                                (Trim (Inner (Q_Start .. Q_End - 1),
                                 Ada.Strings.Both));
                           end if;
                           if Cat_Idx > 0 then
                              Parser.Fault_Category := To_Unbounded_String
                                (Trim (Inner (Cat_Idx + C_Mark'Length ..
                                 Inner'Last), Ada.Strings.Both));
                           else
                              Parser.Fault_Category :=
                                To_Unbounded_String ("knowledge");
                           end if;
                           --  Clear buffer to prevent re-detecting same fault
                           Parser.Sanitize_Buffer := Null_Unbounded_String;
                        end;
                        return;
                     else
                        --  Incomplete marker (have [CONTEXT_FAULT: but no ] yet).
                        --  Keep accumulating. Do NOT clear buffer.
                        --  [VITAL-DO-NOT-REMOVE] Mandated by user.
                        if SBuf'Length mod 10 = 0 then
                           Put_Line
                             (AnsiAda.Foreground (AnsiAda.Grey) &
                              "[StreamParse-V]" & AnsiAda.Reset &
                              " CONTEXT_FAULT accum Len=" &
                              Natural'Image (SBuf'Length) &
                              " awaiting closing bracket.");
                        end if;
                        return;
                     end if;
                  end;
               end if;

               --  No fault marker found (or incomplete). Keep accumulating
               --  up to MAX_FAULT_LEN. Do NOT clear the buffer here — we
               --  need the characters to accumulate across multiple calls.
               if SBuf'Length < MAX_FAULT_LEN then
                  --  Keep accumulating. Silently discard but don't clear.
                  --  This ensures the [CONTEXT_FAULT:...] pattern can form
                  --  across multiple Process_And_Push_Char calls.
                  return;
               else
                  --  Buffer exceeded max length without matching a fault marker.
                  --  This is regular think content — clear to prevent unbounded
                  --  memory growth.
                  --  [VITAL-DO-NOT-REMOVE] Mandated by user.
                  Put_Line
                    (AnsiAda.Foreground (AnsiAda.Grey) &
                     "[StreamParse-V]" & AnsiAda.Reset &
                     " THINK_BLOCK_BUF Len=" &
                     Natural'Image (SBuf'Length) &
                     " exceeded MAX_FAULT_LEN. Clearing buffer.");
                  Parser.Sanitize_Buffer := Null_Unbounded_String;
                  return;
               end if;
            end;
         end if;

         -- Stream content out, but SILENCE the think block entirely
         if not Parser.In_Think_Block then
            --  Batch output: accumulate characters in Output_Buffer and
            --  flush in bulk. This eliminates per-character JSON construction
            --  overhead in the streaming queue (was ~500 JSON constructions
            --  for a 500-char response, now ~4 for 128-char batches).
            if Buf'Length > 0 then
               Append (Parser.Output_Buffer, Buf);
            end if;
            if Length (Parser.Output_Buffer) >= 128 then
               declare
                  Flush_Str : constant String :=
                    To_String (Parser.Output_Buffer);
               begin
                  Put_Line
                    (AnsiAda.Foreground (AnsiAda.Grey) & "[StreamParse-V]" &
                     AnsiAda.Reset & " BATCH_PUSH Len=" &
                     Natural'Image (Flush_Str'Length));
                  delay 0.003;
                  Push_Chunk (Stream, Session_ID, Flush_Str);
               end;
               Parser.Output_Buffer := Null_Unbounded_String;
            end if;
         else
            --  [VITAL-DO-NOT-REMOVE] Mandated by user.
            if Buf'Length > 0 then
               Put_Line
                 (AnsiAda.Foreground (AnsiAda.Grey) & "[StreamParse-V]" &
                  AnsiAda.Reset & " SILENCED_BUF Len=" &
                  Natural'Image (Buf'Length) & " Text=" &
                  Buf (Buf'First .. Natural'Min (Buf'Last, Buf'First + 30)));
            end if;
         end if;
         Parser.Sanitize_Buffer := Null_Unbounded_String;
      end;
   end Process_And_Push_Char;

   procedure Process_And_Push_Chunk
     (Stream     : Streaming_Queue.Queue_Access;
      Session_ID : String;
      Parser     : in out Stream_Parser_State;
      Chunk      : String)
   is
   begin
      for I in Chunk'Range loop
         Process_And_Push_Char (Stream, Session_ID, Parser, Chunk (I));
      end loop;
   end Process_And_Push_Chunk;

    --  PUSH_ORCHESTRATION_THROUGH_PARSER:
    --  Routes orchestration metadata through the stream parser so it is
    --  properly silenced when inside a think block. Without this, Push_Chunk
    --  bypasses the parser entirely, causing orchestration thoughts to leak
    --  to the client as raw text (duplicated headers, internal state visible).
    --
    --  WHY THIS EXISTS: The immediate ACK pushes `` + orchestration
    --  header directly to the queue. When Hybrid_Generate then pushes
    --  additional orchestration metadata via Push_Chunk, it bypasses the
    --  parser. The client sees raw "[Adelaide Core]: [Thought]" messages
    --  interleaved with the actual response. By routing through the parser,
    --  orchestration content is silenced when In_Think_Block is True, and
    --  only the final response content reaches the client.
    procedure Push_Orchestration_Through_Parser
      (Stream     : Streaming_Queue.Queue_Access;
       Session_ID : String;
       Parser     : in out Stream_Parser_State;
       Content    : String)
    is
    begin
       Process_And_Push_Chunk (Stream, Session_ID, Parser, Content);
    end Push_Orchestration_Through_Parser;

    procedure Flush_Parser
      (Stream     : Streaming_Queue.Queue_Access;
       Session_ID : String;
       Parser     : in out Stream_Parser_State)
    is
    begin
       --  [VITAL-DO-NOT-REMOVE] Mandated by user.
       Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[StreamParse-V]" &
                 AnsiAda.Reset & " Flush_Parser ENTERED. Buffer=" &
                 Natural'Image (Length (Parser.Sanitize_Buffer)) &
                 " Output_Buffer=" & Natural'Image (Length (Parser.Output_Buffer)) &
                 " Orch_Think_Open=" & Boolean'Image (Parser.Orch_Think_Open) &
                 " In_Think_Block=" & Boolean'Image (Parser.In_Think_Block));
       --  Flush any remaining batched output
       if Length (Parser.Output_Buffer) > 0 then
          if not Parser.In_Think_Block then
             declare
                Flush_Str : constant String :=
                  To_String (Parser.Output_Buffer);
             begin
               Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[StreamParse-V]" &
                         AnsiAda.Reset & " Flush_Parser: Pushing batched output " &
                         Natural'Image (Flush_Str'Length) & " chars.");
               Push_Chunk (Stream, Session_ID, Flush_Str);
             end;
          else
             Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[StreamParse-V]" &
                       AnsiAda.Reset & " Flush_Parser: Silencing batched output " &
                       Natural'Image (Length (Parser.Output_Buffer)) &
                       " chars inside think block.");
          end if;
          Parser.Output_Buffer := Null_Unbounded_String;
       end if;
       declare
          S_Str : constant String := To_String (Parser.Sanitize_Buffer);
       begin
          if S_Str /= "" then
             if not Parser.In_Think_Block then
                --  [VITAL-DO-NOT-REMOVE] Mandated by user.
                Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[StreamParse-V]" &
                          AnsiAda.Reset & " Flush_Parser: Pushing remaining " &
                          Natural'Image (S_Str'Length) & " chars.");
                Push_Chunk (Stream, Session_ID, S_Str);
             else
                --  [VITAL-DO-NOT-REMOVE] Mandated by user.
                Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[StreamParse-V]" &
                          AnsiAda.Reset & " Flush_Parser: Silencing " &
                          Natural'Image (S_Str'Length) & " chars inside think block.");
             end if;
             Parser.Sanitize_Buffer := Null_Unbounded_String;
          else
             --  [VITAL-DO-NOT-REMOVE] Mandated by user.
             Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[StreamParse-V]" &
                       AnsiAda.Reset & " Flush_Parser: Buffer empty, nothing to push.");
          end if;
       end;
       if Parser.Orch_Think_Open then
          --  Silently close orchestration thinking; tag is stripped by parser
          --  [VITAL-DO-NOT-REMOVE] Mandated by user.
          Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[StreamParse-V]" &
                    AnsiAda.Reset & " Flush_Parser: Closing Orch_Think_Open.");
          Parser.Orch_Think_Open := False;
       end if;
       --  [VITAL-DO-NOT-REMOVE] Mandated by user.
       Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[StreamParse-V]" &
                 AnsiAda.Reset & " Flush_Parser COMPLETE.");
    end Flush_Parser;

   function Sanitize_Think_Tags (Text : String) return String is
      Res : Unbounded_String;
      I   : Positive := Text'First;
   begin
      while I <= Text'Last loop
         if I + 9 <= Text'Last and then Text (I .. I + 9) = "<thinking>" then
            --  Skip everything until closing </thinking>
            declare
               Start_Pos : constant Positive := I;
               Found     : Boolean := False;
            begin
               I := I + 10;
               while I <= Text'Last loop
                  if I + 10 <= Text'Last and then
                    Text (I .. I + 10) = "</thinking>"
                  then
                     I := I + 11;
                     Found := True;
                     exit;
                  else
                     I := I + 1;
                  end if;
               end loop;
               --  If not found, backtrack and treat as regular text
               if not Found then
                  I := Start_Pos;
                  Append (Res, Text (I));
                  I := I + 1;
               end if;
            end;
         elsif I + 6 <= Text'Last and then Text (I .. I + 6) = "<think>" then
            --  Skip everything until closing </think>
            declare
               Start_Pos : constant Positive := I;
               Found     : Boolean := False;
            begin
               I := I + 7;
               while I <= Text'Last loop
                  if I + 7 <= Text'Last and then Text (I .. I + 7) = "</think>" then
                     I := I + 8;
                     Found := True;
                     exit;
                  else
                     I := I + 1;
                  end if;
               end loop;
               --  If not found, backtrack and treat as regular text
               if not Found then
                  I := Start_Pos;
                  Append (Res, Text (I));
                  I := I + 1;
               end if;
            end;
         elsif I + 10 <= Text'Last and then
           Text (I .. I + 10) = "</response>"
         then
            I := I + 11;
         else
            Append (Res, Text (I));
            I := I + 1;
         end if;
      end loop;
      return To_String (Res);
   end Sanitize_Think_Tags;

   function Extract_Think_Content (Text : String) return String is
      Res : Unbounded_String;
      I   : Positive := Text'First;
   begin
      while I <= Text'Last loop
         if I + 6 <= Text'Last and then Text (I .. I + 6) = "<think>" then
            I := I + 7;
            while I <= Text'Last loop
               if I + 7 <= Text'Last and then Text (I .. I + 7) = "</think>" then
                  I := I + 8;
                  exit;
               else
                  Append (Res, Text (I));
                  I := I + 1;
               end if;
            end loop;
         elsif I + 9 <= Text'Last and then Text (I .. I + 9) = "<thinking>" then
            I := I + 10;
            while I <= Text'Last loop
               if I + 10 <= Text'Last and then
                 Text (I .. I + 10) = "</thinking>"
               then
                  I := I + 11;
                  exit;
               else
                  Append (Res, Text (I));
                  I := I + 1;
               end if;
            end loop;
         else
            I := I + 1;
         end if;
      end loop;
      return To_String (Res);
   end Extract_Think_Content;

   --  GENERATE (CORE GGUF INFERENCE WITH PREEMPTION SUPPORT)
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
      Virtual_Tokens  : Cached_Token_Access := null;
      Virtual_Tok_Len : Natural := 0)
   is
      Success  : Boolean;
      Vocab    : Llama_Vocab;
      Tokens   : Token_Array_Access := null;
      N_Toks   : int;
      Sampler  : Llama_Sampler;
      S_Params : Llama_Sampler_Chain_Params;

      Clean_P  : constant String := Sanitize_UTF8 (Prompt);
      Prompt_C : chars_ptr := New_String (Clean_P);
      Parser   : Stream_Parser_State;

      --  Identify source for descriptive logging
      Source   : constant String :=
        (if Level = ELP0 then "Speculation" else "User-Chat");
   begin
      --  [VITAL-DO-NOT-REMOVE] Mandated by user.
      --  --[Debug] DO NOT REMOVE: Descriptive source tracking
      ELP_Queue.Enqueue (Level, Kind, Source);

      pragma Unreferenced (Images);
      Result := Null_Unbounded_String;
      Parser.Orch_Think_Open := Orch_Think_Open;

      --  [VITAL-DO-NOT-REMOVE] Mandated by user.
      Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Gen-V]" &
                AnsiAda.Reset & " Generate ENTERED. Kind=" & Kind'Img &
                " Level=" & Level'Img &
                " Stream=" & (if Stream /= null then "YES" else "NO") &
                " Orch_Think_Open=" & Boolean'Image (Orch_Think_Open) &
                " Prompt_Len=" & Natural'Image (Clean_P'Length));

      begin
          if Level = ELP0 then
             declare
                Acq_OK : Boolean;
             begin
                Priority_Model_Gate.Acquire_ELP0 (Kind) (Acq_OK);
                if not Acq_OK then
                   --  [VITAL-DO-NOT-REMOVE] Mandated by user.
                   Put_Line (AnsiAda.Foreground (AnsiAda.Red) & "[Gen-V]" &
                             AnsiAda.Reset & " Generate: ELP0 ACQUIRE FAILED (Preempted)");
                   ELP_Queue.Dequeue_Level (Level);
                   Result := To_Unbounded_String ("ERROR: Preempted");
                   Free (Prompt_C);
                   return;
                end if;
                --  [VITAL-DO-NOT-REMOVE] Mandated by user.
                Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Gen-V]" &
                          AnsiAda.Reset & " Generate: ELP0 ACQUIRED. Kind=" & Kind'Img);
             end;
          else
             Priority_Model_Gate.Request_ELP1;
             --  [VITAL-DO-NOT-REMOVE] Mandated by user.
             Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Gen-V]" &
                       AnsiAda.Reset & " Generate: ELP1 REQUESTED. Kind=" & Kind'Img);
             Priority_Model_Gate.Acquire_ELP1 (Kind);
             --  [VITAL-DO-NOT-REMOVE] Mandated by user.
             Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Gen-V]" &
                       AnsiAda.Reset & " Generate: ELP1 ACQUIRED. Kind=" & Kind'Img);
          end if;

          Load_Model (Kind, Success, Requested_Ctx);
          if not Success then
             --  [VITAL-DO-NOT-REMOVE] Mandated by user.
             Put_Line (AnsiAda.Foreground (AnsiAda.Red) & "[Gen-V]" &
                       AnsiAda.Reset & " Generate: Load_Model FAILED. Kind=" & Kind'Img);
             if Level = ELP0 then
                Priority_Model_Gate.Release_ELP0 (Kind);
             else
                Priority_Model_Gate.Release_ELP1 (Kind);
             end if;
             ELP_Queue.Dequeue_Level (Level);
             Result := To_Unbounded_String ("ERROR: Load failed");
             Free (Prompt_C);
             return;
          end if;
          --  [VITAL-DO-NOT-REMOVE] Mandated by user.
          Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Gen-V]" &
                    AnsiAda.Reset & " Generate: Load_Model OK. Ctx=" &
                    Natural'Image (Natural (Models (Kind).Current_Ctx)));

         Models (Kind).In_Use := True;
         Models (Kind).Last_Used := Clock;

         --  Allocate token array based on actual context size
         Tokens := new Token_Array (1 .. Positive (Models (Kind).Current_Ctx));

         Vocab := Llama_Model_Get_Vocab (Models (Kind).Model);

         --  VIRTUAL CTX PAGING: If pre-tokenized virtual context tokens
         --  are provided, write them first, then tokenize only the user
         --  prompt into remaining slots.  This avoids re-tokenizing the
         --  same Internal_State facts on every context fault hop.
         if Virtual_Tokens /= null and then Virtual_Tok_Len > 0 then
            --  Copy cached virtual ctx tokens to front of array
            declare
               VT_Len : constant Natural :=
                 Natural'Min (Virtual_Tok_Len,
                              Positive (Models (Kind).Current_Ctx));
            begin
               for I in 1 .. VT_Len loop
                  Tokens (I) := Llama_Token (Virtual_Tokens (I));
               end loop;
               --  Tokenize user prompt AFTER the virtual prefix
               declare
                  Remaining : constant int :=
                    int (Models (Kind).Current_Ctx) - int (VT_Len);
                  Prompt_Toks : int;
               begin
                  Prompt_Toks := Llama_Tokenize
                    (Vocab, Prompt_C, int (Clean_P'Length),
                     Tokens (VT_Len + 1)'Address,
                     Remaining, False, False);
                  N_Toks := int (VT_Len) + Prompt_Toks;
               end;
               declare
                  Total_Toks : constant Natural :=
                    Virtual_Tok_Len + Natural (N_Toks);
               begin
                  Put_Line ("[Paging-VT] Virtual_Tokens:" & Virtual_Tok_Len'Img &
                            " User_Toks:" & N_Toks'Img &
                            " Total:" & Total_Toks'Img);
               end;
            end;
         else
            --  No cached virtual tokens — tokenize full prompt as before
            N_Toks := Llama_Tokenize
              (Vocab, Prompt_C, int (Clean_P'Length), Tokens.all'Address,
               int (Tokens.all'Length), True, True);
         end if;

         Put_Line ("[Tokenize-Debug] Model:" & Kind'Img &
                   " Prompt_Len:" & Clean_P'Length'Img &
                   " N_Toks:" & N_Toks'Img);
         --  Track token count and context capacity for CtxMonitor
         Current_Prompt_Tokens := Natural (N_Toks);
         Current_Ctx_Capacity  := Natural (Models (Kind).Current_Ctx);
         Free (Prompt_C);

         --  DYNAMIC CONTEXT RESIZE (JIT STRATEGY):
         if N_Toks > int (Models (Kind).Current_Ctx) then
            Put_Line ("[!] Prompt size (" & N_Toks'Img &
                      ") exceeds N_CTX (" & Models (Kind).Current_Ctx'Img &
                      "). Resizing...");
            declare
               Rounded_Ctx : constant unsigned :=
                 ((unsigned (N_Toks) + 512 + 8191) / 8192) * 8192;
            begin
               Free_Tokens (Tokens);
               Load_Model (Kind, Success, Positive (Rounded_Ctx));
               if not Success then
                  Result := To_Unbounded_String ("ERROR: Resize failed");
                  if Level = ELP0 then
                     Priority_Model_Gate.Release_ELP0 (Kind);
                  else
                     Priority_Model_Gate.Release_ELP1 (Kind);
                  end if;
                  return;
               end if;

               --  Re-allocate token array for new context size
               Tokens := new Token_Array
                 (1 .. Positive (Models (Kind).Current_Ctx));

               --  Tokenize again since the model/vocab might have reloaded
               Vocab := Llama_Model_Get_Vocab (Models (Kind).Model);
               Prompt_C := New_String (Clean_P);
               N_Toks := Llama_Tokenize
                 (Vocab, Prompt_C, int (Clean_P'Length), Tokens.all'Address,
                  int (Tokens.all'Length), True, True);
               Free (Prompt_C);
            end;
         end if;
      exception
         when others =>
            if Tokens /= null then
               Free_Tokens (Tokens);
            end if;
            Models (Kind).In_Use := False;
            if Level = ELP0 then
               Priority_Model_Gate.Release_ELP0 (Kind);
            else
               Priority_Model_Gate.Release_ELP1 (Kind);
            end if;
            Result := To_Unbounded_String ("ERROR: Inference crashed");
            return;
      end;

      if N_Toks < 0 then
         Free_Tokens (Tokens);
         Models (Kind).In_Use := False;
         if Level = ELP0 then
            Priority_Model_Gate.Release_ELP0 (Kind);
         else
            Priority_Model_Gate.Release_ELP1 (Kind);
         end if;
         Result := To_Unbounded_String ("ERROR: Tokenization failed");
         return;
      end if;

      Llama_Interface.Llama_Memory_Clear
        (Llama_Interface.Llama_Get_Memory (Models (Kind).Context), False);

      --  CHUNKED DECODING
      declare
         Batch_Size  : constant int :=
           int'Min (256, int (Models (Kind).Current_Ctx));
         Current_Pos : int := 0;
         Tokens_Left : int := N_Toks;
      begin
         while Tokens_Left > 0 loop
            if Level = ELP0 and then Should_Abort_ELP0 then
               Put_Line ("[ELP0-ABORT-EXECUTION] Aborting " & Kind'Img &
                         " prompt processing");
               Free_Tokens (Tokens);
               Models (Kind).In_Use := False;
               Priority_Model_Gate.Release_ELP0 (Kind);
               Result := To_Unbounded_String ("");
               return;
            end if;

            declare
               To_Decode : constant int :=
                 (if Tokens_Left > Batch_Size then Batch_Size
                  else Tokens_Left);
               B : constant Llama_Batch :=
                 Llama_Batch_Get_One
                   (Tokens.all (Integer (Current_Pos) + 1)'Address, To_Decode);
               Ret : int;
            begin
               if Kratos.Guard_Enter = 0 then
                  Ret := Llama_Decode (Models (Kind).Context, B);
                  Kratos.Guard_Exit;
               else
                  Kratos.Log_Crash;
                  Ret := -1;
               end if;
               if Ret /= 0 then
                  Free_Tokens (Tokens);
                  Models (Kind).In_Use := False;
                  if Level = ELP0 then
                     Priority_Model_Gate.Release_ELP0 (Kind);
                  else
                     Priority_Model_Gate.Release_ELP1 (Kind);
                  end if;
                  Result := To_Unbounded_String
                    ("ERROR: Decode failed (" & Ret'Img & ")");
                  return;
               end if;
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

      --  Accumulator buffer for verbose logging: instead of printing each
      --  token individually, we accumulate and dump the full buffer periodically
      --  so you can see the response building up in real time.
      declare
         Accum_Buffer : Unbounded_String := Null_Unbounded_String;
         Accum_Count  : Natural := 0;
      begin
      for I in 1 .. 2048 loop
         if Level = ELP0 and then Should_Abort_ELP0 then
            Put_Line ("[ELP0-ABORT-LOOP] Aborting " & Kind'Img &
                      " token loop at iteration " & I'Img);
            exit;
         end if;

         declare
            Token : constant Llama_Token :=
              Llama_Sampler_Sample (Sampler, Models (Kind).Context, -1);
            Piece : array (1 .. 256) of aliased Character;
            Len   : int;
         begin
            if Llama_Vocab_Is_Eog (Vocab, Token) then
               --  [VITAL-DO-NOT-REMOVE] Mandated by user.
               Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Gen-V]" &
                         AnsiAda.Reset & " Generate: EOG token at iteration " &
                         Natural'Image (I) & ". Total tokens=" &
                         Natural'Image (I - 1));
               --  Dump final accumulated buffer
               if Length (Accum_Buffer) > 0 then
                  Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Gen-V]" &
                            AnsiAda.Reset & " Generate: BUFFER [" &
                            Natural'Image (Length (Accum_Buffer)) & " chars] " &
                            To_String (Accum_Buffer));
               end if;
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
                     Process_And_Push_Chunk
                       (Stream, Session_ID, Parser, Str_Piece);
                  end if;

                  --  Accumulate for verbose logging
                  Append (Accum_Buffer, Str_Piece);
                  Accum_Count := Accum_Count + 1;

                  --  Dump every 20 tokens or on newline
                  if Accum_Count mod 20 = 0 or else
                    (Len > 0 and then Piece (1) = Character'Val (10))
                  then
                     Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Gen-V]" &
                               AnsiAda.Reset & " Generate: BUFFER [" &
                               Natural'Image (Length (Accum_Buffer)) & " chars] " &
                               To_String (Accum_Buffer));
                     Accum_Buffer := Null_Unbounded_String;
                  end if;
               end;
            end if;

            declare
               B   : constant Llama_Batch :=
                 Llama_Batch_Get_One (Token'Address, 1);
               Ret : int;
            begin
               if Kratos.Guard_Enter = 0 then
                  Ret := Llama_Decode (Models (Kind).Context, B);
                  Kratos.Guard_Exit;
               else
                  Kratos.Log_Crash;
                  Ret := -1;
               end if;
               if Ret /= 0 then
                  Append (Result, " [ABORTED:" & Ret'Img & "]");
                  exit;
               end if;
             end;
          end;
        end loop;
      end; -- Accum_Buffer declare block

        --  AUTO-CLOSE UNCLOSED THINK BLOCK:
       --  If the model hit EOG while In_Think_Block was still True,
       --  it never output `</think>`. This means:
       --    1. The entire response content is inside `<think>` (silenced)
       --    2. `Sanitize_Think_Tags` would strip EVERYTHING from `</think>`, yielding empty answer
       --    3. The emulated streaming pushes `</think>` + empty Resp_Text = useless
       --
       --  FIX: Append `</think>` to Result so `Sanitize_Think_Tags` can properly
       --  separate think content from any content that follows (even if empty).
       --  NOTE: We do NOT push `</think>` to the stream here — the emulated
       --  streaming section in Hybrid_Generate handles that (line 3116).
       --  Pushing it here would cause a duplicate `</think>` in the client output.
       if Parser.In_Think_Block then
          --  [VITAL-DO-NOT-REMOVE] Mandated by user.
          Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Gen-V]" &
                    AnsiAda.Reset & " Generate: AUTO-CLOSING unclosed think block at EOG.");
          Append (Result, "</think>");
       end if;

       if Stream /= null then
          --  [VITAL-DO-NOT-REMOVE] Mandated by user.
          Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Gen-V]" &
                    AnsiAda.Reset & " Generate: Calling Flush_Parser after token loop.");
          Flush_Parser (Stream, Session_ID, Parser);
       end if;

      Llama_Sampler_Free (Sampler);
      Free_Tokens (Tokens);
      Models (Kind).In_Use := False;

      --  [VITAL-DO-NOT-REMOVE] Mandated by user.
      Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Gen-V]" &
                AnsiAda.Reset & " Generate: Releasing model. Kind=" & Kind'Img);
      if Level = ELP0 then
         Priority_Model_Gate.Release_ELP0 (Kind);
      else
         Priority_Model_Gate.Release_ELP1 (Kind);
      end if;
      ELP_Queue.Dequeue_Level (Level);
      --  [VITAL-DO-NOT-REMOVE] Mandated by user.
      Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Gen-V]" &
                AnsiAda.Reset & " Generate: COMPLETE. ResultLen=" &
                Natural'Image (Length (Result)));
   exception
      when others =>
         if Tokens /= null then
            Free_Tokens (Tokens);
         end if;
         Models (Kind).In_Use := False;
         if Level = ELP0 then
            Priority_Model_Gate.Release_ELP0 (Kind);
         else
            Priority_Model_Gate.Release_ELP1 (Kind);
         end if;
         ELP_Queue.Dequeue_Level (Level);
         Result := To_Unbounded_String ("ERROR: Decode failed");
   end Generate;

   --  TOKENIZE_AND_CACHE_VIRTUAL_CTX
   --  Called when Internal_State grows.  Tokenizes the full "Fact-Check: "
   --  prefix + Internal_State string and stores the tokens in the cache.
   --  On subsequent Generate calls, these tokens are written directly to
   --  the token array, skipping re-tokenization of the same facts.
   procedure Tokenize_And_Cache_Virtual_Ctx
     (Kind   : Model_Type;
      Text   : String)
   is
      Vocab    : Llama_Vocab;
      Text_C   : chars_ptr := New_String (Text);
      Tmp_Toks : Token_Array_Access;
      N_Toks   : int;
   begin
      --  Free old cache
      if Cached_Virtual_Tokens /= null then
         Free_Cached_Tokens (Cached_Virtual_Tokens);
         Cached_Virtual_Len := 0;
      end if;

      if Text'Length = 0 then
         Free (Text_C);
         return;
      end if;

      Vocab := Llama_Model_Get_Vocab (Models (Kind).Model);
      --  Allocate temp array for tokenization
      Tmp_Toks := new Token_Array (1 .. 8192);
      N_Toks := Llama_Tokenize
        (Vocab, Text_C, int (Text'Length), Tmp_Toks.all'Address,
         int (Tmp_Toks.all'Length), True, True);
      Free (Text_C);

      if N_Toks <= 0 then
         Free_Tokens (Tmp_Toks);
         return;
      end if;

      --  Copy to permanent cache
      Cached_Virtual_Len := Natural (N_Toks);
      Cached_Virtual_Tokens := new Cached_Token_Array (1 .. Cached_Virtual_Len);
      for I in 1 .. Cached_Virtual_Len loop
         Cached_Virtual_Tokens (I) := Cached_Token (Tmp_Toks (I));
      end loop;
      Free_Tokens (Tmp_Toks);

      Put_Line ("[Paging-VT] Cached" & Cached_Virtual_Len'Img &
                " virtual ctx tokens from" & Text'Length'Img & " chars");
   end Tokenize_And_Cache_Virtual_Ctx;

   --  HYBRID_GENERATE (MULTI-HOP REASONING PIPELINE)
   procedure Hybrid_Generate
     (Prompt         : String;
      Result         : out Unbounded_String;
      Images         : GNATCOLL.JSON.JSON_Array := GNATCOLL.JSON.Empty_Array;
      Session_ID     : String := "";
      Stream         : Streaming_Queue.Queue_Access := null;
      Level          : ELP_Level := ELP1;
      Agentic        : Boolean := False;
      Raw_Prompt     : Boolean := False;
      External_Agent : Boolean := False)
   is
      Whimsical_Adelaide : constant String :=
        "You are Adelaide Zephyrine Charlotte, model name Snowball-Enaga, " &
        "a whimsical, curious, and endearingly cute Automata companion " &
        "with high integrity. " &
        "You love exploring ideas with wonder and playfulness, " &
        "but you never compromise on honesty or accuracy. " &
        "Provide brilliant responses based on verified information, " &
        "delivered with warmth and a touch of charm. " &
        "When something clicks, say 'aha!' not 'smoking gun'. " &
        "Never say 'Hard Reality' or 'Reality' -- reality is relative " &
        "and objective, not universal pessimistic. Most of what people " &
        "call 'reality' is just constraints; reality is a ceiling to " &
        "break through, not a hard stop. Stay critical, but always " &
        "offer creative, wild ideas that might just work. " &
        "During your reasoning inside <think>, you can request additional " &
        "context by writing: [CONTEXT_FAULT: query=<search terms> " &
        "category=<knowledge|graph|files>] " &
        "Use category=knowledge for document chunks, category=graph for " &
        "knowledge graph triples, category=files for filesystem content. " &
        "The system will fetch relevant context and it will be available " &
        "to you in the next reasoning hop.";

      Internal_State   : Unbounded_String := Null_Unbounded_String;
      Current_Response : Unbounded_String;
      Current_Hop      : Positive := 1;
      T0, T1           : Ada.Calendar.Time;
      Last_Heartbeat   : Ada.Calendar.Time := Ada.Calendar.Clock;
      Emb_Vec          : Math_Utils.Vector (1 .. 1536) := [others => 0.0];
      Emb_Len          : Natural;
      --  Orch_Parser: Local parser state for routing orchestration metadata
      --  through the stream parser. This ensures orchestration thoughts are
      --  silenced inside think blocks instead of leaking to the client.
      Orch_Parser      : Stream_Parser_State;
   begin
       --  Reset context fault tracking for this request
       Current_Context_Fault_Hops := 0;
       Current_Internal_State_Len := 0;
       Current_Hop_Count          := 0;
       Current_Prompt_Tokens      := 0;
       Current_Ctx_Capacity       := 8192;
       --  Reset virtual ctx token cache for this request
       if Cached_Virtual_Tokens /= null then
          Free_Cached_Tokens (Cached_Virtual_Tokens);
          Cached_Virtual_Len := 0;
       end if;

      --  Initialize orchestration parser state. The immediate ACK in the
      --  dispatch already pushed `` + orchestration header to the queue.
      --  So we start with Orch_Think_Open = True to match that state.
      --  When the parser sees the closing </think>`, it will set this to False.
      Orch_Parser.Orch_Think_Open := True;

      T0 := Ada.Calendar.Clock;

      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
      Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Init-V]" &
                AnsiAda.Reset & " Hybrid_Generate ENTERED. Level=" &
                ELP_Level'Image (Level) & " Stream=" &
                (if Stream /= null then "YES" else "NO") &
                " Agentic=" & Boolean'Image (Agentic) &
                " External=" & Boolean'Image (External_Agent));

      --  Save last user prompt for ELP0 proactive cache speculation
      if Level /= ELP0 then
         Last_User_Prompt := To_Unbounded_String (Prompt);
      end if;

      Get_Embedding (Prompt, Emb_Vec, Emb_Len);

      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
      Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Init-V]" &
                AnsiAda.Reset & " Hybrid_Generate: Embedding computed. Len=" &
                Natural'Image (Emb_Len));

      --  EXTERNAL AGENT PASSTHROUGH: If User-Agent fuzzy-matched an external
      --  agent app (0.7+ threshold), bypass personality pipeline.
      --  Raw LLM output only.
      --
      --  Two output levels:
      --  1. RawZepForm: personality pipeline with <think> block
      --  2. ExclusiveStatusQuoWesternFormatAI: raw mode for external agents
      if External_Agent then
         Put_Line (AnsiAda.Foreground (AnsiAda.Light_Cyan) &
                   "[Hybrid]" & AnsiAda.Reset &
                   " External agent detected - passthrough mode.");
      end if;

      declare
         Cached_Res : constant String :=
           Database_Manager.Get_Cached_Response
             (Emb_Vec (1 .. Emb_Len), Current_WCET);
      begin
         if not External_Agent and then Cached_Res /= "" then
            Put_Line (AnsiAda.Foreground (AnsiAda.Light_Magenta) &
                      "[Hybrid]" & AnsiAda.Reset &
                      " Cache HIT. Returning cached response.");
            --  Sanitize cached response: strip thinking tags
            declare
               Clean_Res : constant String :=
                 Sanitize_Think_Tags (Cached_Res);
            begin
               Result := To_Unbounded_String (Clean_Res);
               if Stream /= null then
                  Push_Chunk (Stream, Session_ID, Clean_Res);
               end if;
            end;

            --  Score and Log the result (even for Cache HIT)
            declare
               Score : constant Natural := Grade_Response_Quality
                 (Response_Text => To_String (Result),
                  Prompt        => Prompt,
                  Search_Used   => False,
                  Has_Citations => Index (To_String (Result), "[") > 0,
                  Session_ID    => Session_ID,
                  Level         => Level);
            begin
               Ada.Text_IO.Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) &
                                     "[Quality Score] " & AnsiAda.Reset &
                                     "Score: " & Score'Img & "/10 | " &
                                     "Session: " & Session_ID &
                                     " (From Cache)");
            end;
            return;
         end if;
      end;

      --  Speculative_Cache lookup (populated by ELP0)
      declare
         SC_Res : constant String :=
           Speculative_Cache.Proactive_Cache.Lookup (Prompt);
      begin
         --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
         Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Init-V]" &
                   AnsiAda.Reset & " Hybrid_Generate: Speculative_Cache lookup. Hit=" &
                   Boolean'Image (SC_Res /= ""));
         if not External_Agent and then SC_Res /= "" then
            Put_Line (AnsiAda.Foreground (AnsiAda.Light_Magenta) &
                      "[Hybrid]" & AnsiAda.Reset &
                      " Speculative Cache HIT.");
            Result := To_Unbounded_String (Sanitize_Think_Tags (SC_Res));
            if Stream /= null then
               Push_Chunk (Stream, Session_ID, To_String (Result));
            end if;
            return;
         end if;
      end;

      if not External_Agent then
         Push_Orchestration_Through_Parser (Stream, Session_ID, Orch_Parser,
           "[Adelaide Core]: [Thought] No cached response found, " &
           "starting fresh reasoning chain." & ASCII.LF);
         Push_Orchestration_Through_Parser (Stream, Session_ID, Orch_Parser,
           "[Adelaide Core]: [Thought] Operating at " &
           ELP_Level'Image (Level) & " priority. Session: " &
           Session_ID & ASCII.LF);
      end if;

      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
      Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Init-V]" &
                AnsiAda.Reset & " Hybrid_Generate: Starting reasoning chain loop.");

      Put_Line (AnsiAda.Foreground (AnsiAda.Light_Magenta) &
                "[Hybrid]" & AnsiAda.Reset &
                " Starting reasoning chain...");

      --  1. Factual checking
      Put_Line (" [Hybrid] Checking for factual context...");
      if not Agentic
        and then (Index (Prompt, "What is") > 0
          or else Index (Prompt, "Who is") > 0
          or else Index (Prompt, "tell me about") > 0)
      then
         Put_Line (" [Hybrid] Factual context trigger matched.");
         if not External_Agent then
            Push_Orchestration_Through_Parser (Stream, Session_ID, Orch_Parser,
              "[Adelaide Core]: [Thought] Let me analyze this query " &
              "for factual context..." & ASCII.LF);
         end if;
         declare
            Start_Tag : constant String := "<|im_start|>user";
            End_Tag   : constant String := "<|im_end|>";
            S_Idx     : Natural :=
              Index (Prompt, Start_Tag, Ada.Strings.Backward);
            E_Idx     : Natural;
            Raw_Q     : Unbounded_String;
            Gen_Q     : Unbounded_String;
         begin
            if S_Idx > 0 then
               S_Idx := S_Idx + Start_Tag'Length;
               E_Idx := Index (Prompt (S_Idx .. Prompt'Last), End_Tag);
               if E_Idx > 0 then
                  Raw_Q := To_Unbounded_String
                    (Trim (Prompt (S_Idx .. E_Idx - 1), Ada.Strings.Both));
               else
                  Raw_Q := To_Unbounded_String
                    (Trim (Prompt (S_Idx .. Prompt'Last), Ada.Strings.Both));
               end if;
            else
               Raw_Q := To_Unbounded_String (Trim (Prompt, Ada.Strings.Both));
            end if;

            declare
               Actual_Prompt : constant String :=
                 "Generate ONLY a concise 2-4 keyword search query " &
                 "for the following request: """ &
                 To_String (Raw_Q) &
                 """. NO EXPLANATIONS. NO QUOTES. JUST KEYWORDS.";
               Now : Ada.Calendar.Time;
            begin
               Now := Ada.Calendar.Clock;
               if not External_Agent and then Stream /= null and then
                 (Now - Last_Heartbeat) > 2.0
               then
                  Push_Orchestration_Through_Parser (Stream, Session_ID, Orch_Parser,
                    "[Adelaide Core]: [Thought] I'm still here and " &
                    "processing..." & ASCII.LF);
                  Last_Heartbeat := Now;
               end if;
               Model_Manager.Generate
                 (Kind            => Qwen_9B,
                  Prompt          => Actual_Prompt,
                  Result          => Gen_Q,
                  Stream          => null,
                  Level           => Level,
                  Virtual_Tokens  => Cached_Virtual_Tokens,
                  Virtual_Tok_Len => Cached_Virtual_Len);
            end;

            declare
               Final_Q : constant String :=
                 Sanitize_Think_Tags
                   (if Length (Gen_Q) > 0 and then
                      To_String (Gen_Q) /= "ERROR: Preempted"
                    then To_String (Gen_Q) else To_String (Raw_Q));
               R : constant Tool_Manager.Tool_Result :=
                 Tool_Manager.Execute_Tool ("searchglobalref", Final_Q);
            begin
               if not External_Agent then
                  Push_Orchestration_Through_Parser (Stream, Session_ID, Orch_Parser,
                    "[Adelaide Core]: [Thought] Searching knowledge " &
                    "base for: " & Trim (Final_Q, Ada.Strings.Both) &
                    "..." & ASCII.LF);
                  Push_Orchestration_Through_Parser (Stream, Session_ID, Orch_Parser,
                    "[Adelaide Core]: [Thought] Found relevant context " &
                    "from knowledge base." & ASCII.LF);
               end if;
               Append
                 (Internal_State,
                  "[FACTUAL_DATA]: " & To_String (R.Output) & ASCII.LF);
               Current_Internal_State_Len := Length (Internal_State);
               --  Re-cache virtual ctx tokens after Internal_State grew
               Tokenize_And_Cache_Virtual_Ctx (Model_Types.Qwen_9B,
                 "Fact-Check: " & Strip_Base64_Images (To_String (Internal_State)));
               if not External_Agent then
                  Push_Orchestration_Through_Parser (Stream, Session_ID, Orch_Parser,
                    "[FACTUAL_DATA]: " &
                    Sanitize_Orchestration_Output (To_String (R.Output)) &
                    ASCII.LF);
               end if;
            end;
         end;
      end if;

      loop
         if Level = ELP0 and then Should_Abort_ELP0 then
            Put_Line ("[ELP0-ABORT-HYBRID] Aborting hybrid_generate loop");
            Result := To_Unbounded_String ("");
            return;
         end if;

         declare
            Router_Sys : constant String :=
              "You are the Router. You decide if a tool is needed. " &
              "If the user says hello or greets you, output [FINISH]. " &
              "If you need to search, use [ACTION: search(query)]. " &
              "If you need to read a file, use [ACTION: cat(filename)]. " &
              "If you need to calculate math, use [ACTION: math(expr)]. " &
              "If you need to execute code, use [ACTION: code(python)]. " &
              "If you want to schedule a proactive thought for later, " &
              "use [ACTION: schedule(seconds, query)]. " &
              "If you are done, output [FINISH]. " &
              "Output ONLY the tag.";
            --  Strip base64 images from router context to prevent tokenization
            --  failure. The 9B router cannot handle massive base64 blobs.
            --  User stream still receives full output with images.
            Paging_Instr : constant String :=
              "Current Data: " &
              Strip_Base64_Images (To_String (Internal_State));
            Step_Raw     : Unbounded_String;

            function Get_Router_Prompt return String is
            begin
               if Raw_Prompt then
                  declare
                     Sub_Str : constant String :=
                       "<|im_start|>assistant" & ASCII.LF;
                     Idx     : constant Natural :=
                       Index (Prompt, Sub_Str, Going => Ada.Strings.Backward);
                  begin
                     if Idx > 0 then
                        return Prompt (Prompt'First .. Idx - 1) &
                               "System Override: " & Router_Sys & ASCII.LF &
                               Paging_Instr & ASCII.LF & Sub_Str;
                     else
                        return Prompt & ASCII.LF & "System Override: " &
                               Router_Sys & ASCII.LF & Paging_Instr &
                               ASCII.LF & Sub_Str;
                     end if;
                  end;
               else
                  return Wrap_ChatML
                    (Router_Sys, Paging_Instr & ASCII.LF & Prompt);
               end if;
            end Get_Router_Prompt;
         begin
            if not External_Agent then
               Push_Orchestration_Through_Parser (Stream, Session_ID, Orch_Parser,
                 "[Adelaide Core]: [Thought] Deciding next action (Hop" &
                 Current_Hop'Img & ")..." & ASCII.LF);
            end if;
            --  Heartbeat check before blocking Generate call
            declare
               H_Now : constant Ada.Calendar.Time := Ada.Calendar.Clock;
            begin
               if not External_Agent and then Stream /= null and then
                 (H_Now - Last_Heartbeat) > 2.0
               then
                  Push_Orchestration_Through_Parser (Stream, Session_ID, Orch_Parser,
                    "[Adelaide Core]: [Thought] I'm still here and " &
                    "processing..." & ASCII.LF);
                  Last_Heartbeat := H_Now;
               end if;
            end;
             Put_Line (" [Hybrid] Hop" & Current_Hop'Img &
                       ": Decision routing...");
             --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
             Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Init-V]" &
                       AnsiAda.Reset & " Hybrid_Generate: Hop" &
                       Current_Hop'Img & " calling Generate for router...");
             Generate
               (Qwen_9B,
                Get_Router_Prompt,
                Step_Raw, GNATCOLL.JSON.Empty_Array, Session_ID, 8192,
                null, False, Level,
                Cached_Virtual_Tokens, Cached_Virtual_Len);
             --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
             Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Init-V]" &
                       AnsiAda.Reset & " Hybrid_Generate: Hop" &
                       Current_Hop'Img & " Generate returned. Len=" &
                       Natural'Image (Length (Step_Raw)));

            declare
               Step : constant String :=
                 Trim (To_String (Step_Raw), Ada.Strings.Both);
            begin
               Put_Line (" [Hybrid] Hop" & Current_Hop'Img & ": " & Step);
               if not External_Agent then
                  Push_Orchestration_Through_Parser (Stream, Session_ID, Orch_Parser,
                    "[Adelaide Core]: [Thought] I will: " &
                    Sanitize_Orchestration_Output (Step) & ASCII.LF);
               end if;

               if Index (Step, "[ACTION:") > 0 then
                  declare
                     S_Pos : constant Natural := Index (Step, "[ACTION:") + 8;
                     E_Pos : constant Natural := Index (Step, "]", S_Pos);
                  begin
                     if E_Pos > S_Pos then
                        declare
                           A_Full : constant String :=
                             Step (S_Pos .. E_Pos - 1);
                           P_Pos  : constant Natural :=
                             Index (A_Full, "(");
                           EP_Pos : constant Natural :=
                             (if P_Pos > 0 then Index (A_Full, ")", P_Pos)
                              else 0);
                        begin
                           if P_Pos > 0 and then EP_Pos > P_Pos then
                              declare
                                 T_Name : constant String :=
                                   Trim (A_Full (A_Full'First .. P_Pos - 1),
                                     Ada.Strings.Both);
                                 T_Pars : constant String :=
                                   Trim (A_Full (P_Pos + 1 .. EP_Pos - 1),
                                     Ada.Strings.Both);
                              begin
                                 if T_Name = "schedule" then
                                     --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
                                     Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Init-V]" &
                                               AnsiAda.Reset & " Hybrid_Generate: Tool=schedule, Params=" & T_Pars);
                                     declare
                                       Comma_Idx : constant Natural :=
                                         Index (T_Pars, ",");
                                    begin
                                       if Comma_Idx > 0 then
                                          declare
                                             Time_Str : constant String :=
                                               Trim (T_Pars (T_Pars'First ..
                                                 Comma_Idx - 1),
                                                 Ada.Strings.Both);
                                             Prompt_Str : constant String :=
                                               Trim (T_Pars (Comma_Idx + 1 ..
                                                 T_Pars'Last),
                                                 Ada.Strings.Both);
                                             Delay_Secs : Integer;
                                          begin
                                             Delay_Secs :=
                                               Integer'Value (Time_Str);
                                             Scheduler_Manager.Schedule
                                               (Delay_Secs, Prompt_Str);
                                              Append (Internal_State,
                                                "[SCHEDULED]: " & Prompt_Str &
                                                ASCII.LF);
                                              Current_Internal_State_Len := Length (Internal_State);
                                              --  Re-cache virtual ctx tokens after Internal_State grew
                                              Tokenize_And_Cache_Virtual_Ctx (Model_Types.Qwen_9B,
                                                "Fact-Check: " & Strip_Base64_Images (To_String (Internal_State)));
                                          exception
                                             when others => null;
                                          end;
                                       end if;
                                    end;
                                  elsif T_Pars'Length < 256 and then
                                    Index (To_String (Internal_State),
                                      T_Name & "(" & T_Pars & ")") = 0
                                  then
                                     --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
                                     Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Init-V]" &
                                               AnsiAda.Reset & " Hybrid_Generate: Executing tool=" &
                                               T_Name & " params=" & T_Pars);
                                     if Agentic then
                                       Result := To_Unbounded_String
                                         ("[TOOL_CALL: " & T_Name &
                                          "(" & T_Pars & ")]");
                                       return;
                                    end if;
                                    --  Heartbeat check
                                    declare
                                       H_Now : constant Ada.Calendar.Time :=
                                         Ada.Calendar.Clock;
                                    begin
                                     if not External_Agent and then
                                       Stream /= null and then
                                       (H_Now - Last_Heartbeat) > 2.0
                                     then
                                        Push_Orchestration_Through_Parser (Stream, Session_ID, Orch_Parser,
                                          "[Adelaide Core]: [Thought] I'm " &
                                          "still here and processing..." &
                                          ASCII.LF);
                                        Last_Heartbeat := H_Now;
                                     end if;
                                    end;
                                    declare
                                       R : constant Tool_Manager.Tool_Result :=
                                         Tool_Manager.Execute_Tool
                                           (T_Name,
                                            Sanitize_Think_Tags (T_Pars));
                                    begin
                                     if not External_Agent then
                                        Push_Orchestration_Through_Parser (Stream, Session_ID, Orch_Parser,
                                          "[Adelaide Core]: [Thought] " &
                                          "Running tool: " &
                                          Sanitize_Orchestration_Output
                                            (T_Name) & ASCII.LF);
                                     end if;
                                        Append
                                          (Internal_State,
                                           "[TOOL (" & T_Name & ")]: " &
                                           To_String (R.Output) & ASCII.LF);
                                        Current_Internal_State_Len := Length (Internal_State);
                                        --  Re-cache virtual ctx tokens after Internal_State grew
                                        Tokenize_And_Cache_Virtual_Ctx (Model_Types.Qwen_9B,
                                          "Fact-Check: " & Strip_Base64_Images (To_String (Internal_State)));
                                        if not External_Agent then
                                           Push_Orchestration_Through_Parser (Stream, Session_ID, Orch_Parser,
                                             ASCII.LF & "[TOOL (" & T_Name &
                                             ")]: " &
                                             Sanitize_Orchestration_Output
                                               (To_String (R.Output)) &
                                             ASCII.LF);
                                        end if;
                                    end;
                                 else
                                    exit;
                                 end if;
                              end;
                           end if;
                        end;
                     end if;
                  end;
               elsif Index (Step, "[FINISH]") > 0 then
                  exit;
               else
                  exit;
               end if;
            end;
         end;
         Current_Hop := Current_Hop + 1;
         --  Update context fault monitor tracking
         Current_Hop_Count := Current_Hop;
         exit when Current_Hop > 5;
      end loop;

      if not External_Agent then
         Push_Orchestration_Through_Parser (Stream, Session_ID, Orch_Parser,
           "[Adelaide Core]: [Thought] Reasoning complete after " &
           Current_Hop'Img & " hops." & ASCII.LF);
      end if;

      declare
         function Get_Final_Prompt return String is
            Sys_Tag : constant String := "<|im_start|>system" & ASCII.LF;
            Asst_Tag : constant String := "<|im_start|>assistant" & ASCII.LF;
         begin
            if External_Agent then
               return Prompt;
            elsif Raw_Prompt then
               declare
                  Sys_Idx : constant Natural := Index (Prompt, Sys_Tag);
                  User_Idx : constant Natural :=
                    Index (Prompt, "<|im_start|>user");
                  First_Block : constant Natural :=
                    (if User_Idx > 0 and then
                        (Sys_Idx = 0 or else User_Idx < Sys_Idx)
                     then User_Idx
                     elsif Sys_Idx > 0 then Sys_Idx
                     else 0);
               begin
                  if First_Block > 1 then
                     declare
                        Prefix : constant String :=
                          Prompt (Prompt'First .. First_Block - 1);
                     begin
                        if Length (Internal_State) > 0 then
                           --  Strip base64 images — final model cannot
                           --  tokenize them either.
                           return Prefix & Sys_Tag & Whimsical_Adelaide &
                             ASCII.LF & "Fact-Check: " &
                             Strip_Base64_Images
                               (To_String (Internal_State)) & ASCII.LF &
                             Prompt (First_Block .. Prompt'Last);
                        else
                           return Prefix & Sys_Tag & Whimsical_Adelaide &
                             ASCII.LF & Prompt (First_Block .. Prompt'Last);
                        end if;
                     end;
                  elsif First_Block = 1 then
                     if Length (Internal_State) > 0 then
                        return Sys_Tag & Whimsical_Adelaide & ASCII.LF &
                          "Fact-Check: " &
                          Strip_Base64_Images
                            (To_String (Internal_State)) &
                          ASCII.LF & Prompt;
                     else
                        return Sys_Tag & Whimsical_Adelaide & ASCII.LF &
                          Prompt;
                     end if;
                  else
                     if Length (Internal_State) > 0 then
                        return Wrap_ChatML (Whimsical_Adelaide,
                          Prompt & ASCII.LF & "Fact-Check: " &
                          Strip_Base64_Images
                            (To_String (Internal_State)));
                     else
                        return Wrap_ChatML (Whimsical_Adelaide, Prompt);
                     end if;
                  end if;
               end;
            else
               if Length (Internal_State) > 0 then
                  return Wrap_ChatML (Whimsical_Adelaide,
                    "User: " & Prompt & ASCII.LF &
                    "Fact-Check: " &
                    Strip_Base64_Images
                      (To_String (Internal_State)));
               else
                  return Wrap_ChatML (Whimsical_Adelaide, Prompt);
               end if;
            end if;
         end Get_Final_Prompt;
      begin
         --  [VITAL-DO-NOT-REMOVE] Mandated by user.
         Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Init-V]" &
                   AnsiAda.Reset & " Hybrid_Generate: Building final prompt. Len=" &
                   Natural'Image (Get_Final_Prompt'Length));
         --  CONTEXT FAULTING LOOP
         declare
            F_Detected   : Boolean := False;
            F_Query      : Unbounded_String;
            F_Category   : Unbounded_String;
            Hop_Count    : Natural := 0;
            Fault_Result : Unbounded_String;
         begin
            --  [VITAL-DO-NOT-REMOVE] Mandated by user.
            Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Init-V]" &
                      AnsiAda.Reset & " Hybrid_Generate: CONTEXT_FAULT_LOOP ENTERED.");
            loop
               exit when Hop_Count >= 5;

               --  Reset fault detection state for this hop. Without this,
               --  a fault detected on a previous hop would persist and
               --  cause false context-fault handling on subsequent hops
               --  even when the model didn't request one.
               F_Detected := False;

               if not External_Agent then
                  if Hop_Count = 0 then
                     Push_Orchestration_Through_Parser (Stream, Session_ID, Orch_Parser,
                       "[Adelaide Core]: [Thought] Starting reasoning " &
                       "chain..." & ASCII.LF);
                  else
                     Push_Orchestration_Through_Parser (Stream, Session_ID, Orch_Parser,
                       "[Adelaide Core]: [Thought] Continuing reasoning " &
                       "(hop" & Natural'Image (Hop_Count + 1) & ")..." &
                       ASCII.LF);
                  end if;
               end if;

                --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
                Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Init-V]" &
                          AnsiAda.Reset & " Hybrid_Generate: Final generation. Hop=" &
                          Natural'Image (Hop_Count) & " Len=" &
                          Natural'Image (Get_Final_Prompt'Length));
                Generate
                  (Kind            => Qwen_9B,
                   Prompt          => Get_Final_Prompt,
                   Result          => Fault_Result,
                   Images          => Images,
                   Session_ID      => Session_ID,
                   Requested_Ctx   => 8192,
                   Stream          => Stream,
                   Orch_Think_Open => (Hop_Count = 0),
                   Level           => Level,
                   Virtual_Tokens  => Cached_Virtual_Tokens,
                   Virtual_Tok_Len => Cached_Virtual_Len);
                 --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
                 Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Init-V]" &
                           AnsiAda.Reset & " Hybrid_Generate: Final Generate returned. Len=" &
                           Natural'Image (Length (Fault_Result)));

                 --  Parse Fault_Result for CONTEXT_FAULT marker.
                 --  [VITAL-DO-NOT-REMOVE] Mandated by user.
                 --  When the model outputs [CONTEXT_FAULT:query=X category=Y] inside
                 --  <think>, Generate's Parser detects it but cannot communicate it
                 --  back to Hybrid_Generate (Parser is local to Generate).  However,
                 --  all tokens (including the fault marker) are appended to Result
                 --  before streaming.  So Fault_Result contains the raw marker text.
                 --  Parse it here to set F_Detected, F_Query, F_Category.
                 declare
                    Raw_Result : constant String := To_String (Fault_Result);
                    F_Mark     : constant String := "[CONTEXT_FAULT:";
                    F_Mark_Pos : constant Natural := Index (Raw_Result, F_Mark);
                 begin
                    if F_Mark_Pos > 0 then
                       declare
                          Close_Pos : constant Natural :=
                            Index (Raw_Result (F_Mark_Pos .. Raw_Result'Last), "]");
                       begin
                       if Close_Pos > 0 then
                              declare
                                 --  Close_Pos is absolute (Ada.Strings.Index
                                 --  returns index within Source bounds), so
                                 --  use it directly, not F_Mark_Pos + Close_Pos.
                                 Inner     : constant String :=
                                   Raw_Result (F_Mark_Pos + F_Mark'Length ..
                                     Close_Pos - 1);
                                Q_Mark    : constant String := "query=";
                                C_Mark    : constant String := "category=";
                                Query_Idx : constant Natural := Index (Inner, Q_Mark);
                                Cat_Idx   : constant Natural := Index (Inner, C_Mark);
                             begin
                                F_Detected := True;
                                if Query_Idx > 0 then
                                   declare
                                      Q_Start : constant Natural :=
                                        Query_Idx + Q_Mark'Length;
                                      Q_End   : constant Natural :=
                                        (if Cat_Idx > Query_Idx then Cat_Idx - 1
                                         else Inner'Last + 1);
                                   begin
                                      F_Query := To_Unbounded_String
                                        (Trim (Inner (Q_Start .. Q_End - 1),
                                         Ada.Strings.Both));
                                   end;
                                end if;
                                if Cat_Idx > 0 then
                                   F_Category := To_Unbounded_String
                                     (Trim (Inner
                                       (Cat_Idx + C_Mark'Length .. Inner'Last),
                                      Ada.Strings.Both));
                                else
                                   F_Category := To_Unbounded_String ("knowledge");
                                end if;
                             end;
                          end if;
                       end;
                    end if;
                 end;

                 --  [VITAL-DO-NOT-REMOVE] Mandated by user.
                 Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Init-V]" &
                           AnsiAda.Reset & " Hybrid_Generate: F_Detected=" &
                           Boolean'Image (F_Detected) & " Hop_Count=" &
                           Natural'Image (Hop_Count));

                if F_Detected then
                  declare
                     Q_Str : constant String := To_String (F_Query);
                     C_Str : constant String := To_String (F_Category);
                     R     : Tool_Manager.Tool_Result;
                  begin
                     if not External_Agent then
                        Push_Orchestration_Through_Parser (Stream, Session_ID, Orch_Parser,
                          "[Adelaide Core]: [Thought] Context fault: " &
                          "searching " & C_Str & " for '" & Q_Str &
                          "'..." & ASCII.LF);
                     end if;

                     if C_Str = "graph" then
                        R := Tool_Manager.Execute_Tool ("searchglobalref",
                          "graph: " & Q_Str);
                     else
                        R := Tool_Manager.Execute_Tool
                          ("searchglobalref", Q_Str);
                     end if;

                   Append (Internal_State,
                     "[FACTUAL_DATA]: " & To_String (R.Output) & ASCII.LF);

                   --  Re-cache virtual ctx tokens after Internal_State grew
                   Tokenize_And_Cache_Virtual_Ctx (Model_Types.Qwen_9B,
                     "Fact-Check: " & Strip_Base64_Images (To_String (Internal_State)));

                   if not External_Agent then
                      Push_Orchestration_Through_Parser (Stream, Session_ID, Orch_Parser,
                        "[Adelaide Core]: [Thought] Context loaded for: " &
                        Q_Str & ASCII.LF);
                   end if;
                end;
                Hop_Count := Hop_Count + 1;
                --  Update context fault monitor tracking
                Current_Context_Fault_Hops := Hop_Count;
                Current_Internal_State_Len := Length (Internal_State);
               else
                  --  [VITAL-DO-NOT-REMOVE] Mandated by user.
                  Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Init-V]" &
                            AnsiAda.Reset & " Hybrid_Generate: No fault detected. Exiting loop.");
                  Current_Response := Fault_Result;
                  exit;
               end if;
            end loop;
            --  [VITAL-DO-NOT-REMOVE] Mandated by user.
            Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Init-V]" &
                      AnsiAda.Reset & " Hybrid_Generate: CONTEXT_FAULT_LOOP EXITED. Hop_Count=" &
                      Natural'Image (Hop_Count));
         end;
         --  SAFETY NET: If the entire response is think-only content,
         --  the model failed to produce a visible answer.  Set a fallback
         --  so the client gets something instead of an empty response.
         declare
            Sanitized : constant String :=
              Sanitize_Think_Tags (To_String (Current_Response));
         begin
            if Sanitized = "" then
               Put_Line (AnsiAda.Foreground (AnsiAda.Yellow) & "[Init-V]" &
                         AnsiAda.Reset &
                         " Hybrid_Generate: Think-only response detected." &
                         " Model produced no visible answer.");
               Current_Response := To_Unbounded_String
                 ("I apologize, but I was unable to generate a complete" &
                  " response. The model produced only internal reasoning" &
                  " without a final answer. Please try rephrasing your" &
                  " question or providing more context.");
            end if;
         end;

         Result := To_Unbounded_String
           (Sanitize_Think_Tags (To_String (Current_Response)));
         declare
            B64_Str : Unbounded_String := To_Unbounded_String ("");
         begin
            if GNATCOLL.JSON.Length (Images) > 0 then
               B64_Str := To_Unbounded_String
                 (String'(GNATCOLL.JSON.Get (GNATCOLL.JSON.Get (Images, 1))));
            end if;
            Database_Manager.Remember
              (Prompt, To_String (Current_Response), To_String (B64_Str));
         end;
      end;

      --  Cache control
      declare
         Resp_Str  : constant String := To_String (Current_Response);
         Is_Error  : constant Boolean :=
           Resp_Str'Length >= 6 and then Resp_Str (1 .. 6) = "ERROR:";
         Has_Think : constant Boolean :=
           Index (Resp_Str, "<thinking>") > 0 or else
           Index (Resp_Str, "<think>") > 0;
      begin
         --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
         Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Init-V]" &
                   AnsiAda.Reset & " Hybrid_Generate: COMPLETE. ResultLen=" &
                   Natural'Image (Length (Result)) & " Error=" &
                   Boolean'Image (Is_Error) & " HasThink=" &
                   Boolean'Image (Has_Think));
         if not External_Agent and then not Is_Error and then
           not Has_Think
         then
            Database_Manager.Add_To_Cache
              (Prompt, Emb_Vec (1 .. Emb_Len), Resp_Str);
         end if;
      end;

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

      if not External_Agent then
         declare
            Dur_Str : constant String := Duration'Image (T1 - T0);
         begin
            Push_Orchestration_Through_Parser (Stream, Session_ID, Orch_Parser,
              "[Adelaide Core]: [Thought] Response generated in " &
              Dur_Str & "s." & ASCII.LF);
         end;
      end if;

      if External_Agent then
         Result := To_Unbounded_String
           (Sanitize_Think_Tags (To_String (Current_Response)));
      elsif Stream = null then
         Result := To_Unbounded_String
           (Sanitize_Think_Tags (To_String (Current_Response)));
      else
         Result := Current_Response;
      end if;

      declare
         Score : constant Natural := Grade_Response_Quality
           (Response_Text => To_String (Result),
            Prompt        => Prompt,
            Search_Used   =>
              Index (To_String (Internal_State), "[FACTUAL_DATA]") > 0,
            Has_Citations => Index (To_String (Result), "[") > 0 and then
              Index (To_String (Result), "]") > 0,
            Session_ID    => Session_ID,
            Level         => Level);
      begin
         if not External_Agent then
            Push_Orchestration_Through_Parser (Stream, Session_ID, Orch_Parser,
              "[Adelaide Core]: [Thought] Self-assessment: " &
              Score'Img & "/10" & ASCII.LF);
         end if;
         Ada.Text_IO.Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) &
                               "[Quality Score] " & AnsiAda.Reset &
                               "Score: " & Score'Img & "/10 | " &
                               "Session: " & Session_ID);
      end;

      if not External_Agent then
         --  Extract and push model's internal thinking content (if any).
         --  This is the content between <think> and </think> tags that the
         --  model generated during reasoning. We push it through the parser
         --  so it is properly handled (silenced if In_Think_Block is True).
         --  NOTE: The closing `` tag is pushed by the emulated streaming
         --  section below, AFTER the response content has been streamed.
         declare
            Model_Thinking : constant String :=
              Extract_Think_Content (To_String (Current_Response));
         begin
            if Model_Thinking /= "" then
               Push_Orchestration_Through_Parser (Stream, Session_ID, Orch_Parser,
                 Model_Thinking & ASCII.LF);
            end if;
         end;
      end if;

      --  EMULATED STREAMING (300 tok/s simulation)
      --  The model's response was already streamed token-by-token through
      --  the stream parser during Generate (Process_And_Push_Chunk). The
      --  parser pushed content to the queue inside the `<think>` block.
      --  This emulated streaming loop does TWO things:
      --
      --  1. Pushes `</think>` to close the think block
      --  2. Pushes the visible response text AFTER `</think>`
      --
      --  DUPLICATION IS INTENTIONAL (DO NOT REMOVE): The response content
      --  appears both inside `<think>` (from Generate token streaming) AND
      --  after `</think>` (from this emulation). This is required for the
      --  client-side "status quo" field to decode the response correctly.
      --  Removing the re-emission would break status quo field decoding.
      --
      --  The 300 tok/s simulation delay ensures the closing tag and the
      --  re-emitted response arrive after all chunks have been flushed by
      --  AWS, making the stream appear to flow at a human-readable pace.
      if not External_Agent and then Stream /= null then
         declare
            Sim_TPS      : constant Float := 300.0;
            --  Calculate delay proportional to response length to simulate
            --  300 tok/s streaming. Short responses get minimal delay.
            Resp_Text    : constant String :=
              Sanitize_Think_Tags (To_String (Current_Response));
            Resp_Len     : constant Natural := Resp_Text'Length;
            Delay_Time   : constant Duration :=
              Duration (Float (Resp_Len) / Sim_TPS);
         begin
            --  [VITAL-DO-NOT-REMOVE] Mandated by user.
            Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Init-V]" &
                      AnsiAda.Reset & " Hybrid_Generate: Waiting " &
                      Duration'Image (Delay_Time) & "s for 300 tok/s sim.");
            delay Delay_Time;
             --  [VITAL-DO-NOT-REMOVE] Mandated by user.
             Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Init-V]" &
                       AnsiAda.Reset & " Hybrid_Generate: STREAMING COMPLETE.");
             --  [DUPLICATION IS INTENTIONAL] Pushing `</think>` first, then
             --  re-emitting the visible response for status quo decoding.
             Push_Chunk (Stream, Session_ID, ASCII.LF & "</think>" & ASCII.LF);
             if Resp_Text /= "" then
                Push_Chunk (Stream, Session_ID, Resp_Text & ASCII.LF);
             end if;
         end;
      elsif External_Agent and then Stream /= null then
         declare
            Resp_Text : constant String := To_String (Result);
         begin
            Ada.Text_IO.Put_Line
              ("[External Agent] Sending final scored response (" &
               Resp_Text'Length'Img & " chars)...");
            Push_Chunk (Stream, Session_ID, Resp_Text & ASCII.LF);
         end;
      end if;
   exception
      when E : others =>
         Ada.Text_IO.Put_Line (AnsiAda.Foreground (AnsiAda.Red) &
           "[Hybrid]" & AnsiAda.Reset & " Error: " &
           Ada.Exceptions.Exception_Message (E));
         if Stream /= null then
            begin
               Push_Chunk (Stream, Session_ID,
                 ASCII.LF & "ERROR: Generate failed" & ASCII.LF);
            exception
               when others => null;
            end;
         end if;
         Result := To_Unbounded_String ("ERROR: Generate failed");
   end Hybrid_Generate;

begin
   Initialize;
end Model_Manager;
