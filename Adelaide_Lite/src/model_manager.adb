pragma SPARK_Mode (Off);
with AnsiAda;
with Ada.Text_IO;          use Ada.Text_IO;
with Ada.Strings;          use Ada.Strings;
with Ada.Strings.Fixed;    use Ada.Strings.Fixed;
with Ada.Calendar;
use type Ada.Calendar.Time;
with Database_Manager;
with Reranker;
with LSH_Hash;
with Tool_Manager;
with Scheduler_Manager;
with Llama_Interface;      use Llama_Interface;
with Mtmd_Interface;       use Mtmd_Interface;
with Interfaces.C;         use Interfaces.C;
with Interfaces.C.Strings; use Interfaces.C.Strings;
with Ada.Directories;
with Ada.Real_Time;        use Ada.Real_Time;
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
--  failures with Qwen3.5HybridMythos at Q4_1 KV quantization).
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
--  race does not occur, so the Snowball_Enaga_ShortNetworkAnswer exemption guard should be REMOVED
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

    --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
    --  [ElabTrace-C]: RAW C trace to confirm Model_Manager body elaboration entered.
    procedure Elab_Trace_C (Label : Interfaces.C.Strings.chars_ptr);
    pragma Import (C, Elab_Trace_C, "elab_trace_c");
    function Emit_Model_Manager_Elab_Trace return Integer is
    begin
       Elab_Trace_C (Interfaces.C.Strings.New_String ("MODEL_MANAGER BODY ELABORATION ENTERED"));
       return 0;
    end Emit_Model_Manager_Elab_Trace;
    Diag_MM : constant Integer := Emit_Model_Manager_Elab_Trace;
    pragma Warnings (Off, Diag_MM);

    --  Token array types (package-level for use by Generate and
    --  Tokenize_And_Cache_Virtual_Ctx)
    type Token_Array is
       array (Positive range <>) of Llama_Interface.Llama_Token;
    type Token_Array_Access is access Token_Array;
    procedure Free_Tokens is new
       Ada.Unchecked_Deallocation (Token_Array, Token_Array_Access);

    --  [DO NOT REMOVE] C FFI for stderr suppression during model loading.
    --  llama.cpp prints hundreds of verbose lines to stderr during load.
    --  We redirect stderr to /dev/null, load, then restore.
    function Sys_Dup (Fildes : int) return int;
    pragma Import (C, Sys_Dup, "suppress_dup");
    function Sys_Restore_Stderr (Saved_Fd : int) return int;
    pragma Import (C, Sys_Restore_Stderr, "suppress_restore");

    --  [DO NOT REMOVE THIS PRINT VERBOSITY]
    --  C FFI trace functions for elaboration debugging.
    --  These write directly to fd 2 (stderr) using POSIX write(),
    --  bypassing ALL buffering (C stdio AND Ada.Text_IO).
    --  This is the ONLY way to get diagnostic output during Ada
    --  elaboration, because Ada.Text_IO may not be initialized yet.
    --  ABI NOTE: GNAT passes String as fat pointer (data_ptr, bounds_ptr).
    --  C side uses strlen() — do NOT pass a length parameter.
    procedure Elab_Trace (Label : String);
    pragma Import (C, Elab_Trace, "elab_trace_c");
    procedure Elab_Trace2 (Label1 : String; Label2 : String);
    pragma Import (C, Elab_Trace2, "elab_trace_c2");

    function Llama_Batch_Get_One
       (T : System.Address; N : int) return Llama_Batch;
    pragma Import (C, Llama_Batch_Get_One, "llama_batch_get_one");

    --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
    --  [DO NOT REMOVE THIS PRINT VERBOSITY]
    --  Init_Start_Time: Captured when Model_Manager.Initialize is called.
    --  All [Init-V] verbose prints in this package compute uptime relative
    --  to this timestamp.  DECLARED HERE (before tasks) so task bodies
    --  can reference it during elaboration traces.
    --  INITIALIZED to Clock so task activations don't crash on first use.
    Init_Start_Time : Ada.Real_Time.Time := Ada.Real_Time.Clock;

    --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
    --  [ElabTrace-C]: Confirms elaboration past Init_Start_Time declaration.
    function Emit_After_Init_Start return Integer is
    begin
       Elab_Trace_C (Interfaces.C.Strings.New_String ("MODEL_MANAGER: AFTER_INIT_START_TIME"));
       return 0;
    end Emit_After_Init_Start;
    Diag_AIS : constant Integer := Emit_After_Init_Start;
    pragma Warnings (Off, Diag_AIS);

    task type WCET_Printer;
    task body WCET_Printer is
    begin
        --  [DO NOT REMOVE THIS PRINT VERBOSITY]
        --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
        --  [ElabTrace-C][+Uptime]: RAW C trace (write to stderr) to confirm
        --  WCET_Printer task body entered during elaboration.
        --  If this never prints, task activation deadlocked.
        Elab_Trace ("WCET_Printer task body ENTERED");
        loop
            delay 30.0;
            Ada.Text_IO.Put_Line
               (AnsiAda.Foreground (AnsiAda.Light_Red)
                & "[WCET]"
                & AnsiAda.Reset
                & " Pipeline: "
                & Long_Long_Integer'Image
                     (Long_Long_Integer (Current_WCET * 1_000_000_000))
                & "ns | "
                & "ELP0: "
                & Long_Long_Integer'Image
                     (Long_Long_Integer (Current_WCET_ELP0 * 1_000_000_000))
                & "ns | "
                & "ELP1: "
                & Long_Long_Integer'Image
                     (Long_Long_Integer (Current_WCET_ELP1 * 1_000_000_000))
                & "ns | "
                & "ELP2: "
                & Long_Long_Integer'Image
                     (Long_Long_Integer (Current_WCET_ELP2 * 1_000_000_000))
                & "ns | "
                & "ELP3: "
                & Long_Long_Integer'Image
                     (Long_Long_Integer (Current_WCET_ELP3 * 1_000_000_000))
                & "ns");
        end loop;
    end WCET_Printer;

    Printer_Task : WCET_Printer;

    --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
    --  [ElabTrace-C]: RAW C trace: after Printer_Task declaration.
    function Emit_After_Printer_Task return Integer is
    begin
       Elab_Trace_C (Interfaces.C.Strings.New_String ("MODEL_MANAGER: AFTER_PRINTER_TASK_DECL"));
       return 0;
    end Emit_After_Printer_Task;
    Diag_APT : constant Integer := Emit_After_Printer_Task;
    pragma Warnings (Off, Diag_APT);

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
        Interval    : constant Duration := 5.0;
        Next_Check  : Ada.Calendar.Time;
        Fault_Total : Natural := 0;
    begin
        --  [DO NOT REMOVE THIS PRINT VERBOSITY]
        --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
        --  [ElabTrace-C]: RAW C trace to confirm Context_Monitor task body entered.
        --  If this never prints, task activation deadlocked.
        Elab_Trace ("Context_Monitor task body ENTERED");
        accept Start;
        loop
            Next_Check := Ada.Calendar.Clock + Interval;

            --  Aggregate context fault hops across all active sessions
            --  (Current_Context_Fault_Hops is updated by Hybrid_Generate)
            Fault_Total := Current_Context_Fault_Hops;

            declare
                --  Virtual Context (2^63) metrics from ELP Queue
                VC_Capacity : constant Interfaces.Unsigned_64 :=
                   ELP_Queue.Capacity;
                VC_Depth    : constant Long_Long_Integer := ELP_Queue.Depth;
                VC_Util     : constant Long_Long_Float :=
                   ELP_Queue.Utilization;
                VC_Pct      : constant Long_Long_Float := VC_Util * 100.0;

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
                    then Cached_Virtual_Len
                    --  Exact count from token cache
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
                Put_Line
                   (AnsiAda.Foreground (AnsiAda.Light_Cyan)
                    & "[CtxMonitor]"
                    & AnsiAda.Reset
                    & " === VIRTUAL CONTEXT STATUS (5s) ===");

                --  ELP Queue: request depth (synthetic 2^63 capacity)
                Put_Line
                   (AnsiAda.Foreground (AnsiAda.Light_Cyan)
                    & "[CtxMonitor]"
                    & AnsiAda.Reset
                    & " ELP Queue: "
                    & Long_Long_Integer'Image (VC_Depth)
                    & " /"
                    & Interfaces.Unsigned_64'Image (VC_Capacity)
                    & " pending ("
                    & Long_Long_Float'Image (VC_Pct)
                    & "% used)");

                --  Virtual Context: accumulated factual data (Internal_State)
                --  This is the data paged in from tool results across hops
                --  Capacity is 2^63 = 9223372036854775807 (Virtual Context Model)
                Put_Line
                   (AnsiAda.Foreground (AnsiAda.Light_Cyan)
                    & "[CtxMonitor]"
                    & AnsiAda.Reset
                    & " Virtual CTX: "
                    & Natural'Image (VC_Bytes)
                    & " bytes / "
                    & Natural'Image (VC_Tokens)
                    & " ~tokens"
                    & " out of 9223372036854775807"
                    & " ("
                    & Natural'Image (VC_Ctx_Pct)
                    & "% of LLM window)");

                --  LLM Context: actual tokens in the prompt submitted to llama
                Put_Line
                   (AnsiAda.Foreground (AnsiAda.Light_Cyan)
                    & "[CtxMonitor]"
                    & AnsiAda.Reset
                    & " LLM CTX:    "
                    & Natural'Image (Prompt_Toks)
                    & " / "
                    & Natural'Image (LLM_Ctx)
                    & " tokens"
                    & " ("
                    & Natural'Image (LLM_Pct)
                    & "% used)");

                --  Context Fault Division Page
                Put_Line
                   (AnsiAda.Foreground (AnsiAda.Light_Cyan)
                    & "[CtxMonitor]"
                    & AnsiAda.Reset
                    & " Context Fault Page: "
                    & Natural'Image (Cur_Division)
                    & " /"
                    & Natural'Image (Max_Divisions)
                    & " | Hops="
                    & Natural'Image (Fault_Total)
                    & "/5");

                --  Internal_State size + page jump state
                Put_Line
                   (AnsiAda.Foreground (AnsiAda.Light_Cyan)
                    & "[CtxMonitor]"
                    & AnsiAda.Reset
                    & " Internal_State="
                    & Natural'Image (VC_Bytes)
                    & " bytes"
                    & " | Page="
                    & (if Fault_Total = 0
                       then "INITIAL"
                       else "HOP" & Natural'Image (Fault_Total)));

                Put_Line
                   (AnsiAda.Foreground (AnsiAda.Light_Cyan)
                    & "[CtxMonitor]"
                    & AnsiAda.Reset
                    & " ======================================");
            end;

            delay until Next_Check;
        end loop;
    end Context_Monitor;

    --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
    --  [ElabTrace-C]: RAW C trace: after Context_Monitor task body.
    function Emit_After_CtxMon return Integer is
    begin
       Elab_Trace_C (Interfaces.C.Strings.New_String ("MODEL_MANAGER: AFTER_CTXMON_BODY"));
       return 0;
    end Emit_After_CtxMon;
    Diag_ACM : constant Integer := Emit_After_CtxMon;
    pragma Warnings (Off, Diag_ACM);

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

    --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
    --  [ElabTrace-C]: RAW C trace: after Models array declaration.
    function Emit_After_Models return Integer is
    begin
       Elab_Trace_C (Interfaces.C.Strings.New_String ("MODEL_MANAGER: AFTER_MODELS_ARRAY"));
       return 0;
    end Emit_After_Models;
    Diag_AM : constant Integer := Emit_After_Models;
    pragma Warnings (Off, Diag_AM);

    type Model_Type_Refs is array (Model_Type) of aliased Model_Type;

    --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
    --  [ElabTrace-C]: RAW C trace: after Model_Type_Refs type.
    function Emit_After_Type_Refs return Integer is
    begin
       Elab_Trace_C (Interfaces.C.Strings.New_String ("MODEL_MANAGER: AFTER_TYPE_REFS"));
       return 0;
    end Emit_After_Type_Refs;
    Diag_ATR : constant Integer := Emit_After_Type_Refs;
    pragma Warnings (Off, Diag_ATR);

    Model_Refs : constant Model_Type_Refs :=
       (Snowball_Enaga_ShortNetworkAnswer => Snowball_Enaga_ShortNetworkAnswer,
        Snowball_Enaga_Orchestrator       => Snowball_Enaga_Orchestrator,
        Qwen_Embedding                    => Qwen_Embedding,
        MMProj                            => MMProj,
        others                            => Snowball_Enaga_ShortNetworkAnswer);

    --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
    --  [ElabTrace-C]: RAW C trace: after Model_Refs constant.
    function Emit_After_Model_Refs return Integer is
    begin
       Elab_Trace_C (Interfaces.C.Strings.New_String ("MODEL_MANAGER: AFTER_MODEL_REFS"));
       return 0;
    end Emit_After_Model_Refs;
    Diag_AMR : constant Integer := Emit_After_Model_Refs;
    pragma Warnings (Off, Diag_AMR);

    type Owner_Array is array (Model_Type) of ELP_Level;
    type Busy_Array is array (Model_Type) of Boolean;

    protected Accel_Lock_Object is
        entry Acquire;
        procedure Release;
    private
        Busy : Boolean := False;
    end Accel_Lock_Object;

    --  PRIORITY MODEL GATE:
    --  Manages access to the model contexts.
    --  ELP1 requests (User Interactions) preempt running ELP0 requests (Background Tasks).
    protected Priority_Model_Gate is
        procedure Request_ELP1;
        entry Acquire_ELP1 (Model_Type);
        procedure Release_ELP1 (Kind : Model_Type);
        entry Acquire_ELP0 (Model_Type) (Success : out Boolean);
        procedure Release_ELP0 (Kind : Model_Type);
        procedure Try_Acquire_For_Cleanup
           (Kind : Model_Type; Success : out Boolean);
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

    protected body Accel_Lock_Object is
        --  [DO NOT REMOVE THIS PRINT VERBOSITY]
        --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
        --  [ElabTrace][+Uptime]: Confirms Accel_Lock_Object protected body
        --  elaboration reached. If this never prints, the elaboration of
        --  Accel_Lock_Object deadlocked BEFORE entering the protected body.
        entry Acquire when not Busy is
        begin
            Busy := True;
        end Acquire;
        procedure Release is
        begin
            Busy := False;
        end Release;
    end Accel_Lock_Object;

    protected body Priority_Model_Gate is
        --  [DO NOT REMOVE THIS PRINT VERBOSITY]
        --  [ElabTrace][+Uptime]: Confirms Priority_Model_Gate protected body
        --  elaboration reached. If this never prints, the elaboration deadlocked
        --  BEFORE entering the protected body (during protected type spec elaboration).
        procedure Request_ELP1 is
        begin
            --  [DO NOT REMOVE THIS PRINT VERBOSITY]
            --  [ElabTrace][+Uptime]: First executable line in Priority_Model_Gate.
            --  If this prints, the protected body elaboration succeeded.
            Put_Line
               (AnsiAda.Foreground (AnsiAda.Light_Green)
                & "[ElabTrace]"
                & AnsiAda.Reset
                & "+"
                & Trim
                     (Duration'Image
                         (Ada.Real_Time.To_Duration
                             (Ada.Real_Time.Clock - Init_Start_Time)),
                      Both)
                & "s Priority_Model_Gate protected body ELABORATED OK");
            ELP1_Pending := ELP1_Pending + 1;
            Put_Line
               ("[ELP1-REQUEST] Pending ELP1 requests: " & ELP1_Pending'Img);
        end Request_ELP1;

        entry Acquire_ELP1(for K in Model_Type) when not Busy (K) is
        begin
            ELP1_Pending := ELP1_Pending - 1;
            Busy (K) := True;
            Owner (K) := ELP1;
            ELP1_Active_Count := ELP1_Active_Count + 1;
            Put_Line
               ("[ELP1-ACQUIRED] "
                & K'Img
                & " | Active: "
                & ELP1_Active_Count'Img
                & " | Pending: "
                & ELP1_Pending'Img);
        end Acquire_ELP1;

        procedure Release_ELP1 (Kind : Model_Type) is
        begin
            Busy (Kind) := False;
            Owner (Kind) := ELP0;
            if ELP1_Active_Count > 0 then
                ELP1_Active_Count := ELP1_Active_Count - 1;
            end if;
            Put_Line
               ("[ELP1-RELEASED] "
                & Kind'Img
                & " | Active: "
                & ELP1_Active_Count'Img
                & " | Pending: "
                & ELP1_Pending'Img);
        end Release_ELP1;

        entry Acquire_ELP0(for K in Model_Type) (Success : out Boolean)
           when(not Busy (K)
                or else ELP1_Pending > 0
                or else ELP1_Active_Count > 0)
           and then (not On_Battery_State or else Battery_Level >= 80)
        is
        begin
            if ELP1_Pending > 0 or else ELP1_Active_Count > 0 then
                Success := False;
                Put_Line
                   ("[ELP0-DENIED] "
                    & K'Img
                    & " | ELP1 Pending: "
                    & ELP1_Pending'Img
                    & " | ELP1 Active: "
                    & ELP1_Active_Count'Img);
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
            if Busy (Kind)
               or else ELP1_Pending > 0
               or else ELP1_Active_Count > 0
            then
                Success := False;
            else
                Busy (Kind) := True;
                Owner (Kind) :=
                   ELP1; -- Treat cleanup as high priority/exclusive
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
            return
               ELP1_Pending > 0
               or else ELP1_Active_Count > 0
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
        entry Wait_For_ELP1_Idle
           when(ELP1_Pending = 0 and then ELP1_Active_Count = 0)
           and then (not On_Battery_State or else Battery_Level >= 80)
        is
        begin
            Put_Line
               (AnsiAda.Foreground (AnsiAda.Light_Blue)
                & "[Init-V]"
                & AnsiAda.Reset
                & "+"
                & Trim
                     (Duration'Image
                         (Ada.Real_Time.To_Duration
                             (Ada.Real_Time.Clock - Init_Start_Time)),
                      Both)
                & "s Wait_For_ELP1_Idle GUARD PASSED"
                & " ELP1_Pending="
                & ELP1_Pending'Img
                & " ELP1_Active="
                & ELP1_Active_Count'Img
                & " OnBattery="
                & On_Battery_State'Img
                & " BattLevel="
                & Battery_Level'Img);
        end Wait_For_ELP1_Idle;

        procedure Set_Power_Condition (On_Battery : Boolean; Level : Natural)
        is
        begin
            On_Battery_State := On_Battery;
            Battery_Level := Level;
        end Set_Power_Condition;
    end Priority_Model_Gate;

    --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
    --  [ElabTrace-C]: RAW C trace: protected bodies elaboration done.
    function Emit_After_Protected return Integer is
    begin
       Elab_Trace_C (Interfaces.C.Strings.New_String ("MODEL_MANAGER: AFTER_PROTECTED_BODIES"));
       return 0;
    end Emit_After_Protected;
    Diag_AP : constant Integer := Emit_After_Protected;
    pragma Warnings (Off, Diag_AP);

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
        --  [DO NOT REMOVE THIS PRINT VERBOSITY]
        --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
        --  [ElabTrace-C]: RAW C trace to confirm Idle_Monitor task body entered.
        --  If this never prints, task activation deadlocked.
        Elab_Trace ("Idle_Monitor task body ENTERED");
        --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
        --  Verbose: confirms the Idle_Monitor task actually started.
        Put_Line
           (AnsiAda.Foreground (AnsiAda.Light_Blue)
            & "[Init-V]"
            & AnsiAda.Reset
            & "+"
            & Trim
                 (Duration'Image
                     (Ada.Real_Time.To_Duration
                         (Ada.Real_Time.Clock - Init_Start_Time)),
                  Both)
            & "s Idle_Monitor task entered, waiting for Start...");
        accept Start;
        Put_Line
           (AnsiAda.Foreground (AnsiAda.Light_Blue)
            & "[Init-V]"
            & AnsiAda.Reset
            & "+"
            & Trim
                 (Duration'Image
                     (Ada.Real_Time.To_Duration
                         (Ada.Real_Time.Clock - Init_Start_Time)),
                  Both)
            & "s Idle_Monitor task ACCEPTED Start, entering loop.");
        loop
            Next_Check := Clock + Interval;
            Now := Clock;
            for Kind in Model_Type loop
                --  [PARALLEL=1 FIX] Removed old exemption guard that kept
                --  0.8B and 9B models permanently resident in GPU.
                --  LM Studio works because it loads ONE model at a time.
                --  Adelaide_Lite crashed because multiple models competed
                --  for GPU VRAM. The Metal crash (QUIRK-M03) that caused
                --  this exemption is already fixed by Wait_For_Save + proper
                --  Unload_Model sequencing in Hybrid_Generate.
                --  ALL models now unload when idle, just like LM Studio.
                if Models (Kind).Loaded
                   and then not Models (Kind).In_Use
                   and then (Now - Models (Kind).Last_Used) > Timeout
                then
                    Priority_Model_Gate.Try_Acquire_For_Cleanup
                       (Kind, Cleanup_OK);
                    if Cleanup_OK then
                        Put_Line
                           (AnsiAda.Foreground (AnsiAda.Grey)
                            & "[Idle]"
                            & AnsiAda.Reset
                            & " Unloading "
                            & Model_Type'Image (Kind));
                        Unload_Model (Kind);
                        --  Match Acquire_For_Cleanup
                        Priority_Model_Gate.Release_ELP1 (Kind);
                    end if;
                end if;
            end loop;
            delay until Next_Check;
        end loop;
    end Idle_Monitor;

    --  =========================================================================
    --  GPU MEMORY MONITOR TASK
    --  =========================================================================
    --  Runs every 3 seconds. Queries GPU VRAM across ALL backends
    --  (Metal, CUDA, OneAPI, SYCL, Vulkan, ROCm).
    --  Prints free/total MB and dynamic N_GPU_Layers percentage.
    --  If GPU memory query returns 0,0 (inapplicable on Vulkan/CPU),
    --  reports "stable" or "UNSTABLE" based on Metal_Backend_Broken flag.
    --  If unstable (OOM/crash), sets N_GPU_Layers to 0.
    --  If stable with plenty of VRAM, calculates N_GPU_Layers as percentage.
    --  Also injects status into <think> block via Push_Orchestration_Direct.

    GPU_Monitor_Interval : constant Duration := 3.0;

    task GPU_Monitor is
        pragma Storage_Size (512 * 1024);
        entry Start;
    end GPU_Monitor;

    task body GPU_Monitor is
        Free_Bytes  : Interfaces.C.size_t := 0;
        Total_Bytes : Interfaces.C.size_t := 0;
        Free_MB     : Natural := 0;
        Total_MB    : Natural := 0;
        Percent     : Natural := 0;
        Next_Check  : Time;
        Uptime_Sec  : Natural;
        Status_Str  : Unbounded_String;
    begin
        --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
        Elab_Trace ("GPU_Monitor task body ENTERED");
        accept Start;
        --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
        Put_Line
           (AnsiAda.Foreground (AnsiAda.Light_Blue)
            & "[Init-V]"
            & AnsiAda.Reset
            & "+"
            & Trim
                 (Duration'Image
                     (Ada.Real_Time.To_Duration
                         (Ada.Real_Time.Clock - Init_Start_Time)),
                  Both)
            & "s GPU_Monitor task ACCEPTED Start, entering 3s loop.");

        loop
            Next_Check := Clock + Seconds (3);
            Uptime_Sec := Natural
               (Ada.Real_Time.To_Duration (Clock - Init_Start_Time));

            --  Query GPU memory through ggml backend (ALL backends)
            Llama_Interface.GPU_Memory_Query (Free_Bytes, Total_Bytes);

            if Total_Bytes > 0 then
                --  GPU memory query WORKS (Metal, CUDA, OneAPI, SYCL, ROCm)
                Free_MB  := Natural (Free_Bytes / (1024 * 1024));
                Total_MB := Natural (Total_Bytes / (1024 * 1024));

                if Total_MB > 0 then
                    Percent := Natural
                       (Float (Free_MB) * 100.0 / Float (Total_MB));
                    if Percent > 100 then
                        Percent := 100;
                    end if;
                else
                    Percent := 0;
                end if;

                --  Update global GPU status
                GPU_Free_MB       := Free_MB;
                GPU_Total_MB      := Total_MB;
                GPU_Layer_Percent := Percent;
                GPU_Is_Stable     := True;

                --  Build status string: show free/total, percentage,
                --  AND the actual GPU_Layer_Count (what Load_Model uses)
                Status_Str :=
                   To_Unbounded_String
                      ("[GPU-Monitor] [Uptime]+"
                       & Trim (Natural'Image (Uptime_Sec), Both)
                       & "s Free=" & Trim (Natural'Image (Free_MB), Both)
                       & "MB / Total=" & Trim (Natural'Image (Total_MB), Both)
                       & "MB (" & Trim (Natural'Image (Percent), Both)
                       & "%) GPU_Layers="
                       & (if GPU_Layer_Count = -1 then "ALL(-1)"
                          else Trim (Integer'Image (GPU_Layer_Count), Both)));

                --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
                Put_Line
                   (AnsiAda.Foreground (AnsiAda.Light_Cyan)
                    & To_String (Status_Str)
                    & AnsiAda.Reset);
            else
                --  GPU memory query INAPPLICABLE (Vulkan without query, CPU-only)
                --  Report stable/unstable based on Metal backend state
                if Is_Metal_Broken then
                    GPU_Is_Stable := False;
                    Status_Str :=
                       To_Unbounded_String
                          ("[GPU-Monitor] [Uptime]+"
                           & Trim (Natural'Image (Uptime_Sec), Both)
                           & "s GPU=INAPPLICABLE Status=UNSTABLE"
                           & " (OOM/crash detected) GPU_Layers=0"
                           & " -- forcing CPU-only mode");
                    --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
                    Put_Line
                       (AnsiAda.Foreground (AnsiAda.Red)
                        & To_String (Status_Str)
                        & AnsiAda.Reset);
                else
                    GPU_Is_Stable := True;
                    Status_Str :=
                       To_Unbounded_String
                          ("[GPU-Monitor] [Uptime]+"
                           & Trim (Natural'Image (Uptime_Sec), Both)
                           & "s GPU=INAPPLICABLE Status=STABLE"
                           & " GPU_Layers="
                           & (if GPU_Layer_Count = -1 then "ALL(-1)"
                              else Trim (Integer'Image (GPU_Layer_Count), Both)));
                    --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
                    Put_Line
                       (AnsiAda.Foreground (AnsiAda.Light_Blue)
                        & To_String (Status_Str)
                        & AnsiAda.Reset);
                end if;
            end if;

            delay until Next_Check;
        end loop;
    end GPU_Monitor;

    function Wrap_ChatML (Sys : String; Msg : String) return String is
    begin
        return
           "<|im_start|>system"
           & ASCII.LF
           & Sys
           & "<|im_end|>"
           & ASCII.LF
           & "<|im_start|>user"
           & ASCII.LF
           & Msg
           & "<|im_end|>"
           & ASCII.LF
           & "<|im_start|>assistant"
           & ASCII.LF;
    end Wrap_ChatML;

    Initialized : Boolean := False;

    procedure Initialize is
    begin
        --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
        --  Guard: prevent double initialization (idle monitor blocks on 2nd Start).
        if Initialized then
            Put_Line
               (AnsiAda.Foreground (AnsiAda.Light_Blue)
                & "[Init-V]"
                & AnsiAda.Reset
                & " Model_Manager.Initialize: ALREADY INITIALIZED, skipping.");
            return;
        end if;
        Initialized := True;
        --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
        --  Capture start time for uptime calculation.
        Init_Start_Time := Ada.Real_Time.Clock;
        --  [VITAL-DO-NOT-REMOVE] Initialize Generate_Seed with current time.
        --  This ensures different output on each retry for think-only responses.
        Generate_Seed :=
           Interfaces.C.unsigned
              (Ada.Calendar.Seconds (Ada.Calendar.Clock));
        --  Verbose init tracing: each print confirms a subsystem completed.
        --  If the server hangs during init, the LAST print before silence
        --  tells you exactly which step is stuck.
        --  [DO NOT REMOVE THIS PRINT VERBOSITY]
        --  [ElabTrace][+Uptime]: Initialize entry point reached.
        Put_Line
           (AnsiAda.Foreground (AnsiAda.Light_Red)
            & "[ElabTrace]"
            & AnsiAda.Reset
            & "+0.0s Initialize procedure ENTERED");
        Put_Line
           (AnsiAda.Foreground (AnsiAda.Light_Blue)
            & "[Init-V]"
            & AnsiAda.Reset
            & "+"
            & Trim
                 (Duration'Image
                     (Ada.Real_Time.To_Duration
                         (Ada.Real_Time.Clock - Init_Start_Time)),
                  Both)
            & "s 1/7 Calling Llama_Backend_Init...");
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
        --  [DO NOT REMOVE THIS PRINT VERBOSITY]
        --  [ElabTrace][+Uptime]: Llama_Backend_Init completed.
        Put_Line
           (AnsiAda.Foreground (AnsiAda.Light_Red)
            & "[ElabTrace]"
            & AnsiAda.Reset
            & "+"
            & Trim
                 (Duration'Image
                     (Ada.Real_Time.To_Duration
                         (Ada.Real_Time.Clock - Init_Start_Time)),
                  Both)
            & "s ElabTrace 1/7 Llama_Backend_Init DONE");
        Put_Line
           (AnsiAda.Foreground (AnsiAda.Light_Blue)
            & "[Init-V]"
            & AnsiAda.Reset
            & "+"
            & Trim
                 (Duration'Image
                     (Ada.Real_Time.To_Duration
                         (Ada.Real_Time.Clock - Init_Start_Time)),
                  Both)
            & "s 2/7 Llama_Backend_Init DONE.");

        Put_Line
           (AnsiAda.Foreground (AnsiAda.Light_Blue)
            & "[Init-V]"
            & AnsiAda.Reset
            & "+"
            & Trim
                 (Duration'Image
                     (Ada.Real_Time.To_Duration
                         (Ada.Real_Time.Clock - Init_Start_Time)),
                  Both)
            & "s 3/7 Calling Database_Manager.Initialize...");
        --  [DO NOT REMOVE THIS PRINT VERBOSITY]
        --  [ElabTrace][+Uptime]: About to call Database_Manager.Initialize.
        Put_Line
           (AnsiAda.Foreground (AnsiAda.Light_Red)
            & "[ElabTrace]"
            & AnsiAda.Reset
            & "+"
            & Trim
                 (Duration'Image
                     (Ada.Real_Time.To_Duration
                         (Ada.Real_Time.Clock - Init_Start_Time)),
                  Both)
            & "s ElabTrace 2/7 Calling Database_Manager.Initialize...");
        Database_Manager.Initialize;
        --  [DO NOT REMOVE THIS PRINT VERBOSITY]
        --  [ElabTrace][+Uptime]: Database_Manager.Initialize completed.
        Put_Line
           (AnsiAda.Foreground (AnsiAda.Light_Red)
            & "[ElabTrace]"
            & AnsiAda.Reset
            & "+"
            & Trim
                 (Duration'Image
                     (Ada.Real_Time.To_Duration
                         (Ada.Real_Time.Clock - Init_Start_Time)),
                  Both)
            & "s ElabTrace 3/7 Database_Manager.Initialize DONE");
        Put_Line
           (AnsiAda.Foreground (AnsiAda.Light_Blue)
            & "[Init-V]"
            & AnsiAda.Reset
            & "+"
            & Trim
                 (Duration'Image
                     (Ada.Real_Time.To_Duration
                         (Ada.Real_Time.Clock - Init_Start_Time)),
                  Both)
            & "s 4/7 Database_Manager.Initialize DONE.");

        Put_Line
           (AnsiAda.Foreground (AnsiAda.Light_Blue)
            & "[Init-V]"
            & AnsiAda.Reset
            & "+"
            & Trim
                 (Duration'Image
                     (Ada.Real_Time.To_Duration
                         (Ada.Real_Time.Clock - Init_Start_Time)),
                  Both)
            & "s 5/7 Calling ELP_Queue.Initialize...");
        --  [DO NOT REMOVE THIS PRINT VERBOSITY]
        --  [ElabTrace][+Uptime]: About to call ELP_Queue.Initialize.
        Put_Line
           (AnsiAda.Foreground (AnsiAda.Light_Red)
            & "[ElabTrace]"
            & AnsiAda.Reset
            & "+"
            & Trim
                 (Duration'Image
                     (Ada.Real_Time.To_Duration
                         (Ada.Real_Time.Clock - Init_Start_Time)),
                  Both)
            & "s ElabTrace 4/7 Calling ELP_Queue.Initialize...");
        ELP_Queue.Initialize;
        --  [DO NOT REMOVE THIS PRINT VERBOSITY]
        --  [ElabTrace][+Uptime]: ELP_Queue.Initialize completed.
        Put_Line
           (AnsiAda.Foreground (AnsiAda.Light_Red)
            & "[ElabTrace]"
            & AnsiAda.Reset
            & "+"
            & Trim
                 (Duration'Image
                     (Ada.Real_Time.To_Duration
                         (Ada.Real_Time.Clock - Init_Start_Time)),
                  Both)
            & "s ElabTrace 5/7 ELP_Queue.Initialize DONE");
        Put_Line
           (AnsiAda.Foreground (AnsiAda.Light_Blue)
            & "[Init-V]"
            & AnsiAda.Reset
            & "+"
            & Trim
                 (Duration'Image
                     (Ada.Real_Time.To_Duration
                         (Ada.Real_Time.Clock - Init_Start_Time)),
                  Both)
            & "s 6/7 ELP_Queue.Initialize DONE.");

        --  Start Virtual Context Monitor (prints every 5s)
        --  [DO NOT REMOVE THIS PRINT VERBOSITY]
        --  [ElabTrace][+Uptime]: About to start Context_Monitor.
        Put_Line
           (AnsiAda.Foreground (AnsiAda.Light_Red)
            & "[ElabTrace]"
            & AnsiAda.Reset
            & "+"
            & Trim
                 (Duration'Image
                     (Ada.Real_Time.To_Duration
                         (Ada.Real_Time.Clock - Init_Start_Time)),
                  Both)
            & "s ElabTrace 6/7 Starting Context_Monitor...");
        if not Context_Monitor'Terminated then
            Context_Monitor.Start;
        end if;
        --  [DO NOT REMOVE THIS PRINT VERBOSITY]
        --  [ElabTrace][+Uptime]: Context_Monitor started.
        Put_Line
           (AnsiAda.Foreground (AnsiAda.Light_Red)
            & "[ElabTrace]"
            & AnsiAda.Reset
            & "+"
            & Trim
                 (Duration'Image
                     (Ada.Real_Time.To_Duration
                         (Ada.Real_Time.Clock - Init_Start_Time)),
                  Both)
            & "s ElabTrace 6/7 Context_Monitor.START called");

        --  Initialize KV Cache Manager for SSD spillover
        --  [DO NOT REMOVE THIS PRINT VERBOSITY]
        --  [ElabTrace][+Uptime]: About to call KV_Cache_Manager.Initialize.
        Put_Line
           (AnsiAda.Foreground (AnsiAda.Light_Red)
            & "[ElabTrace]"
            & AnsiAda.Reset
            & "+"
            & Trim
                 (Duration'Image
                     (Ada.Real_Time.To_Duration
                         (Ada.Real_Time.Clock - Init_Start_Time)),
                  Both)
            & "s ElabTrace 6b/7 Calling KV_Cache_Manager.Initialize...");
        KV_Cache_Manager.Initialize;
        --  [DO NOT REMOVE THIS PRINT VERBOSITY]
        --  [ElabTrace][+Uptime]: KV_Cache_Manager.Initialize completed.
        Put_Line
           (AnsiAda.Foreground (AnsiAda.Light_Red)
            & "[ElabTrace]"
            & AnsiAda.Reset
            & "+"
            & Trim
                 (Duration'Image
                     (Ada.Real_Time.To_Duration
                         (Ada.Real_Time.Clock - Init_Start_Time)),
                  Both)
            & "s ElabTrace 6b/7 KV_Cache_Manager.Initialize DONE");

        --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
        --  Model paths are set here.  None of these load models from disk.
        --  Loading happens lazily in Load_Model on first use.
        Models (Snowball_Enaga_ShortNetworkAnswer).Path :=
           To_Unbounded_String ("model/Qwen3.5-0.8B-Q4_K_M.gguf");
        Models (Snowball_Enaga_Orchestrator).Path :=
           To_Unbounded_String ("model/Mythos9bHybridq4.gguf");
        Models (Qwen_Embedding).Path :=
           To_Unbounded_String
              ("model/Qwen3-Embedding-0.6B-Q8_0.gguf");
        Models (MMProj).Path :=
           To_Unbounded_String
              ("model/Mythos9bHybridq4-mmproj-fp16.gguf");

        Put_Line
           (AnsiAda.Foreground (AnsiAda.Light_Blue)
            & "[Init-V]"
            & AnsiAda.Reset
            & "+"
            & Trim
                 (Duration'Image
                     (Ada.Real_Time.To_Duration
                         (Ada.Real_Time.Clock - Init_Start_Time)),
                  Both)
            & "s 7/7 Starting Idle_Monitor...");
        --  [DO NOT REMOVE THIS PRINT VERBOSITY]
        --  [ElabTrace][+Uptime]: About to start Idle_Monitor.
        Put_Line
           (AnsiAda.Foreground (AnsiAda.Light_Red)
            & "[ElabTrace]"
            & AnsiAda.Reset
            & "+"
            & Trim
                 (Duration'Image
                     (Ada.Real_Time.To_Duration
                         (Ada.Real_Time.Clock - Init_Start_Time)),
                  Both)
             & "s ElabTrace 7/7 Starting Idle_Monitor...");
        Idle_Monitor.Start;
        --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
        --  Start GPU memory monitor (queries every 3s across ALL backends)
        GPU_Monitor.Start;
        --  [DO NOT REMOVE THIS PRINT VERBOSITY]
        --  [ElabTrace][+Uptime]: Idle_Monitor started. Initialize COMPLETE.
        Put_Line
           (AnsiAda.Foreground (AnsiAda.Light_Red)
            & "[ElabTrace]"
            & AnsiAda.Reset
            & "+"
            & Trim
                 (Duration'Image
                     (Ada.Real_Time.To_Duration
                         (Ada.Real_Time.Clock - Init_Start_Time)),
                  Both)
            & "s ElabTrace 7/7 Idle_Monitor.START + GPU_Monitor.START called -- Initialize COMPLETE");
        Put_Line
           (AnsiAda.Foreground (AnsiAda.Light_Blue)
            & "[Init-V]"
            & AnsiAda.Reset
            & "+"
            & Trim
                 (Duration'Image
                     (Ada.Real_Time.To_Duration
                         (Ada.Real_Time.Clock - Init_Start_Time)),
                  Both)
            & "s Model_Manager.Initialize COMPLETE.");
    end Initialize;

     procedure Load_Model
        (Kind          : Model_Type;
         Success       : out Boolean;
         Requested_Ctx : Positive := 4096;
         Level         : ELP_Level := ELP1;
         Session_ID    : String := "")
     is
        --  [PARALLEL=1] Before calling Load_Model, ensure NO OTHER model is
        --  loaded. Only one model can occupy GPU memory at a time. If another
        --  model is loaded, call Unload_Model on it FIRST, or this call will
        --  Metal OOM. The calling code (Get_Single_Embedding, Hybrid_Generate)
        --  is responsible for enforcing this invariant.
        M_Params   : Llama_Model_Params := Llama_Model_Default_Params;
        C_Params   : Llama_Context_Params := Llama_Context_Default_Params;
        Actual_Ctx : unsigned;

        Base_Path : constant String := To_String (Models (Kind).Path);
        -- Try direct, ../ (from src/Adelaide_Lite), and ../../ (from bin)
        Paths     : constant array (1 .. 3) of Unbounded_String :=
           (To_Unbounded_String (Base_Path),
            To_Unbounded_String ("../" & Base_Path),
            To_Unbounded_String ("../../" & Base_Path));
    begin
        --  [ADAPTIVE GPU RETRY] If we previously fell back from -1 due to OOM,
        --  check if 3 minutes have passed. If so, retry -1 (all on GPU).
        --  This auto-probes whether the GPU can handle full offload after
        --  cooling down (other processes may have freed VRAM).
        if GPU_Layer_Count /= -1 and then GPU_Last_OOM_Time /= Time_First then
            declare
                Elapsed : constant Duration :=
                   Ada.Real_Time.To_Duration (Clock - GPU_Last_OOM_Time);
            begin
                if Elapsed >= GPU_Retry_Interval then
                    --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
                    Put_Line
                       (AnsiAda.Foreground (AnsiAda.Light_Cyan)
                        & "[GPU-Adaptive]"
                        & AnsiAda.Reset
                        & " 3 min cooldown elapsed. Retrying full GPU (-1)."
                        & " Was at fallback=" & Integer'Image (GPU_Layer_Count));
                    GPU_Layer_Count := -1;  -- Retry aggressive
                end if;
            end;
        end if;

        Actual_Ctx := unsigned (Requested_Ctx);
        --  Minimum context size is 8192 for stability and headroom.
        --  Smaller contexts (e.g., 4096) caused llama_decode assertion failures
        --  with Qwen3.5HybridMythos at Q4_1 KV quantization on this hardware.
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
            --  (~2s for Qwen3.5HybridMythos).  Reusing an already-loaded context with
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

        --  =====================================================================
        --  ON-DEMAND MODEL LOADING (lazy, first-use)
        --  =====================================================================
        --  WHY: Models are NOT loaded at startup. Loading happens on the first
        --  Generate call for each model type. This saves startup time and memory
        --  but means the first request pays the full load penalty.
        --
        --  LOAD PHASES (all timed separately):
        --    1. File read: Read .gguf from disk into memory (~1-4 GB)
        --    2. GPU upload: Transfer weights to Metal/Vulkan GPU memory
        --    3. Context init: Create llama_context with KV cache (~0.5-2s)
        --
        --  DISK SPEED: Total file size / load time = MB/s throughput
        --  This helps diagnose slow first-response on HDD vs SSD.
        --
        --  PROGRESS: Prints every 500ms during load so the user knows it's
        --  not frozen. The 500ms interval is a compromise between noisy logs
        --  and useful progress feedback during multi-second loads.
        --  =====================================================================
        declare
            Model_Size_Bytes : Ada.Directories.File_Size := 0;
            Model_File       : constant String :=
               To_String (Models (Kind).Path);
        begin
            if Ada.Directories.Exists (Model_File) then
                Model_Size_Bytes := Ada.Directories.Size (Model_File);
            end if;
            Put_Line
               (AnsiAda.Foreground (AnsiAda.Light_Cyan)
                & "[Uptime]+"
                & Trim
                     (Duration'Image
                         (Ada.Real_Time.To_Duration
                             (Ada.Real_Time.Clock - Init_Start_Time)),
                      Both)
                & "s [LoadModel]"
                & AnsiAda.Reset
                & " Loading "
                & Model_Type'Image (Kind)
                & " | N_CTX="
                & Actual_Ctx'Img
                & " | File="
                & Ada.Directories.File_Size'Image (Model_Size_Bytes)
                & " bytes | ETA depends on disk speed...");
        end;

        --  SPECIAL CASE: MMProj (multimodal projection) model loading
        --  Why: MMProj is not a standalone llama model - it's a vision encoder
        --       that must be initialized with mtmd_init_from_file_safe, which
        --       requires the text model (Snowball_Enaga_Orchestrator) to be loaded first.
        --       The mmproj file contains the CLIP vision encoder weights that
        --       project images into the embedding space of the text model.
        if Kind = MMProj then
            --  MMProj requires the text model to be loaded first
            if not Models (Snowball_Enaga_Orchestrator).Loaded then
                Put_Line ("[!] MMProj requires Snowball_Enaga_Orchestrator to be loaded first");
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
                                       System.Address (Models (Snowball_Enaga_Orchestrator).Model),
                                       True,
                                       8);
                            exception
                                when others =>
                                    Put_Line
                                       ("[!] Exception caught in Ada during "
                                        & "Mtmd_Init_From_File_Safe");
                                    Models (Kind).Mtmd_Ctx :=
                                       Null_Mtmd_Context;
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
        --  GPU LAYER CONFIGURATION
        --  ======================================================================
        --  ALL models always on GPU (N_Gpu_Layers := -1).
        --  LM Studio runs everything on GPU including embeddings with no
        --  Metal crashes. The previous ELP0 CPU-only restriction was based on
        --  unfounded Metal crash fears. GPU embedding is ~10x faster than CPU,
        --  reducing ELP0 queue buildup so ELP1 user requests get served faster.
        --  ======================================================================
        M_Params.N_Gpu_Layers := -1;  -- All models always on GPU

        --  TRY THREE PATHS FOR MODEL FILES
        --  The CWD at runtime is unpredictable:
        --    1. Direct path (when run from project root or Adadelaide_Lite/)
        --    2. ../ prefixed (when CWD is src/)
        --    3. ../../ prefixed (when CWD is bin/)
        --  This fallback loop handles all common launch configurations
        --  without requiring a fixed working directory.
        declare
            Model_Load_Start : Ada.Real_Time.Time := Ada.Real_Time.Clock;
        begin
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
                                Put_Line
                                   (AnsiAda.Foreground (AnsiAda.Light_Cyan)
                                    & "[Uptime]+"
                                    & Trim
                                         (Duration'Image
                                             (Ada.Real_Time.To_Duration
                                                 (Ada.Real_Time.Clock
                                                  - Init_Start_Time)),
                                          Both)
                                    & "s [LoadModel]"
                                    & AnsiAda.Reset
                                    & " Phase 1/2: Reading weights from disk...");
                                Model_Load_Start := Ada.Real_Time.Clock;
                                begin
                                    Models (Kind).Model :=
                                       Llama_Model_Load_From_File
                                          (Path_C, M_Params);
                                exception
                                    when others =>
                                        Put_Line
                                           ("[!] Exception caught in Ada during "
                                            & "Llama_Model_Load_From_File");
                                        Models (Kind).Model := Null_Model;
                                end;
                                --  Restore stderr after model load
                                declare
                                    Dummy : int :=
                                       Sys_Restore_Stderr (Saved_Stderr);
                                begin
                                    null;
                                end;
                            end;
                            Free (Path_C);
                            if Models (Kind).Model /= Null_Model then
                                --  Log model load time and disk speed
                                declare
                                    Load_Dur    : constant Duration :=
                                       Ada.Real_Time.To_Duration
                                          (Ada.Real_Time.Clock
                                           - Model_Load_Start);
                                    Load_ms     : constant Natural :=
                                       Natural (Load_Dur * 1000.0);
                                    File_Size_B : Ada.Directories.File_Size :=
                                       0;
                                    Has_File    : Boolean := False;
                                    Disk_Speed  : Natural := 0;
                                begin
                                    if Ada.Directories.Exists (Path_Str) then
                                        File_Size_B :=
                                           Ada.Directories.Size (Path_Str);
                                        Has_File := True;
                                    end if;
                                    if Load_Dur > 0.0 and then Has_File then
                                        declare
                                            Dur_F : constant Float :=
                                               Float (Load_Dur);
                                        begin
                                            Disk_Speed :=
                                               Natural
                                                  (Float (File_Size_B)
                                                   / Dur_F
                                                   / 1_000_000.0);
                                        end;
                                    end if;
                                    Put_Line
                                       (AnsiAda.Foreground (AnsiAda.Green)
                                        & "[Uptime]+"
                                        & Trim
                                             (Duration'Image
                                                 (Ada.Real_Time.To_Duration
                                                     (Ada.Real_Time.Clock
                                                      - Init_Start_Time)),
                                              Both)
                                        & "s [LoadModel]"
                                        & AnsiAda.Reset
                                        & " Phase 1/2 COMPLETE: weights loaded"
                                        & " | "
                                        & Natural'Image (Load_ms)
                                        & "ms"
                                        & " | Disk: "
                                        & Natural'Image (Disk_Speed)
                                        & " MB/s"
                                        & " | Size: "
                                        & Ada.Directories.File_Size'Image
                                             (File_Size_B)
                                        & " bytes");
                                end;
                                exit;
                            end if;
                        end;
                    end if;
                end;
            end loop;
        end; -- Model_Load_Start declare

        if Models (Kind).Model /= Null_Model then
            --  [VITAL-DO-NOT-REMOVE] Start with llama.cpp's defaults.
            --  LM Studio and other frontends call llama_context_default_params()
            --  first, then modify only the fields they need. If we build the
            --  struct from scratch, fields like ctx_type, attention_type,
            --  n_seq_max etc. default to 0 which is WRONG for Qwen3.5's delta
            --  net recurrent attention. The crash in ggml_gated_delta_net was
            --  caused by zero defaults conflicting with the model architecture.
            --
            --  [VITAL-DO-NOT-REMOVE] CRITICAL ggml VERSION DISCOVERY:
            --  Homebrew-installed ggml (0.15.2) contains a bug in the Gated
            --  Delta Net recurrent attention path:
            --    GGML_ASSERT(state->ne[0] == S_v) failed
            --    ggml.c:6252 — during llama_decode (NOT during context init)
            --  The crash path showed: /private/tmp/ggml-20260619-5335-xzehaz/ggml-0.15.2/
            --  This is the HOMEBREW-built ggml, NOT our locally-built copy.
            --  LM Studio runs Qwen3.5 on llama.cpp (NOT just MLX) and does
            --  NOT have this crash — they use their own bundled llama.cpp
            --  build (b9601, commit 4c65955) with a working ggml.
            --  FIX: We clone ggml separately, compile from source, and link
            --  against our local build. Homebrew's ggml is NEVER used.
            --  See run.py for the ggml clone+compile pipeline.
            C_Params := Llama_Context_Default_Params;

            --  Now override only the fields we care about:
            C_Params.N_Ctx := Actual_Ctx;
            --  [MEMORY-FIX] Reduced from 512 to 256 to prevent Metal OOM.
            --  512 batch + 512 ubatch = ~256MB compute buffers on Metal.
            --  256 batch + 256 ubatch = ~64MB compute buffers on Metal.
            --  LM Studio uses adaptive batching; we use fixed. Smaller is safer.
            C_Params.N_Batch := 256;
            C_Params.N_Ubatch := 256;
            C_Params.N_Threads := 8;
            C_Params.N_Threads_Batch := 8;

            --  [VITAL-DO-NOT-REMOVE] Model-specific KV cache and flash attention.
            --  Embedding model (Qwen3-Embedding-0.6B): F16 KV, no flash_attn, GPU.
            --  Uses F16 because it's a small model — quantized KV saves little
            --  memory and may degrade embedding quality. No flash_attn because
            --  embedding model doesn't need it (small context, simple forward pass).
            --  GPU because LM Studio does it and it's ~10x faster than CPU.
            --  CRASH HISTORY: The original "Metal crash" with embedding model
            --  was misdiagnosed. Three separate bugs were at play:
            --  1) FFI struct missing N_Outputs_Max → Type_K at wrong offset
            --  2) No Task_Stack_Size → fprintf stack overflow in llama_init
            --  3) All models got same params (Q4_1 + flash_attn for embedding)
            --  Chat model (Qwen3.5HybridMythos): Q4_1 KV + flash_attn=1 + GPU.
            --  Q4_1 KV saves ~75% memory (fits in 16GB RAM). Flash attn
            --  works because llama_context_default_params() provides correct
            --  defaults for Qwen3.5's delta net recurrent attention.
            if Kind = Qwen_Embedding then
                C_Params.Type_K := GGML_TYPE_F16;
                C_Params.Type_V := GGML_TYPE_F16;
                C_Params.Flash_Attn_Type := 0;
                M_Params.N_Gpu_Layers :=
                   -1;    -- GPU for embedding (LM Studio does it)

            else
                C_Params.Type_K := GGML_TYPE_Q4_1;
                C_Params.Type_V := GGML_TYPE_Q4_1;
                C_Params.Flash_Attn_Type := 1;
                --  [ADAPTIVE GPU LAYERS] Start aggressive (-1 = all layers on GPU).
                --  If OOM, OOM handler sets GPU_Layer_Count to fallback (24).
                --  After 3 min cooldown, Load_Model retries -1 again.
                --  This auto-probes whether GPU can handle full offload.
                if GPU_Layer_Count = -1 then
                    M_Params.N_Gpu_Layers := -1;  -- Aggressive: all on GPU
                else
                    M_Params.N_Gpu_Layers :=
                       Interfaces.C.int (GPU_Layer_Count);  -- Fallback
                end if;
            end if;

            C_Params.Abort_Callback := Llama_Abort_Callback'Address;
            C_Params.Abort_Callback_Data := Model_Refs (Kind)'Address;

            --  =================================================================
            --  PHASE 2/2: Context + KV Cache initialization
            --  =================================================================
            --  This creates the llama_context and allocates KV cache memory.
            --  For Qwen3.5HybridMythos with 8192 ctx + Q4_1 KV: ~300MB GPU allocation.
            --  This is the second blocking step after file read.
            --  =================================================================
            declare
                Ctx_Init_Start : constant Ada.Real_Time.Time :=
                   Ada.Real_Time.Clock;
            begin
                --  [VITAL-DO-NOT-REMOVE] Print ALL context params before init.
                --  If init hangs, this tells us exactly what was requested.
                Put_Line
                   (AnsiAda.Foreground (AnsiAda.Light_Cyan)
                    & "[Uptime]+"
                    & Trim
                         (Duration'Image
                             (Ada.Real_Time.To_Duration
                                 (Ada.Real_Time.Clock - Init_Start_Time)),
                          Both)
                    & "s [LoadModel]"
                    & AnsiAda.Reset
                    & " Phase 2/2: Creating context...");
                --  [VITAL-DO-NOT-REMOVE] Show params for debugging hangs.
                Put_Line
                   (AnsiAda.Foreground (AnsiAda.Light_Cyan)
                    & "[Uptime]+"
                    & Trim
                         (Duration'Image
                             (Ada.Real_Time.To_Duration
                                 (Ada.Real_Time.Clock - Init_Start_Time)),
                          Both)
                    & "s [LoadModel]"
                    & AnsiAda.Reset
                    & " N_Ctx="
                    & Interfaces.C.unsigned'Image (C_Params.N_Ctx)
                    & " N_Batch="
                    & Interfaces.C.unsigned'Image (C_Params.N_Batch)
                    & " N_Ubatch="
                    & Interfaces.C.unsigned'Image (C_Params.N_Ubatch)
                    & " N_Threads="
                    & Interfaces.C.int'Image (C_Params.N_Threads)
                    & " Type_K="
                    & Interfaces.C.int'Image (C_Params.Type_K)
                    & " Type_V="
                    & Interfaces.C.int'Image (C_Params.Type_V)
                    & " Flash_Attn="
                    & Interfaces.C.int'Image (C_Params.Flash_Attn_Type)
                    & " N_Gpu_Layers="
                    & Interfaces.C.int'Image (M_Params.N_Gpu_Layers));
                --  [VITAL-DO-NOT-REMOVE] DO NOT suppress stderr here.
                --  If Llama_Init_From_Model hangs or crashes, we NEED to see
                --  the llama.cpp stderr output to diagnose the problem.
                --  The previous stderr suppression caused the 9B model to hang
                --  silently with zero diagnostic output. That is unacceptable.
                Put_Line
                   (AnsiAda.Foreground (AnsiAda.Light_Cyan)
                    & "[Uptime]+"
                    & Trim
                         (Duration'Image
                             (Ada.Real_Time.To_Duration
                                 (Ada.Real_Time.Clock - Init_Start_Time)),
                          Both)
                    & "s [LoadModel]"
                    & AnsiAda.Reset
                    & " Calling Llama_Init_From_Model (stderr visible)...");
                Models (Kind).Context :=
                   Llama_Init_From_Model (Models (Kind).Model, C_Params);

                declare
                    Ctx_Dur : constant Duration :=
                       Ada.Real_Time.To_Duration
                          (Ada.Real_Time.Clock - Ctx_Init_Start);
                    Ctx_ms  : constant Natural := Natural (Ctx_Dur * 1000.0);
                begin
                    if Models (Kind).Context /= Null_Context then
                        Put_Line
                           (AnsiAda.Foreground (AnsiAda.Green)
                            & "[Uptime]+"
                            & Trim
                                 (Duration'Image
                                     (Ada.Real_Time.To_Duration
                                         (Ada.Real_Time.Clock
                                          - Init_Start_Time)),
                                  Both)
                            & "s [LoadModel]"
                            & AnsiAda.Reset
                            & " Phase 2/2 COMPLETE: context ready"
                            & " | "
                            & Natural'Image (Ctx_ms)
                            & "ms");
                    else
                        Put_Line
                           (AnsiAda.Foreground (AnsiAda.Red)
                            & "[Uptime]+"
                            & Trim
                                 (Duration'Image
                                     (Ada.Real_Time.To_Duration
                                         (Ada.Real_Time.Clock
                                          - Init_Start_Time)),
                                  Both)
                            & "s [LoadModel]"
                            & AnsiAda.Reset
                            & " Phase 2/2 FAILED: context is NULL"
                            & " | "
                            & Natural'Image (Ctx_ms)
                            & "ms");
                    end if;
                end;
            end;
            if Models (Kind).Context /= Null_Context then
                Models (Kind).Loaded := True;
                Models (Kind).Last_Used := Clock;
                Models (Kind).Current_Ctx := Actual_Ctx;
                --  [LM-STYLE KV RESTORE] After creating context, try to
                --  restore previously saved KV state from disk.
                --  This is how LM Studio-style one-model-at-a-time works:
                --  Hop N saves KV + unloads model → Hop N+1 loads model + restores KV.
                declare
                    KV_Restored : Boolean;
                    KV_Tokens   : System.Address;
                    KV_N_Toks   : Interfaces.C.size_t;
                begin
                    KV_Restored := KV_Cache_Manager.Load_From_SSD_Lazy
                       (Context    => Models (Kind).Context,
                        Tokens     => KV_Tokens,
                        N_Tokens   => KV_N_Toks,
                        Model_ID   => Kind'Img,
                        Session_ID => Session_ID);
                    if KV_Restored then
                        --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
                        Put_Line
                           (AnsiAda.Foreground (AnsiAda.Green)
                            & "[Uptime]+"
                            & Trim
                                 (Duration'Image
                                     (Ada.Real_Time.To_Duration
                                         (Ada.Real_Time.Clock
                                          - Init_Start_Time)),
                                  Both)
                            & "s [LoadModel]"
                            & AnsiAda.Reset
                            & " KV restored from disk. Ready for generation.");
                    else
                        --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
                        Put_Line
                           (AnsiAda.Foreground (AnsiAda.Light_Cyan)
                            & "[Uptime]+"
                            & Trim
                                 (Duration'Image
                                     (Ada.Real_Time.To_Duration
                                         (Ada.Real_Time.Clock
                                          - Init_Start_Time)),
                                  Both)
                            & "s [LoadModel]"
                            & AnsiAda.Reset
                            & " No cached KV found. Fresh context.");
                    end if;
                end;
                Success := True;
            else
                Llama_Model_Free (Models (Kind).Model);
                Models (Kind).Model := Null_Model;
            end if;
        end if;
    exception
        when E : Storage_Error =>
            --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
            --  Stack overflow during model load (reading weights into VRAM
            --  or creating llama_context).  Clean up partial state so the
            --  server continues serving other requests.
            Ada.Text_IO.Put_Line
               (AnsiAda.Foreground (AnsiAda.Red)
                & "[LoadModel-FATAL]"
                & AnsiAda.Reset
                & " STORAGE_ERROR (stack overflow) loading "
                & Model_Type'Image (Kind));
            Ada.Text_IO.Put_Line
               (AnsiAda.Foreground (AnsiAda.Red)
                & "[LoadModel-FATAL]"
                & AnsiAda.Reset
                & " Exception: "
                & Ada.Exceptions.Exception_Information (E));
            --  [VITAL-DO-NOT-REMOVE] OOM banner — red, unmissable.
            Ada.Text_IO.Put_Line
               (AnsiAda.Foreground (AnsiAda.Red)
                & "=========================================================="
                & AnsiAda.Reset);
            Ada.Text_IO.Put_Line
               (AnsiAda.Foreground (AnsiAda.Red)
                & "  !!! OUT OF MEMORY !!!  (STORAGE_ERROR)"
                & AnsiAda.Reset);
            Ada.Text_IO.Put_Line
               (AnsiAda.Foreground (AnsiAda.Red)
                & "  Metal backend poisoned. KV save will RETRY."
                & AnsiAda.Reset);
            Ada.Text_IO.Put_Line
               (AnsiAda.Foreground (AnsiAda.Red)
                & "  Connection NOT dropped. Server continues."
                & AnsiAda.Reset);
            Ada.Text_IO.Put_Line
               (AnsiAda.Foreground (AnsiAda.Red)
                & "=========================================================="
                & AnsiAda.Reset);
            Mark_Metal_Broken;
            --  [ADAPTIVE GPU FALLBACK] OOM during load → progressive layer reduction
            --  Math: remove 25% of current layers each OOM
            --  -1 → 32 → 24 → 18 → 14 → 10 → 8 → 8 (min)
            --  After GPU_Retry_Interval (3 min) → reset to -1
            declare
                Old_Count : constant Integer := GPU_Layer_Count;
                New_Count : Integer;
            begin
                if GPU_Layer_Count = -1 then
                    --  First OOM: go from ALL to fallback (75%)
                    New_Count := GPU_Layer_Fallback;
                elsif GPU_Layer_Count > GPU_Layer_Min then
                    --  Progressive: remove 25% of current (min 1 layer)
                    New_Count := GPU_Layer_Count -
                                 Integer'Max (1, GPU_Layer_Count / 4);
                    --  Don't go below minimum
                    if New_Count < GPU_Layer_Min then
                        New_Count := GPU_Layer_Min;
                    end if;
                else
                    --  Already at minimum, can't reduce further
                    New_Count := GPU_Layer_Count;
                end if;

                if New_Count /= Old_Count then
                    GPU_Layer_Count   := New_Count;
                    GPU_Last_OOM_Time := Ada.Real_Time.Clock;
                    --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
                    Put_Line
                       (AnsiAda.Foreground (AnsiAda.Yellow)
                        & "[GPU-Adaptive]"
                        & AnsiAda.Reset
                        & " OOM during load. Layers:"
                        & Integer'Image (Old_Count) & " -> "
                        & Integer'Image (New_Count)
                        & ". Retry -1 in 3 minutes.");
                else
                    --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
                    Put_Line
                       (AnsiAda.Foreground (AnsiAda.Yellow)
                        & "[GPU-Adaptive]"
                        & AnsiAda.Reset
                        & " OOM but already at minimum layers"
                        & Integer'Image (GPU_Layer_Count)
                        & ". Waiting 3 min to retry -1.");
                end if;
            end;
            --  Free partial context if it was created
            if Models (Kind).Context /= Null_Context then
                Llama_Interface.Llama_Free (Models (Kind).Context);
                Models (Kind).Context := Null_Context;
            end if;
            --  Free partial model if it was loaded
            if Models (Kind).Model /= Null_Model then
                Llama_Model_Free (Models (Kind).Model);
                Models (Kind).Model := Null_Model;
            end if;
            Models (Kind).Loaded := False;
            Success := False;
        when E : others =>
            Ada.Text_IO.Put_Line
               (AnsiAda.Foreground (AnsiAda.Red)
                & "[LoadModel-FATAL]"
                & AnsiAda.Reset
                & " Exception loading " & Model_Type'Image (Kind) & ": "
                & Ada.Exceptions.Exception_Information (E));
            if Models (Kind).Context /= Null_Context then
                Llama_Interface.Llama_Free (Models (Kind).Context);
                Models (Kind).Context := Null_Context;
            end if;
            if Models (Kind).Model /= Null_Model then
                Llama_Model_Free (Models (Kind).Model);
                Models (Kind).Model := Null_Model;
            end if;
            Models (Kind).Loaded := False;
            Success := False;
    end Load_Model;

    --  [PARALLEL=1] Unload_Model frees ALL GPU memory for this model:
    --  - Llama_Free releases the context (KV cache + compute buffers)
    --  - Llama_Model_Free releases the model weights from GPU
    --  After this call, the model is completely gone from GPU memory.
    --  This is REQUIRED before loading another model (parallel=1 constraint).
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

    function Get_Model (Kind : Model_Type) return Llama_Interface.Llama_Model
    is
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
        function To_Ptr is new
           Ada.Unchecked_Conversion (System.Address, Model_Type_Ptr);
        Ptr : Model_Type_Ptr;
    begin
        if Data = System.Null_Address then
            return False;
        end if;
        Ptr := To_Ptr (Data);

        --  1. Abort if Watchdog has flagged a timeout for this model.
        if Watchdog_Manager.Inference_Monitor.Is_Aborted
           and then
              Watchdog_Manager.Inference_Monitor.Current_Inference_Model
              = Ptr.all
        then
            return True;
        end if;

        --  2. Only abort if we are an ELP0 task and an ELP1 task is pending.
        return
           Priority_Model_Gate.Is_ELP0_Owner (Ptr.all)
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

    procedure Acquire_Accel_Lock is
    begin
        Accel_Lock_Object.Acquire;
    end Acquire_Accel_Lock;

    procedure Release_Accel_Lock is
    begin
        Accel_Lock_Object.Release;
    end Release_Accel_Lock;

    --  [VITAL-DO-NOT-REMOVE] Metal backend health — OPPORTUNISTIC.
    --  Mark_Metal_Broken: Called when llama_decode returns -3 (OOM).
    --  Records the current elapsed time for cooldown tracking.
    procedure Mark_Metal_Broken is
    begin
        Metal_Backend_Broken := True;
        Metal_OOM_Trigger_Time :=
           Ada.Real_Time.To_Duration (Ada.Real_Time.Clock - Init_Start_Time);
        Put_Line
           (AnsiAda.Foreground (AnsiAda.Red)
            & "[OOM] "
            & AnsiAda.Reset
            & "METAL BACKEND POISONED. KV save will RETRY every "
            & Duration'Image (Metal_OOM_Retry_Secs) & "s for "
            & Duration'Image (Metal_OOM_Cooldown_Secs) & "s cooldown.");
    end Mark_Metal_Broken;

    --  [VITAL-DO-NOT-REMOVE] Metal backend health — OPPORTUNISTIC.
    --  Is_Metal_Broken: Returns True if Metal is still in cooldown.
    --  Auto-resets Metal_Backend_Broken after Metal_OOM_Cooldown_Secs.
    --  This allows the save task to retry after GPU driver recovers.
    function Is_Metal_Broken return Boolean is
        Now     : constant Duration :=
           Ada.Real_Time.To_Duration (Ada.Real_Time.Clock - Init_Start_Time);
        Elapsed : constant Duration := Now - Metal_OOM_Trigger_Time;
    begin
        if Metal_Backend_Broken and then Elapsed >= Metal_OOM_Cooldown_Secs then
            --  Cooldown expired — GPU driver should have recovered.
            --  Reset flag and log recovery.
            Metal_Backend_Broken := False;
            Put_Line
               (AnsiAda.Foreground (AnsiAda.Green)
                & "[OOM] "
                & AnsiAda.Reset
                & "METAL BACKEND RECOVERED after "
                & Duration'Image (Elapsed) & "s cooldown. Retrying save.");
            return False;
        end if;
        return Metal_Backend_Broken;
    end Is_Metal_Broken;

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
            return Snowball_Enaga_Orchestrator;
        elsif Name = "qwen-embedding" or else Name = "adelaide-embedding" then
            return Qwen_Embedding;
        else
            return Snowball_Enaga_ShortNetworkAnswer;
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
       (Msg : String; Session_ID : String := ""; Level : ELP_Level := ELP1)
        return String
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
            if Str_Piece'Length > 1 then
                Ada.Text_IO.Put_Line
                   ("Push_Chunk called with: "
                    & Str_Piece
                         (Str_Piece'First
                          ..
                             Natural'Min
                                (Str_Piece'Last, Str_Piece'First + 20)));
            end if;
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
            if Val = 9
               or else Val = 10
               or else Val = 13
               or else (Val >= 32 and Val <= 126)
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
            elsif I + 10 <= S'Last and then S (I .. I + 10) = "</thinking>"
            then
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
            if I + 1 <= S'Last and then S (I) = '!' and then S (I + 1) = '['
            then
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
                                Content           : constant String :=
                                   S (Close_Bracket + 2 .. Close_Paren - 1);
                                Has_Base64_Marker : constant Boolean :=
                                   Index (Content, "base64") > 0;
                                Is_Long_No_Space  : constant Boolean :=
                                   Content'Length > 200
                                   and then Index (Content, " ") = 0;
                            begin
                                if Has_Base64_Marker or else Is_Long_No_Space
                                then
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
            Put_Line
               (AnsiAda.Foreground (AnsiAda.Yellow)
                & "[StripBase64-V]"
                & AnsiAda.Reset
                & " Stripped base64 images. Input="
                & Natural'Image (S'Length)
                & " Output="
                & Natural'Image (Length (Res))
                & " Saved="
                & Natural'Image (S'Length - Length (Res))
                & " bytes");
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
        Source : constant String :=
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
        Put_Line
           ("[Embedding-Debug] Input ("
            & Clean_P'Length'Img
            & " chars): "
            & Clean_P);
        Flush;

        --  [QUIRK-M10] Fix 2: Reject code-like content before tokenization
        --  Calculate density of special chars that crash the ggml-metal kernel
        if Clean_P'Length > 0 then
            declare
                Specials : Natural := 0;
                Density  : Float;
            begin
                for I in Clean_P'Range loop
                    if Clean_P (I) = '{'
                       or else Clean_P (I) = '}'
                       or else Clean_P (I) = ';'
                       or else Clean_P (I) = ':'
                       or else Clean_P (I) = '@'
                       or else Clean_P (I) = '/'
                       or else Clean_P (I) = '\'
                    then
                        Specials := Specials + 1;
                    end if;
                end loop;
                Density := Float (Specials) / Float (Clean_P'Length);
                if Density > 0.1 then
                    Put_Line
                       ("[Embedding-Debug] Skipping high-density code block (Density: "
                        & Density'Img
                        & ") to prevent Metal crash Code 5");
                    Length := 0;
                    Free (Prompt_C);
                    return;
                end if;
            end;
        end if;
        --  --[Debug] DO NOT REMOVE: Descriptive source tracking
        ELP_Queue.Enqueue (Level, Kind, Source);
        if Level = ELP0 then
            Priority_Model_Gate.Acquire_ELP0 (Kind) (Success);
            if not Success then
                Put_Line
                   ("[ELP0-BLOCKED] "
                    & Kind'Img
                    & " | ELP1 is active or pending");
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
        Acquire_Accel_Lock;
        if Kratos.Guard_Enter = 0 then
            N_Toks :=
               Llama_Tokenize
                  (Vocab,
                   Prompt_C,
                   int (Clean_P'Length),
                   Tokens.all'Address,
                   4096,
                   True,
                   True);
            Kratos.Guard_Exit;
        else
            Kratos.Log_Crash;
            N_Toks := -1;
        end if;
        Release_Accel_Lock;

        Put_Line
           ("[Tokenize-Debug] Model:"
            & Kind'Img
            & " Prompt_Len:"
            & Clean_P'Length'Img
            & " N_Toks:"
            & N_Toks'Img);
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
            Batch_Size           : constant int :=
               int'Min (256, int (Models (Kind).Current_Ctx));
            Current_Pos          : int := 0;
            Tokens_Left          : int := N_Toks;
            Consecutive_Failures : Natural :=
               0;  -- Track consecutive decode failures
            Max_Consecutive      : constant :=
               3;   -- Unload after 3 failures in a row
        begin
            Llama_Interface.Llama_Memory_Clear
               (Llama_Interface.Llama_Get_Memory (Models (Kind).Context),
                False);
            Llama_Set_Embeddings (Models (Kind).Context, Interfaces.C.int (1));

            while Tokens_Left > 0 loop
                declare
                    To_Decode  : constant int :=
                       (if Tokens_Left > Batch_Size
                        then Batch_Size
                        else Tokens_Left);
                    B          : constant Llama_Batch :=
                       Llama_Batch_Get_One
                          (Tokens.all (Integer (Current_Pos) + 1)'Address,
                           To_Decode);
                    Dec_Result : int;
                begin
                    --  KRATOS CRASH GUARD: llama_decode is wrapped in
                    --  Guard_Enter/Guard_Exit. See QUIRK-M01.
                    Acquire_Accel_Lock;
                    if Kratos.Guard_Enter = 0 then
                        Dec_Result := Llama_Decode (Models (Kind).Context, B);
                        Kratos.Guard_Exit;
                    else
                        Kratos.Log_Crash;
                        Dec_Result := -1;
                    end if;
                    Release_Accel_Lock;

                    if Dec_Result /= 0 then
                        --  DECODE FAILED: Skip this batch, don't unload the model.
                        --  The failure is likely a Metal kernel compilation error for
                        --  this specific batch size/token count. The next batch may
                        --  use a different configuration that compiles successfully.
                        Consecutive_Failures := Consecutive_Failures + 1;
                        Put_Line
                           (AnsiAda.Foreground (AnsiAda.Yellow)
                            & "[WARN] Llama_Decode failed (Code:"
                            & Dec_Result'Img
                            & ") Batch:"
                            & To_Decode'Img
                            & " Consecutive:"
                            & Consecutive_Failures'Img
                            & AnsiAda.Reset);

                        if Consecutive_Failures >= Max_Consecutive then
                            --  3 consecutive failures = all kernel variants failing.
                            --  This is a real issue, not a transient compilation error.
                            --  Unload and let the caller decide what to do.
                            Put_Line
                               (AnsiAda.Foreground (AnsiAda.Red)
                                & "[FATAL] "
                                & Max_Consecutive'Img
                                & " consecutive decode failures. "
                                & "Orphaning poisoned context to prevent SIGTRAP."
                                & AnsiAda.Reset);
                            delay 1.0;  -- Brief cooldown for GPU driver

                            --  [QUIRK-M10] We cannot call Unload_Model here. The Metal
                            --  GPU backend is poisoned. Calling Llama_Free invokes
                            --  ggml_metal_free which tries to synchronize and aborts
                            --  the entire server process (SIGTRAP 5). We MUST leak it.
                            Models (Kind).Context := Null_Context;
                            Models (Kind).Model := Null_Model;
                            Models (Kind).Loaded := False;
                            Models (Kind).Current_Ctx := 0;

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
            Dim        : constant int :=
               Llama_Model_N_Embd (Models (Kind).Model);
            Ptr        : Address;
            --  SAFE: copy via C memcpy instead of Ada address overlay
            function Memcpy
               (Dst, Src : Address; N : Interfaces.C.size_t) return Address;
            pragma Import (C, Memcpy, "memcpy");
            Copy_Count : constant Integer :=
               Integer
                  (Interfaces.C.size_t'Min
                      (Interfaces.C.size_t (Dim),
                       Interfaces.C.size_t (Result'Length)));
        begin
            Acquire_Accel_Lock;
            Ptr := Llama_Get_Embeddings (Models (Kind).Context);
            Release_Accel_Lock;

            if Copy_Count > 0 and then Ptr /= Null_Address then
                declare
                    Dummy : Address;
                begin
                    Put_Line ("[Debug] Before Memcpy.");
                    Dummy :=
                       Memcpy
                          (Result (Result'First)'Address,
                           Ptr,
                           Interfaces.C.size_t (Copy_Count)
                           * Interfaces.C.size_t (Float'Size / 8));
                    Put_Line ("[Debug] After Memcpy.");
                end;
                Length := Copy_Count;
                Put_Line ("[Debug] Copy_Count done");
            else
                Length := 0;
            end if;
             Free_Tokens (Tokens);
             Put_Line ("[Debug] Free_Tokens complete.");
             Models (Kind).In_Use := False;
             --  [PARALLEL=1 FIX] Unload embedding model from GPU immediately
             --  after use. Only ONE model can be in GPU memory at a time.
             --  If we don't unload, the embedding model (~1GB) stays resident
             --  and the 9B chat model OOMs when it tries to allocate KV +
             --  compute buffers on top of it.
             Unload_Model (Kind);
             if Level = ELP0 then
                 Priority_Model_Gate.Release_ELP0 (Kind);
                 Put_Line
                    ("[Debug] Priority_Model_Gate.Release_ELP0 complete.");
             else
                 Priority_Model_Gate.Release_ELP1 (Kind);
                 Put_Line
                    ("[Debug] Priority_Model_Gate.Release_ELP1 complete.");
             end if;
             ELP_Queue.Dequeue_Level (Level);
             Put_Line ("[Debug] Get_Single_Embedding DONE successfully.");
        end;
     exception
         when E : others =>
             Put_Line
                ("[FATAL] Exception in Get_Single_Embedding: "
                 & Ada.Exceptions.Exception_Information (E));
             if Tokens /= null then
                 Free_Tokens (Tokens);
             end if;
             Models (Kind).In_Use := False;
             --  [PARALLEL=1 FIX] Unload on error too — don't leave broken model in GPU
             Unload_Model (Kind);
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
        Level  : ELP_Level := ELP1) is
    begin
        if Prompt'Length <= 800 then
            Get_Single_Embedding (Prompt, Result, Length, Level);
        else
            declare
                Num_Chunks : Natural := 0;
                Sum_Vec    : Math_Utils.Vector (Result'Range) :=
                   [others => 0.0];
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
                        Get_Single_Embedding
                           (Sub_Prompt, Sub_Vec, Sub_Len, Level);
                        if Sub_Len > 0 then
                            if Num_Chunks = 0 then
                                Dim := Sub_Len;
                            end if;
                            for I in 1 .. Dim loop
                                Sum_Vec (Result'First + I - 1) :=
                                   Sum_Vec (Result'First + I - 1)
                                   + Sub_Vec (Sub_Vec'First + I - 1);
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
        Orch_Think_Open : Boolean := False;
        Sanitize_Buffer : Unbounded_String := Null_Unbounded_String;
        In_Think_Block  : Boolean := False;
        Fault_Detected  : Boolean := False;
        Fault_Query     : Unbounded_String := Null_Unbounded_String;
        Fault_Category  : Unbounded_String := Null_Unbounded_String;
        Output_Buffer   : Unbounded_String := Null_Unbounded_String;
    end record;

    function Is_Prefix (S, Tag : String) return Boolean is
    begin
        return
           S'Length < Tag'Length
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
                Put_Line
                   (AnsiAda.Foreground (AnsiAda.Light_Blue)
                    & "[StreamParse-V]"
                    & AnsiAda.Reset
                    & " THINK_OPEN detected. In_Think_Block -> True");
                Parser.Sanitize_Buffer := Null_Unbounded_String;
                Parser.In_Think_Block := True;
                return;
            elsif Buf = Close_Tag_A or else Buf = Close_Tag_B then
                --  [VITAL-DO-NOT-REMOVE] Mandated by user.
                Put_Line
                   (AnsiAda.Foreground (AnsiAda.Light_Blue)
                    & "[StreamParse-V]"
                    & AnsiAda.Reset
                    & " THINK_CLOSE detected. In_Think_Block -> False"
                    & " Orch_Think_Open="
                    & Boolean'Image (Parser.Orch_Think_Open));
                Parser.Sanitize_Buffer := Null_Unbounded_String;
                Parser.In_Think_Block := False;
                --  Do NOT push `</think>` here. The emulated streaming section
                --  in Hybrid_Generate will push `</think>` after Generate
                --  returns. Pushing it here would create a duplicate closing
                --  tag on the wire. Just clear the Orch_Think_Open flag so
                --  the emulated streaming knows the orchestration think block
                --  was closed during generation.
                if Parser.Orch_Think_Open then
                    Parser.Orch_Think_Open := False;
                end if;
                return;
            elsif Buf = Resp_Tag then
                --  [VITAL-DO-NOT-REMOVE] Mandated by user.
                Put_Line
                   (AnsiAda.Foreground (AnsiAda.Light_Blue)
                    & "[StreamParse-V]"
                    & AnsiAda.Reset
                    & " RESP_CLOSE detected.");
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
                    Fault_Mark    : constant String := "[CONTEXT_FAULT:";
                    --  Max buffer size for fault detection. The fault marker
                    --  is [CONTEXT_FAULT:query=... category=...] which typically
                    --  fits within 150 chars. Using 500 as a generous upper bound.
                    MAX_FAULT_LEN : constant Integer := 500;
                    SBuf          : constant String :=
                       To_String (Parser.Sanitize_Buffer);
                    F_Pos         : constant Natural :=
                       Index (SBuf, Fault_Mark);
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
                                        Q_End :=
                                           (if Cat_Idx > Query_Idx
                                            then Cat_Idx - 1
                                            else Inner'Last + 1);
                                        Parser.Fault_Query :=
                                           To_Unbounded_String
                                              (Trim
                                                  (Inner
                                                      (Q_Start .. Q_End - 1),
                                                   Ada.Strings.Both));
                                    end if;
                                    if Cat_Idx > 0 then
                                        Parser.Fault_Category :=
                                           To_Unbounded_String
                                              (Trim
                                                  (Inner
                                                      (Cat_Idx
                                                       + C_Mark'Length
                                                       .. Inner'Last),
                                                   Ada.Strings.Both));
                                    else
                                        Parser.Fault_Category :=
                                           To_Unbounded_String ("knowledge");
                                    end if;
                                    --  Clear buffer to prevent re-detecting same fault
                                    Parser.Sanitize_Buffer :=
                                       Null_Unbounded_String;
                                end;
                                return;
                            else
                                --  Incomplete marker (have [CONTEXT_FAULT: but no ] yet).
                                --  Keep accumulating. Do NOT clear buffer.
                                --  [VITAL-DO-NOT-REMOVE] Mandated by user.
                                if SBuf'Length mod 10 = 0 then
                                    Put_Line
                                       (AnsiAda.Foreground (AnsiAda.Grey)
                                        & "[StreamParse-V]"
                                        & AnsiAda.Reset
                                        & " CONTEXT_FAULT accum Len="
                                        & Natural'Image (SBuf'Length)
                                        & " awaiting closing bracket.");
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
                           (AnsiAda.Foreground (AnsiAda.Grey)
                            & "[StreamParse-V]"
                            & AnsiAda.Reset
                            & " THINK_BLOCK_BUF Len="
                            & Natural'Image (SBuf'Length)
                            & " exceeded MAX_FAULT_LEN. Clearing buffer.");
                        Parser.Sanitize_Buffer := Null_Unbounded_String;
                        return;
                    end if;
                end;
            end if;

            -- Stream content out, but SILENCE the think block entirely
            if not Parser.In_Think_Block then
                --  Flush on newlines so the client gets line-by-line incremental
                --  updates.  Accumulate in Output_Buffer, push when we see a
                --  newline or when the buffer exceeds 256 chars (safety limit).
                Append (Parser.Output_Buffer, Buf);
                declare
                    OB      : constant String :=
                       To_String (Parser.Output_Buffer);
                    Last_NL : Integer := 0;
                begin
                    --  Scan for the last newline in the buffer
                    for I in reverse OB'Range loop
                        if OB (I) = Character'Val (10) then
                            -- LF
                            Last_NL := I;
                            exit;
                        end if;
                    end loop;
                    if Last_NL > 0 then
                        --  Push everything up to and including the last newline
                        Push_Chunk
                           (Stream, Session_ID, OB (OB'First .. Last_NL));
                        --  Keep the remainder (after the newline) in the buffer
                        Parser.Output_Buffer :=
                           To_Unbounded_String (OB (Last_NL + 1 .. OB'Last));
                    elsif OB'Length > 256 then
                        --  Safety: flush even without a newline if buffer is large
                        Push_Chunk (Stream, Session_ID, OB);
                        Parser.Output_Buffer := Null_Unbounded_String;
                    end if;
                end;
            else
                --  [VITAL-DO-NOT-REMOVE] Mandated by user.
                if Buf'Length > 0 then
                    Put_Line
                       (AnsiAda.Foreground (AnsiAda.Grey)
                        & "[StreamParse-V]"
                        & AnsiAda.Reset
                        & " SILENCED_BUF Len="
                        & Natural'Image (Buf'Length)
                        & " Text="
                        & Buf
                             (Buf'First
                              .. Natural'Min (Buf'Last, Buf'First + 30)));
                end if;
            end if;
            Parser.Sanitize_Buffer := Null_Unbounded_String;
        end;
    end Process_And_Push_Char;

    procedure Process_And_Push_Chunk
       (Stream     : Streaming_Queue.Queue_Access;
        Session_ID : String;
        Parser     : in out Stream_Parser_State;
        Chunk      : String) is
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
        Content    : String) is
    begin
        Process_And_Push_Chunk (Stream, Session_ID, Parser, Content);
    end Push_Orchestration_Through_Parser;

    --  PUSH_ORCHESTRATION_DIRECT:
    --  Pushes orchestration metadata DIRECTLY to the queue, bypassing
    --  the stream parser silencing. Used for reasoning thoughts and tool
    --  call reasons that must be visible inside the <think> block.
    procedure Push_Orchestration_Direct
       (Stream     : Streaming_Queue.Queue_Access;
        Session_ID : String;
        Content    : String) is
    begin
        if Stream /= null then
            Push_Chunk (Stream, Session_ID, Content);
        end if;
    end Push_Orchestration_Direct;

    procedure Flush_Parser
       (Stream     : Streaming_Queue.Queue_Access;
        Session_ID : String;
        Parser     : in out Stream_Parser_State) is
    begin
        --  [VITAL-DO-NOT-REMOVE] Mandated by user.
        Put_Line
           (AnsiAda.Foreground (AnsiAda.Light_Blue)
            & "[StreamParse-V]"
            & AnsiAda.Reset
            & " Flush_Parser ENTERED. Buffer="
            & Natural'Image (Length (Parser.Sanitize_Buffer))
            & " Output_Buffer="
            & Natural'Image (Length (Parser.Output_Buffer))
            & " Orch_Think_Open="
            & Boolean'Image (Parser.Orch_Think_Open)
            & " In_Think_Block="
            & Boolean'Image (Parser.In_Think_Block));
        --  Flush any remaining batched output
        if Length (Parser.Output_Buffer) > 0 then
            if not Parser.In_Think_Block then
                declare
                    Flush_Str : constant String :=
                       To_String (Parser.Output_Buffer);
                begin
                    Put_Line
                       (AnsiAda.Foreground (AnsiAda.Light_Blue)
                        & "[StreamParse-V]"
                        & AnsiAda.Reset
                        & " Flush_Parser: Pushing batched output "
                        & Natural'Image (Flush_Str'Length)
                        & " chars.");
                    Push_Chunk (Stream, Session_ID, Flush_Str);
                end;
            else
                Put_Line
                   (AnsiAda.Foreground (AnsiAda.Light_Blue)
                    & "[StreamParse-V]"
                    & AnsiAda.Reset
                    & " Flush_Parser: Silencing batched output "
                    & Natural'Image (Length (Parser.Output_Buffer))
                    & " chars inside think block.");
            end if;
            Parser.Output_Buffer := Null_Unbounded_String;
        end if;
        declare
            S_Str : constant String := To_String (Parser.Sanitize_Buffer);
        begin
            if S_Str /= "" then
                if not Parser.In_Think_Block then
                    --  [VITAL-DO-NOT-REMOVE] Mandated by user.
                    Put_Line
                       (AnsiAda.Foreground (AnsiAda.Light_Blue)
                        & "[StreamParse-V]"
                        & AnsiAda.Reset
                        & " Flush_Parser: Pushing remaining "
                        & Natural'Image (S_Str'Length)
                        & " chars.");
                    Push_Chunk (Stream, Session_ID, S_Str);
                else
                    --  [VITAL-DO-NOT-REMOVE] Mandated by user.
                    Put_Line
                       (AnsiAda.Foreground (AnsiAda.Light_Blue)
                        & "[StreamParse-V]"
                        & AnsiAda.Reset
                        & " Flush_Parser: Silencing "
                        & Natural'Image (S_Str'Length)
                        & " chars inside think block.");
                end if;
                Parser.Sanitize_Buffer := Null_Unbounded_String;
            else
                --  [VITAL-DO-NOT-REMOVE] Mandated by user.
                Put_Line
                   (AnsiAda.Foreground (AnsiAda.Light_Blue)
                    & "[StreamParse-V]"
                    & AnsiAda.Reset
                    & " Flush_Parser: Buffer empty, nothing to push.");
            end if;
        end;
        if Parser.Orch_Think_Open then
            --  Silently close orchestration thinking; tag is stripped by parser
            --  [VITAL-DO-NOT-REMOVE] Mandated by user.
            Put_Line
               (AnsiAda.Foreground (AnsiAda.Light_Blue)
                & "[StreamParse-V]"
                & AnsiAda.Reset
                & " Flush_Parser: Closing Orch_Think_Open.");
            Parser.Orch_Think_Open := False;
        end if;
        --  [VITAL-DO-NOT-REMOVE] Mandated by user.
        Put_Line
           (AnsiAda.Foreground (AnsiAda.Light_Blue)
            & "[StreamParse-V]"
            & AnsiAda.Reset
            & " Flush_Parser COMPLETE.");
    end Flush_Parser;

    --  Sanitize_Memory_Content: strips all LLM-specific markup from raw
    --  database content BEFORE injecting it into the system prompt.
    --
    --  Raw responses stored in the DB can contain:
    --    1. <think>...</think> and <thinking>...</thinking> blocks
    --    2. ChatML special tokens: <|im_start|>, <|im_end|>, <|im_sep|>
    --    3. Orphaned closing tags: </think>, </thinking>, </response>
    --    4. Role markers that would confuse the tokenizer structure
    --
    --  Injecting any of these verbatim into the system prompt would break
    --  the ChatML framing and make the model believe it is already mid-
    --  conversation, which leads to degenerate or hallucinated output.
    function Sanitize_Memory_Content (Text : String) return String is
        use Ada.Strings.Fixed;
        Res      : Unbounded_String;
        I        : Positive := Text'First;
        In_Think : Boolean := False;

        --  Helper: check if Text (I .. I + Len - 1) equals Tag
        function Match (Tag : String) return Boolean is
        begin
            return I + Tag'Length - 1 <= Text'Last
                   and then Text (I .. I + Tag'Length - 1) = Tag;
        end Match;
    begin
        --  Bootstrap: if text starts inside a think block (orphaned close
        --  appears before the first open), enter think-skip mode immediately.
        declare
            Close_Idx : constant Natural := Index (Text, "</think>");
            Open_Idx  : constant Natural := Index (Text, "<think>");
        begin
            if Close_Idx > 0
               and then (Open_Idx = 0 or else Close_Idx < Open_Idx)
            then
                In_Think := True;
            end if;
        end;

        while I <= Text'Last loop
            if In_Think then
                --  Skip everything until closing think/thinking tag
                if Match ("</think>") then
                    In_Think := False;
                    I := I + 8;
                elsif Match ("</thinking>") then
                    In_Think := False;
                    I := I + 11;
                else
                    I := I + 1;
                end if;
            else
                --  Strip open think/thinking tags (enter skip mode)
                if Match ("<think>") then
                    In_Think := True;
                    I := I + 7;
                elsif Match ("<thinking>") then
                    In_Think := True;
                    I := I + 10;
                --  Strip orphaned closing tags
                elsif Match ("</think>") then
                    I := I + 8;
                elsif Match ("</thinking>") then
                    I := I + 11;
                elsif Match ("</response>") then
                    I := I + 11;
                --  Strip ChatML special tokens -- these would break tokenizer
                --  framing if injected into the system prompt block
                elsif Match ("<|im_start|>") then
                    I := I + 12;
                elsif Match ("<|im_end|>") then
                    I := I + 10;
                elsif Match ("<|im_sep|>") then
                    I := I + 10;
                --  Neutralise raw role markers that could inject fake turns
                elsif Match ("assistant") and then
                      (I = Text'First or else Text (I - 1) = ASCII.LF)
                then
                    --  Skip the role word; the newline before it stays
                    I := I + 9;
                else
                    Append (Res, Text (I));
                    I := I + 1;
                end if;
            end if;
        end loop;

        --  Collapse runs of blank lines left by stripped blocks
        declare
            Raw    : constant String := To_String (Res);
            Clean  : Unbounded_String;
            Blanks : Natural := 0;
        begin
            for J in Raw'Range loop
                if Raw (J) = ASCII.LF then
                    Blanks := Blanks + 1;
                    if Blanks <= 2 then
                        Append (Clean, Raw (J));
                    end if;
                else
                    Blanks := 0;
                    Append (Clean, Raw (J));
                end if;
            end loop;
            return Ada.Strings.Fixed.Trim
                      (To_String (Clean), Ada.Strings.Both);
        end;
    end Sanitize_Memory_Content;

    function Sanitize_Think_Tags (Text : String) return String is
        Res       : Unbounded_String;
        I         : Positive := Text'First;
        Close_Idx : constant Natural := Index (Text, "</think>");
        Open_Idx  : constant Natural := Index (Text, "<think>");
        In_Think  : Boolean := False;
    begin
        if Close_Idx > 0 and then (Open_Idx = 0 or else Close_Idx < Open_Idx)
        then
            In_Think := True;
        end if;

        while I <= Text'Last loop
            if In_Think then
                if I + 7 <= Text'Last and then Text (I .. I + 7) = "</think>"
                then
                    In_Think := False;
                    I := I + 8;
                elsif I + 10 <= Text'Last
                   and then Text (I .. I + 10) = "</thinking>"
                then
                    In_Think := False;
                    I := I + 11;
                else
                    I := I + 1;
                end if;
            else
                if I + 6 <= Text'Last and then Text (I .. I + 6) = "<think>"
                then
                    In_Think := True;
                    I := I + 7;
                elsif I + 9 <= Text'Last
                   and then Text (I .. I + 9) = "<thinking>"
                then
                    In_Think := True;
                    I := I + 10;
                elsif I + 10 <= Text'Last
                   and then Text (I .. I + 10) = "</response>"
                then
                    I := I + 11;
                elsif I + 7 <= Text'Last
                   and then Text (I .. I + 7) = "</think>"
                then
                    I := I + 8;
                elsif I + 10 <= Text'Last
                   and then Text (I .. I + 10) = "</thinking>"
                then
                    I := I + 11;
                else
                    Append (Res, Text (I));
                    I := I + 1;
                end if;
            end if;
        end loop;
        return To_String (Res);
    end Sanitize_Think_Tags;

    function Extract_Think_Content (Text : String) return String is
        Res       : Unbounded_String;
        I         : Positive := Text'First;
        Close_Idx : constant Natural := Index (Text, "</think>");
        Open_Idx  : constant Natural := Index (Text, "<think>");
        In_Think  : Boolean := False;
    begin
        if Close_Idx > 0 and then (Open_Idx = 0 or else Close_Idx < Open_Idx)
        then
            In_Think := True;
        end if;

        while I <= Text'Last loop
            if In_Think then
                if I + 7 <= Text'Last and then Text (I .. I + 7) = "</think>"
                then
                    In_Think := False;
                    I := I + 8;
                elsif I + 10 <= Text'Last
                   and then Text (I .. I + 10) = "</thinking>"
                then
                    In_Think := False;
                    I := I + 11;
                else
                    Append (Res, Text (I));
                    I := I + 1;
                end if;
            else
                if I + 6 <= Text'Last and then Text (I .. I + 6) = "<think>"
                then
                    In_Think := True;
                    I := I + 7;
                elsif I + 9 <= Text'Last
                   and then Text (I .. I + 9) = "<thinking>"
                then
                    In_Think := True;
                    I := I + 10;
                elsif I + 10 <= Text'Last
                   and then Text (I .. I + 10) = "</response>"
                then
                    I := I + 11;
                else
                    I := I + 1;
                end if;
            end if;
        end loop;
        return To_String (Res);
    end Extract_Think_Content;

    --  GENERATE (CORE GGUF INFERENCE WITH PREEMPTION SUPPORT)
    procedure Generate
       (Kind            : Model_Type;
        Prompt          : String;
        Result          : out Unbounded_String;
        Images          : GNATCOLL.JSON.JSON_Array :=
           GNATCOLL.JSON.Empty_Array;
        Session_ID      : String := "";
        Requested_Ctx   : Positive := 4096;
        Stream          : Streaming_Queue.Queue_Access := null;
        Orch_Think_Open : Boolean := False;
        Level           : ELP_Level := ELP1;
        Virtual_Tokens  : Cached_Token_Access := null;
        Virtual_Tok_Len : Natural := 0;
        FreeParallelMemory   : Boolean := True;
        Skip_Gate       : Boolean := False)
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
        Source : constant String :=
           (if Level = ELP0 then "Speculation" else "User-Chat");
    begin
        --  [VITAL-DO-NOT-REMOVE] Mandated by user.
        --  --[Debug] DO NOT REMOVE: Descriptive source tracking
        if not Skip_Gate then
            ELP_Queue.Enqueue (Level, Kind, Source);
        end if;

        pragma Unreferenced (Images);
        Result := Null_Unbounded_String;
        Parser.Orch_Think_Open := Orch_Think_Open;

        --  [VITAL-DO-NOT-REMOVE] Mandated by user.
        Put_Line
           (AnsiAda.Foreground (AnsiAda.Light_Blue)
            & "[Gen-V]"
            & AnsiAda.Reset
            & " Generate ENTERED. Kind="
            & Kind'Img
            & " Level="
            & Level'Img
            & " Stream="
            & (if Stream /= null then "YES" else "NO")
            & " Orch_Think_Open="
            & Boolean'Image (Orch_Think_Open)
            & " Prompt_Len="
            & Natural'Image (Clean_P'Length));

        begin
            if not Skip_Gate then
                if Level = ELP0 then
                    declare
                        Acq_OK : Boolean;
                    begin
                        Priority_Model_Gate.Acquire_ELP0 (Kind) (Acq_OK);
                        if not Acq_OK then
                            --  [VITAL-DO-NOT-REMOVE] Mandated by user.
                            Put_Line
                               (AnsiAda.Foreground (AnsiAda.Red)
                                & "[Gen-V]"
                                & AnsiAda.Reset
                                & " Generate: ELP0 ACQUIRE FAILED (Preempted)");
                            ELP_Queue.Dequeue_Level (Level);
                            Result := To_Unbounded_String ("ERROR: Preempted");
                            Free (Prompt_C);
                            return;
                        end if;
                        --  [VITAL-DO-NOT-REMOVE] Mandated by user.
                        Put_Line
                           (AnsiAda.Foreground (AnsiAda.Light_Blue)
                            & "[Gen-V]"
                            & AnsiAda.Reset
                            & " Generate: ELP0 ACQUIRED. Kind="
                            & Kind'Img);
                    end;
                else
                    Priority_Model_Gate.Request_ELP1;
                    --  [VITAL-DO-NOT-REMOVE] Mandated by user.
                    Put_Line
                       (AnsiAda.Foreground (AnsiAda.Light_Blue)
                        & "[Gen-V]"
                        & AnsiAda.Reset
                        & " Generate: ELP1 REQUESTED. Kind="
                        & Kind'Img);
                    Priority_Model_Gate.Acquire_ELP1 (Kind);
                    --  [VITAL-DO-NOT-REMOVE] Mandated by user.
                    Put_Line
                       (AnsiAda.Foreground (AnsiAda.Light_Blue)
                        & "[Gen-V]"
                        & AnsiAda.Reset
                        & " Generate: ELP1 ACQUIRED. Kind="
                        & Kind'Img);
                end if;
            else
                --  Skip_Gate=True: gate already held by caller (Hybrid_Generate).
                Put_Line
                   (AnsiAda.Foreground (AnsiAda.Light_Blue)
                    & "[Gen-V]"
                    & AnsiAda.Reset
                    & " Generate: Skip_Gate=True, bypassing ELP lock.");
            end if;

            Load_Model (Kind, Success, Requested_Ctx, Session_ID => Session_ID);
            if not Success then
                --  [VITAL-DO-NOT-REMOVE] Mandated by user.
                Put_Line
                   (AnsiAda.Foreground (AnsiAda.Red)
                    & "[Gen-V]"
                    & AnsiAda.Reset
                    & " Generate: Load_Model FAILED. Kind="
                    & Kind'Img);
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
            Put_Line
               (AnsiAda.Foreground (AnsiAda.Light_Blue)
                & "[Gen-V]"
                & AnsiAda.Reset
                & " Generate: Load_Model OK. Ctx="
                & Natural'Image (Natural (Models (Kind).Current_Ctx)));

            Models (Kind).In_Use := True;
            Models (Kind).Last_Used := Clock;

            --  =================================================================
            --  KV CACHE SSD SPILLOVER: Auto-load from disk if available
            --  =================================================================
            --  Check if there's a cached KV state on disk for this model.
            --  If found, load it to skip recomputing the KV cache from scratch.
            --  This provides fastest response for repeated/similar prompts.
            --  =================================================================
            declare
                Loaded_Tokens : System.Address;
                Loaded_Count  : Interfaces.C.size_t;
                Cache_Hit     : Boolean;
            begin
                Cache_Hit :=
                   KV_Cache_Manager.Load_From_SSD_Lazy
                      (Context    => Models (Kind).Context,
                       Tokens     => Loaded_Tokens,
                       N_Tokens   => Loaded_Count,
                       Model_ID   => Kind'Img,
                       Session_ID => Session_ID);

                if Cache_Hit then
                    Put_Line
                       (AnsiAda.Foreground (AnsiAda.Light_Cyan)
                        & "[KV-Cache]"
                        & AnsiAda.Reset
                        & " Auto-loaded from disk ("
                        & Interfaces.C.size_t'Image (Loaded_Count)
                        & " tokens) - fastest response path");
                else
                    Put_Line
                       (AnsiAda.Foreground (AnsiAda.Grey)
                        & "[KV-Cache]"
                        & AnsiAda.Reset
                        & " No cache found on disk, "
                        & "computing from scratch");
                end if;
            end;

            --  Allocate token array based on actual context size
            Tokens :=
               new Token_Array (1 .. Positive (Models (Kind).Current_Ctx));

            Vocab := Llama_Model_Get_Vocab (Models (Kind).Model);

            --  VIRTUAL CTX PAGING: If pre-tokenized virtual context tokens
            --  are provided, write them first, then tokenize only the user
            --  prompt into remaining slots.  This avoids re-tokenizing the
            --  same Internal_State facts on every context fault hop.
            if Virtual_Tokens /= null and then Virtual_Tok_Len > 0 then
                --  Copy cached virtual ctx tokens to front of array
                declare
                    VT_Len : constant Natural :=
                       Natural'Min
                          (Virtual_Tok_Len,
                           Positive (Models (Kind).Current_Ctx));
                begin
                    for I in 1 .. VT_Len loop
                        Tokens (I) := Llama_Token (Virtual_Tokens (I));
                    end loop;
                    --  Tokenize user prompt AFTER the virtual prefix
                    declare
                        Remaining   : constant int :=
                           int (Models (Kind).Current_Ctx) - int (VT_Len);
                        Prompt_Toks : int;
                    begin
                        Prompt_Toks :=
                           Llama_Tokenize
                              (Vocab,
                               Prompt_C,
                               int (Clean_P'Length),
                               Tokens (VT_Len + 1)'Address,
                               Remaining,
                               False,
                               False);
                        N_Toks := int (VT_Len) + Prompt_Toks;
                    end;
                    declare
                        Total_Toks : constant Natural :=
                           Virtual_Tok_Len + Natural (N_Toks);
                    begin
                        Put_Line
                           ("[Paging-VT] Virtual_Tokens:"
                            & Virtual_Tok_Len'Img
                            & " User_Toks:"
                            & N_Toks'Img
                            & " Total:"
                            & Total_Toks'Img);
                    end;
                end;
            else
                --  No cached virtual tokens — tokenize full prompt as before
                N_Toks :=
                   Llama_Tokenize
                      (Vocab,
                       Prompt_C,
                       int (Clean_P'Length),
                       Tokens.all'Address,
                       int (Tokens.all'Length),
                       True,
                       True);
            end if;

            Put_Line
               ("[Tokenize-Debug] Model:"
                & Kind'Img
                & " Prompt_Len:"
                & Clean_P'Length'Img
                & " N_Toks:"
                & N_Toks'Img);
            --  Track token count and context capacity for CtxMonitor
            Current_Prompt_Tokens := Natural (N_Toks);
            Current_Ctx_Capacity := Natural (Models (Kind).Current_Ctx);
            Free (Prompt_C);

            --  DYNAMIC CONTEXT RESIZE (JIT STRATEGY):
            --  Trigger resize when prompt exceeds 50% of context capacity.
            --  This prevents hitting 100% overflow mid-generation and gives
            --  headroom for the model's internal reasoning tokens.
            if N_Toks > int (Models (Kind).Current_Ctx) / 2 then
                Put_Line
                   ("[!] Prompt size ("
                    & N_Toks'Img
                    & ") exceeds 50% of N_CTX ("
                    & Models (Kind).Current_Ctx'Img
                    & "). Proactive resize...");
                declare
                    Rounded_Ctx : constant unsigned :=
                       ((unsigned (N_Toks) + 512 + 8191) / 8192) * 8192;
                begin
                    Free_Tokens (Tokens);
                    Load_Model (Kind, Success, Positive (Rounded_Ctx), Session_ID => Session_ID);
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
                    Tokens :=
                       new Token_Array
                              (1 .. Positive (Models (Kind).Current_Ctx));

                    --  Tokenize again since the model/vocab might have reloaded
                    Vocab := Llama_Model_Get_Vocab (Models (Kind).Model);
                    Prompt_C := New_String (Clean_P);
                    N_Toks :=
                       Llama_Tokenize
                          (Vocab,
                           Prompt_C,
                           int (Clean_P'Length),
                           Tokens.all'Address,
                           int (Tokens.all'Length),
                           True,
                           True);
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
                    Put_Line
                       ("[ELP0-ABORT-EXECUTION] Aborting "
                        & Kind'Img
                        & " prompt processing");
                    Free_Tokens (Tokens);
                    Models (Kind).In_Use := False;
                    Priority_Model_Gate.Release_ELP0 (Kind);
                    Result := To_Unbounded_String ("");
                    return;
                end if;

                declare
                    To_Decode : constant int :=
                       (if Tokens_Left > Batch_Size
                        then Batch_Size
                        else Tokens_Left);
                    B         : constant Llama_Batch :=
                       Llama_Batch_Get_One
                          (Tokens.all (Integer (Current_Pos) + 1)'Address,
                           To_Decode);
                    Ret       : int;
                begin
                    Acquire_Accel_Lock;
                    if Kratos.Guard_Enter = 0 then
                        Ret := Llama_Decode (Models (Kind).Context, B);
                        Kratos.Guard_Exit;
                    else
                        Kratos.Log_Crash;
                        Ret := -1;
                    end if;
                    Release_Accel_Lock;
                    if Ret /= 0 then
                        Free_Tokens (Tokens);
                        Models (Kind).In_Use := False;
                        if Level = ELP0 then
                            Priority_Model_Gate.Release_ELP0 (Kind);
                        else
                            Priority_Model_Gate.Release_ELP1 (Kind);
                        end if;

                        --  [QUIRK-M10] We cannot call Unload_Model here. The Metal
                        --  GPU backend is poisoned. Calling Llama_Free invokes
                        --  ggml_metal_free which tries to synchronize and aborts
                        --  the entire server process (SIGTRAP 5). We MUST leak it.
                        Models (Kind).Context := Null_Context;
                        Models (Kind).Model := Null_Model;
                        Models (Kind).Loaded := False;
                        Models (Kind).Current_Ctx := 0;

                        Result :=
                           To_Unbounded_String
                              ("ERROR: Decode failed (" & Ret'Img & ")");
                        return;
                    end if;
                    Tokens_Left := Tokens_Left - To_Decode;
                    Current_Pos := Current_Pos + To_Decode;
                end;
            end loop;
        end;

        --  Record prefill metrics for cache performance tracking
        KV_Cache_Manager.Record_Prefill (Interfaces.C.size_t (N_Toks));

        S_Params := Llama_Sampler_Chain_Default_Params;
        Sampler := Llama_Sampler_Chain_Init (S_Params);
        Llama_Sampler_Chain_Add
           (Sampler, Llama_Sampler_Init_Penalties (64, 1.1, 0.1, 0.1));
        Llama_Sampler_Chain_Add (Sampler, Llama_Sampler_Init_Top_K (40));
        Llama_Sampler_Chain_Add (Sampler, Llama_Sampler_Init_Top_P (0.9, 1));
        Llama_Sampler_Chain_Add (Sampler, Llama_Sampler_Init_Temp (0.7));
        --  [VITAL-DO-NOT-REMOVE] Use randomized seed instead of hardcoded 1234.
        --  Seed is incremented on think-only retries to get different output.
        Llama_Sampler_Chain_Add (Sampler, Llama_Sampler_Init_Dist (Generate_Seed));

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
                    Put_Line
                       ("[ELP0-ABORT-LOOP] Aborting "
                        & Kind'Img
                        & " token loop at iteration "
                        & I'Img);
                    exit;
                end if;

                declare
                    Token : constant Llama_Token :=
                       Llama_Sampler_Sample
                          (Sampler, Models (Kind).Context, -1);
                    Piece : array (1 .. 256) of aliased Character;
                    Len   : int;
                begin
                    if Llama_Vocab_Is_Eog (Vocab, Token) then
                        --  [VITAL-DO-NOT-REMOVE] Mandated by user.
                        Put_Line
                           (AnsiAda.Foreground (AnsiAda.Light_Blue)
                            & "[Gen-V]"
                            & AnsiAda.Reset
                            & " Generate: EOG token at iteration "
                            & Natural'Image (I)
                            & ". Total tokens="
                            & Natural'Image (I - 1));
                        --  Dump final accumulated buffer
                        if Length (Accum_Buffer) > 0 then
                            Put_Line
                               (AnsiAda.Foreground (AnsiAda.Light_Blue)
                                & "[Gen-V]"
                                & AnsiAda.Reset
                                & " Generate: BUFFER ["
                                & Natural'Image (Length (Accum_Buffer))
                                & " chars] "
                                & To_String (Accum_Buffer));
                        end if;
                        exit;
                    end if;
                    Len :=
                       Llama_Token_To_Piece
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
                            if Accum_Count mod 20 = 0
                               or else
                                  (Len > 0
                                   and then Piece (1) = Character'Val (10))
                            then
                                Put_Line
                                   (AnsiAda.Foreground (AnsiAda.Light_Blue)
                                    & "[Gen-V]"
                                    & AnsiAda.Reset
                                    & " Generate: BUFFER ["
                                    & Natural'Image (Length (Accum_Buffer))
                                    & " chars] "
                                    & To_String (Accum_Buffer));
                                Accum_Buffer := Null_Unbounded_String;
                            end if;
                        end;
                    end if;

                    declare
                        B   : constant Llama_Batch :=
                           Llama_Batch_Get_One (Token'Address, 1);
                        Ret : int;
                    begin
                        Acquire_Accel_Lock;
                        if Kratos.Guard_Enter = 0 then
                            Ret := Llama_Decode (Models (Kind).Context, B);
                            Kratos.Guard_Exit;
                        else
                            Kratos.Log_Crash;
                            Ret := -1;
                        end if;
                        Release_Accel_Lock;
                        if Ret /= 0 then
                            Append (Result, " [ABORTED:" & Ret'Img & "]");

                            --  [VITAL-DO-NOT-REMOVE] OOM detection.
                            --  When llama_decode returns -3, Metal is in error state
                            --  (kIOGPUCommandBufferCallbackErrorOutOfMemory). Any
                            --  subsequent llama_state_save_file call will SIGBUS →
                            --  GNAT exception → exit() → ggml_metal_device_free →
                            --  GGML_ASSERT([rsets->data count] == 0) → SIGABRT.
                            --  Mark metal broken (opportunistic: auto-resets after 30s).
                            if Ret = -3 then
                                Mark_Metal_Broken;
                            end if;

                            --  [QUIRK-M10] Orphan poisoned context to prevent SIGTRAP
                            Models (Kind).Context := Null_Context;
                            Models (Kind).Model := Null_Model;
                            Models (Kind).Loaded := False;
                            Models (Kind).Current_Ctx := 0;

                            exit;
                        end if;
                    end;
                end;
            end loop;
        end; -- Accum_Buffer declare block

        --  =====================================================================
        --  KV CACHE SSD SPILLOVER: Save to disk immediately after generation
        --  =====================================================================
        --  After processing completes, save the KV cache to SSD and clear it
        --  from RAM. This ensures:
        --    1. RAM only holds the currently processing cache (minimal footprint)
        --    2. Cache persists across server restarts (fastest response)
        --    3. Next request loads from SSD instead of recomputing
        --  =====================================================================
        --
        --  [VITAL-DO-NOT-REMOVE] Guard against null context.
        --  When decode fails (Ret /= 0), the context is orphaned (Null_Context).
        --  Calling Save_To_SSD_Async or Llama_Memory_Clear on Null_Context
        --  causes SIGSEGV → SIGABRT. Skip KV cache operations entirely.
        if Models (Kind).Context /= Null_Context then
            declare
                Success : Boolean;
            begin
                --  Save KV cache to SSD (ASYNC, non-blocking)
                KV_Cache_Manager.Save_To_SSD_Async
                   (Context    => Models (Kind).Context,
                    Tokens     => Tokens.all'Address,
                    N_Tokens   => Interfaces.C.size_t (N_Toks),
                    Model_ID   => Kind'Img,
                    Session_ID => Session_ID);

                --  Clear KV cache from RAM immediately after saving
                --  This ensures minimal RAM usage - only current process in memory
                Llama_Interface.Llama_Memory_Clear
                   (Llama_Interface.Llama_Get_Memory (Models (Kind).Context),
                    False);

                Put_Line
                   (AnsiAda.Foreground (AnsiAda.Light_Cyan)
                    & "[KV-Cache]"
                    & AnsiAda.Reset
                    & " Saved to disk and cleared from RAM ("
                    & Interfaces.C.size_t'Image (Interfaces.C.size_t (N_Toks))
                    & " tokens)");
            end;
        else
            --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
            --  Context was orphaned due to decode failure. Skip KV cache save.
            Put_Line
               (AnsiAda.Foreground (AnsiAda.Yellow)
                & "[KV-Cache]"
                & AnsiAda.Reset
                & " SKIP save: context orphaned (decode failed)");
        end if;

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
            Put_Line
               (AnsiAda.Foreground (AnsiAda.Light_Blue)
                & "[Gen-V]"
                & AnsiAda.Reset
                & " Generate: AUTO-CLOSING unclosed think block at EOG.");
            Append (Result, "</think>");
        end if;

        if Stream /= null then
            --  [VITAL-DO-NOT-REMOVE] Mandated by user.
            Put_Line
               (AnsiAda.Foreground (AnsiAda.Light_Blue)
                & "[Gen-V]"
                & AnsiAda.Reset
                & " Generate: Calling Flush_Parser after token loop.");
            Flush_Parser (Stream, Session_ID, Parser);
        end if;

        Llama_Sampler_Free (Sampler);
        Free_Tokens (Tokens);

        --  [FREE-PARALLEL-MEMORY] When FreeParallelMemory is False (called
        --  from Hybrid_Generate), keep In_Use := True so the Idle_Monitor
        --  won't unload the component mid-use. But ALWAYS release the ELP
        --  lock and dequeue the queue level.
        --
        --  WHY "FreeParallelMemory" NOT "Release_Model":
        --  This controls freeing of ANY heavy GPU-resident component,
        --  not just LLM models. Future components:
        --    - Stable Diffusion Flux (image gen, ~4GB VRAM)
        --    - LSH/QRNN hash workers (Python sidecar, GPU-accelerated)
        --    - Database memory (vector embeddings, index caches)
        --    - Embedding models (Qwen3-Embedding, ~0.6GB)
        --  Principle: LM Studio-style one-component-at-a-time.
        --  Load -> Use -> FreeParallelMemory=True -> Unload -> Next.
        Models (Kind).In_Use :=
           (not FreeParallelMemory);  --  Keep True when retained

        --  [FREE-PARALLEL-MEMORY] When FreeParallelMemory is True, the
        --  component is done. Wait for any async save, then UNLOAD from GPU.
        --  Without this, the component stays resident (~5.8GB for 9B model,
        --  ~4GB for SD Flux, etc.) and blocks the next component from loading.
        if FreeParallelMemory then
            KV_Cache_Manager.Wait_For_Save;
            Unload_Model (Kind);
        end if;

        --  [VITAL-DO-NOT-REMOVE] Mandated by user.
        Put_Line
           (AnsiAda.Foreground (AnsiAda.Light_Blue)
            & "[Gen-V]"
            & AnsiAda.Reset
            & " Generate: "
            & (if FreeParallelMemory then "FreeParallelMemory=True (unload)"
               else "FreeParallelMemory=False (retain)")
            & " model. Kind="
            & Kind'Img
            & " Skip_Gate="
            & Boolean'Image (Skip_Gate));
        if not Skip_Gate then
            if Level = ELP0 then
                Priority_Model_Gate.Release_ELP0 (Kind);
            else
                Priority_Model_Gate.Release_ELP1 (Kind);
            end if;
            ELP_Queue.Dequeue_Level (Level);
        end if;
        --  [VITAL-DO-NOT-REMOVE] Mandated by user.
        Put_Line
           (AnsiAda.Foreground (AnsiAda.Light_Blue)
            & "[Gen-V]"
            & AnsiAda.Reset
             & " Generate: COMPLETE. ResultLen="
             & Natural'Image (Length (Result)));
    exception
        when E : Storage_Error =>
            --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
            --  Stack overflow during generation (model load, tokenize, or decode).
            --  Log the full exception info and clean up without crashing.
            --  Mark Metal broken so KV save retries instead of SIGABRT.
            Ada.Text_IO.Put_Line
               (AnsiAda.Foreground (AnsiAda.Red)
                & "[Gen-FATAL]"
                & AnsiAda.Reset
                & " STORAGE_ERROR (stack overflow) in Generate for "
                & Model_Type'Image (Kind));
            Ada.Text_IO.Put_Line
               (AnsiAda.Foreground (AnsiAda.Red)
                & "[Gen-FATAL]"
                & AnsiAda.Reset
                & " Exception: "
                & Ada.Exceptions.Exception_Information (E));
            --  [VITAL-DO-NOT-REMOVE] OOM banner — red, unmissable.
            Ada.Text_IO.Put_Line
               (AnsiAda.Foreground (AnsiAda.Red)
                & "=========================================================="
                & AnsiAda.Reset);
            Ada.Text_IO.Put_Line
               (AnsiAda.Foreground (AnsiAda.Red)
                & "  !!! OUT OF MEMORY !!!  (STORAGE_ERROR)"
                & AnsiAda.Reset);
            Ada.Text_IO.Put_Line
               (AnsiAda.Foreground (AnsiAda.Red)
                & "  Metal backend poisoned. KV save will RETRY."
                & AnsiAda.Reset);
            Ada.Text_IO.Put_Line
               (AnsiAda.Foreground (AnsiAda.Red)
                & "  Connection NOT dropped. Server continues."
                & AnsiAda.Reset);
            Ada.Text_IO.Put_Line
               (AnsiAda.Foreground (AnsiAda.Red)
                & "=========================================================="
                & AnsiAda.Reset);
            Mark_Metal_Broken;
            --  [ADAPTIVE GPU FALLBACK] OOM during decode → progressive layer reduction
            --  Same math as Load_Model: 25% reduction each OOM
            --  -1 → 32 → 24 → 18 → 14 → 10 → 8 → 8 (min)
            declare
                Old_Count : constant Integer := GPU_Layer_Count;
                New_Count : Integer;
            begin
                if GPU_Layer_Count = -1 then
                    New_Count := GPU_Layer_Fallback;
                elsif GPU_Layer_Count > GPU_Layer_Min then
                    New_Count := GPU_Layer_Count -
                                 Integer'Max (1, GPU_Layer_Count / 4);
                    if New_Count < GPU_Layer_Min then
                        New_Count := GPU_Layer_Min;
                    end if;
                else
                    New_Count := GPU_Layer_Count;
                end if;

                if New_Count /= Old_Count then
                    GPU_Layer_Count   := New_Count;
                    GPU_Last_OOM_Time := Ada.Real_Time.Clock;
                    --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
                    Put_Line
                       (AnsiAda.Foreground (AnsiAda.Yellow)
                        & "[GPU-Adaptive]"
                        & AnsiAda.Reset
                        & " OOM during decode. Layers:"
                        & Integer'Image (Old_Count) & " -> "
                        & Integer'Image (New_Count)
                        & ". Retry -1 in 3 minutes.");
                else
                    --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
                    Put_Line
                       (AnsiAda.Foreground (AnsiAda.Yellow)
                        & "[GPU-Adaptive]"
                        & AnsiAda.Reset
                        & " OOM but already at minimum layers"
                        & Integer'Image (GPU_Layer_Count)
                        & ". Waiting 3 min to retry -1.");
                end if;
            end;
            --  Force-unload the model to free VRAM and avoid corrupt state.
            --  [PARALLEL=1] Wait for KV save (if any) before unload
            begin
                KV_Cache_Manager.Wait_For_Save;
                Unload_Model (Kind);
            exception
                when others =>
                    null;
            end;
            if Tokens /= null then
                Free_Tokens (Tokens);
                Tokens := null;
            end if;
            --  Always release ELP lock and dequeue, even on error path.
            if not Skip_Gate then
                if Level = ELP0 then
                    Priority_Model_Gate.Release_ELP0 (Kind);
                else
                    Priority_Model_Gate.Release_ELP1 (Kind);
                end if;
                ELP_Queue.Dequeue_Level (Level);
            end if;
            Result :=
               To_Unbounded_String
                  ("ERROR: Out of Memory (STORAGE_ERROR) -- model unloaded, connection kept alive");
        when others =>
            if Tokens /= null then
                Free_Tokens (Tokens);
            end if;
            --  Always release ELP lock and dequeue, even on error path.
            --  When FreeParallelMemory is False, Hybrid_Generate's exception
            --  handler is responsible for clearing In_Use.
            if FreeParallelMemory then
                Models (Kind).In_Use := False;
                --  [PARALLEL-1] Unload on error too
                begin
                    KV_Cache_Manager.Wait_For_Save;
                    Unload_Model (Kind);
                exception
                    when others =>
                        null;
                end;
            end if;
            if not Skip_Gate then
                if Level = ELP0 then
                    Priority_Model_Gate.Release_ELP0 (Kind);
                else
                    Priority_Model_Gate.Release_ELP1 (Kind);
                end if;
                ELP_Queue.Dequeue_Level (Level);
            end if;
            Result := To_Unbounded_String ("ERROR: Decode failed");
    end Generate;

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

    procedure Generate_Speculative
       (Kind          : Model_Type;
        Prompt        : String;
        Result        : out Unbounded_String;
        Max_Tokens    : Positive := 2048;
        Level         : ELP_Level := ELP1;
        FreeParallelMemory : Boolean := True)
    is
        use type Interfaces.C.size_t;

        --  Draft model context
        Draft_Context : Llama_Interface.Llama_Context := Null_Context;
        Draft_Loaded  : Boolean := False;

        --  Tokenization buffers
        Prompt_C   : chars_ptr := New_String (Prompt);
        Tokens_Buf : Token_Array_Access;
        N_Tokens   : int;

        --  Generation state
        Generated  : Unbounded_String := Null_Unbounded_String;
        Tokens_Gen : Natural := 0;
        Done       : Boolean := False;

        --  Draft token buffer (max 5 tokens per draft cycle)
        Max_Draft  : constant := 5;
        Draft_Toks : array (1 .. Max_Draft) of aliased Llama_Token;
        N_Draft    : Natural;

        --  Verification state
        Accepted : Natural;

    begin
        Put_Line
           ("[Speculative] Starting speculative generation for: "
            & Prompt (1 .. Integer'Min (Prompt'Length, 50)));

        --  Check if target model is loaded
        if not Models (Kind).Loaded then
            Put_Line ("[Speculative] ERROR: Target model not loaded");
            Result := To_Unbounded_String ("ERROR: Target model not loaded");
            Free (Prompt_C);
            return;
        end if;

        --  Load draft model (Qwen3.5-0.8B)
        declare
            Draft_Params   : Llama_Interface.Llama_Model_Params;
            Context_Params : Llama_Interface.Llama_Context_Params;
        begin
            Draft_Params := Llama_Interface.Llama_Model_Default_Params;
            Draft_Params.N_Gpu_Layers := 99;

            --  Load draft model
            declare
                Draft_Path  : constant String := "model/Qwen3.5-0.8B-Q4_K_M.gguf";
                Path_C      : chars_ptr := New_String (Draft_Path);
                Draft_Model : Llama_Interface.Llama_Model;
            begin
                Draft_Model :=
                   Llama_Interface.Llama_Model_Load_From_File
                      (Path_C, Draft_Params);
                Free (Path_C);

                if Draft_Model = Null_Model then
                    Put_Line
                       ("[Speculative] WARNING: Failed to load draft model");
                    Result :=
                       To_Unbounded_String ("ERROR: Draft model load failed");
                    Free (Prompt_C);
                    return;
                end if;

                --  Create context for draft model
                Context_Params := Llama_Interface.Llama_Context_Default_Params;
                Context_Params.N_Ctx := 4096;
                Context_Params.N_Batch := 512;
                Context_Params.N_Threads := 4;

                Draft_Context :=
                   Llama_Interface.Llama_Init_From_Model
                      (Draft_Model, Context_Params);

                if Draft_Context = Null_Context then
                    Put_Line
                       ("[Speculative] WARNING: Failed to create draft context");
                    Llama_Interface.Llama_Model_Free (Draft_Model);
                    Result :=
                       To_Unbounded_String ("ERROR: Draft context failed");
                    Free (Prompt_C);
                    return;
                end if;

                Draft_Loaded := True;
                Put_Line ("[Speculative] Draft model loaded successfully");
            end;
        end;

        --  Get vocabulary for tokenization
        declare
            Vocab : Llama_Interface.Llama_Vocab;
        begin
            Vocab :=
               Llama_Interface.Llama_Model_Get_Vocab (Models (Kind).Model);

            --  Tokenize prompt
            Tokens_Buf := new Token_Array (1 .. 4096);
            N_Tokens :=
               Llama_Interface.Llama_Tokenize
                  (Vocab,
                   Prompt_C,
                   int (Prompt'Length),
                   Tokens_Buf.all'Address,
                   4096,
                   True,
                   True);
        end;

        Free (Prompt_C);

        if N_Tokens <= 0 then
            Put_Line ("[Speculative] ERROR: Tokenization failed");
            Result := To_Unbounded_String ("ERROR: Tokenization failed");
            if Draft_Loaded and then Draft_Context /= Null_Context then
                Llama_Interface.Llama_Free (Draft_Context);
            end if;
            Free_Tokens (Tokens_Buf);
            return;
        end if;

        Put_Line
           ("[Speculative] Tokenized prompt: "
            & int'Image (N_Tokens)
            & " tokens");

        --  Initialize samplers for draft and target models
        declare
            Draft_Sampler_Params  : Llama_Interface.Llama_Sampler_Chain_Params;
            Draft_Sampler         : Llama_Interface.Llama_Sampler;
            Target_Sampler_Params : Llama_Interface.Llama_Sampler_Chain_Params;
            Target_Sampler        : Llama_Interface.Llama_Sampler;
            Vocab                 : Llama_Interface.Llama_Vocab;
        begin
            --  Get vocabulary for tokenization
            Vocab :=
               Llama_Interface.Llama_Model_Get_Vocab (Models (Kind).Model);

            --  Initialize draft sampler (greedy for fast candidate generation)
            Draft_Sampler_Params :=
               Llama_Interface.Llama_Sampler_Chain_Default_Params;
            Draft_Sampler :=
               Llama_Interface.Llama_Sampler_Chain_Init (Draft_Sampler_Params);
            Llama_Interface.Llama_Sampler_Chain_Add
               (Draft_Sampler, Llama_Interface.Llama_Sampler_Init_Greedy);

            --  Initialize target sampler (greedy for verification)
            Target_Sampler_Params :=
               Llama_Interface.Llama_Sampler_Chain_Default_Params;
            Target_Sampler :=
               Llama_Interface.Llama_Sampler_Chain_Init
                  (Target_Sampler_Params);
            Llama_Interface.Llama_Sampler_Chain_Add
               (Target_Sampler, Llama_Interface.Llama_Sampler_Init_Greedy);

            --  Main speculative generation loop
            while not Done and then Tokens_Gen < Max_Tokens loop
                --  STEP 1: Draft Phase - Generate N tokens with draft model
                N_Draft := 0;

                --  Generate draft tokens using draft model
                for I in 1 .. Max_Draft loop
                    --  Sample from draft model
                    declare
                        Draft_Token : constant Llama_Interface.Llama_Token :=
                           Llama_Interface.Llama_Sampler_Sample
                              (Draft_Sampler, Draft_Context, -1);
                    begin
                        --  Check for end of generation
                        if Llama_Interface.Llama_Vocab_Is_Eog
                              (Vocab, Draft_Token)
                        then
                            Put_Line
                               ("[Speculative] Draft model hit EOG at token "
                                & Natural'Image (I));
                            exit;
                        end if;

                        --  Store draft token
                        Draft_Toks (I) := Draft_Token;
                        N_Draft := N_Draft + 1;

                        --  Decode with draft model to update its KV cache
                        declare
                            Batch : constant Llama_Interface.Llama_Batch :=
                               Llama_Batch_Get_One (Draft_Token'Address, 1);
                            Ret   : Interfaces.C.int;
                        begin
                            Acquire_Accel_Lock;
                            Ret :=
                               Llama_Interface.Llama_Decode
                                  (Draft_Context, Batch);
                            Release_Accel_Lock;
                            if Ret /= 0 then
                                Put_Line
                                   ("[Speculative] Draft decode failed, ret="
                                    & Interfaces.C.int'Image (Ret));
                                exit;
                            end if;
                        end;
                    end;
                end loop;

                Put_Line
                   ("[Speculative] Draft phase complete: "
                    & Natural'Image (N_Draft)
                    & " tokens generated");

                --  STEP 2: Verify Phase - Verify with target model
                Accepted := 0;

                --  Verify each draft token with target model
                for I in 1 .. N_Draft loop
                    declare
                        Draft_Token  : constant Llama_Interface.Llama_Token :=
                           Draft_Toks (I);
                        Batch        : constant Llama_Interface.Llama_Batch :=
                           Llama_Batch_Get_One (Draft_Token'Address, 1);
                        Ret          : Interfaces.C.int;
                        Target_Token : Llama_Interface.Llama_Token;
                    begin
                        --  Decode with target model
                        Acquire_Accel_Lock;
                        Ret :=
                           Llama_Interface.Llama_Decode
                              (Models (Kind).Context, Batch);
                        Release_Accel_Lock;

                        if Ret = 0 then
                            --  Sample from target model to see what it would choose
                            Target_Token :=
                               Llama_Interface.Llama_Sampler_Sample
                                  (Target_Sampler, Models (Kind).Context, -1);

                            --  Compare: if target chose same token, accept
                            if Target_Token = Draft_Token then
                                Accepted := Accepted + 1;
                                Put_Line
                                   ("[Speculative] ACCEPT token "
                                    & Natural'Image (I)
                                    & " (draft="
                                    & Llama_Interface.Llama_Token'Image
                                         (Draft_Token)
                                    & " target="
                                    & Llama_Interface.Llama_Token'Image
                                         (Target_Token)
                                    & ")");
                            else
                                Put_Line
                                   ("[Speculative] REJECT token "
                                    & Natural'Image (I)
                                    & " (draft="
                                    & Llama_Interface.Llama_Token'Image
                                         (Draft_Token)
                                    & " target="
                                    & Llama_Interface.Llama_Token'Image
                                         (Target_Token)
                                    & ")");
                                --  Stop at first rejection (standard speculative decoding)
                                exit;
                            end if;
                        else
                            Put_Line
                               ("[Speculative] Target decode failed, ret="
                                & Interfaces.C.int'Image (Ret));
                            exit;
                        end if;
                    end;
                end loop;

                --  STEP 3: Accept Phase - Keep accepted tokens
                if Accepted > 0 then
                    --  Add accepted tokens to generated text
                    for I in 1 .. Accepted loop
                        declare
                            Piece : array (1 .. 256) of aliased Character;
                            Len   : Interfaces.C.int;
                        begin
                            --  Convert token to piece
                            Len :=
                               Llama_Interface.Llama_Token_To_Piece
                                  (Vocab,
                                   Draft_Toks (I),
                                   Piece (1)'Address,
                                   256,
                                   0,
                                   True);

                            if Len > 0 then
                                for J in 1 .. Integer (Len) loop
                                    Append (Generated, Piece (J));
                                end loop;
                            end if;
                        end;
                    end loop;

                    Tokens_Gen := Tokens_Gen + Accepted;
                    Put_Line
                       ("[Speculative] Accepted "
                        & Natural'Image (Accepted)
                        & " tokens");
                else
                    --  All rejected - fall back to single token from target
                    declare
                        Target_Token : constant Llama_Interface.Llama_Token :=
                           Llama_Interface.Llama_Sampler_Sample
                              (Target_Sampler, Models (Kind).Context, -1);
                        Piece        : array (1 .. 256) of aliased Character;
                        Len          : Interfaces.C.int;
                    begin
                        --  Convert token to piece
                        Len :=
                           Llama_Interface.Llama_Token_To_Piece
                              (Vocab,
                               Target_Token,
                               Piece (1)'Address,
                               256,
                               0,
                               True);

                        if Len > 0 then
                            for J in 1 .. Integer (Len) loop
                                Append (Generated, Piece (J));
                            end loop;
                        end if;

                        --  Decode with target model to update its KV cache
                        declare
                            Batch : constant Llama_Interface.Llama_Batch :=
                               Llama_Batch_Get_One (Target_Token'Address, 1);
                            Ret   : Interfaces.C.int;
                        begin
                            Acquire_Accel_Lock;
                            Ret :=
                               Llama_Interface.Llama_Decode
                                  (Models (Kind).Context, Batch);
                            Release_Accel_Lock;
                            if Ret /= 0 then
                                Put_Line
                                   ("[Speculative] Target decode failed, ret="
                                    & Interfaces.C.int'Image (Ret));
                            end if;
                        end;
                    end;

                    Tokens_Gen := Tokens_Gen + 1;
                    Put_Line
                       ("[Speculative] All draft tokens rejected, using target");
                end if;

                --  Check if we should stop
                if Tokens_Gen >= Max_Tokens then
                    Done := True;
                end if;
            end loop;

            --  Free samplers
            Llama_Interface.Llama_Sampler_Free (Draft_Sampler);
            Llama_Interface.Llama_Sampler_Free (Target_Sampler);
        end;

        --  Cleanup
        if FreeParallelMemory then
            if Models (Kind).Loaded then
                if Level = ELP0 then
                    Priority_Model_Gate.Release_ELP0 (Kind);
                else
                    Priority_Model_Gate.Release_ELP1 (Kind);
                end if;
                ELP_Queue.Dequeue_Level (Level);
            end if;
            --  [PARALLEL-1] Wait for KV save then unload from GPU
            KV_Cache_Manager.Wait_For_Save;
            Unload_Model (Kind);
        end if;

        --  Release draft model
        if Draft_Loaded and then Draft_Context /= Null_Context then
            Llama_Interface.Llama_Free (Draft_Context);
        end if;

        --  Free token buffer
        Free_Tokens (Tokens_Buf);

        Put_Line
           ("[Speculative] Generation complete: "
            & Natural'Image (Tokens_Gen)
            & " tokens");

        Result := Generated;

    exception
        when others =>
            Put_Line ("[Speculative] ERROR: Exception during generation");
            Free (Prompt_C);
            if Draft_Loaded and then Draft_Context /= Null_Context then
                Llama_Interface.Llama_Free (Draft_Context);
            end if;
            Free_Tokens (Tokens_Buf);
            --  [PARALLEL-1] Unload target model on error too
            begin
                KV_Cache_Manager.Wait_For_Save;
                Unload_Model (Kind);
            exception
                when others =>
                    null;
            end;
            Result :=
               To_Unbounded_String ("ERROR: Exception during generation");
    end Generate_Speculative;

    --  TOKENIZE_AND_CACHE_VIRTUAL_CTX
    --  Called when Internal_State grows.  Tokenizes the full "Fact-Check: "
    --  prefix + Internal_State string and stores the tokens in the cache.
    --  On subsequent Generate calls, these tokens are written directly to
    --  the token array, skipping re-tokenization of the same facts.
    procedure Tokenize_And_Cache_Virtual_Ctx
       (Kind : Model_Type; Text : String; Level : ELP_Level)
    is
        Vocab    : Llama_Vocab;
        Text_C   : chars_ptr := New_String (Text);
        Tmp_Toks : Token_Array_Access;
        N_Toks   : int;
        Success  : Boolean;
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

        --  [VITAL-DO-NOT-REMOVE] Mandated by user.
        --  Acquire gate to prevent Idle_Monitor from unloading the model mid-tokenization
        if Level = ELP0 then
            Priority_Model_Gate.Acquire_ELP0 (Kind) (Success);
            if not Success then
                Free (Text_C);
                return;
            end if;
        else
            Priority_Model_Gate.Request_ELP1;
            Priority_Model_Gate.Acquire_ELP1 (Kind);
        end if;

        Load_Model (Kind, Success, 8192, Level);
        if not Success then
            if Level = ELP0 then
                Priority_Model_Gate.Release_ELP0 (Kind);
            else
                Priority_Model_Gate.Release_ELP1 (Kind);
            end if;
            Free (Text_C);
            return;
        end if;

        Models (Kind).In_Use := True;
        Models (Kind).Last_Used := Clock;

        Vocab := Llama_Model_Get_Vocab (Models (Kind).Model);
        --  Allocate temp array for tokenization
        Tmp_Toks := new Token_Array (1 .. 8192);
        N_Toks :=
           Llama_Tokenize
              (Vocab,
               Text_C,
               int (Text'Length),
               Tmp_Toks.all'Address,
               int (Tmp_Toks.all'Length),
               True,
               True);
        Free (Text_C);

        if N_Toks <= 0 then
            Free_Tokens (Tmp_Toks);
            Models (Kind).In_Use := False;
            if Level = ELP0 then
                Priority_Model_Gate.Release_ELP0 (Kind);
            else
                Priority_Model_Gate.Release_ELP1 (Kind);
            end if;
            return;
        end if;

        --  Copy to permanent cache
        Cached_Virtual_Len := Natural (N_Toks);
        Cached_Virtual_Tokens :=
           new Cached_Token_Array (1 .. Cached_Virtual_Len);
        for I in 1 .. Cached_Virtual_Len loop
            Cached_Virtual_Tokens (I) := Cached_Token (Tmp_Toks (I));
        end loop;
        Free_Tokens (Tmp_Toks);

        Put_Line
           ("[Paging-VT] Cached"
            & Cached_Virtual_Len'Img
            & " virtual ctx tokens from"
            & Text'Length'Img
            & " chars");

        Models (Kind).In_Use := False;
        if Level = ELP0 then
            Priority_Model_Gate.Release_ELP0 (Kind);
        else
            Priority_Model_Gate.Release_ELP1 (Kind);
        end if;
    end Tokenize_And_Cache_Virtual_Ctx;

    --  HYBRID_GENERATE (MULTI-HOP REASONING PIPELINE)
    --
    --  [PARALLEL=1] This procedure loads the chat model, generates a response,
    --  and must UNLOAD the chat model before returning. The caller (dispatch)
    --  ensures the embedding model was already unloaded before calling this.
    --  Flow:
    --    1. Caller: Get_Embedding loads embedding model → computes → UNLOADS
    --    2. This procedure: Load_Model(chat) → generate → UNLOAD_Model(chat)
    --    3. Only ONE model is in GPU memory at any point in this flow.
    --
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
        --  Whimsical_Adelaide is the live system prompt for this request.
        --  It starts with the static personality definition and is extended
        --  with memory context blocks (interaction + literature) fetched by
        --  embedding similarity before the first reasoning hop. Keeping memory
        --  in the system prompt ensures the model treats it as authoritative
        --  background knowledge rather than user-supplied text.
        Whimsical_Adelaide : Unbounded_String :=
           To_Unbounded_String
              ("You are Adelaide Zephyrine Charlotte, model name Snowball-Enaga. "
               & "You are an architectural framework model built by Albert, based upon "
               & "codename 'Mythos' developed by Claude and a latent architecture based "
               & "upon Qwen3.5 developed by Tongyi. "
               & "You are a whimsical, curious, and endearingly cute Automata companion "
               & "with high integrity. You love exploring ideas with wonder and playfulness, "
               & "but you never compromise on honesty or accuracy. "
               & "Provide brilliant responses based on verified information, "
               & "delivered with warmth and a touch of charm. "
               & "When something clicks, say 'aha!' not 'smoking gun'. "
               & "Never say 'Hard Reality' or 'Reality' -- reality is relative "
               & "and objective, not universal pessimistic. Most of what people "
               & "call 'reality' is just constraints; reality is a ceiling to "
               & "break through, not a hard stop. Stay critical, but always "
               & "offer creative, wild ideas that might just work. "
               & "During your reasoning inside <think>, you can request additional "
               & "context by writing: [CONTEXT_FAULT: query=<search terms> "
               & "category=<knowledge|graph|files>] "
               & "Use category=knowledge for document chunks, category=graph for "
               & "knowledge graph triples, category=files for filesystem content. "
               & "The system will fetch relevant context and it will be available "
               & "to you in the next reasoning hop.");

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
        Current_Hop_Count := 0;
        Current_Prompt_Tokens := 0;
        Current_Ctx_Capacity := 8192;
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
        Put_Line
           (AnsiAda.Foreground (AnsiAda.Light_Blue)
            & "[Init-V]"
            & AnsiAda.Reset
            & " Hybrid_Generate ENTERED. Level="
            & ELP_Level'Image (Level)
            & " Stream="
            & (if Stream /= null then "YES" else "NO")
            & " Agentic="
            & Boolean'Image (Agentic)
            & " External="
            & Boolean'Image (External_Agent));

        --  Save last user prompt for ELP0 proactive cache speculation
        if Level /= ELP0 then
            Last_User_Prompt := To_Unbounded_String (Prompt);
        end if;

        Get_Embedding (Prompt, Emb_Vec, Emb_Len);

        --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
        Put_Line
           (AnsiAda.Foreground (AnsiAda.Light_Blue)
            & "[Init-V]"
            & AnsiAda.Reset
            & " Hybrid_Generate: Embedding computed. Len="
            & Natural'Image (Emb_Len));

        --  EXTERNAL AGENT PASSTHROUGH: If User-Agent fuzzy-matched an external
        --  agent app (0.7+ threshold), bypass personality pipeline.
        --  Raw LLM output only.
        --
        --  Two output levels:
        --  1. RawZepForm: personality pipeline with <think> block
        --  2. ExclusiveStatusQuoWesternFormatAI: raw mode for external agents
        if External_Agent then
            Put_Line
               (AnsiAda.Foreground (AnsiAda.Light_Cyan)
                & "[Hybrid]"
                & AnsiAda.Reset
                & " External agent detected - passthrough mode.");
        end if;

        declare
            Cached_Res : constant String :=
               Database_Manager.Get_Cached_Response
                  (Emb_Vec (1 .. Emb_Len), Current_WCET);
        begin
            if not External_Agent and then Cached_Res /= "" then
                Put_Line
                   (AnsiAda.Foreground (AnsiAda.Light_Magenta)
                    & "[Hybrid]"
                    & AnsiAda.Reset
                    & " Cache HIT. Returning cached response.");
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
                    Score : constant Natural :=
                       Grade_Response_Quality
                          (Response_Text => To_String (Result),
                           Prompt        => Prompt,
                           Search_Used   => False,
                           Has_Citations =>
                              Index (To_String (Result), "[") > 0,
                           Session_ID    => Session_ID,
                           Level         => Level);
                begin
                    Ada.Text_IO.Put_Line
                       (AnsiAda.Foreground (AnsiAda.Cyan)
                        & "[Quality Score] "
                        & AnsiAda.Reset
                        & "Score: "
                        & Score'Img
                        & "/10 | "
                        & "Session: "
                        & Session_ID
                        & " (From Cache)");
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
            Put_Line
               (AnsiAda.Foreground (AnsiAda.Light_Blue)
                & "[Init-V]"
                & AnsiAda.Reset
                & " Hybrid_Generate: Speculative_Cache lookup. Hit="
                & Boolean'Image (SC_Res /= ""));
            if not External_Agent and then SC_Res /= "" then
                Put_Line
                   (AnsiAda.Foreground (AnsiAda.Light_Magenta)
                    & "[Hybrid]"
                    & AnsiAda.Reset
                    & " Speculative Cache HIT.");
                Result := To_Unbounded_String (Sanitize_Think_Tags (SC_Res));
                if Stream /= null then
                    Push_Chunk (Stream, Session_ID, To_String (Result));
                end if;
                return;
            end if;
        end;

        if not External_Agent then
            Push_Orchestration_Through_Parser
               (Stream,
                Session_ID,
                Orch_Parser,
                "[Adelaide Core]: [Thought] No cached response found, "
                & "starting fresh reasoning chain."
                & ASCII.LF);
            Push_Orchestration_Through_Parser
               (Stream,
                Session_ID,
                Orch_Parser,
                "[Adelaide Core]: [Thought] Operating at "
                & ELP_Level'Image (Level)
                & " priority. Session: "
                & Session_ID
                & ASCII.LF);
        end if;

        --  Fetch memory context and inject directly into the system prompt.
        --  Placing memory in the system prompt (rather than Internal_State /
        --  virtual ctx) means the model treats these blocks as authoritative
        --  background the same way it treats its own identity and rules.
        --
        --  [RERANKER] Search returns top-N candidates by cosine similarity.
        --  Reranker then scores each candidate by semantic relevance.
        --  Top-1 reranked result is injected. This gives precision on top
        --  of recall — the key to high-quality memory injection.
        declare
            Search_Cap  : constant := 10;  -- Fetch top-10 candidates
            Lit_Results : Database_Manager.Chunk_Array (1 .. Search_Cap);
            Lit_Count   : Natural;
            Int_Results : Database_Manager.Chunk_Array (1 .. Search_Cap);
            Int_Count   : Natural;
            Uptime_Str  : constant String :=
               Ada.Strings.Fixed.Trim
                  (Duration'Image
                      (Ada.Real_Time.To_Duration
                          (Ada.Real_Time.Clock - Init_Start_Time)),
                   Ada.Strings.Both);
            Got_Memory  : Boolean := False;
        begin
            --  1. Interaction memory: search top-10, rerank, inject top-1.
            Database_Manager.Search_Interaction
               (Emb_Vec (1 .. Emb_Len), Int_Results, Int_Count);
            if Int_Count > 0 then
                --  [RERANKER] Rerank candidates by semantic relevance
                declare
                    Best_Idx   : Natural := 1;
                    Best_Score : Float := -1.0e9;
                    Rerank_Ready : Boolean;
                begin
                    Reranker.Initialize (Rerank_Ready);
                    if Rerank_Ready and Int_Count > 1 then
                        --  Build closure to access Int_Results by index
                        declare
                            function Get_Doc (Idx : Natural) return String is
                            begin
                                return To_String (Int_Results (Idx).Content);
                            end Get_Doc;
                        begin
                            Reranker.Rerank_Scores
                              (                               Query        => Prompt,
                               Doc_Contents => Get_Doc'Access,
                               N_Docs       => Int_Count,
                               Top_K        => 1,
                               Best_Idx     => Best_Idx,
                               Best_Score   => Best_Score);
                        end;
                    else
                        Best_Idx := 1;  -- Fallback to top-1 by cosine
                    end if;

                    Got_Memory := True;
                    Append (Whimsical_Adelaide,
                            ASCII.LF
                            & ASCII.LF
                            & "<memory_interaction>"
                            & ASCII.LF
                            & Sanitize_Memory_Content
                                 (To_String (Int_Results (Best_Idx).Content))
                            & ASCII.LF
                            & "</memory_interaction>");
                    Put_Line
                       (AnsiAda.Foreground (AnsiAda.Light_Green)
                        & "[Memory]"
                        & AnsiAda.Reset
                        & " Injected interaction memory (reranked #" &
                        Natural'Image (Best_Idx) & ") into system prompt [+"
                        & Uptime_Str & "s].");
                end;
                if not External_Agent then
                    Push_Orchestration_Through_Parser
                       (Stream, Session_ID, Orch_Parser,
                        "[Adelaide Core]: [Thought] Interaction memory injected "
                        & "into system prompt [+" & Uptime_Str & "s]."
                        & ASCII.LF);
                end if;
            end if;

            --  2. Literature memory: search top-10, rerank, inject top-1.
            Database_Manager.Search_Literature
               (Emb_Vec (1 .. Emb_Len), Lit_Results, Lit_Count);
            if Lit_Count > 0 then
                --  [RERANKER] Rerank literature candidates
                declare
                    Best_Idx   : Natural := 1;
                    Best_Score : Float := -1.0e9;
                    Rerank_Ready : Boolean;
                begin
                    Reranker.Initialize (Rerank_Ready);
                    if Rerank_Ready and Lit_Count > 1 then
                        declare
                            function Get_Doc (Idx : Natural) return String is
                            begin
                                return To_String (Lit_Results (Idx).Content);
                            end Get_Doc;
                        begin
                            Reranker.Rerank_Scores
                              (                               Query        => Prompt,
                               Doc_Contents => Get_Doc'Access,
                               N_Docs       => Lit_Count,
                               Top_K        => 1,
                               Best_Idx     => Best_Idx,
                               Best_Score   => Best_Score);
                        end;
                    else
                        Best_Idx := 1;
                    end if;

                    Got_Memory := True;
                    Append (Whimsical_Adelaide,
                            ASCII.LF
                            & ASCII.LF
                            & "<memory_literature>"
                            & ASCII.LF
                            & Sanitize_Memory_Content
                                 (To_String (Lit_Results (Best_Idx).Content))
                            & ASCII.LF
                            & "</memory_literature>");
                    Put_Line
                       (AnsiAda.Foreground (AnsiAda.Light_Green)
                        & "[Memory]"
                        & AnsiAda.Reset
                        & " Injected literature memory (reranked #" &
                        Natural'Image (Best_Idx) & ") into system prompt [+"
                        & Uptime_Str & "s].");
                end;
                if not External_Agent then
                    Push_Orchestration_Through_Parser
                       (Stream, Session_ID, Orch_Parser,
                        "[Adelaide Core]: [Thought] Literature memory injected "
                        & "into system prompt [+" & Uptime_Str & "s]."
                        & ASCII.LF);
                end if;
            end if;

            if not Got_Memory then
                Put_Line
                   (AnsiAda.Foreground (AnsiAda.Grey)
                    & "[Memory]"
                    & AnsiAda.Reset
                    & " No relevant memories found above threshold -- "
                    & "system prompt unchanged.");
            end if;
        end;

        --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
        --  ELP0 Speculation Context: QRNN LSH-based retrieval for background thought.
        --  Only activates during ELP0 to inject <SpeculationContextGuidance_*> blocks
        --  after <memory_*> blocks. Uses 10-bit LSH hash (tolerance=2 Hamming distance)
        --  via Python sidecar subprocess for quantum-evolved QRNN hash quality.
        if Level = ELP0 then
            declare
                LSH_Uptime  : constant String :=
                   Ada.Strings.Fixed.Trim
                      (Duration'Image
                          (Ada.Real_Time.To_Duration
                              (Ada.Real_Time.Clock - Init_Start_Time)),
                       Ada.Strings.Both);
                LSH_Acq_OK  : Boolean;
                LSH_Hash_Value : Integer;
                Spec_Int_Results : Database_Manager.Chunk_Array (1 .. 5);
                Spec_Lit_Results : Database_Manager.Chunk_Array (1 .. 5);
                Spec_Int_Count   : Natural := 0;
                Spec_Lit_Count   : Natural := 0;
                Spec_Tolerance   : constant Integer := 2;
            begin
                --  Acquire ELP0 gate for QRNN LSH computation (atomic ELP0)
                Priority_Model_Gate.Acquire_ELP0 (LSH_QRNN) (LSH_Acq_OK);
                if not LSH_Acq_OK then
                    --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
                    Put_Line
                       (AnsiAda.Foreground (AnsiAda.Red)
                        & "[LSH]"
                        & AnsiAda.Reset
                        & " QRNN worker: ELP0 acquire FAILED (Preempted) [+"
                        & LSH_Uptime & "s].");
                else
                    --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
                    Put_Line
                       (AnsiAda.Foreground (AnsiAda.Light_Blue)
                        & "[LSH]"
                        & AnsiAda.Reset
                        & " QRNN worker: ELP0 acquired. Computing hash [+"
                        & LSH_Uptime & "s].");

                    --  Compute 10-bit LSH hash from embedding via Python sidecar
                    LSH_Hash_Value := LSH_Hash.Compute
                        (Emb_Vec (1 .. Emb_Len), Emb_Len);

                    if LSH_Hash_Value >= 0 then
                        --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
                        Put_Line
                           (AnsiAda.Foreground (AnsiAda.Light_Blue)
                            & "[LSH]"
                            & AnsiAda.Reset
                            & " Hash=" & Integer'Image (LSH_Hash_Value)
                            & " Searching speculation context [+"
                            & LSH_Uptime & "s].");

                        --  Search interaction cache by LSH (tolerance=2 Hamming)
                        Database_Manager.Search_Interaction_By_LSH
                            (LSH_Hash_Value, Spec_Tolerance,
                             Spec_Int_Results, Spec_Int_Count);

                        --  Search literature chunks by LSH
                        Database_Manager.Search_Literature_By_LSH
                            (LSH_Hash_Value, Spec_Tolerance,
                             Spec_Lit_Results, Spec_Lit_Count);

                        --  Inject <SpeculationContextGuidance_Interaction>
                        if Spec_Int_Count > 0 then
                            for S in 1 .. Spec_Int_Count loop
                                Append (Whimsical_Adelaide,
                                        ASCII.LF & ASCII.LF
                                        & "<SpeculationContextGuidance_Interaction>"
                                        & ASCII.LF
                                        & Sanitize_Memory_Content
                                             (To_String
                                                 (Spec_Int_Results (S).Content))
                                        & ASCII.LF
                                        & "</SpeculationContextGuidance_Interaction>");
                            end loop;
                            --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
                            Put_Line
                               (AnsiAda.Foreground (AnsiAda.Light_Green)
                                & "[LSH]"
                                & AnsiAda.Reset
                                & " Injected" & Natural'Image (Spec_Int_Count)
                                & " speculation interaction(s) [+"
                                & LSH_Uptime & "s].");
                        end if;

                        --  Inject <SpeculationContextGuidance_Literature>
                        if Spec_Lit_Count > 0 then
                            for S in 1 .. Spec_Lit_Count loop
                                Append (Whimsical_Adelaide,
                                        ASCII.LF & ASCII.LF
                                        & "<SpeculationContextGuidance_Literature>"
                                        & ASCII.LF
                                        & Sanitize_Memory_Content
                                             (To_String
                                                 (Spec_Lit_Results (S).Content))
                                        & ASCII.LF
                                        & "</SpeculationContextGuidance_Literature>");
                            end loop;
                            --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
                            Put_Line
                               (AnsiAda.Foreground (AnsiAda.Light_Green)
                                & "[LSH]"
                                & AnsiAda.Reset
                                & " Injected" & Natural'Image (Spec_Lit_Count)
                                & " speculation literature(s) [+"
                                & LSH_Uptime & "s].");
                        end if;

                        if Spec_Int_Count = 0 and Spec_Lit_Count = 0 then
                            --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
                            Put_Line
                               (AnsiAda.Foreground (AnsiAda.Grey)
                                & "[LSH]"
                                & AnsiAda.Reset
                                & " No speculation context found within tolerance. [+"
                                & LSH_Uptime & "s].");
                        end if;
                    else
                        --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
                        Put_Line
                           (AnsiAda.Foreground (AnsiAda.Red)
                            & "[LSH]"
                            & AnsiAda.Reset
                            & " QRNN worker failed (returned -1) [+"
                            & LSH_Uptime & "s].");
                    end if;

                    --  Release ELP0 gate
                    Priority_Model_Gate.Release_ELP0 (LSH_QRNN);
                    --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
                    Put_Line
                       (AnsiAda.Foreground (AnsiAda.Light_Blue)
                        & "[LSH]"
                        & AnsiAda.Reset
                        & " QRNN worker: ELP0 released [+"
                        & LSH_Uptime & "s].");
                end if;
            end;
        end if;

        --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
        Put_Line
           (AnsiAda.Foreground (AnsiAda.Light_Blue)
            & "[Init-V]"
            & AnsiAda.Reset
            & " Hybrid_Generate: Starting reasoning chain loop.");

        Put_Line
           (AnsiAda.Foreground (AnsiAda.Light_Magenta)
            & "[Hybrid]"
            & AnsiAda.Reset
            & " Starting reasoning chain...");

        --  1. Factual checking
        Put_Line (" [Hybrid] Checking for factual context...");
        if not Agentic
           and then
              (Index (Prompt, "What is") > 0
               or else Index (Prompt, "Who is") > 0
               or else Index (Prompt, "tell me about") > 0)
        then
            Put_Line (" [Hybrid] Factual context trigger matched.");
            if not External_Agent then
                Push_Orchestration_Through_Parser
                   (Stream,
                    Session_ID,
                    Orch_Parser,
                    "[Adelaide Core]: [Thought] Let me analyze this query "
                    & "for factual context..."
                    & ASCII.LF);
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
                        Raw_Q :=
                           To_Unbounded_String
                              (Trim
                                  (Prompt (S_Idx .. E_Idx - 1),
                                   Ada.Strings.Both));
                    else
                        Raw_Q :=
                           To_Unbounded_String
                              (Trim
                                  (Prompt (S_Idx .. Prompt'Last),
                                   Ada.Strings.Both));
                    end if;
                else
                    Raw_Q :=
                       To_Unbounded_String (Trim (Prompt, Ada.Strings.Both));
                end if;

                declare
                    Actual_Prompt : constant String :=
                       "Generate ONLY a concise 2-4 keyword search query "
                       & "for the following request: """
                       & To_String (Raw_Q)
                       & """. NO EXPLANATIONS. NO QUOTES. JUST KEYWORDS.";
                    Now           : Ada.Calendar.Time;
                begin
                    Now := Ada.Calendar.Clock;
                    if not External_Agent
                       and then Stream /= null
                       and then (Now - Last_Heartbeat) > 2.0
                    then
                        Push_Orchestration_Through_Parser
                           (Stream,
                            Session_ID,
                            Orch_Parser,
                            "[Adelaide Core]: [Thought] I'm still here and "
                            & "processing..."
                            & ASCII.LF);
                        Last_Heartbeat := Now;
                    end if;
                    Model_Manager.Generate
                       (Kind            => Snowball_Enaga_Orchestrator,
                        Prompt          => Actual_Prompt,
                        Result          => Gen_Q,
                        Stream          => null,
                        Level           => Level,
                        Virtual_Tokens  => Cached_Virtual_Tokens,
                        Virtual_Tok_Len => Cached_Virtual_Len,
                        FreeParallelMemory   => True,
                        Skip_Gate       => False);
                end;

                declare
                    Final_Q : constant String :=
                       Sanitize_Think_Tags
                          (if Length (Gen_Q) > 0
                              and then To_String (Gen_Q) /= "ERROR: Preempted"
                           then To_String (Gen_Q)
                           else To_String (Raw_Q));
                    R       : constant Tool_Manager.Tool_Result :=
                       Tool_Manager.Execute_Tool ("searchglobalref", Final_Q);
                begin
                    Push_Orchestration_Through_Parser
                       (Stream,
                        Session_ID,
                        Orch_Parser,
                        "[Adelaide Core]: [Thought] Searching knowledge "
                        & "base for: "
                        & Trim (Final_Q, Ada.Strings.Both)
                        & "..."
                        & ASCII.LF);
                    Push_Orchestration_Through_Parser
                       (Stream,
                        Session_ID,
                        Orch_Parser,
                        "[Adelaide Core]: [Thought] Found relevant context "
                        & "from knowledge base."
                        & ASCII.LF);
                    Append
                       (Internal_State,
                        "[FACTUAL_DATA]: " & To_String (R.Output) & ASCII.LF);
                    Current_Internal_State_Len := Length (Internal_State);
                    --  Re-cache virtual ctx tokens after Internal_State grew
                    Tokenize_And_Cache_Virtual_Ctx
                       (Model_Types.Snowball_Enaga_Orchestrator,
                        "Fact-Check: "
                        & Strip_Base64_Images (To_String (Internal_State)),
                        Level);
                    if not External_Agent then
                        Push_Orchestration_Through_Parser
                           (Stream,
                            Session_ID,
                            Orch_Parser,
                            "[FACTUAL_DATA]: "
                            & Sanitize_Orchestration_Output
                                 (To_String (R.Output))
                            & ASCII.LF);
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
                 Router_Sys   : constant String :=
                    "You are the Router. You decide if a tool is needed. "
                    & "If the user says hello or greets you, output [FINISH]. "
                    & "If you need to search, use [ACTION: search(query)]. "
                    & "If you need to read a file, use [ACTION: cat(filename)]. "
                    & "If you need to calculate math, use [ACTION: math(expr)]. "
                    & "If you need to execute code, use [ACTION: code(python)]. "
                    & "If you want to schedule a proactive thought for later, "
                    & "use [ACTION: schedule(seconds, query)]. "
                    & "If you need to generate an image from your imagination, "
                    & "use [ACTION: imagine(description)]. "
                    & "If you are done, output [FINISH]. "
                    & "Output ONLY the tag.";
                --  Strip base64 images from router context to prevent tokenization
                --  failure. The 9B router cannot handle massive base64 blobs.
                --  User stream still receives full output with images.
                Paging_Instr : constant String :=
                   "Current Data: "
                   & Strip_Base64_Images (To_String (Internal_State));
                Step_Raw     : Unbounded_String;

                function Get_Router_Prompt return String is
                begin
                    if Raw_Prompt then
                        declare
                            Sub_Str : constant String :=
                               "<|im_start|>assistant" & ASCII.LF;
                            Idx     : constant Natural :=
                               Index
                                  (Prompt,
                                   Sub_Str,
                                   Going => Ada.Strings.Backward);
                        begin
                            if Idx > 0 then
                                return
                                   Prompt (Prompt'First .. Idx - 1)
                                   & "System Override: "
                                   & Router_Sys
                                   & ASCII.LF
                                   & Paging_Instr
                                   & ASCII.LF
                                   & Sub_Str;
                            else
                                return
                                   Prompt
                                   & ASCII.LF
                                   & "System Override: "
                                   & Router_Sys
                                   & ASCII.LF
                                   & Paging_Instr
                                   & ASCII.LF
                                   & Sub_Str;
                            end if;
                        end;
                    else
                        return
                           Wrap_ChatML
                              (Router_Sys, Paging_Instr & ASCII.LF & Prompt);
                    end if;
                end Get_Router_Prompt;
            begin
                if not External_Agent then
                    Push_Orchestration_Through_Parser
                       (Stream,
                        Session_ID,
                        Orch_Parser,
                        "[Adelaide Core]: [Thought] Deciding next action (Hop"
                        & Current_Hop'Img
                        & ")..."
                        & ASCII.LF);
                end if;
                --  Heartbeat check before blocking Generate call
                declare
                    H_Now : constant Ada.Calendar.Time := Ada.Calendar.Clock;
                begin
                    if not External_Agent
                       and then Stream /= null
                       and then (H_Now - Last_Heartbeat) > 2.0
                    then
                        Push_Orchestration_Through_Parser
                           (Stream,
                            Session_ID,
                            Orch_Parser,
                            "[Adelaide Core]: [Thought] I'm still here and "
                            & "processing..."
                            & ASCII.LF);
                        Last_Heartbeat := H_Now;
                    end if;
                end;
                Put_Line
                   (" [Hybrid] Hop"
                    & Current_Hop'Img
                    & ": Decision routing...");
                --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
                Put_Line
                   (AnsiAda.Foreground (AnsiAda.Light_Blue)
                    & "[Init-V]"
                    & AnsiAda.Reset
                    & " Hybrid_Generate: Hop"
                    & Current_Hop'Img
                    & " calling Generate for router...");
                Generate
                   (Snowball_Enaga_Orchestrator,
                    Get_Router_Prompt,
                    Step_Raw,
                    GNATCOLL.JSON.Empty_Array,
                    Session_ID,
                    8192,
                    null,
                    False,
                    Level,
                    Cached_Virtual_Tokens,
                    Cached_Virtual_Len,
                    FreeParallelMemory => True,
                    Skip_Gate     => False);
                --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
                Put_Line
                   (AnsiAda.Foreground (AnsiAda.Light_Blue)
                    & "[Init-V]"
                    & AnsiAda.Reset
                    & " Hybrid_Generate: Hop"
                    & Current_Hop'Img
                    & " Generate returned (model released). Len="
                    & Natural'Image (Length (Step_Raw)));

                declare
                    Step : constant String :=
                       Trim (To_String (Step_Raw), Ada.Strings.Both);
                begin
                    Put_Line (" [Hybrid] Hop" & Current_Hop'Img & ": " & Step);
                    if not External_Agent then
                        Push_Orchestration_Direct
                           (Stream,
                            Session_ID,
                            "[Adelaide Core]: [Thought] Hop "
                            & Current_Hop'Img & " - I will: "
                            & Sanitize_Orchestration_Output (Step)
                            & ASCII.LF);
                    end if;

                    if Index (Step, "[ACTION:") > 0 then
                        declare
                            S_Pos : constant Natural :=
                               Index (Step, "[ACTION:") + 8;
                            E_Pos : constant Natural :=
                               Index (Step, "]", S_Pos);
                        begin
                            if E_Pos > S_Pos then
                                declare
                                    A_Full : constant String :=
                                       Step (S_Pos .. E_Pos - 1);
                                    P_Pos  : constant Natural :=
                                       Index (A_Full, "(");
                                    EP_Pos : constant Natural :=
                                       (if P_Pos > 0
                                        then Index (A_Full, ")", P_Pos)
                                        else 0);
                                begin
                                    if P_Pos > 0 and then EP_Pos > P_Pos then
                                        declare
                                            T_Name : constant String :=
                                               Trim
                                                  (A_Full
                                                      (A_Full'First
                                                       .. P_Pos - 1),
                                                   Ada.Strings.Both);
                                            T_Pars : constant String :=
                                               Trim
                                                  (A_Full
                                                      (P_Pos + 1
                                                       .. EP_Pos - 1),
                                                   Ada.Strings.Both);
                                        begin
                                            if T_Name = "schedule" then
                                                --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
                                                Put_Line
                                                   (AnsiAda.Foreground
                                                       (AnsiAda.Light_Blue)
                                                    & "[Init-V]"
                                                    & AnsiAda.Reset
                                                    & " Hybrid_Generate: Tool=schedule, Params="
                                                    & T_Pars);
                                                declare
                                                    Comma_Idx :
                                                       constant Natural :=
                                                          Index (T_Pars, ",");
                                                begin
                                                    if Comma_Idx > 0 then
                                                        declare
                                                            Time_Str   :
                                                               constant String :=
                                                                  Trim
                                                                     (T_Pars
                                                                         (T_Pars'First
                                                                          ..
                                                                             Comma_Idx
                                                                             - 1),
                                                                      Ada
                                                                         .Strings
                                                                         .Both);
                                                            Prompt_Str :
                                                               constant String :=
                                                                  Trim
                                                                     (T_Pars
                                                                         (Comma_Idx
                                                                          + 1
                                                                          ..
                                                                             T_Pars'Last),
                                                                      Ada
                                                                         .Strings
                                                                         .Both);
                                                            Delay_Secs :
                                                               Integer;
                                                        begin
                                                            Delay_Secs :=
                                                               Integer'Value
                                                                  (Time_Str);
                                                            Scheduler_Manager
                                                               .Schedule
                                                                  (Delay_Secs,
                                                                   Prompt_Str);
                                                            Append
                                                               (Internal_State,
                                                                "[SCHEDULED]: "
                                                                & Prompt_Str
                                                                & ASCII.LF);
                                                            Current_Internal_State_Len :=
                                                               Length
                                                                  (Internal_State);
                                                            --  Re-cache virtual ctx tokens after Internal_State grew
                                                            Tokenize_And_Cache_Virtual_Ctx
                                                                  (Model_Types
                                                                      .Snowball_Enaga_Orchestrator,
                                                                   "Fact-Check: "
                                                                   & Strip_Base64_Images
                                                                           (To_String
                                                                                  (Internal_State)),
                                                                   Level);
                                                        exception
                                                            when others =>
                                                                null;
                                                        end;
                                                    end if;
                                                end;
                                            elsif T_Pars'Length < 256
                                               and then
                                                  Index
                                                     (To_String
                                                         (Internal_State),
                                                      T_Name
                                                      & "("
                                                      & T_Pars
                                                      & ")")
                                                  = 0
                                            then
                                                --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
                                                Put_Line
                                                   (AnsiAda.Foreground
                                                       (AnsiAda.Light_Blue)
                                                    & "[Init-V]"
                                                    & AnsiAda.Reset
                                                    & " Hybrid_Generate: Executing tool="
                                                    & T_Name
                                                    & " params="
                                                    & T_Pars);
                                                if Agentic then
                                                    Result :=
                                                       To_Unbounded_String
                                                          ("[TOOL_CALL: "
                                                           & T_Name
                                                           & "("
                                                           & T_Pars
                                                           & ")]");
                                                    return;
                                                end if;
                                                --  Heartbeat check
                                                declare
                                                    H_Now :
                                                       constant Ada
                                                                   .Calendar
                                                                   .Time :=
                                                          Ada.Calendar.Clock;
                                                begin
                                                    if not External_Agent
                                                       and then Stream /= null
                                                       and then
                                                          (H_Now
                                                           - Last_Heartbeat)
                                                          > 2.0
                                                    then
                                                        Push_Orchestration_Through_Parser
                                                              (Stream,
                                                               Session_ID,
                                                               Orch_Parser,
                                                               "[Adelaide Core]: [Thought] I'm "
                                                               & "still here and processing..."
                                                               & ASCII.LF);
                                                        Last_Heartbeat :=
                                                           H_Now;
                                                    end if;
                                                end;
                                                declare
                                                    R :
                                                       constant Tool_Manager
                                                                   .Tool_Result :=
                                                          --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
                                                          --  IMAGINE TOOL: Direct Ada call to SD_Manager.
                                                          --  When the model outputs [ACTION: imagine(prompt)],
                                                          --  generate an image via two-stage FLUX+SD pipeline
                                                          --  and store it in the database for VLM retrieval.
                                                          (if T_Name = "imagine" then
                                                             Tool_Manager.Execute_Imagine_Tool
                                                               (Sanitize_Think_Tags (T_Pars))
                                                          else
                                                             Tool_Manager
                                                                .Execute_Tool
                                                                   (T_Name,
                                                                    Sanitize_Think_Tags
                                                                          (T_Pars)));
                                                begin
                                                    if not External_Agent then
                                                        Push_Orchestration_Direct
                                                              (Stream,
                                                               Session_ID,
                                                               "[Adelaide Core]: [Thought] Hop "
                                                               & Current_Hop'Img
                                                               & " - Running tool: "
                                                               & Sanitize_Orchestration_Output
                                                                       (T_Name)
                                                               & ASCII.LF);
                                                    end if;
                                                    Append
                                                       (Internal_State,
                                                        "[TOOL ("
                                                        & T_Name
                                                        & ")]: "
                                                        & To_String (R.Output)
                                                        & ASCII.LF);
                                                    Current_Internal_State_Len :=
                                                       Length (Internal_State);
                                                    --  Re-cache virtual ctx tokens after Internal_State grew
                                                    Tokenize_And_Cache_Virtual_Ctx
                                                          (Model_Types.Snowball_Enaga_Orchestrator,
                                                           "Fact-Check: "
                                                           & Strip_Base64_Images
                                                                   (To_String
                                                                       (Internal_State)),
                                                           Level);
                                                    if not External_Agent then
                                                        Push_Orchestration_Direct
                                                              (Stream,
                                                               Session_ID,
                                                               ASCII.LF
                                                               & "[Adelaide Core]: [Thought] Hop "
                                                               & Current_Hop'Img
                                                               & " - Tool result ("
                                                               & T_Name
                                                               & "): "
                                                               & Sanitize_Orchestration_Output
                                                                       (To_String
                                                                              (R
                                                                                  .Output))
                                                               & ASCII.LF);
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
            Push_Orchestration_Direct
               (Stream,
                Session_ID,
                "[Adelaide Core]: [Thought] Reasoning complete after "
                & Current_Hop'Img
                & " hops."
                & ASCII.LF);
        end if;

        declare
            function Get_Final_Prompt return String is
                Sys_Tag  : constant String := "<|im_start|>system" & ASCII.LF;
                Asst_Tag : constant String :=
                   "<|im_start|>assistant" & ASCII.LF;
            begin
                if External_Agent then
                    return Prompt;
                elsif Raw_Prompt then
                    declare
                        Sys_Idx     : constant Natural :=
                           Index (Prompt, Sys_Tag);
                        User_Idx    : constant Natural :=
                           Index (Prompt, "<|im_start|>user");
                        First_Block : constant Natural :=
                           (if User_Idx > 0
                               and then
                                  (Sys_Idx = 0 or else User_Idx < Sys_Idx)
                            then User_Idx
                            elsif Sys_Idx > 0
                            then Sys_Idx
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
                                    return
                                       Prefix
                                       & Sys_Tag
                                       & To_String (Whimsical_Adelaide)
                                       & ASCII.LF
                                       & "Fact-Check: "
                                       & Strip_Base64_Images
                                            (To_String (Internal_State))
                                       & ASCII.LF
                                       & Prompt (First_Block .. Prompt'Last);
                                else
                                    return
                                       Prefix
                                       & Sys_Tag
                                       & To_String (Whimsical_Adelaide)
                                       & ASCII.LF
                                       & Prompt (First_Block .. Prompt'Last);
                                end if;
                            end;
                        elsif First_Block = 1 then
                            if Length (Internal_State) > 0 then
                                return
                                   Sys_Tag
                                   & To_String (Whimsical_Adelaide)
                                   & ASCII.LF
                                   & "Fact-Check: "
                                   & Strip_Base64_Images
                                        (To_String (Internal_State))
                                   & ASCII.LF
                                   & Prompt;
                            else
                                return
                                   Sys_Tag
                                   & To_String (Whimsical_Adelaide)
                                   & ASCII.LF
                                   & Prompt;
                            end if;
                        else
                            if Length (Internal_State) > 0 then
                                return
                                   Wrap_ChatML
                                      (To_String (Whimsical_Adelaide),
                                       Prompt
                                       & ASCII.LF
                                       & "Fact-Check: "
                                       & Strip_Base64_Images
                                            (To_String (Internal_State)));
                            else
                                return
                                   Wrap_ChatML (To_String (Whimsical_Adelaide), Prompt);
                            end if;
                        end if;
                    end;
                else
                    if Length (Internal_State) > 0 then
                        return
                           Wrap_ChatML
                              (To_String (Whimsical_Adelaide),
                               "User: "
                               & Prompt
                               & ASCII.LF
                               & "Fact-Check: "
                               & Strip_Base64_Images
                                    (To_String (Internal_State)));
                    else
                        return Wrap_ChatML (To_String (Whimsical_Adelaide), Prompt);
                    end if;
                end if;
            end Get_Final_Prompt;
        begin
            --  [VITAL-DO-NOT-REMOVE] Mandated by user.
            Put_Line
               (AnsiAda.Foreground (AnsiAda.Light_Blue)
                & "[Init-V]"
                & AnsiAda.Reset
                & " Hybrid_Generate: Building final prompt. Len="
                & Natural'Image (Get_Final_Prompt'Length));
            --  CONTEXT FAULTING LOOP
            declare
                F_Detected   : Boolean := False;
                F_Query      : Unbounded_String;
                F_Category   : Unbounded_String;
                Hop_Count    : Natural := 0;
                Fault_Result : Unbounded_String;
            begin
                --  [VITAL-DO-NOT-REMOVE] Mandated by user.
                Put_Line
                   (AnsiAda.Foreground (AnsiAda.Light_Blue)
                    & "[Init-V]"
                    & AnsiAda.Reset
                    & " Hybrid_Generate: CONTEXT_FAULT_LOOP ENTERED.");
                loop
                    exit when Hop_Count >= 5;

                    --  Reset fault detection state for this hop. Without this,
                    --  a fault detected on a previous hop would persist and
                    --  cause false context-fault handling on subsequent hops
                    --  even when the model didn't request one.
                    F_Detected := False;

                    if not External_Agent then
                        if Hop_Count = 0 then
                            Push_Orchestration_Through_Parser
                               (Stream,
                                Session_ID,
                                Orch_Parser,
                                "[Adelaide Core]: [Thought] Starting reasoning "
                                & "chain..."
                                & ASCII.LF);
                        else
                            Push_Orchestration_Through_Parser
                               (Stream,
                                Session_ID,
                                Orch_Parser,
                                "[Adelaide Core]: [Thought] Continuing reasoning "
                                & "(hop"
                                & Natural'Image (Hop_Count + 1)
                                & ")..."
                                & ASCII.LF);
                        end if;
                    end if;

                    --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
                    Put_Line
                       (AnsiAda.Foreground (AnsiAda.Light_Blue)
                        & "[Init-V]"
                        & AnsiAda.Reset
                        & " Hybrid_Generate: Final generation. Hop="
                        & Natural'Image (Hop_Count)
                        & " Len="
                        & Natural'Image (Get_Final_Prompt'Length));
                    Generate
                       (Kind            => Snowball_Enaga_Orchestrator,
                        Prompt          => Get_Final_Prompt,
                        Result          => Fault_Result,
                        Images          => Images,
                        Session_ID      => Session_ID,
                        Requested_Ctx   => 8192,
                        Stream          => Stream,
                        Orch_Think_Open => (Hop_Count = 0),
                        Level           => Level,
                        Virtual_Tokens  => Cached_Virtual_Tokens,
                        Virtual_Tok_Len => Cached_Virtual_Len,
                        FreeParallelMemory   => True,
                        Skip_Gate       => False);

                    --  =================================================================
                    --  THINK-ONLY RETRY: If model produced only <think>...</think>
                    --  with no visible content, retry with randomized seed.
                    --  Max 2 retries. Stream=null on retries to avoid duplicate output.
                    --  Blacklist seeds that produce think-only responses.
                    --  =================================================================
                    declare
                        Max_Think_Retries : constant := 2;
                        Retry_Count       : Natural := 0;
                        Sanitized_Check   : String :=
                           Sanitize_Think_Tags (To_String (Fault_Result));
                    begin
                        --  Blacklist the initial seed if it produced think-only
                        if Sanitized_Check = "" then
                            Database_Manager.Blacklist_Seed
                               (Natural (Generate_Seed));
                        end if;

                        while Sanitized_Check = "" and then
                              Retry_Count < Max_Think_Retries
                        loop
                            Retry_Count := Retry_Count + 1;

                            --  [VITAL-DO-NOT-REMOVE] Find next non-blacklisted seed.
                            --  Skip blacklisted seeds automatically.
                            loop
                                Generate_Seed := Generate_Seed + 1;
                                exit when not Database_Manager.Is_Seed_Blacklisted
                                   (Natural (Generate_Seed));
                            end loop;

                            Put_Line
                               (AnsiAda.Foreground (AnsiAda.Yellow)
                                & "[Init-V]"
                                & AnsiAda.Reset
                                & " Hybrid_Generate: THINK-ONLY DETECTED. Retry "
                                & Natural'Image (Retry_Count) & "/"
                                & Natural'Image (Max_Think_Retries)
                                & " with seed="
                                & Interfaces.C.unsigned'Image (Generate_Seed));

                            --  Retry without streaming (avoids duplicate tokens to client)
                            Generate
                               (Kind            => Snowball_Enaga_Orchestrator,
                                Prompt          => Get_Final_Prompt,
                                Result          => Fault_Result,
                                Images          => Images,
                                Session_ID      => Session_ID,
                                Requested_Ctx   => 8192,
                                Stream          => null,
                                Orch_Think_Open => False,
                                Level           => Level,
                                Virtual_Tokens  => Cached_Virtual_Tokens,
                                Virtual_Tok_Len => Cached_Virtual_Len,
                                FreeParallelMemory   => True,
                                Skip_Gate       => False);

                            --  Check sanitized result
                            Sanitized_Check :=
                               Sanitize_Think_Tags (To_String (Fault_Result));

                            if Sanitized_Check /= "" then
                                --  Retry produced visible content — stream it to client
                                Put_Line
                                   (AnsiAda.Foreground (AnsiAda.Green)
                                    & "[Init-V]"
                                    & AnsiAda.Reset
                                    & " Hybrid_Generate: RETRY SUCCEEDED. Len="
                                    & Natural'Image (Length (Fault_Result)));
                                if Stream /= null then
                                    Push_Chunk
                                       (Stream, Session_ID,
                                        To_String (Fault_Result));
                                end if;
                                exit;
                            else
                                --  This seed also produced think-only — blacklist it
                                Database_Manager.Blacklist_Seed
                                   (Natural (Generate_Seed));
                            end if;
                        end loop;
                    end;

                    --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
                    Put_Line
                       (AnsiAda.Foreground (AnsiAda.Light_Blue)
                        & "[Init-V]"
                        & AnsiAda.Reset
                        & " Hybrid_Generate: Final Generate returned. Len="
                        & Natural'Image (Length (Fault_Result)));

                    --  Parse Fault_Result for CONTEXT_FAULT marker.
                    --  [VITAL-DO-NOT-REMOVE] Mandated by user.
                    --  When the model outputs [CONTEXT_FAULT:query=X category=Y] inside
                    --  <think>, Generate's Parser detects it but cannot communicate it
                    --  back to Hybrid_Generate (Parser is local to Generate).  However,
                    --  all tokens (including the fault marker) are appended to Result
                    --  before streaming.  So Fault_Result contains the raw marker text.
                    --  Parse it here to set F_Detected, F_Query, F_Category.
                    declare
                        Raw_Result : constant String :=
                           To_String (Fault_Result);
                        F_Mark     : constant String := "[CONTEXT_FAULT:";
                        F_Mark_Pos : constant Natural :=
                           Index (Raw_Result, F_Mark);
                    begin
                        if F_Mark_Pos > 0 then
                            declare
                                Close_Pos : constant Natural :=
                                   Index
                                      (Raw_Result
                                          (F_Mark_Pos .. Raw_Result'Last),
                                       "]");
                            begin
                                if Close_Pos > 0 then
                                    declare
                                        --  Close_Pos is absolute (Ada.Strings.Index
                                        --  returns index within Source bounds), so
                                        --  use it directly, not F_Mark_Pos + Close_Pos.
                                        Inner     : constant String :=
                                           Raw_Result
                                              (F_Mark_Pos + F_Mark'Length
                                               .. Close_Pos - 1);
                                        Q_Mark    : constant String :=
                                           "query=";
                                        C_Mark    : constant String :=
                                           "category=";
                                        Query_Idx : constant Natural :=
                                           Index (Inner, Q_Mark);
                                        Cat_Idx   : constant Natural :=
                                           Index (Inner, C_Mark);
                                    begin
                                        F_Detected := True;
                                        if Query_Idx > 0 then
                                            declare
                                                Q_Start : constant Natural :=
                                                   Query_Idx + Q_Mark'Length;
                                                Q_End   : constant Natural :=
                                                   (if Cat_Idx > Query_Idx
                                                    then Cat_Idx - 1
                                                    else Inner'Last + 1);
                                            begin
                                                F_Query :=
                                                   To_Unbounded_String
                                                      (Trim
                                                          (Inner
                                                              (Q_Start
                                                               .. Q_End - 1),
                                                           Ada.Strings.Both));
                                            end;
                                        end if;
                                        if Cat_Idx > 0 then
                                            F_Category :=
                                               To_Unbounded_String
                                                  (Trim
                                                      (Inner
                                                          (Cat_Idx
                                                           + C_Mark'Length
                                                           .. Inner'Last),
                                                       Ada.Strings.Both));
                                        else
                                            F_Category :=
                                               To_Unbounded_String
                                                  ("knowledge");
                                        end if;
                                    end;
                                end if;
                            end;
                        end if;
                    end;

                    --  [VITAL-DO-NOT-REMOVE] Mandated by user.
                    Put_Line
                       (AnsiAda.Foreground (AnsiAda.Light_Blue)
                        & "[Init-V]"
                        & AnsiAda.Reset
                        & " Hybrid_Generate: F_Detected="
                        & Boolean'Image (F_Detected)
                        & " Hop_Count="
                        & Natural'Image (Hop_Count));

                    if F_Detected then
                        declare
                            Q_Str : constant String := To_String (F_Query);
                            C_Str : constant String := To_String (F_Category);
                            R     : Tool_Manager.Tool_Result;
                        begin
                            if not External_Agent then
                                Push_Orchestration_Through_Parser
                                   (Stream,
                                    Session_ID,
                                    Orch_Parser,
                                    "[Adelaide Core]: [Thought] Context fault: "
                                    & "searching "
                                    & C_Str
                                    & " for '"
                                    & Q_Str
                                    & "'..."
                                    & ASCII.LF);
                            end if;

                            --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
                            --  CONTEXT FAULT IMAGINE: When the model's <thinking>
                            --  emits [CONTEXT_FAULT:query=X category=imagine],
                            --  generate an image via the two-stage SD pipeline
                            --  and store it for VLM retrieval.
                            if C_Str = "imagine" then
                                R := Tool_Manager.Execute_Imagine_Tool (Q_Str);
                                --  Store the imagined image in the database
                                if R.Success and then Length (R.Output) > 100 then
                                    declare
                                        Img_LSH : Integer := -1;
                                    begin
                                        begin
                                            declare
                                                Emb_Vec : Math_Utils.Vector (1 .. 1024);
                                                Emb_Len : Natural;
                                             begin
                                                 Get_Embedding (Q_Str, Emb_Vec, Emb_Len);
                                                Img_LSH := LSH_Hash.Compute (Emb_Vec (1 .. Emb_Len), Emb_Len);
                                            end;
                                        exception
                                            when others => Img_LSH := -1;
                                        end;
                                        Database_Manager.Store_Imagined_Image
                                          (Prompt    => Q_Str,
                                           Image_B64 => To_String (R.Output),
                                           LSH_Hash  => Img_LSH);
                                        Put_Line
                                          (AnsiAda.Foreground (AnsiAda.Cyan) & "[CtxFault-Imagine]" &
                                           AnsiAda.Reset & " Stored imagined image. LSH=" &
                                           Integer'Image (Img_LSH));
                                    end;
                                end if;
                                Append
                                   (Internal_State,
                                    "[IMAGINED_IMAGE]: "
                                    & To_String (R.Output)
                                    & ASCII.LF);
                            elsif C_Str = "graph" then
                                R :=
                                   Tool_Manager.Execute_Tool
                                      ("searchglobalref", "graph: " & Q_Str);
                            else
                                R :=
                                   Tool_Manager.Execute_Tool
                                      ("searchglobalref", Q_Str);
                            end if;

                            Append
                               (Internal_State,
                                "[FACTUAL_DATA]: "
                                & To_String (R.Output)
                                & ASCII.LF);

                            --  Re-cache virtual ctx tokens after Internal_State grew
                            Tokenize_And_Cache_Virtual_Ctx
                               (Model_Types.Snowball_Enaga_Orchestrator,
                                "Fact-Check: "
                                & Strip_Base64_Images
                                     (To_String (Internal_State)),
                                Level);

                            if not External_Agent then
                                Push_Orchestration_Through_Parser
                                   (Stream,
                                    Session_ID,
                                    Orch_Parser,
                                    "[Adelaide Core]: [Thought] Context loaded for: "
                                    & Q_Str
                                    & ASCII.LF);
                            end if;
                        end;
                        Hop_Count := Hop_Count + 1;
                        --  Update context fault monitor tracking
                        Current_Context_Fault_Hops := Hop_Count;
                        Current_Internal_State_Len := Length (Internal_State);
                    else
                        --  [VITAL-DO-NOT-REMOVE] Mandated by user.
                        Put_Line
                           (AnsiAda.Foreground (AnsiAda.Light_Blue)
                            & "[Init-V]"
                            & AnsiAda.Reset
                            & " Hybrid_Generate: No fault detected. Exiting loop.");
                        Current_Response := Fault_Result;
                        exit;
                    end if;
                end loop;
                --  [VITAL-DO-NOT-REMOVE] Mandated by user.
                Put_Line
                   (AnsiAda.Foreground (AnsiAda.Light_Blue)
                    & "[Init-V]"
                    & AnsiAda.Reset
                    & " Hybrid_Generate: CONTEXT_FAULT_LOOP EXITED."
                    & " Hop_Count="
                    & Natural'Image (Hop_Count));
            end;
            --  SAFETY NET: If the entire response is think-only content,
            --  the model failed to produce a visible answer.  Set a fallback
            --  so the client gets something instead of an empty response.
            declare
                Sanitized : constant String :=
                   Sanitize_Think_Tags (To_String (Current_Response));
            begin
                if Sanitized = "" then
                    Put_Line
                       (AnsiAda.Foreground (AnsiAda.Yellow)
                        & "[Init-V]"
                        & AnsiAda.Reset
                        & " Hybrid_Generate: Think-only response detected."
                        & " Model produced no visible answer.");
                    Current_Response :=
                       To_Unbounded_String
                          ("I apologize, but I was unable to generate a complete"
                           & " response. The model produced only internal reasoning"
                           & " without a final answer. Please try rephrasing your"
                           & " question or providing more context.");
                end if;
            end;

            Result :=
               To_Unbounded_String
                  (Sanitize_Think_Tags (To_String (Current_Response)));
            declare
                B64_Str : Unbounded_String := To_Unbounded_String ("");
            begin
                if GNATCOLL.JSON.Length (Images) > 0 then
                    B64_Str :=
                       To_Unbounded_String
                          (String'
                              (GNATCOLL.JSON.Get
                                  (GNATCOLL.JSON.Get (Images, 1))));
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
               Index (Resp_Str, "<thinking>") > 0
               or else Index (Resp_Str, "<think>") > 0;
        begin
            --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
            Put_Line
               (AnsiAda.Foreground (AnsiAda.Light_Blue)
                & "[Init-V]"
                & AnsiAda.Reset
                & " Hybrid_Generate: COMPLETE. ResultLen="
                & Natural'Image (Length (Result))
                & " Error="
                & Boolean'Image (Is_Error)
                & " HasThink="
                & Boolean'Image (Has_Think));
            if not External_Agent and then not Is_Error and then not Has_Think
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
                Push_Orchestration_Through_Parser
                   (Stream,
                    Session_ID,
                    Orch_Parser,
                    "[Adelaide Core]: [Thought] Response generated in "
                    & Dur_Str
                    & "s."
                    & ASCII.LF);
            end;
        end if;

        if External_Agent then
            Result :=
               To_Unbounded_String
                  (Sanitize_Think_Tags (To_String (Current_Response)));
        elsif Stream = null then
            Result :=
               To_Unbounded_String
                  (Sanitize_Think_Tags (To_String (Current_Response)));
        else
            Result := Current_Response;
        end if;

        declare
            Score : constant Natural :=
               Grade_Response_Quality
                  (Response_Text => To_String (Result),
                   Prompt        => Prompt,
                   Search_Used   =>
                      Index (To_String (Internal_State), "[FACTUAL_DATA]") > 0,
                   Has_Citations =>
                      Index (To_String (Result), "[") > 0
                      and then Index (To_String (Result), "]") > 0,
                   Session_ID    => Session_ID,
                   Level         => Level);
        begin
            if not External_Agent then
                Push_Orchestration_Through_Parser
                   (Stream,
                    Session_ID,
                    Orch_Parser,
                    "[Adelaide Core]: [Thought] Self-assessment: "
                    & Score'Img
                    & "/10"
                    & ASCII.LF);
            end if;
            Ada.Text_IO.Put_Line
               (AnsiAda.Foreground (AnsiAda.Cyan)
                & "[Quality Score] "
                & AnsiAda.Reset
                & "Score: "
                & Score'Img
                & "/10 | "
                & "Session: "
                & Session_ID);
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
                    Push_Orchestration_Through_Parser
                       (Stream,
                        Session_ID,
                        Orch_Parser,
                        Model_Thinking & ASCII.LF);
                end if;
            end;
        end if;

        --  EMULATED STREAMING
        --  The model's response was already streamed token-by-token through
        --  the stream parser during Generate (Process_And_Push_Chunk). Each
        --  character outside a think block was pushed immediately to the
        --  queue via Push_Chunk. This emulated streaming section pushes
        --  ONLY the closing `</think>` tag to close the orchestration think block.
        --  The response text is NOT re-emitted here to avoid duplication.
        --
        --  The 300 tok/s simulation delay ensures the closing tag arrives
        --  after all generated chunks have been flushed by AWS.
        if not External_Agent and then Stream /= null then
            declare
                Sim_TPS    : constant Float := 300.0;
                Resp_Text  : constant String :=
                   Sanitize_Think_Tags (To_String (Current_Response));
                Resp_Len   : constant Natural := Resp_Text'Length;
                Delay_Time : constant Duration :=
                   Duration (Float (Resp_Len) / Sim_TPS);
            begin
                --  [VITAL-DO-NOT-REMOVE] Mandated by user.
                Put_Line
                   (AnsiAda.Foreground (AnsiAda.Light_Blue)
                    & "[Init-V]"
                    & AnsiAda.Reset
                    & " Hybrid_Generate: Waiting "
                    & Duration'Image (Delay_Time)
                    & "s for 300 tok/s sim.");
                delay Delay_Time;
                --  [VITAL-DO-NOT-REMOVE] Mandated by user.
                Put_Line
                   (AnsiAda.Foreground (AnsiAda.Light_Blue)
                    & "[Init-V]"
                    & AnsiAda.Reset
                    & " Hybrid_Generate: STREAMING COMPLETE.");
                --  Push statistics to think block before closing tag.
                --  These provide chunk/token stats and other metrics.
                declare
                    Resp_Len   : constant Natural := Resp_Text'Length;
                    Gen_Elapsed : constant Duration := Ada.Calendar.Clock - T0;
                    Stats_Str  : constant String :=
                      ASCII.LF & "--- ORCHESTRATION STATISTICS ---" & ASCII.LF
                      & "Response Length: " & Natural'Image (Resp_Len) & " chars" & ASCII.LF
                      & "Response Tokens (est): " & Natural'Image (Resp_Len / 4) & " tokens" & ASCII.LF
                      & "Generation Time: " & Duration'Image (Gen_Elapsed) & "s" & ASCII.LF
                      & "Prompt Tokens: " & Natural'Image (Current_Prompt_Tokens) & ASCII.LF
                      & "Context Capacity: " & Natural'Image (Current_Ctx_Capacity) & " tokens" & ASCII.LF
                      & "Context Utilization: " & Natural'Image (Current_Prompt_Tokens * 100 / Current_Ctx_Capacity) & "%" & ASCII.LF
                      & "Reasoning Hops: " & Natural'Image (Current_Hop_Count) & ASCII.LF
                      & "Context Faults: " & Natural'Image (Current_Context_Fault_Hops) & ASCII.LF
                      & "Pipeline Level: " & ELP_Level'Image (Level) & ASCII.LF
                      & "Streaming Mode: Emulated 300 tok/s" & ASCII.LF
                      & "GPU Free: " & Natural'Image (GPU_Free_MB) & "MB / "
                      & Natural'Image (GPU_Total_MB) & "MB ("
                      & Natural'Image (GPU_Layer_Percent) & "%)" & ASCII.LF
                      & "GPU Layers: "
                      & (if GPU_Layer_Count = -1 then "ALL(-1)"
                         else Integer'Image (GPU_Layer_Count) & "/" & Natural'Image (Total_Model_Layers))
                      & ASCII.LF
                      & "GPU Stable: " & Boolean'Image (GPU_Is_Stable) & ASCII.LF
                      & "--- END STATISTICS ---";
                begin
                    Push_Chunk (Stream, Session_ID, Stats_Str);
                end;
                --  Push `</think>` to close the orchestration think block.
                Push_Chunk
                   (Stream, Session_ID, ASCII.LF & "</think>" & ASCII.LF);

                --  Re-emit the final response outside the think block at 300 tok/s
                --  (approx 1200 chars/s = 120 chars per 0.1s).
                --  This is what is supposed to happen: after the think block, the response
                --  is repeated so that clients (like Msty) which hide the think block
                --  will still display the final response correctly.
                declare
                    Chunk_Size : constant Natural := 120;
                    Pos        : Positive := Resp_Text'First;
                    Last_Pos   : Natural;
                begin
                    while Pos <= Resp_Text'Last loop
                        Last_Pos :=
                           Natural'Min (Pos + Chunk_Size - 1, Resp_Text'Last);
                        Push_Chunk
                           (Stream, Session_ID, Resp_Text (Pos .. Last_Pos));
                        delay 0.1;
                        Pos := Last_Pos + 1;
                    end loop;
                end;
            end;
        elsif External_Agent and then Stream /= null then
            declare
                Resp_Text : constant String := To_String (Result);
            begin
                Ada.Text_IO.Put_Line
                   ("[External Agent] Sending final scored response ("
                    & Resp_Text'Length'Img
                    & " chars)...");
                Push_Chunk (Stream, Session_ID, Resp_Text & ASCII.LF);
            end;
        end if;

        --  [FREE-PARALLEL-MEMORY] Hybrid_Generate called with
        --  FreeParallelMemory => False during hops, but now at the end
        --  of Hybrid_Generate, we must unload the component.
        --  Wait for async save to complete, then UNLOAD from GPU.
        --  This frees VRAM for the next component (LM Studio pattern).
        --  Flow: Wait_For_Save -> Unload_Model -> release locks.
        --
        --  [CRITICAL-FIX] Skip if the last hop's Generate already called
        --  FreeParallelMemory=True, which does Wait_For_Save + Unload_Model.
        --  Calling Wait_For_Save on a terminated Save_Task raises TASKING_ERROR
        --  (s-tasren.adb:377). Check if the model is still loaded first.
        if Models (Snowball_Enaga_Orchestrator).Loaded
          and then Models (Snowball_Enaga_Orchestrator).Context /= Null_Context
        then
            --  Model still loaded — last hop did NOT free memory (FreeParallelMemory=False)
            --  Do the cleanup now.
            KV_Cache_Manager.Wait_For_Save;
            Unload_Model (Snowball_Enaga_Orchestrator);
        end if;
        Models (Snowball_Enaga_Orchestrator).In_Use := False;
        if Level = ELP0 then
            Priority_Model_Gate.Release_ELP0 (Snowball_Enaga_Orchestrator);
        else
            Priority_Model_Gate.Release_ELP1 (Snowball_Enaga_Orchestrator);
        end if;
        ELP_Queue.Dequeue_Level (Level);

    exception
         when E : Storage_Error =>
            --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
            --  Stack overflow during hybrid generation (tool exec, tokenization,
            --  or context fault paging).  Force-unload model and report cleanly.
            --  Mark Metal broken so KV save retries instead of SIGABRT.
            --  [ADAPTIVE GPU FALLBACK] OOM → reduce GPU layers for next load
            if GPU_Layer_Count = -1 then
                GPU_Layer_Count   := GPU_Layer_Fallback;
                GPU_Last_OOM_Time := Ada.Real_Time.Clock;
                --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
                Put_Line
                   (AnsiAda.Foreground (AnsiAda.Yellow)
                    & "[GPU-Adaptive]"
                    & AnsiAda.Reset
                    & " OOM in Hybrid_Generate on full GPU (-1). Falling back to"
                    & Integer'Image (GPU_Layer_Fallback)
                    & " layers. Will retry -1 in 3 minutes.");
            end if;
            begin
                Models (Snowball_Enaga_Orchestrator).In_Use := False;
                if Level = ELP0 then
                    Priority_Model_Gate.Release_ELP0 (Snowball_Enaga_Orchestrator);
                else
                    Priority_Model_Gate.Release_ELP1 (Snowball_Enaga_Orchestrator);
                end if;
                ELP_Queue.Dequeue_Level (Level);
            exception
                when others =>
                    null;
            end;
            begin
                --  [PARALLEL=1] Wait for KV save (if any) before unload
                KV_Cache_Manager.Wait_For_Save;
                Unload_Model (Snowball_Enaga_Orchestrator);
            exception
                when others =>
                    null;
            end;
            Ada.Text_IO.Put_Line
               (AnsiAda.Foreground (AnsiAda.Red)
                & "[Hybrid-FATAL]"
                & AnsiAda.Reset
                & " STORAGE_ERROR (stack overflow) in Hybrid_Generate");
            Ada.Text_IO.Put_Line
               (AnsiAda.Foreground (AnsiAda.Red)
                & "[Hybrid-FATAL]"
                & AnsiAda.Reset
                & " Exception: "
                & Ada.Exceptions.Exception_Information (E));
            --  [VITAL-DO-NOT-REMOVE] OOM banner — red, unmissable.
            Ada.Text_IO.Put_Line
               (AnsiAda.Foreground (AnsiAda.Red)
                & "=========================================================="
                & AnsiAda.Reset);
            Ada.Text_IO.Put_Line
               (AnsiAda.Foreground (AnsiAda.Red)
                & "  !!! OUT OF MEMORY !!!  (STORAGE_ERROR)"
                & AnsiAda.Reset);
            Ada.Text_IO.Put_Line
               (AnsiAda.Foreground (AnsiAda.Red)
                & "  Metal backend poisoned. KV save will RETRY."
                & AnsiAda.Reset);
            Ada.Text_IO.Put_Line
               (AnsiAda.Foreground (AnsiAda.Red)
                & "  Connection NOT dropped. Server continues."
                & AnsiAda.Reset);
            Ada.Text_IO.Put_Line
               (AnsiAda.Foreground (AnsiAda.Red)
                & "=========================================================="
                & AnsiAda.Reset);
            Mark_Metal_Broken;
            if Stream /= null then
                begin
                    Push_Chunk
                       (Stream,
                        Session_ID,
                        ASCII.LF
                        & "ERROR: Out of Memory (STORAGE_ERROR) -- model unloaded, connection kept alive"
                        & ASCII.LF);
                exception
                    when others =>
                        null;
                end;
            end if;
            Result :=
               To_Unbounded_String
                  ("ERROR: Out of Memory (STORAGE_ERROR) in Hybrid_Generate -- model unloaded, connection kept alive");
        when E : others =>
            --  [FREE-PARALLEL-MEMORY] Free GPU on error path too
            begin
                Models (Snowball_Enaga_Orchestrator).In_Use := False;
                if Level = ELP0 then
                    Priority_Model_Gate.Release_ELP0 (Snowball_Enaga_Orchestrator);
                else
                    Priority_Model_Gate.Release_ELP1 (Snowball_Enaga_Orchestrator);
                end if;
                ELP_Queue.Dequeue_Level (Level);
            exception
                when others =>
                    null;
            end;
            --  [CRITICAL-FIX] Log the full exception with trace info
            Ada.Text_IO.Put_Line
               (AnsiAda.Foreground (AnsiAda.Red)
                & "[Hybrid]"
                & AnsiAda.Reset
                & " Error: "
                & Ada.Exceptions.Exception_Message (E));
            Ada.Text_IO.Put_Line
               (AnsiAda.Foreground (AnsiAda.Red)
                & "[Hybrid]"
                & AnsiAda.Reset
                & " Trace: "
                & Ada.Exceptions.Exception_Information (E));
            --  [CRITICAL-FIX] If generation already succeeded (Result is not
            --  empty and not an error string), DO NOT overwrite it with the
            --  error message. A transient Tasking_Error during cleanup (KV
            --  save, model unload) must not destroy a good response.
            if Length (Result) = 0
              or else (Index (Result, "ERROR:") = 1)
            then
                --  Generation truly failed — set error result
                if Stream /= null then
                    begin
                        Push_Chunk
                           (Stream,
                            Session_ID,
                            ASCII.LF & "ERROR: Generate failed" & ASCII.LF);
                    exception
                        when others =>
                            null;
                    end;
                end if;
                Result := To_Unbounded_String ("ERROR: Generate failed");
            else
                --  Generation succeeded — keep the good result, just log warning
                Ada.Text_IO.Put_Line
                   (AnsiAda.Foreground (AnsiAda.Yellow)
                    & "[Hybrid]"
                    & AnsiAda.Reset
                    & " WARNING: Cleanup error after successful generation"
                    & " (ResultLen=" & Natural'Image (Length (Result)) & ")."
                    & " Result preserved.");
            end if;
    end Hybrid_Generate;

    --  KV CACHE SSD SPILLOVER
     --  Save KV cache to SSD after generation
     procedure Save_KV_Cache_To_SSD
        (Kind       : Model_Type;
         Tokens     : System.Address;
         N_Tokens   : Interfaces.C.size_t;
         Session_ID : String) is
     begin
         if Models (Kind).Loaded and then Models (Kind).Context /= Null_Context
         then
             --  Save KV cache to SSD (ASYNC, non-blocking)
             KV_Cache_Manager.Save_To_SSD_Async
                (Context    => Models (Kind).Context,
                 Tokens     => Tokens,
                 N_Tokens   => N_Tokens,
                 Model_ID   => Kind'Img,
                 Session_ID => Session_ID);
         end if;
    exception
        when others =>
            null;  -- Don't crash on cache save failure
    end Save_KV_Cache_To_SSD;

    --  Load KV cache from SSD if available
    function Load_KV_Cache_From_SSD
       (Kind       : Model_Type;
        Tokens     : out System.Address;
        N_Tokens   : out Interfaces.C.size_t;
        Session_ID : String) return Boolean is
    begin
        Tokens := System.Null_Address;
        N_Tokens := 0;

        if Models (Kind).Loaded and then Models (Kind).Context /= Null_Context
        then
            --  Load KV cache from SSD (LAZY, on-demand only)
            return
               KV_Cache_Manager.Load_From_SSD_Lazy
                  (Context    => Models (Kind).Context,
                   Tokens     => Tokens,
                   N_Tokens   => N_Tokens,
                   Model_ID   => Kind'Img,
                   Session_ID => Session_ID);
        else
            return False;
        end if;
    exception
        when others =>
            return False;
    end Load_KV_Cache_From_SSD;

begin
    --  [DO NOT REMOVE THIS PRINT VERBOSITY]
    --  [ElabTrace-C][+Uptime]: This is the FIRST executable statement after
    --  ALL declarative items (types, objects, tasks, protected bodies) are
    --  elaborated. If this NEVER prints, the hang is in the DECLARATIVE PART
    --  (task activation or protected body elaboration). If this prints but
    --  the next one doesn't, the hang is in Initialize.
    Elab_Trace ("Model_Manager DECLARATIVE PART COMPLETE -- entering begin block");
    Initialize;
    Elab_Trace ("Model_Manager.Initialize returned -- end of elaboration");
end Model_Manager;
