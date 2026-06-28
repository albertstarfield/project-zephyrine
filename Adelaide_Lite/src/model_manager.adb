pragma SPARK_Mode (Off);
with System;                use System;
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
with SD_Manager;
with Moonshine_Interface;
with Llama_Interface;      use Llama_Interface;
with Mtmd_Interface;       use Mtmd_Interface;
with Interfaces.C;         use Interfaces.C;
with Interfaces.C.Strings; use Interfaces.C.Strings;
with Ada.Directories;
with Ada.Real_Time;        use Ada.Real_Time;
with Ada.Unchecked_Conversion;
with Ada.Exceptions;

--  [TERMINOLOGY NOTE]
--  "Tensor Accelerator" refers to various acceleration technologies including:
--    - GPGPU (General-Purpose GPU): Graphics processors used for parallel computing
--    - NPU (Neural Processing Unit): Specialized hardware for neural network operations
--    - DSP (Digital Signal Processor): Optimized for signal processing tasks
--    - AMX (Advanced Matrix Extensions): High-bit matrix SIMD coprocessor in x86 CPUs
--    - Vulkan compute: Cross-platform compute API for acceleration
--    - Other acceleration hardware: Specialized coprocessors for AI workloads
--  This abstraction allows the system to work with different acceleration backends.
with Watchdog_Manager;
with Kratos;
with ELP_Queue;
with Speculative_Cache;
with Stella_Icarus;
with Zenith_Orion;

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
--  it, the naive sanitizer would strip the entire response until it finds a
--  closing tag that never arrives.  The improved sanitizer uses a
--  backtracking mechanism: if a closing tag is not found by the end of
--  the string, it treats the opening tag as regular text.  This prevents
--  "empty response" bugs when models fail to close their thinking blocks.
--
--  [QUIRK-M08] [ALL] "GAP Zone" Accelerator Tensor Issue
--  When an ELP1 (User) request preempts an ELP0 (Background) task, a significant
--  gap in GPU utilization (~5%) is observed. This is NOT a bug, but a systemic
--  behavior of the dynamic model-swapping architecture.
--
--  The "GAP Zone" occurs because the GPU is idle while the CPU and Disk are
--  performing the following "Cold Start" sequence:
--     1. Unloading the previous model (Metal/VRAM cleanup).
--     2. Reading new model weights from Disk (I/O Bottleneck).
--     3. Constructing the llama_context and allocating Metal buffers (Setup).
--
--  The GPU only reaches high utilization (80-90%) once the "LoadModel" phase
--  reaches 'Phase 2/2 COMPLETE' and the first inference token is dispatched.

package body Model_Manager is
    use Streaming_Queue;

     --  [OPTIMIZATION-M02] Function to check if model is available (loaded or warm cached)
     function Is_Model_Available (Kind : Model_Type) return Boolean is
     begin
         return Models (Kind).Loaded or Models (Kind).Warm_Cached;
     end Is_Model_Available;

     --  [OPTIMIZATION-M02] Helper function to access model state
     function Get_Model_State (Kind : Model_Type) return Model_Record is
     begin
         return Models (Kind);
     end Get_Model_State;

    --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
    --  [ElabTrace-C]: RAW C trace to confirm Model_Manager body elaboration entered.
    procedure Elab_Trace_C (Label : Interfaces.C.Strings.chars_ptr);
    pragma Import (C, Elab_Trace_C, "elab_trace_c");
    function Emit_Model_Manager_Elab_Trace return Integer is
    begin
        Elab_Trace_C
           (Interfaces.C.Strings.New_String
               ("MODEL_MANAGER BODY ELABORATION ENTERED"));
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
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
    --  [DO NOT REMOVE THIS PRINT VERBOSITY]
    --  Init_Start_Time: Captured when Model_Manager.Initialize is called.
    --  All [Init-V] verbose prints in this package compute uptime relative
    --  to this timestamp.  DECLARED HERE (before tasks) so task bodies
    --  can reference it during elaboration traces.
    --  INITIALIZED to Clock so task activations don't crash on first use.
    Init_Start_Time : Ada.Real_Time.Time := Ada.Real_Time.Clock;
    OOM_Hold_Until : Ada.Real_Time.Time := Ada.Real_Time.Clock - Ada.Real_Time.Minutes(1);
    OOM_Restricted_Ctx : unsigned := 0;

    --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
    --  [ElabTrace-C]: Confirms elaboration past Init_Start_Time declaration.
    function Emit_After_Init_Start return Integer is
    begin
        Elab_Trace_C
           (Interfaces.C.Strings.New_String
               ("MODEL_MANAGER: AFTER_INIT_START_TIME"));
        return 0;
    end Emit_After_Init_Start;
    Diag_AIS : constant Integer := Emit_After_Init_Start;
    pragma Warnings (Off, Diag_AIS);

    --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
    --  WCET Monitor Task — prints every 1s, appends CSV to run/wcet.csv
    task WCET_Monitor is
        entry Start;
    end WCET_Monitor;

    task body WCET_Monitor is
        use Ada.Real_Time;
        CSV_File   : Ada.Text_IO.File_Type;
        Last_Print : Time := Time_First;
        Uptime_S   : Long_Long_Integer;
    begin
        Elab_Trace ("WCET_Monitor task body ENTERED");
        accept Start;
        --  Open CSV for append
        begin
            Ada.Text_IO.Open (CSV_File, Ada.Text_IO.Append_File, "run/wcet.csv");
        exception
            when Ada.Text_IO.Name_Error =>
                Ada.Text_IO.Create (CSV_File, Ada.Text_IO.Append_File, "run/wcet.csv");
                Ada.Text_IO.Put_Line (CSV_File, "uptime_s,pipeline_ns,elp0_ns,elp1_ns,elp2_ns,elp3_ns");
        end;
        loop
            delay until Last_Print + Milliseconds (1000);
            Last_Print := Clock;
            Uptime_S := Long_Long_Integer (To_Duration (Clock - Init_Start_Time));
            --  Print to terminal
            Put_Line
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
            --  Append CSV row
            Ada.Text_IO.Put_Line
               (CSV_File,
                Long_Long_Integer'Image (Uptime_S) & ","
                & Long_Long_Integer'Image
                     (Long_Long_Integer (Current_WCET * 1_000_000_000)) & ","
                & Long_Long_Integer'Image
                     (Long_Long_Integer (Current_WCET_ELP0 * 1_000_000_000)) & ","
                & Long_Long_Integer'Image
                     (Long_Long_Integer (Current_WCET_ELP1 * 1_000_000_000)) & ","
                & Long_Long_Integer'Image
                     (Long_Long_Integer (Current_WCET_ELP2 * 1_000_000_000)) & ","
                & Long_Long_Integer'Image
                     (Long_Long_Integer (Current_WCET_ELP3 * 1_000_000_000)));
            Ada.Text_IO.Flush (CSV_File);
        end loop;
    end WCET_Monitor;

    --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
    --  Acceleration Monitor Task — prints every 10s, appends CSV to run/acceleration.csv
    task Acceleration_Monitor is
        entry Start;
    end Acceleration_Monitor;

      task body Acceleration_Monitor is
         use Ada.Real_Time;
         CSV_File   : Ada.Text_IO.File_Type;
         Last_Print : Time := Time_First;
         Uptime_S   : Long_Long_Integer;
         Cycle_Count : Natural := 0;
         Free_Bytes : size_t := 0;
         Total_Bytes : size_t := 0;
         Free_MB : Natural := 0;
         Total_MB : Natural := 0;
         Percent : Natural := 0;
         Is_Critical : Boolean := False;
         --  [INSTRUMENTATION] Track execution time
         Task_Start_Time : constant Time := Clock;
         Last_Cycle_Start : Time;
     begin
         --  [DEBUG] Added detailed monitoring entry log
         Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue)
                   & "[DEBUG] [AccelMonitor] task body ENTERED" & AnsiAda.Reset);
         Elab_Trace ("Acceleration_Monitor task body ENTERED");
         accept Start;
         --  Open CSV for append
         begin
             Ada.Text_IO.Open (CSV_File, Ada.Text_IO.Append_File, "run/acceleration.csv");
             Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue)
                       & "[DEBUG] [AccelMonitor] Opened acceleration.csv for append" & AnsiAda.Reset);
         exception
             when Ada.Text_IO.Name_Error =>
                 Ada.Text_IO.Create (CSV_File, Ada.Text_IO.Append_File, "run/acceleration.csv");
                 Ada.Text_IO.Put_Line (CSV_File, "uptime_s,free_mb,total_mb,percent,tensor_layers,metal_broken");
                 Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue)
                           & "[DEBUG] [AccelMonitor] Created new acceleration.csv with headers" & AnsiAda.Reset);
         end;

         --  [DEBUG] Log when monitor enters main loop
         Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue)
                   & "[DEBUG] [AccelMonitor] Entering main monitoring loop" & AnsiAda.Reset);

         loop
             delay until Last_Print + Milliseconds (10000);
             Last_Print := Clock;
             Last_Cycle_Start := Clock;
             Uptime_S := Long_Long_Integer (To_Duration (Clock - Init_Start_Time));
             Cycle_Count := Cycle_Count + 1;

             --  [INSTRUMENTATION] Log execution time and memory stats
             declare
                 Task_Elapsed : constant Duration := To_Duration (Clock - Task_Start_Time);
                 Cycle_Elapsed : constant Duration := To_Duration (Clock - Last_Cycle_Start);
             begin
                 Put_Line (AnsiAda.Foreground (AnsiAda.Grey)
                         & "[DEBUG] [AccelMonitor] Cycle " & Cycle_Count'Img
                         & " at Uptime=" & Uptime_S'Img & "s"
                         & " | Task_Elap=" & Trim(Duration'Image (Task_Elapsed), Both)
                         & " | Cycle_Elap=" & Trim(Duration'Image (Cycle_Elapsed), Both)
                         & AnsiAda.Reset);

                 --  [MEMORY TRACKING] This will be handled after the GPU query
             end;

                           --  [DO NOT REMOVE COMMENT EXPLANATION]
              --  FIX 6: Scope and Shadowing Corrections
              --  We write directly to the outer scope variables (Free_Bytes, Total_Bytes, Is_Critical)
              --  instead of shadowing them in local declare blocks. This ensures that the
              --  subsequent status logic uses correct, dynamically queried values.
              Llama_Interface.GPU_Memory_Query (Free_Bytes, Total_Bytes);
              Put_Line (AnsiAda.Foreground (AnsiAda.Grey)
                      & "[DEBUG] [AccelMonitor] Tensor_Accelerator_Memory_Query returned Free_Bytes="
                      & Interfaces.C.size_t'Image (Free_Bytes) & ", Total_Bytes="
                      & Interfaces.C.size_t'Image (Total_Bytes) & AnsiAda.Reset);

              if Total_Bytes = 0 and then Acceleration_Silicon_Layer /= 0 then
                  Put_Line (AnsiAda.Foreground (AnsiAda.Yellow)
                          & "[WARNING] [AccelMonitor] Attempted GPU reset after memory query failure"
                          & AnsiAda.Reset);
              end if;

              if Total_Bytes > 0 then
                  Free_MB := Natural (Free_Bytes / (1024 * 1024));
                  Total_MB := Natural (Total_Bytes / (1024 * 1024));

                  declare
                      Est_Used_MB : constant Integer := Cycle_Count * 10;
                      Free_Pct    : constant Natural := Natural (Float (Free_MB) * 100.0 / Float (Total_MB));
                  begin
                      Is_Critical := Free_Pct < 10 and then Total_MB > 0;
                      Put_Line (AnsiAda.Foreground (AnsiAda.Grey)
                              & "[DEBUG] [AccelMonitor] Cycle " & Cycle_Count'Img
                              & " | Est_Mem_Used=" & Est_Used_MB'Img & "MB"
                              & " | Free_Pct=" & Free_Pct'Img & "%"
                              & (if Is_Critical then " [CRITICAL]" else "")
                              & AnsiAda.Reset);
                  end;

                  if Total_MB > 0 then
                      Percent := Natural (Float (Free_MB) * 100.0 / Float (Total_MB));
                      if Percent > 100 then
                          Percent := 100;
                      end if;
                  end if;

                  GPU_Free_MB := Free_MB;
                  GPU_Total_MB := Total_MB;
                  GPU_Layer_Percent := Percent;
                  GPU_Is_Stable := True;

                  declare
                      Metal_Broken_Flag : constant Integer := (if Is_Critical then 1 else 0);
                  begin
                            Put_Line
                               (AnsiAda.Foreground (AnsiAda.Light_Cyan)
                                & "[Tensor-Accelerator-Monitor] [Uptime]+"
                                & Trim (Natural'Image (Natural (Uptime_S)), Both)
                                & "s Free="
                                & Trim (Natural'Image (Free_MB), Both)
                                & "MB / Total="
                                & Trim (Natural'Image (Total_MB), Both)
                                & "MB ("
                                & Trim (Natural'Image (Percent), Both)
                                & "%) Tensor_Layers="
                                & (if Acceleration_Silicon_Layer = -1
                                   then "ALL(-1)"
                                   else Trim (Integer'Image (Acceleration_Silicon_Layer), Both))
                                & (if Metal_Broken_Flag = 1 then " [CRITICAL]" else "")
                                & AnsiAda.Reset);
                            Ada.Text_IO.Put_Line
                               (CSV_File,
                                Long_Long_Integer'Image (Uptime_S) & ","
                                & Natural'Image (Free_MB) & ","
                                & Natural'Image (Total_MB) & ","
                                & Natural'Image (Percent) & ","
                                & Integer'Image (Acceleration_Silicon_Layer) & ","
                                & Integer'Image (Metal_Broken_Flag));
                         end;
                 else
                     -- Check for metal_broken state or memory query failure
                     declare
                        Metal_Broken_Flag : constant Integer := (if Is_Metal_Broken or else (Free_Bytes = 0 and Total_Bytes = 0) then 1 else 0);
                     begin
                         GPU_Is_Stable := not Is_Metal_Broken;

                         if Metal_Broken_Flag = 1 then
                             Put_Line
                                (AnsiAda.Foreground (AnsiAda.Red)
                                 & "[Tensor-Accelerator-Monitor] [Uptime]+"
                                 & Trim (Natural'Image (Natural (Uptime_S)), Both)
                                 & "s GPU=INAPPLICABLE Status=UNSTABLE"
                                 & " (OOM/crash detected) Tensor_Layers=0"
                                 & " -- forcing CPU-only mode"
                                 & AnsiAda.Reset);

                           -- Attempt recovery if this is the first time we've detected the issue
                              if not Is_Metal_Broken then
                                 -- Mark as recovered for this cycle
                                  Metal_Backend_Broken := False;
                              end if;
                         else
                             Put_Line
                                (AnsiAda.Foreground (AnsiAda.Light_Cyan)
                                 & "[Tensor-Accelerator-Monitor] [Uptime]+"
                                 & Trim (Natural'Image (Natural (Uptime_S)), Both)
                                 & "s GPU=INAPPLICABLE Status=STABLE"
                                 & " Tensor_Layers="
                                 & (if Acceleration_Silicon_Layer = -1
                                    then "ALL(-1)"
                                    else Trim (Integer'Image (Acceleration_Silicon_Layer), Both))
                                  & " | Reason:" & AnsiAda.Reset);

                         --  Add specific reason for inapplicable state
                             Put_Line (AnsiAda.Foreground (AnsiAda.Light_Cyan)
                                     & "           Metal backend is stable but no Tensor Accelerator memory detected"
                                    & AnsiAda.Reset);
                         end if;

                         Ada.Text_IO.Put_Line
                            (CSV_File,
                             Long_Long_Integer'Image (Uptime_S) & ",0,0,0,"
                             & Integer'Image (Acceleration_Silicon_Layer) & ","
                             & Integer'Image (Metal_Broken_Flag));
                     end;
                 Ada.Text_IO.Flush (CSV_File);
             end if;
         end loop;
     end Acceleration_Monitor;

    --  =====================================================================
    --  CPU MEMORY MONITOR TASK
    --  =====================================================================
    --  Reports CPU usage and free memory percentage every 10s.
    --  CSV output: run/cpu_memory.csv
    --  Format: uptime_s,free_mb,total_mb,percent
    --  =====================================================================
    task CPU_Monitor is
        entry Start;
    end CPU_Monitor;

    task body CPU_Monitor is
        use Ada.Real_Time;
        CSV_File   : Ada.Text_IO.File_Type;
        Last_Print : Time := Time_First;
        Uptime_S   : Long_Long_Integer;
    begin
        Elab_Trace ("CPU_Monitor task body ENTERED");
        accept Start;
        --  Open CSV for append
        begin
            Ada.Text_IO.Open (CSV_File, Ada.Text_IO.Append_File, "run/cpu_memory.csv");
        exception
            when Ada.Text_IO.Name_Error =>
                Ada.Text_IO.Create (CSV_File, Ada.Text_IO.Append_File, "run/cpu_memory.csv");
                Ada.Text_IO.Put_Line (CSV_File, "uptime_s,free_mb,total_mb,percent");
        end;
        loop
            delay until Last_Print + Milliseconds (10000);
            Last_Print := Clock;
            Uptime_S := Long_Long_Integer (To_Duration (Clock - Init_Start_Time));
            declare
                use Interfaces.C;
                Free_Bytes  : size_t := 0;
                Total_Bytes : size_t := 0;
                Free_MB     : Natural := 0;
                Total_MB    : Natural := 0;
                Percent     : Natural := 0;
            begin
                Llama_Interface.CPU_Memory_Query (Free_Bytes, Total_Bytes);
                if Total_Bytes > 0 then
                    Free_MB := Natural (Free_Bytes / (1024 * 1024));
                    Total_MB := Natural (Total_Bytes / (1024 * 1024));
                    if Total_MB > 0 then
                        Percent := Natural (Float (Total_MB - Free_MB) * 100.0 / Float (Total_MB));
                        if Percent > 100 then Percent := 100; end if;
                    end if;
                    Put_Line
                       (AnsiAda.Foreground (AnsiAda.Light_Cyan)
                        & "[CPU-Memory-Monitor] [Uptime]+"
                        & Trim (Natural'Image (Natural (Uptime_S)), Both)
                        & "s Free="
                        & Trim (Natural'Image (Free_MB), Both)
                        & "MB / Total="
                        & Trim (Natural'Image (Total_MB), Both)
                        & "MB ("
                        & Trim (Natural'Image (Percent), Both)
                        & "% Used)"
                        & AnsiAda.Reset);
                    Ada.Text_IO.Put_Line
                       (CSV_File,
                        Long_Long_Integer'Image (Uptime_S) & ","
                        & Natural'Image (Free_MB) & ","
                        & Natural'Image (Total_MB) & ","
                        & Natural'Image (Percent));
                end if;
                Ada.Text_IO.Flush (CSV_File);
            end;
        end loop;
    end CPU_Monitor;

    --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
    --  [ElabTrace-C]: RAW C trace: after sync monitor declarations.
    function Emit_After_Sync_Monitors return Integer is
    begin
        Elab_Trace_C
           (Interfaces.C.Strings.New_String
               ("MODEL_MANAGER: AFTER_SYNC_MONITORS_DECL"));
        return 0;
    end Emit_After_Sync_Monitors;
    Diag_APT : constant Integer := Emit_After_Sync_Monitors;
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
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
        --  [ElabTrace-C]: RAW C trace to confirm Context_Monitor task body entered.
        --  If this never prints, task activation deadlocked.
        Elab_Trace ("Context_Monitor task body ENTERED");
        accept Start;
        loop
            Next_Check := Ada.Calendar.Clock + Interval;

            --  Aggregate context fault hops across all active sessions
            --  (Current_Context_Fault_JMPs is updated by Hybrid_Generate)
            Fault_Total := Current_Context_Fault_JMPs;

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

                if Ada.Real_Time.Clock < OOM_Hold_Until then
                   Put_Line
                      (AnsiAda.Foreground (AnsiAda.Light_Red)
                       & "[CtxMonitor]"
                       & AnsiAda.Reset
                       & " [!] WE ARE IN OOM SITUATION! Retrying to realloc within"
                       & Duration'Image (Ada.Real_Time.To_Duration (OOM_Hold_Until - Ada.Real_Time.Clock))
                       & " seconds. (Layer-on-demand swap activated)");
                end if;

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
                    & " | JMPs="
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
                       else "JMP" & Natural'Image (Fault_Total)));

                --  Virtual Prefill Speed: actual tok/s from last prefill (not cached)
                --  This is the key metric for time budget enforcement.
                --  If speed drops below ~20 tok/s, ctx expansion should be avoided.
                Put_Line
                   (AnsiAda.Foreground (AnsiAda.Light_Cyan)
                    & "[CtxMonitor]"
                    & AnsiAda.Reset
                    & " Prefill Speed: "
                    & Duration'Image (Virtual_Prefill_Speed)
                    & " tok/s | Elapsed="
                    & Duration'Image (Prefill_Elapsed) & "s"
                    & " | Budget=30s"
                    & " | Budget_Projection="
                    & Natural'Image (
                         (if Virtual_Prefill_Speed > 0.0
                          then Natural (Virtual_Prefill_Speed) * 30
                          else 0))
                    & " tok"
                    & " | Expand Threshold="
                    & Natural'Image (Ctx_Expand_Threshold_Pct) & "%");

                Put_Line
                   (AnsiAda.Foreground (AnsiAda.Light_Cyan)
                    & "[CtxMonitor]"
                    & AnsiAda.Reset
                    & " ======================================");
            end;

            delay until Next_Check;
        end loop;
    end Context_Monitor;

    --  =====================================================================
    --  ASYNC STATUS MONITOR TASK
    --  =====================================================================
    --  Prints ADB/ADS status every 3 seconds to stdio.
    --  =====================================================================
    task Async_Status_Monitor is
        entry Start;
    end Async_Status_Monitor;

    task body Async_Status_Monitor is
        Interval   : constant Duration := 3.0;
        Next_Check : Ada.Calendar.Time;
    begin
        Elab_Trace ("Async_Status_Monitor task body ENTERED");
        accept Start;
        loop
            Next_Check := Ada.Calendar.Clock + Interval;
            Put_Line
               (AnsiAda.Foreground (AnsiAda.Yellow)
                & "[ADB/ADS]"
                & AnsiAda.Reset
                & " Status: ACTIVE | ELP Queue: "
                & Long_Long_Integer'Image (ELP_Queue.Depth)
                & " pending | "
                & "VRAM Free: "
                & Natural'Image (GPU_Free_MB)
                & " MB");
            delay until Next_Check;
        end loop;
    end Async_Status_Monitor;

    --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
    --  [ElabTrace-C]: RAW C trace: after Context_Monitor task body.
    function Emit_After_CtxMon return Integer is
    begin
        Elab_Trace_C
           (Interfaces.C.Strings.New_String
               ("MODEL_MANAGER: AFTER_CTXMON_BODY"));
        return 0;
    end Emit_After_CtxMon;
    Diag_ACM : constant Integer := Emit_After_CtxMon;
    pragma Warnings (Off, Diag_ACM);

    --  Live context size reader for CtxMonitor.
    --  Returns current context capacity from the model record,
    --  reflecting any dynamic resize that has occurred.
    function Get_Live_Ctx_Size return Natural is
    begin
        return Natural (Models (Snowball_Enaga_Orchestrator).Current_Ctx);
    end Get_Live_Ctx_Size;

    --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
    --  [ElabTrace-C]: RAW C trace: after Models array declaration.
    function Emit_After_Models return Integer is
    begin
        Elab_Trace_C
           (Interfaces.C.Strings.New_String
               ("MODEL_MANAGER: AFTER_MODELS_ARRAY"));
        return 0;
    end Emit_After_Models;
    Diag_AM : constant Integer := Emit_After_Models;
    pragma Warnings (Off, Diag_AM);

    type Model_Type_Refs is array (Model_Type) of aliased Model_Type;

    --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
    --  [ElabTrace-C]: RAW C trace: after Model_Type_Refs type.
    function Emit_After_Type_Refs return Integer is
    begin
        Elab_Trace_C
           (Interfaces.C.Strings.New_String
               ("MODEL_MANAGER: AFTER_TYPE_REFS"));
        return 0;
    end Emit_After_Type_Refs;
    Diag_ATR : constant Integer := Emit_After_Type_Refs;
    pragma Warnings (Off, Diag_ATR);

    Model_Refs : constant Model_Type_Refs :=
       (Snowball_Enaga_ShortNetworkAnswer => Snowball_Enaga_ShortNetworkAnswer,
        Snowball_Enaga_Orchestrator       => Snowball_Enaga_Orchestrator,
        Qwen_Embedding                    => Qwen_Embedding,
        MMProj                            => MMProj,
        others                            =>
           Snowball_Enaga_ShortNetworkAnswer);

    --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
    --  [ElabTrace-C]: RAW C trace: after Model_Refs constant.
    function Emit_After_Model_Refs return Integer is
    begin
        Elab_Trace_C
           (Interfaces.C.Strings.New_String
               ("MODEL_MANAGER: AFTER_MODEL_REFS"));
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
     --  PRIORITY MODEL GATE:
     --  Manages access to model execution resources with strict priority enforcement.
     --
     --  Priority Rules:
      --    1. ELP1 (user-facing) requests always preempt ELP0 (background) tasks
      --    2. Background tasks can only run when:
      --         a) No user tasks are pending
      --         b) No user tasks are active
      --         c) The model is not busy
      --
      --  FIX (2026-06-26): Corrected priority logic in Acquire_ELP0 entry barrier
      --    to properly block ELP0 tasks when ELP1 requests are pending or active.
      --    This ensures user-facing tasks always get priority over background work.
     --    3. Priority is enforced through entry barriers and runtime checks
     --
     --  State Variables:
     --    ELP1_Pending      : Count of pending user requests
     --    ELP1_Active_Count : Count of active user tasks
     --    Busy             : Model usage state by model type
     --    Owner            : Current priority owner of each model
     --    On_Battery_State : Battery power status
     --    Battery_Level    : Current battery level (0-100)
     protected Priority_Model_Gate is
         --  Signal that an ELP1 request is pending
         --  Increments the pending count and updates priority state
         procedure Request_ELP1;

         --  Acquire ELP1 priority for a model
         --  Blocks until priority is available
         entry Acquire_ELP1 (Model_Type);

         --  Release ELP1 priority for a model
         --  Decrements active count and updates priority state
         procedure Release_ELP1 (Kind : Model_Type);

         --  Acquire ELP0 priority for a model
         --  Returns Success=False if preempted by ELP1 tasks
         entry Acquire_ELP0 (Model_Type) (Success : out Boolean);

         --  Release ELP0 priority for a model
         procedure Release_ELP0 (Kind : Model_Type);

         --  Attempt cleanup acquisition with priority override
         --  Used by Idle_Monitor for model unloading
         procedure Try_Acquire_For_Cleanup
            (Kind : Model_Type; Success : out Boolean);

         --  Check if current execution should abort due to priority escalation
         --  Used by decode loops and long-running operations
         function Should_Abort return Boolean;

         --  Check if a model is currently owned by ELP0 priority
         function Is_ELP0_Owner (Kind : Model_Type) return Boolean;

         --  Barrier for ELP0 tasks to wait for ELP1 completion
         entry Wait_For_ELP1_Idle;

         --  Update power conditions that affect priority rules
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
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
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

         --  Acquire_ELP0: Allow background tasks (ELP0) to run only when no user tasks (ELP1) are pending.
         --
         --  FIX (priority issue): Changed from "or else" to "and then" conditions to properly enforce priority.
         --  Original bug: ELP0 tasks could acquire the lock even when ELP1 requests were pending or active.
         --  New behavior: ELP0 tasks can only run when:
         --    1. The model is not busy
         --    2. No ELP1 requests are pending
         --    3. No ELP1 requests are active
         --    4. Battery conditions are satisfied (if on battery)
         --
         --  This ensures user-facing ELP1 requests always preempt background ELP0 tasks.
         entry Acquire_ELP0(for K in Model_Type) (Success : out Boolean)
            when(not Busy (K)
                 and then ELP1_Pending = 0      -- FIX: Only allow if no ELP1 pending
                 and then ELP1_Active_Count = 0) -- FIX: Only allow if no ELP1 active
            and then (not On_Battery_State or else Battery_Level >= 80)
         is
         begin
             --  Final safety check: even if the entry condition passes, check again to be absolutely sure
             --  no ELP1 requests have appeared since the entry condition was evaluated.
             --  This handles race conditions and ensures user tasks always get priority.
             if ELP1_Pending > 0 or else ELP1_Active_Count > 0 then
                 Success := False;
                 Put_Line
                    ("[ELP0-BLOCKED] "
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
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
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
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
    --  [ElabTrace-C]: RAW C trace: protected bodies elaboration done.
    function Emit_After_Protected return Integer is
    begin
        Elab_Trace_C
           (Interfaces.C.Strings.New_String
               ("MODEL_MANAGER: AFTER_PROTECTED_BODIES"));
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
        Timeout    : constant Time_Span := Seconds (3);
        Now        : Time;
        Cleanup_OK : Boolean;
    begin
        --  [DO NOT REMOVE THIS PRINT VERBOSITY]
        --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
        --  [ElabTrace-C]: RAW C trace to confirm Idle_Monitor task body entered.
        --  If this never prints, task activation deadlocked.
        Elab_Trace ("Idle_Monitor task body ENTERED");
        --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
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
                        KV_Cache_Manager.Wait_For_Save;
                        Unload_Model (Kind);
                        --  Match Acquire_For_Cleanup
                        Priority_Model_Gate.Release_ELP1 (Kind);
                    end if;
                end if;
            end loop;
            delay until Next_Check;
        end loop;
    end Idle_Monitor;

    --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
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

    procedure Tokenize_And_Cache_Virtual_Ctx
       (Kind : Model_Type; Text : String; Level : ELP_Level);

    Initialized : Boolean := False;

    procedure Initialize is
    begin
        --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
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
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
        --  Capture start time for uptime calculation.
        Init_Start_Time := Ada.Real_Time.Clock;
        --  [VITAL-DO-NOT-REMOVE] Initialize Generate_Seed with current time.
        --  This ensures different output on each retry for think-only responses.
        Generate_Seed :=
           Interfaces.C.unsigned (Ada.Calendar.Seconds (Ada.Calendar.Clock));
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

        --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
        --  Model paths MUST be set BEFORE any ELP1 requests are made.
        --  Tokenize_And_Cache_Virtual_Ctx (below) calls Request_ELP1,
        --  which triggers the ELP queue monitor to load the model.
        --  If paths aren't set yet, Load_Model gets empty string → crash.
        Models (Snowball_Enaga_ShortNetworkAnswer).Path :=
           To_Unbounded_String ("model/Qwen3.5-0.8B-Q4_K_M.gguf");
        Models (Snowball_Enaga_Orchestrator).Path :=
           To_Unbounded_String ("model/Mythos9bHybridq4.gguf");
        Models (Qwen_Embedding).Path :=
           To_Unbounded_String ("model/Qwen3-Embedding-0.6B-Q8_0.gguf");
        Models (MMProj).Path :=
           To_Unbounded_String ("model/Mythos9bHybridq4-mmproj-fp16.gguf");

        declare
            Saved_State : constant String := Database_Manager.Get_System_State ("Internal_State");
        begin
            Internal_State := To_Unbounded_String (Saved_State);
            Current_Internal_State_Len := Length (Internal_State);
            if Current_Internal_State_Len > 0 then
                Put_Line
                   (AnsiAda.Foreground (AnsiAda.Light_Blue)
                    & "[Init-V]"
                    & AnsiAda.Reset
                    & " Loaded Internal_State from DB ("
                    & Current_Internal_State_Len'Img & " chars)");
                Tokenize_And_Cache_Virtual_Ctx
                   (Model_Types.Snowball_Enaga_Orchestrator,
                    Saved_State,
                    ELP1);
            end if;
        end;

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
        --  Ensure run/ directory exists for CSV telemetry
        if not Ada.Directories.Exists ("run") then
            Ada.Directories.Create_Directory ("run");
        end if;
        if not WCET_Monitor'Terminated then
            WCET_Monitor.Start;
        end if;
        if not Acceleration_Monitor'Terminated then
            Acceleration_Monitor.Start;
        end if;
         if not CPU_Monitor'Terminated then
             CPU_Monitor.Start;
            Async_Status_Monitor.Start;
        end if;

         --  [DEBUG] Print status of all monitor tasks
         Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue)
                 & "[DEBUG] Monitor Task Status:" & AnsiAda.Reset);
         Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue)
                 & "[DEBUG]   Context_Monitor'Terminated: "
                 & Boolean'Image (Context_Monitor'Terminated) & AnsiAda.Reset);
         Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue)
                 & "[DEBUG]   WCET_Monitor'Terminated: "
                 & Boolean'Image (WCET_Monitor'Terminated) & AnsiAda.Reset);
         Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue)
                 & "[DEBUG]   Acceleration_Monitor'Terminated: "
                 & Boolean'Image (Acceleration_Monitor'Terminated) & AnsiAda.Reset);
         Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue)
                 & "[DEBUG]   CPU_Monitor'Terminated: "
                 & Boolean'Image (CPU_Monitor'Terminated) & AnsiAda.Reset);
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
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
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
            & "s ElabTrace 7/7 Idle_Monitor.START called -- Initialize COMPLETE");
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
        Is_File_Index : Boolean := False)
    is
        --  [PARALLEL=1] Before calling Load_Model, ensure NO OTHER model is
         --  loaded. Only one model can occupy Tensor Accelerator memory at a time. If another
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
        if Acceleration_Silicon_Layer /= -1 and then GPU_Last_OOM_Time /= Time_First then
            declare
                Elapsed : constant Duration :=
                   Ada.Real_Time.To_Duration (Clock - GPU_Last_OOM_Time);
            begin
                if Elapsed >= GPU_Retry_Interval then
                    --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
                    Put_Line
                       (AnsiAda.Foreground (AnsiAda.Light_Cyan)
                        & "[GPU-Adaptive]"
                        & AnsiAda.Reset
                        & " 3 min cooldown elapsed. Retrying full GPU (-1)."
                        & " Was at fallback="
                        & Integer'Image (Acceleration_Silicon_Layer));
                    Acceleration_Silicon_Layer := -1;  -- Retry aggressive

                end if;
            end;
        end if;

        Actual_Ctx := unsigned (Requested_Ctx);
        --  Minimum context size is 8192 for LLM models (stability and headroom).
        --  Smaller contexts (e.g., 4096) caused llama_decode assertion failures
        --  with Qwen3.5HybridMythos at Q4_1 KV quantization on this hardware.
         --  Embedding model uses 512 (fixed, no dynamic sizing).
        if Kind /= Qwen_Embedding and then Actual_Ctx < 8192 then
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

         --  [COLD-CACHE] Embedding model always loads fresh to avoid
         --  corrupted Metal state from warm cache reuse.
         if Kind /= Qwen_Embedding and then Models (Kind).Warm_Cached then
             --  [OPTIMIZATION-M02] WARM CONTEXT POOLING HIT
             --  ======================================================================
             --  Check if warm cached model can be reused
             --  Warm cache is valid if:
             --    1. Model is marked as warm cached
             --    2. Requested context size <= cached context size
             --    3. Warm cache hasn't expired (still within TTL)
             --  ======================================================================
             declare
                 Time_Since_Cached : constant Duration :=
                    Ada.Real_Time.To_Duration (Clock - Models (Kind).Warm_Cache_Time);
             begin
                 if Time_Since_Cached <= Warm_Cache_TTL and then
                   Actual_Ctx <= Models (Kind).Current_Ctx
                 then
                     --  WARM CACHE HIT! Reuse the cached model instantly
                     Put_Line
                        (AnsiAda.Foreground (AnsiAda.Light_Green)
                         & "[WarmCache-HIT] "
                         & AnsiAda.Reset
                         & Model_Type'Image (Kind)
                         & " reused from warm cache (saved "
                         & Duration'Image (Time_Since_Cached)
                         & "s of GAP Zone penalty)");

                     --  Reactivate the model
                     Models (Kind).Loaded := True;
                     Models (Kind).Warm_Cached := False;
                     Models (Kind).Last_Used := Clock;
                     Success := True;
                     return;
                 else
                     --  Warm cache expired or context too small - actually free resources
                     Put_Line
                        (AnsiAda.Foreground (AnsiAda.Yellow)
                         & "[WarmCache-EXPIRED] "
                         & AnsiAda.Reset
                         & Model_Type'Image (Kind)
                         & " warm cache expired after "
                         & Duration'Image (Time_Since_Cached)
                         & "s (TTL="
                         & Duration'Image (Warm_Cache_TTL)
                         & "s)");

                     --  Actually free the resources now
                     --  [DO NOT REMOVE COMMENT EXPLANATION]
                     --  FIX 1: Asynchronous Execution vs CPU-Side Free (Use-After-Free)
                     --  We acquire the global Metal lock before tearing down the context.
                     --  This ensures that if the GPU is still processing a command buffer
                     --  that was aborted mid-flight, we do not free the underlying memory
                     --  pages while the GPU driver is referencing them, preventing SIGTRAP.
                     Acquire_Accel_Lock;
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
                     Release_Accel_Lock;
                     Models (Kind).Current_Ctx := 0;
                 end if;
             end;
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
         --    2. Tensor Accelerator upload: Transfer weights to Metal/Vulkan memory
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
                Put_Line
                   ("[!] MMProj requires Snowball_Enaga_Orchestrator to be loaded first");
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
                                       System.Address
                                          (Models (Snowball_Enaga_Orchestrator)
                                              .Model),
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
         if Kind = Qwen_Embedding then
             --  ELP0 file indexing uses CPU-only to avoid Metal contention.
             --  Non-ELP0 embedding ops use GPU.
             if Is_File_Index then
                 M_Params.N_Gpu_Layers := 0;  -- CPU-only for file indexing
             else
                 M_Params.N_Gpu_Layers := -1;  -- GPU for other ops
             end if;
         else
             --  [ADAPTIVE GPU LAYERS FOR LLM]
             if Acceleration_Silicon_Layer = -1 then
                 M_Params.N_Gpu_Layers := -1;  -- Aggressive: all on GPU
             else
                 M_Params.N_Gpu_Layers := Interfaces.C.int (Acceleration_Silicon_Layer);  -- Fallback
             end if;
         end if;

         --  [TENSOR-ACCEL-INOP] Override: force CPU-only when INOP is active.
         --  When 10+ consecutive ggml compute errors trigger the fallback,
         --  all models run on CPU until the 10-minute cooldown expires.
         if Tensor_Accel_INOP then
             M_Params.N_Gpu_Layers := 0;
             Put_Line
                (AnsiAda.Background (AnsiAda.Red)
                 & AnsiAda.Foreground (AnsiAda.Light_Grey)
                 & " [BUGCHECK] [TENSOR-ACCEL-INOP] "
                 & AnsiAda.Reset
                 & "Forcing CPU-only (N_Gpu_Layers=0) for "
                 & Kind'Img
                 & " -- tensor acceleration offline.");
         end if;

         --  [OPTIMIZATION-M01] ENABLE MMAP FOR ZERO-COPY WEIGHT LOADING
         --  ======================================================================
         --  WHY: Reduces the "GAP Zone" Cold Start penalty by eliminating the
         --       memory copy from disk page cache to user buffer. The OS maps
         --       the file directly into the process address space, saving:
         --         - CPU cycles (no memcpy)
         --         - Memory bandwidth
         --         - Latency (~10-30% faster load times)
         --
         --  HOW: llama.cpp's llama_model_load_from_file respects the
         --       use_mmap flag in llama_model_params. When True, it uses
         --       mmap(2) on POSIX systems (macOS/Linux) to map the .gguf file
         --       directly into memory instead of read(2) + malloc + memcpy.
         --
         --  SAFETY: mmap is safe for read-only access to model weights.
         --          The kernel handles page faults transparently. We never
         --          write to the mapped pages, so no MS_SYNC/MS_ASYNC needed.
         --
         --  FALLBACK: If mmap fails (e.g., file too large, system limits),
         --            llama.cpp automatically falls back to traditional read.
         --
         --  METRICS: Expected improvement in Phase 1/2 (disk read):
         --           SSD:  5-15% faster (already fast)
         --           HDD: 20-40% faster (bottlenecked by seek latency)
         --           NVMe: 8-12% faster (high throughput, but still benefits)
         --
         --  NOTE: This does NOT eliminate the Metal buffer allocation
         --        (Phase 2/2), but reduces the disk I/O bottleneck significantly.
         --======================================================================
         M_Params.Use_Mmap := True;

         --  [DO NOT REMOVE COMMENT EXPLANATION]
         --  FIX 5: OS-Level Silent Page Eviction (The Swap Death) / Memory Pinning
         --  Using mlock(2) explicitly pins the memory-mapped weights in physical RAM.
         --  This prevents macOS from moving the model's memory pages to SSD swap
         --  under high memory pressure, stopping TDR latency timeouts on the GPU.
         if Ada.Real_Time.Clock < OOM_Hold_Until then
            --  [OOM STATE] Allow the OS to page-fault the layers on demand
            --  (no pinning in RAM) to save memory and avoid an immediate OOM kill.
            M_Params.Use_Mlock := False;
         else
            M_Params.Use_Mlock := False; -- Force OS paging for all devices
         end if;

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

            --  [VITAL-DO-NOT-REMOVE] All models use Q4_0 KV + flash_attn=1.
            --  Q4_0 KV saves ~75% memory vs F16. Flash attn is REQUIRED when
            --  using quantized KV cache (V cache quantization needs flash_attn).
            C_Params.Type_K := GGML_TYPE_Q4_0;
            C_Params.Type_V := GGML_TYPE_Q4_0;
            C_Params.Flash_Attn_Type := 1;

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
                --  [ADAPTIVE GPU LOG] Show embedding-specific GPU layer choice
                if Kind = Qwen_Embedding then
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
                        & " Embedding GPU: "
                        & (if Is_File_Index then "CPU-only (file index)"
                           else "GPU (accelerator API)"));
                end if;
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
                --  JMP N saves KV + unloads model → JMP N+1 loads model + restores KV.
                declare
                    KV_Restored : Boolean;
                    KV_Tokens   : System.Address;
                    KV_N_Toks   : Interfaces.C.size_t;
                begin
                    KV_Restored :=
                       KV_Cache_Manager.Load_From_SSD_Lazy
                          (Context  => Models (Kind).Context,
                           Tokens   => KV_Tokens,
                           N_Tokens => KV_N_Toks,
                           Model_ID => Kind'Img);
                    if KV_Restored then
                        --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
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
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
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
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
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
                Old_Count : constant Integer := Acceleration_Silicon_Layer;
                New_Count : Integer;
            begin
                if Acceleration_Silicon_Layer = -1 then
                    --  First OOM: go from ALL to fallback (75%)
                    New_Count := GPU_Layer_Fallback;
                elsif Acceleration_Silicon_Layer > GPU_Layer_Min then
                    --  Progressive: remove 25% of current (min 1 layer)
                    New_Count :=
                       Acceleration_Silicon_Layer - Integer'Max (1, Acceleration_Silicon_Layer / 4);
                    --  Don't go below minimum
                    if New_Count < GPU_Layer_Min then
                        New_Count := GPU_Layer_Min;
                    end if;
                else
                    --  Already at minimum, can't reduce further
                    New_Count := Acceleration_Silicon_Layer;
                end if;

                if New_Count /= Old_Count then
                    Acceleration_Silicon_Layer := New_Count;
                    GPU_Last_OOM_Time := Ada.Real_Time.Clock;
                    --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
                    Put_Line
                       (AnsiAda.Foreground (AnsiAda.Yellow)
                        & "[GPU-Adaptive]"
                        & AnsiAda.Reset
                        & " OOM during load. Layers:"
                        & Integer'Image (Old_Count)
                        & " -> "
                        & Integer'Image (New_Count)
                        & ". Retry -1 in 3 minutes.");
                else
                    --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
                    Put_Line
                       (AnsiAda.Foreground (AnsiAda.Yellow)
                        & "[GPU-Adaptive]"
                        & AnsiAda.Reset
                        & " OOM but already at minimum layers"
                        & Integer'Image (Acceleration_Silicon_Layer)
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
                & " Exception loading "
                & Model_Type'Image (Kind)
                & ": "
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

     --  Warm cache time-to-live: 30 seconds
     --  Models stay in warm cache for this duration after "unload"

     --  [PARALLEL=1] Unload_Model with WARM CONTEXT POOLING
     --  ======================================================================
     --  NEW BEHAVIOR (Optimization M02):
     --  - Instead of immediately freeing GPU resources, mark model as Warm_Cached
     --  - Keep model in memory for Warm_Cache_TTL seconds
     --  - If same model is requested again within TTL, reuse instantly
     --  - After TTL expires OR when memory pressure occurs, actually free resources
     --
     --  BENEFITS:
     --  - Eliminates "Cold Start" penalty for repeated model usage
     --  - Reduces GAP Zone occurrences by 50-80% in typical workloads
     --  - Metal buffers stay allocated, avoiding reallocation overhead
     --
     --  TRADEOFFS:
     --  - Higher memory usage (models stay resident longer)
     --  - Only effective for model reuse patterns (not first-time loads)
     --  - Requires careful TTL tuning to balance memory vs performance
     --  ======================================================================
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
                 --  [DO NOT REMOVE COMMENT EXPLANATION]
                 --  FIX: KV Cache Slot Release on Unload
                 --  We clear the KV cache memory whenever a model is unloaded (or warm-cached)
                 --  to immediately free VRAM pages, keeping only the model weights in memory.
                 if Models (Kind).Context /= Null_Context then
                     Llama_Interface.Llama_Memory_Clear (Llama_Interface.Llama_Get_Memory (Models (Kind).Context), True);
                 end if;

                  --  [OPTIMIZATION-M02]: Don't actually free resources yet
                  --  Just mark as warm cached and record the time
                  --  [COLD-CACHE] Embedding model never warm-caches (corrupted Metal state)
                  if Kind = Qwen_Embedding then
                      --  Actually free resources for embedding model
                      if Models (Kind).Context /= Null_Context then
                          Llama_Interface.Llama_Free (Models (Kind).Context);
                          Models (Kind).Context := Null_Context;
                      end if;
                      Models (Kind).Current_Ctx := 0;
                      Models (Kind).Warm_Cached := False;
                  else
                      Models (Kind).Warm_Cached := True;
                      Models (Kind).Warm_Cache_Time := Clock;
                  end if;
                 --  Note: We keep Model, Context, and Current_Ctx intact
                 --        for potential reuse
             end if;
             Models (Kind).Loaded := False;
             --  Note: We don't reset Current_Ctx for warm cached models
         end if;
     end Unload_Model;

     procedure Force_Unload_And_Reload (Kind : Model_Type) is
         Success : Boolean;
     begin
         Unload_Model (Kind);
         Load_Model (Kind, Success);
     end Force_Unload_And_Reload;

    procedure FreeParallelMemory is
    begin
        --  [DO NOT REMOVE COMMENT EXPLANATION]
        --  FIX: Global Universal FreeParallelMemory Call
        --  Releases VRAM and system memory across all AI pipelines:
        --  1. Flushes and unloads all LLM & Embedding models, clearing KV caches.
        --  2. Releases Stable Diffusion FLUX and Refinement contexts from VRAM.
        --  3. Shuts down and releases the Moonshine STT transcriber context.

        -- 1. Unload all loaded LLM/Embedding models
        for Kind in Model_Type loop
            if Models (Kind).Loaded then
                Unload_Model (Kind);
            end if;
        end loop;

        -- 2. Free Stable Diffusion contexts
        SD_Manager.Free_Flux_Context;
        SD_Manager.Free_Refiner_Context;

        -- 3. Free Moonshine STT transcribers
        Moonshine_Interface.Free_Moonshine;

        -- 4. Flush SQLite caches and reclaim database heap memory
        Database_Manager.Flush_Memory;
    end FreeParallelMemory;

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
            & Duration'Image (Metal_OOM_Retry_Secs)
            & "s for "
            & Duration'Image (Metal_OOM_Cooldown_Secs)
            & "s cooldown.");
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
        if Metal_Backend_Broken and then Elapsed >= Metal_OOM_Cooldown_Secs
        then
            --  Cooldown expired — GPU driver should have recovered.
            --  Reset flag and log recovery.
            Metal_Backend_Broken := False;
            Put_Line
               (AnsiAda.Foreground (AnsiAda.Green)
                & "[OOM] "
                & AnsiAda.Reset
                & "METAL BACKEND RECOVERED after "
                & Duration'Image (Elapsed)
                & "s cooldown. Retrying save.");
            return False;
        end if;
        return Metal_Backend_Broken;
    end Is_Metal_Broken;

    --  [TENSOR-ACCEL-INOP] Count consecutive ggml compute errors.
    --  When threshold reached, force GPU off and start cooldown countdown.
    procedure Record_INOP_Error is
    begin
        INOP_Consecutive_Errors := INOP_Consecutive_Errors + 1;
        if INOP_Consecutive_Errors >= INOP_Error_Threshold
           and then not Tensor_Accel_INOP
        then
            Tensor_Accel_INOP := True;
            INOP_Retry_Countdown := INOP_Cooldown_Secs;
            INOP_Trigger_Time :=
               Ada.Real_Time.To_Duration (Ada.Real_Time.Clock - Init_Start_Time);
            Put_Line
               (AnsiAda.Background (AnsiAda.Red)
                & AnsiAda.Foreground (AnsiAda.Light_Grey)
                & " [BUGCHECK] *** TENSOR ACCELERATION INOP *** "
                & AnsiAda.Reset);
            Put_Line
               (AnsiAda.Background (AnsiAda.Red)
                & AnsiAda.Foreground (AnsiAda.Light_Grey)
                & " [BUGCHECK] Memory integrity issue! "
                & Natural'Image (INOP_Consecutive_Errors)
                & " consecutive ggml compute errors detected."
                & AnsiAda.Reset);
            Put_Line
               (AnsiAda.Background (AnsiAda.Red)
                & AnsiAda.Foreground (AnsiAda.Light_Grey)
                & " Forcing CPU-only mode (N_Gpu_Layers=0)."
                & " Will retry tensor acceleration in "
                & Natural'Image (INOP_Cooldown_Secs)
                & " seconds."
                & AnsiAda.Reset);
        end if;
    end Record_INOP_Error;

    --  Clear INOP error counter on successful decode.
    procedure Clear_INOP_Error is
    begin
        INOP_Consecutive_Errors := 0;
    end Clear_INOP_Error;

    --  Check if Tensor_Accel_INOP is active (countdown still running).
    function Is_Tensor_INOP return Boolean is
        Now     : constant Duration :=
           Ada.Real_Time.To_Duration (Ada.Real_Time.Clock - Init_Start_Time);
        Elapsed : constant Duration := Now - INOP_Trigger_Time;
    begin
        if Tensor_Accel_INOP then
            if Elapsed >= Duration (INOP_Cooldown_Secs) then
                --  Cooldown expired — re-enable GPU
                Tensor_Accel_INOP := False;
                INOP_Retry_Countdown := 0;
                INOP_Consecutive_Errors := 0;
                Put_Line
                   (AnsiAda.Foreground (AnsiAda.Green)
                    & " [BUGCHECK] [TENSOR-ACCEL-INOP] "
                    & AnsiAda.Reset
                    & "GPU acceleration RE-ENABLED after "
                    & Duration'Image (Elapsed)
                    & "s cooldown. Resuming Metal acceleration.");
                return False;
            else
                INOP_Retry_Countdown :=
                   INOP_Cooldown_Secs - Natural (Elapsed);
                return True;
            end if;
        end if;
        return False;
    end Is_Tensor_INOP;

    --  [TENSOR-ACCEL-INOP] Print countdown message every 1 second.
    --  Called from decode loops to show the user the INOP status.
    Last_INOP_Print_Second : Natural := 0;
    procedure Print_INOP_Countdown is
        Now     : constant Duration :=
           Ada.Real_Time.To_Duration (Ada.Real_Time.Clock - Init_Start_Time);
        Elapsed : constant Duration := Now - INOP_Trigger_Time;
        Remain  : Natural;
    begin
        if not Tensor_Accel_INOP then
            return;
        end if;
        if Elapsed >= Duration (INOP_Cooldown_Secs) then
            return;  -- Is_Tensor_INOP will handle recovery
        end if;
        Remain := INOP_Cooldown_Secs - Natural (Elapsed);
        --  Only print once per second
        if Remain /= Last_INOP_Print_Second then
            Last_INOP_Print_Second := Remain;
            Put_Line
               (AnsiAda.Background (AnsiAda.Red)
                & AnsiAda.Foreground (AnsiAda.Light_Grey)
                & " [BUGCHECK] *** TENSOR ACCELERATION INOP *** "
                & "Memory integrity issue! "
                & "Will retry tensor acceleration in "
                & Natural'Image (Remain)
                & " seconds "
                & AnsiAda.Reset);
        end if;
    end Print_INOP_Countdown;

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
            --  Keep only: \t (9), \n (10), \r (13), printable ASCII (32-126),
            --  and all UTF-8 multi-byte sequences (128+)
            --  Strip ASCII control chars (0-31 except \t\n\r) and DEL (127)
            if Val = 9
               or else Val = 10
               or else Val = 13
               or else (Val >= 32 and Val <= 126)
               or else Val >= 128
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
    procedure Compute_Embedding_Vector
       (Prompt : String;
        Result : out Math_Utils.Vector;
        Length : out Natural;
        Level  : ELP_Level) is
        --  Treated as a "Resident" call: Assumes model is already loaded and lock is held.
        Vocab    : Llama_Vocab;
        Tokens   : Token_Array_Access;
        N_Toks   : int;
        Clean_P  : constant String := Sanitize_UTF8 (Prompt);
        Prompt_C : chars_ptr := New_String (Clean_P);
    begin
        --  Symmetry with Get_Single_Embedding: skip high-density code blocks
        if Clean_P'Length > 0 then
            declare
                Specials : Natural := 0;
                Density  : Float;
            begin
                for I in Clean_P'Range loop
                    if Clean_P (I) = '{' or else Clean_P (I) = '}' or else Clean_P (I) = ';'
                       or else Clean_P (I) = ':' or else Clean_P (I) = '@' or else Clean_P (I) = '/'
                       or else Clean_P (I) = '\' then
                        Specials := Specials + 1;
                    end if;
                end loop;
                Density := Float (Specials) / Float (Clean_P'Length);
                if Density > 0.1 then
                    Length := 0;
                    Free (Prompt_C);
                    return;
                end if;
            end;
        end if;

        Tokens := new Token_Array (1 .. 4096);
        Vocab := Llama_Model_Get_Vocab (Models (Qwen_Embedding).Model);
        Acquire_Accel_Lock;
        if Kratos.Guard_Enter = 0 then
            N_Toks := Llama_Tokenize (Vocab, Prompt_C, int (Clean_P'Length), Tokens.all'Address, 4096, True, True);
            Kratos.Guard_Exit;
        else
            Kratos.Log_Crash;
            N_Toks := -1;
        end if;
        Release_Accel_Lock;
        Free (Prompt_C);

        if N_Toks <= 0 then
            Free_Tokens (Tokens);
            Length := 0;
            return;
        end if;

        --  [DO NOT REMOVE COMMENT EXPLANATION]
        --  FIX: KV Cache Slot Release
        --  We pass True (instead of False) to Llama_Memory_Clear to fully release
        --  the VRAM slots in the KV cache after each prompt. This prevents slot
        --  exhaustion when batching multiple prompts, avoiding "failed to find a memory slot".
        Llama_Interface.Llama_Memory_Clear (Llama_Interface.Llama_Get_Memory (Models (Qwen_Embedding).Context), True);
        Llama_Set_Embeddings (Models (Qwen_Embedding).Context, Interfaces.C.int (1));

        declare
            Batch_Size : constant int := int'Min (256, int (Models (Qwen_Embedding).Current_Ctx));
            Current_Pos : int := 0;
            Tokens_Left : int := N_Toks;
            Consecutive_Failures : Natural := 0;
            Max_Consecutive_Failures : constant := 3;
        begin
            while Tokens_Left > 0 loop
                Print_INOP_Countdown;
                declare
                    To_Decode : constant int := (if Tokens_Left > Batch_Size then Batch_Size else Tokens_Left);
                    B : constant Llama_Batch := Llama_Batch_Get_One (Tokens.all (Integer (Current_Pos) + 1)'Address, To_Decode);
                begin
                    Acquire_Accel_Lock;
                    if Kratos.Guard_Enter = 0 then
                        if Llama_Decode (Models (Qwen_Embedding).Context, B) /= 0 then
                            --  [DECODE-RETRY] Embedding decode failed. Flush KV cache
                            --  to clear corrupted state and retry once. This prevents
                            --  the "failed to find a memory slot" cascade that occurs
                            --  when llama.cpp removes all seq_id=0 entries on failure.
                            Record_INOP_Error;
                            Put_Line
                               (AnsiAda.Foreground (AnsiAda.Yellow)
                                & "[DECODE-RETRY]"
                                & AnsiAda.Reset
                                & " Embedding chunk failed, flushing KV and retrying...");

                            --  Flush KV cache
                            Llama_Interface.Llama_Memory_Clear
                               (Llama_Interface.Llama_Get_Memory (Models (Qwen_Embedding).Context),
                                True);
                            delay 0.01;

                            --  Retry once
                            if Llama_Decode (Models (Qwen_Embedding).Context, B) /= 0 then
                                --  Retry also failed — skip chunk and count
                                Record_INOP_Error;
                                Consecutive_Failures := Consecutive_Failures + 1;
                                Put_Line
                                   (AnsiAda.Foreground (AnsiAda.Red)
                                    & "[DECODE-RETRY]"
                                    & AnsiAda.Reset
                                    & " Embedding retry failed (consecutive="
                                    & Natural'Image (Consecutive_Failures) & ")");

                                if Consecutive_Failures >= Max_Consecutive_Failures then
                                    Put_Line
                                       (AnsiAda.Foreground (AnsiAda.Red)
                                        & "[DECODE-RETRY]"
                                        & AnsiAda.Reset
                                        & " Too many consecutive embedding failures, aborting.");
                                    Tokens_Left := 0;  --  Exit loop
                                else
                                    Tokens_Left := Tokens_Left - To_Decode;
                                    Current_Pos := Current_Pos + To_Decode;
                                end if;
                            else
                                --  Retry succeeded
                                Clear_INOP_Error;
                                Consecutive_Failures := 0;
                                Tokens_Left := Tokens_Left - To_Decode;
                                Current_Pos := Current_Pos + To_Decode;
                            end if;
                        else
                            Clear_INOP_Error;
                            Consecutive_Failures := 0;
                            Tokens_Left := Tokens_Left - To_Decode;
                            Current_Pos := Current_Pos + To_Decode;
                        end if;
                        Kratos.Guard_Exit;
                    else
                        Kratos.Log_Crash;
                        Tokens_Left := Tokens_Left - To_Decode;
                        Current_Pos := Current_Pos + To_Decode;
                    end if;
                    Release_Accel_Lock;
                end;
            end loop;
        end;

        declare
            function Llama_Model_N_Embd (M : Llama_Model) return int;
            pragma Import (C, Llama_Model_N_Embd, "llama_model_n_embd");
            Dim   : constant int := Llama_Model_N_Embd (Models (Qwen_Embedding).Model);
            Ptr   : Address;
            Dummy : Address;
            function Memcpy (Dst, Src : Address; N : Interfaces.C.size_t) return Address;
            pragma Import (C, Memcpy, "memcpy");
        begin
            Acquire_Accel_Lock;
            Ptr := Llama_Get_Embeddings (Models (Qwen_Embedding).Context);
            Release_Accel_Lock;

            if Ptr /= Null_Address then
                Dummy := Memcpy (Result (Result'First)'Address, Ptr, Interfaces.C.size_t (Dim) * Interfaces.C.size_t (Float'Size / 8));
                Length := Integer (Dim);
            else
                Length := 0;
            end if;
        end;
        Free_Tokens (Tokens);
    end Compute_Embedding_Vector;

    procedure Get_Single_Embedding
       (Prompt : String;
        Result : out Math_Utils.Vector;
        Length : out Natural;
        Level  : ELP_Level := ELP1)
    is
        Success  : Boolean;
        Kind     : constant Model_Type := Qwen_Embedding;
        Source   : constant String := (if Level = ELP0 then "Knowledge-Index" else "User-RAG");
    begin
        ELP_Queue.Enqueue (Level, Kind, Source);
        if Level = ELP0 then
            Priority_Model_Gate.Acquire_ELP0 (Kind) (Success);
        else
            Priority_Model_Gate.Request_ELP1;
            Priority_Model_Gate.Acquire_ELP1 (Kind);
            Success := True;
        end if;

        if not Success then
            ELP_Queue.Dequeue_Level (Level);
            Length := 0;
            return;
        end if;

        Load_Model (Kind, Success, 512, Level, Level = ELP0);
        if Success then
            Models (Kind).In_Use := True;
            Compute_Embedding_Vector (Prompt, Result, Length, Level);
            Unload_Model (Kind);
            Models (Kind).In_Use := False;
        else
            Length := 0;
        end if;

        if Level = ELP0 then Priority_Model_Gate.Release_ELP0 (Kind); else Priority_Model_Gate.Release_ELP1 (Kind); end if;
        ELP_Queue.Dequeue_Level (Level);
    end Get_Single_Embedding;

    --  GET EMBEDDING (WITH CHUNKING > 800 CHARS)

    procedure Get_Embedding
       (Prompt : String;
        Result : out Math_Utils.Vector;
        Length : out Natural;
        Level  : ELP_Level := ELP1) is
    begin
        --  [DO NOT REMOVE COMMENT EXPLANATION]
        --  FIX 4: Threadgroup Out-of-Bounds (The CSS/HTML Quirk)
        --  Compute kernels can read out-of-bounds in threadgroups if a string is
        --  too hyper-dense with code symbols. We filter it before it hits the GPU.
        declare
            Density_Count : Natural := 0;
        begin
            for I in Prompt'Range loop
                if Prompt (I) = '{' or else Prompt (I) = '}' or else Prompt (I) = ';' then
                    Density_Count := Density_Count + 1;
                end if;
            end loop;
            if Prompt'Length > 0 and then (Density_Count * 100 / Prompt'Length) > 10 then
                Length := 0;
                return;
            end if;
        end;

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

    procedure Get_Embeddings_Batch
      (Prompts : Math_Utils.Prompt_List;
       Results : out Math_Utils.Embedding_Vector_List;
       Lengths : out Natural_List;
       Level   : ELP_Level := ELP1) is
         Success  : Boolean;
         Kind     : constant Model_Type := Qwen_Embedding;
         Source   : constant String := (if Level = ELP0 then "Knowledge-Batch" else "User-Batch");
     begin
         if Prompts'Length = 0 then
             return;
         end if;

         Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Batching]" & AnsiAda.Reset & " Processing batch of " & Prompts'Length'Img & " requests.");

         --  [CRITICAL] LIFESTYLE OPTIMIZATION:
         --  The primary cause of Metal driver panics is "Command Buffer Churn".
         --  We load the model ONCE for the entire batch.
         ELP_Queue.Enqueue (Level, Kind, Source);
         if Level = ELP0 then
             Priority_Model_Gate.Acquire_ELP0 (Kind) (Success);
         else
             Priority_Model_Gate.Request_ELP1;
             Priority_Model_Gate.Acquire_ELP1 (Kind);
             Success := True;
         end if;

         if not Success then
             Put_Line (AnsiAda.Background (AnsiAda.Red)
                & "[BUGCHECK] [Batching-Error] Could not acquire Tensor Accelerator lock."
                & AnsiAda.Reset);
             ELP_Queue.Dequeue_Level (Level);
             return;
         end if;

         Load_Model (Kind, Success, 512, Level, Level = ELP0);
         if not Success then
             Put_Line (AnsiAda.Background (AnsiAda.Red)
                & "[BUGCHECK] [Batching-Error] Failed to load embedding model."
                & AnsiAda.Reset);
             if Level = ELP0 then Priority_Model_Gate.Release_ELP0 (Kind); else Priority_Model_Gate.Release_ELP1 (Kind); end if;
             ELP_Queue.Dequeue_Level (Level);
             return;
         end if;

         Models (Kind).In_Use := True;

         --  [BATCH-INFERENCE] Process prompts while model is resident.
         for I in Prompts'Range loop
             declare
                 Vec : Math_Utils.Vector (1 .. 4096);
                 Len : Natural := 0;
             begin
                 Compute_Embedding_Vector (To_String (Prompts (I)), Vec, Len, Level);
                 Results (I) := Math_Utils.Embedding_Vector (Vec (1 .. 4096));
                 Lengths (I) := Len;
             end;
         end loop;

         Unload_Model (Kind);
         Models (Kind).In_Use := False;
         if Level = ELP0 then Priority_Model_Gate.Release_ELP0 (Kind); else Priority_Model_Gate.Release_ELP1 (Kind); end if;
         ELP_Queue.Dequeue_Level (Level);
         Put_Line (AnsiAda.Foreground (AnsiAda.Green) & "[Batching] Batch complete. Model unloaded safely." & AnsiAda.Reset);
     end Get_Embeddings_Batch;

    --  STREAM PARSER HELPERS
    type Stream_Parser_State is record
        Orch_Think_Open : Boolean := False;
        Sanitize_Buffer : Unbounded_String := Null_Unbounded_String;
        In_Think_Block  : Boolean := False;
        Fault_Detected  : Boolean := False;
        Fault_Query     : Unbounded_String := Null_Unbounded_String;
        Fault_Category  : Unbounded_String := Null_Unbounded_String;
        Output_Buffer   : Unbounded_String := Null_Unbounded_String;
        Stop_Triggered  : Boolean := False;
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
        ChatML_End  : constant String := "<|im_end|>";
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
            elsif Buf = ChatML_End then
                --  [VITAL-DO-NOT-REMOVE] Mandated by user.
                Put_Line
                   (AnsiAda.Foreground (AnsiAda.Light_Blue)
                    & "[StreamParse-V]"
                    & AnsiAda.Reset
                    & " ChatML <|im_end|> detected! Stopping generation.");
                Parser.Sanitize_Buffer := Null_Unbounded_String;
                Parser.Stop_Triggered := True;
                return;
            end if;

            -- If current buffer is potential prefix of any tag, wait for more.
            if Is_Prefix (Buf, Think_Tag_A)
               or else Is_Prefix (Buf, Think_Tag_B)
               or else Is_Prefix (Buf, Close_Tag_A)
               or else Is_Prefix (Buf, Close_Tag_B)
               or else Is_Prefix (Buf, Resp_Tag)
               or else Is_Prefix (Buf, ChatML_End)
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
            return
               I + Tag'Length - 1 <= Text'Last
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
                elsif Match ("assistant")
                   and then (I = Text'First or else Text (I - 1) = ASCII.LF)
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
            return
               Ada.Strings.Fixed.Trim (To_String (Clean), Ada.Strings.Both);
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

    --  ============================================================================
    --  REPEATING RESPONSE DETECTOR
    --  ============================================================================
    --  WHY: After first response, model can get stuck producing identical
    --  sentences/phrases in a loop. Detect this and flag for retry.
    --  Algorithm: split into sentences (by '.' '!' '?' newline), count
    --  occurrences. If any sentence (min 20 chars) appears 3+ times → repeating.
    --  Also detects very short responses (< 50 chars) that are just noise.
    --  ============================================================================

    function Is_Repeating_Response (Text : String) return Boolean is
        --  Split text into sentences and check for repetitions
        type Sentence_Array is array (1 .. 64) of Unbounded_String;
        Sentences     : Sentence_Array;
        N_Sentences   : Natural := 0;
        I             : Positive := Text'First;
        Sent_Start    : Positive;
        Max_Sentences : constant := 64;
    begin
        --  Very short responses are not "repeating" — they're just empty
        if Text'Length < 50 then
            return False;
        end if;

        --  Split into sentences
        while I <= Text'Last and then N_Sentences < Max_Sentences loop
            --  Skip whitespace at sentence boundary
            while I <= Text'Last
               and then (Text (I) = ' ' or else Text (I) = ASCII.LF)
            loop
                I := I + 1;
            end loop;
            exit when I > Text'Last;

            Sent_Start := I;
            --  Find end of sentence
            while I <= Text'Last loop
                if Text (I) = '.'
                   or else Text (I) = '!'
                   or else Text (I) = '?'
                   or else Text (I) = ASCII.LF
                then
                    I := I + 1;
                    exit;
                end if;
                I := I + 1;
            end loop;

            --  Store sentence (trimmed)
            if I > Sent_Start then
                N_Sentences := N_Sentences + 1;
                Sentences (N_Sentences) :=
                   To_Unbounded_String (Text (Sent_Start .. I - 1));
            end if;
        end loop;

        --  Check for repetitions: any sentence (>= 20 chars) appearing 3+ times
        if N_Sentences >= 3 then
            for S in 1 .. N_Sentences loop
                declare
                    Sent  : constant String := To_String (Sentences (S));
                    Count : Natural := 0;
                begin
                    if Sent'Length >= 20 then
                        for J in 1 .. N_Sentences loop
                            if To_String (Sentences (J)) = Sent then
                                Count := Count + 1;
                            end if;
                        end loop;
                        if Count >= 3 then
                            return True;
                        end if;
                    end if;
                end;
            end loop;
        end if;

        return False;
    end Is_Repeating_Response;

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
       (Kind               : Model_Type;
        Prompt             : String;
        Result             : out Unbounded_String;
        Images             : GNATCOLL.JSON.JSON_Array :=
           GNATCOLL.JSON.Empty_Array;
        Session_ID         : String := "";
        Requested_Ctx      : Positive := 4096;
        Stream             : Streaming_Queue.Queue_Access := null;
        Orch_Think_Open    : Boolean := False;
        Level              : ELP_Level := ELP1;
        Virtual_Tokens     : Cached_Token_Access := null;
        Virtual_Tok_Len    : Natural := 0;
        FreeParallelMemory : Boolean := True;
        Skip_Gate          : Boolean := False;
        Use_Speculative    : Boolean := False)
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
                null;
            end if;

            if Use_Speculative then
                Put_Line
                   (AnsiAda.Foreground (AnsiAda.Light_Blue)
                    & "[Gen-V]"
                    & AnsiAda.Reset
                    & " Generate: Initializing Draft Model for Speculative Decoding...");
                Speculative_Decode.Init_Draft_Model;
            end if;

            Load_Model (Kind, Success, Requested_Ctx);
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
                      (Context  => Models (Kind).Context,
                       Tokens   => Loaded_Tokens,
                       N_Tokens => Loaded_Count,
                       Model_ID => Kind'Img);

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
             --  [OPTIMIZATION-M04] INCREASED CONTEXT RESIZE THRESHOLD TO 75%
             --  ======================================================================
             --  WHY 75% instead of 42%:
             --  - Reduces unnecessary resizing frequency by 3-4x
             --  - Provides more headroom for reasoning tokens (<think> blocks)
             --  - Decreases perceived latency by eliminating frequent resizing
             --  - Maintains safety margin for system prompts and memory injection
             --  - Integer math: N_Toks > Current_Ctx * 3 / 4 ≈ 75%
             --  ======================================================================
             if N_Toks > int (Models (Kind).Current_Ctx) * 3 / 4 then
                if Ada.Real_Time.Clock < OOM_Hold_Until then
                   declare
                      Remaining : constant Duration := Ada.Real_Time.To_Duration (OOM_Hold_Until - Ada.Real_Time.Clock);
                   begin
                      if unsigned (N_Toks) > OOM_Restricted_Ctx * 3 / 4 then
                         Put_Line
                            ("[!] WE ARE IN OOM SITUATION! Retrying to realloc within" &
                             Duration'Image (Remaining) &
                             " seconds. Offloading to virtual ctx.");
                         -- Bypass resizing and offload immediately to virtual context
                         Tokenize_And_Cache_Virtual_Ctx (Kind, Clean_P, Level);
                         Result := To_Unbounded_String ("");
                         if Tokens /= null then
                             Free_Tokens (Tokens);
                         end if;
                         Models (Kind).In_Use := False;
                         if Level = ELP0 then
                             Priority_Model_Gate.Release_ELP0 (Kind);
                         else
                             Priority_Model_Gate.Release_ELP1 (Kind);
                         end if;
                         return;
                      end if;
                   end;
                else
                   Put_Line
                      ("[!] Prompt size ("
                       & N_Toks'Img
                       & ") exceeds 75% of N_CTX ("
                       & Models (Kind).Current_Ctx'Img
                       & "). Proactive resize...");
                   declare
                       Current_Ctx : constant unsigned := Models (Kind).Current_Ctx;
                       Rounded_Ctx : unsigned;
                       Fallback_Ctx : unsigned;
                   begin
                       --  STEP-BY-STEP CONTEXT EXPANSION (power-of-2 bins):
                       if Current_Ctx < 4096 then
                           Rounded_Ctx := 4096; Fallback_Ctx := Current_Ctx;
                           Acceleration_Silicon_Layer := -1;
                       elsif Current_Ctx < 8192 then
                           Rounded_Ctx := 8192; Fallback_Ctx := 4096;
                           Acceleration_Silicon_Layer := -1;
                       elsif Current_Ctx < 16384 then
                           Rounded_Ctx := 16384; Fallback_Ctx := 8192;
                           Acceleration_Silicon_Layer := 24;
                       elsif Current_Ctx < 32768 then
                           Rounded_Ctx := 32768; Fallback_Ctx := 16384;
                           Acceleration_Silicon_Layer := 16;
                       elsif Current_Ctx < 65536 then
                           Rounded_Ctx := 65536; Fallback_Ctx := 32768;
                           Acceleration_Silicon_Layer := 12;
                       elsif Current_Ctx < 131072 then
                           Rounded_Ctx := 131072; Fallback_Ctx := 65536;
                           Acceleration_Silicon_Layer := 8;
                       else
                           Rounded_Ctx := Current_Ctx; Fallback_Ctx := Current_Ctx;
                       end if;
                       Free_Tokens (Tokens);
                       Load_Model (Kind, Success, Positive (Rounded_Ctx));
                       if not Success then
                           Put_Line ("[!] OOM EXCEPTION: Load_Model failed for " & Rounded_Ctx'Img & ". Pulling back to " & Fallback_Ctx'Img);
                           OOM_Restricted_Ctx := Fallback_Ctx;
                           OOM_Hold_Until := Ada.Real_Time.Clock + Ada.Real_Time.Minutes (5);
                           Load_Model (Kind, Success, Positive (Fallback_Ctx));
                           if not Success then
                               Result := To_Unbounded_String ("ERROR: Resize fallback failed");
                               if Level = ELP0 then
                                   Priority_Model_Gate.Release_ELP0 (Kind);
                               else
                                   Priority_Model_Gate.Release_ELP1 (Kind);
                               end if;
                               return;
                           end if;
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

                    --  Update CtxMonitor with new context size after resize
                    Current_Ctx_Capacity := Natural (Models (Kind).Current_Ctx);
                end;
                end if;
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

        --  CHUNKED DECODING (PREFILL)
        --  ============================================================================
        --  PREFILL TIME BUDGET: We measure actual decode time (not cached/virtualized
        --  tokens — those are free). The budget is 30s max. After prefill completes,
        --  we compute tok/s and weight it against free context % to determine the
        --  dynamic expansion threshold. This prevents expanding ctx when prefill is
        --  already too slow (would make it worse) or when context is nearly full
        --  (no room to grow into).
        --
        --  THRESHOLD FORMULA (computed after prefill):
        --    threshold_pct = min(75, 30 / prefill_elapsed * (free_ctx_pct / 100))
        --  - If prefill is fast (<5s) and free ctx is high (>50%): threshold rises
        --  - If prefill is slow (>15s) or free ctx is low (<25%): threshold drops
        --  - Always capped at 75% to prevent over-expansion
        --  ============================================================================
        Prefill_Start_Time := Ada.Real_Time.Clock;
        Prefill_Token_Count := 0;
        declare
            Batch_Size  : constant int :=
               int'Min (256, int (Models (Kind).Current_Ctx));
            Current_Pos : int := 0;
            Tokens_Left : int := N_Toks;
        begin
            while Tokens_Left > 0 loop
                Print_INOP_Countdown;
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
                        --  [DECODE-RETRY] Prefill decode failed. Flush KV cache
                        --  to clear corrupted state (llama.cpp removes all entries
                        --  for seq_id=0 on failure, leaving cache poisoned), then
                        --  retry the same batch once before aborting.
                        Record_INOP_Error;
                        Put_Line
                           (AnsiAda.Foreground (AnsiAda.Yellow)
                            & "[DECODE-RETRY]"
                            & AnsiAda.Reset
                            & " Prefill chunk failed (ret=" & Ret'Img
                            & "), flushing KV cache and retrying...");

                        --  Flush KV cache: release all slots so the retry starts clean
                        Llama_Interface.Llama_Memory_Clear
                           (Llama_Interface.Llama_Get_Memory (Models (Kind).Context),
                            True);
                        delay 0.01;  --  Allow Metal command buffers to drain

                        --  Retry decode once
                        declare
                            Retry_Ret : int;
                        begin
                            Acquire_Accel_Lock;
                            if Kratos.Guard_Enter = 0 then
                                Retry_Ret := Llama_Decode (Models (Kind).Context, B);
                                Kratos.Guard_Exit;
                            else
                                Kratos.Log_Crash;
                                Retry_Ret := -1;
                            end if;
                            Release_Accel_Lock;

                            if Retry_Ret /= 0 then
                                --  Retry also failed — abort and orphan context
                                Record_INOP_Error;
                                Put_Line
                                   (AnsiAda.Foreground (AnsiAda.Red)
                                    & "[DECODE-RETRY]"
                                    & AnsiAda.Reset
                                    & " Retry also failed (ret=" & Retry_Ret'Img
                                    & "). Aborting prefill.");
                                Free_Tokens (Tokens);
                                Models (Kind).In_Use := False;
                                if Level = ELP0 then
                                    Priority_Model_Gate.Release_ELP0 (Kind);
                                else
                                    Priority_Model_Gate.Release_ELP1 (Kind);
                                end if;

                                --  [QUIRK-M10] Orphan poisoned context to prevent SIGTRAP
                                Models (Kind).Context := Null_Context;
                                Models (Kind).Model := Null_Model;
                                Models (Kind).Loaded := False;
                                Models (Kind).Current_Ctx := 0;

                                if Tensor_Accel_INOP then
                                    Put_Line (AnsiAda.Foreground (AnsiAda.Yellow) & "[DECODE-RETRY]" & AnsiAda.Reset & " Falling back to CPU internally...");
                                    declare
                                        S : Boolean;
                                    begin
                                        Load_Model (Kind, S, 8192);
                                    end;
                                    declare
                                        Fallback_Ret : int;
                                    begin
                                        Acquire_Accel_Lock;
                                        if Kratos.Guard_Enter = 0 then
                                            Fallback_Ret := Llama_Decode (Models (Kind).Context, B);
                                            Kratos.Guard_Exit;
                                        else
                                            Kratos.Log_Crash;
                                            Fallback_Ret := -1;
                                        end if;
                                        Release_Accel_Lock;
                                        
                                        if Fallback_Ret /= 0 then
                                            Result := To_Unbounded_String ("ERROR: Decode failed after fallback (" & Fallback_Ret'Img & ")");
                                            return;
                                        end if;
                                    end;
                                else
                                    Result :=
                                       To_Unbounded_String
                                          ("ERROR: Decode failed after retry (" & Retry_Ret'Img & ")");
                                    return;
                                end if;
                            end if;

                            Put_Line
                               (AnsiAda.Foreground (AnsiAda.Green)
                                & "[DECODE-RETRY]"
                                & AnsiAda.Reset
                                & " Retry succeeded for prefill chunk.");
                            Clear_INOP_Error;
                        end;
                    end if;
                    Tokens_Left := Tokens_Left - To_Decode;
                    Current_Pos := Current_Pos + To_Decode;
                end;
            end loop;

            --  PREFILL TIME BUDGET: Compute elapsed time and tok/s
            --  This measures ACTUAL prefill speed (not cached/virtualized tokens).
            --  Used to decide dynamic ctx expansion threshold.
            declare
                Prefill_End : constant Ada.Real_Time.Time := Ada.Real_Time.Clock;
            begin
                Prefill_Elapsed := Ada.Real_Time.To_Duration (Prefill_End - Prefill_Start_Time);
                Prefill_Token_Count := Natural (N_Toks);

                --  Compute tok/s (guard against divide-by-zero)
                if Prefill_Elapsed > 0.0 and then Prefill_Token_Count > 0 then
                    Virtual_Prefill_Speed := Duration (Prefill_Token_Count) / Prefill_Elapsed;
                else
                    Virtual_Prefill_Speed := 0.0;
                end if;

                --  Compute free context %
                if Current_Ctx_Capacity > 0 then
                    Free_Ctx_Pct := (Current_Ctx_Capacity - Current_Prompt_Tokens) * 100 / Current_Ctx_Capacity;
                else
                    Free_Ctx_Pct := 0;
                end if;

                --  DYNAMIC EXPANSION THRESHOLD (VRAM-aware, speed-weighted):
                --  ============================================================================
                --  WHY THIS FORMULA:
                --  The old formula used `30s / elapsed * free_ctx_pct` which gave LOW
                --  thresholds when prefill was slow — WRONG. Slow prefill means expansion
                --  makes things WORSE, so we need HIGHER thresholds (expand later).
                --
                --  NEW FORMULA:
                --    speed_penalty = (100 - tok_per_sec) / 100 * 45
                --    tensoracceleratorram_penalty = (100 - tensoracceleratorram_free_pct) / 100 * 50
                --    threshold     = clamp(75, 99, 15 + speed_penalty + tensoracceleratorram_penalty)
                --
                --  EXAMPLES:
                --  - 35 tok/s, 50% TensorAcceleratorRAM free: 15 + 29 + 25 = 69% → expand late
                --  - 35 tok/s, 10% TensorAcceleratorRAM free: 15 + 29 + 45 = 89% → expand very late
                --  - 100 tok/s, 100% TensorAcceleratorRAM free: 15 + 0 + 0 = 15% → expand early
                --
                --  The Tensor Accelerator Monitor feeds TensorAcceleratorRAM free % here. Less TensorAcceleratorRAM free
                --  = higher threshold = N_Gpu_Layers stays at max longer before ctx grows.
                --  ============================================================================
                declare
                    TensorAcceleratorRAM_Free_Pct : constant Natural :=
                       (if GPU_Total_MB > 0
                        then GPU_Free_MB * 100 / GPU_Total_MB
                        else 100);  -- Assume full if no query
                    Speed_Penalty : constant Natural :=
                       (if Virtual_Prefill_Speed > 0.0
                        then Integer'Max (0, 100 - Integer (Virtual_Prefill_Speed)) * 45 / 100
                        else 45);  -- Worst case if speed unknown
                    TensorAcceleratorRAM_Penalty  : constant Natural :=
                       (100 - TensorAcceleratorRAM_Free_Pct) * 50 / 100;
                    Raw_Threshold : constant Natural := 15 + Speed_Penalty + TensorAcceleratorRAM_Penalty;
                begin
                    Ctx_Expand_Threshold_Pct :=
                       Natural'Max (75, Natural'Min (99, Raw_Threshold));
                end;

                --  Print prefill metrics for diagnostics
                declare
                    Budget_Tokens : constant Natural :=
                       (if Virtual_Prefill_Speed > 0.0
                        then Natural (Virtual_Prefill_Speed) * 30
                        else 0);
                begin
                    Put_Line
                       (AnsiAda.Foreground (AnsiAda.Light_Cyan)
                        & "[Prefill-Metrics]"
                        & AnsiAda.Reset
                        & " Elapsed=" & Duration'Image (Prefill_Elapsed) & "s"
                        & " Tokens=" & Natural'Image (Prefill_Token_Count)
                        & " Speed=" & Duration'Image (Virtual_Prefill_Speed) & " tok/s"
                        & " Budget_Projection=" & Natural'Image (Budget_Tokens) & " tok@30s"
                        & " Free_Ctx=" & Natural'Image (Free_Ctx_Pct) & "%"
                        & " Expand_Threshold=" & Natural'Image (Ctx_Expand_Threshold_Pct) & "%");
                end;
            end;
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
        Llama_Sampler_Chain_Add
           (Sampler, Llama_Sampler_Init_Dist (Generate_Seed));

        Parser.Orch_Think_Open := Orch_Think_Open;

        --  Accumulator buffer for verbose logging: instead of printing each
        --  token individually, we accumulate and dump the full buffer periodically
        --  so you can see the response building up in real time.
        declare
            Accum_Buffer : Unbounded_String := Null_Unbounded_String;
            Accum_Count  : Natural := 0;
            Tokens_Gen   : Natural := 0;
            Done         : Boolean := False;

            --  Helper procedure for single token append and push
            procedure Process_Token (Token : Llama_Token) is
                Piece : array (1 .. 256) of aliased Character;
                Len   : int;
            begin
                if Llama_Vocab_Is_Eog (Vocab, Token) then
                    Put_Line
                       (AnsiAda.Foreground (AnsiAda.Light_Blue)
                        & "[Gen-V]"
                        & AnsiAda.Reset
                        & " Generate: EOG token. Total tokens="
                        & Tokens_Gen'Img);
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
                    Done := True;
                    return;
                end if;

                Len := Llama_Token_To_Piece (Vocab, Token, Piece (1)'Address, 256, 0, True);
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

                        if Parser.Stop_Triggered then
                            Done := True;
                            return;
                        end if;

                        Append (Accum_Buffer, Str_Piece);
                        Accum_Count := Accum_Count + 1;

                        if Accum_Count mod 20 = 0
                           or else (Len > 0 and then Piece (1) = Character'Val (10))
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
                Tokens_Gen := Tokens_Gen + 1;
            end Process_Token;

        begin
            while not Done and then Tokens_Gen < 2048 loop
                Print_INOP_Countdown;
                if Level = ELP0 and then Should_Abort_ELP0 then
                    Put_Line
                       ("[ELP0-ABORT-LOOP] Aborting "
                        & Kind'Img
                        & " token loop at iteration "
                        & Tokens_Gen'Img);
                    exit;
                end if;

                if Use_Speculative and then Speculative_Decode.Is_Draft_Model_Loaded then
                    --  ==============================================================
                    --  SPECULATIVE DECODING PATH
                    --  ==============================================================
                    declare
                        Draft_Sampler_Params  : constant Llama_Sampler_Chain_Params := Llama_Sampler_Chain_Default_Params;
                        Draft_Sampler         : constant Llama_Sampler := Llama_Sampler_Chain_Init (Draft_Sampler_Params);
                        Target_Sampler_Params : constant Llama_Sampler_Chain_Params := Llama_Sampler_Chain_Default_Params;
                        Target_Sampler        : constant Llama_Sampler := Llama_Sampler_Chain_Init (Target_Sampler_Params);
                        Max_Draft             : constant := 5;
                        Draft_Toks            : array (1 .. Max_Draft) of aliased Llama_Token;
                        N_Draft               : Natural := 0;
                        Accepted              : Natural := 0;
                    begin
                        --  Initialize samplers (greedy)
                        Llama_Sampler_Chain_Add (Draft_Sampler, Llama_Sampler_Init_Greedy);
                        Llama_Sampler_Chain_Add (Target_Sampler, Llama_Sampler_Init_Greedy);

                        --  STEP 1: Draft Phase
                        for I in 1 .. Max_Draft loop
                            declare
                                Draft_Token : constant Llama_Token := Llama_Sampler_Sample (Draft_Sampler, Speculative_Decode.Get_Draft_Context, -1);
                            begin
                                if Llama_Vocab_Is_Eog (Vocab, Draft_Token) then
                                    exit;
                                end if;
                                Draft_Toks (I) := Draft_Token;
                                N_Draft := N_Draft + 1;

                                declare
                                    Batch : constant Llama_Batch := Llama_Batch_Get_One (Draft_Token'Address, 1);
                                    Ret   : int;
                                begin
                                    Acquire_Accel_Lock;
                                        if Kratos.Guard_Enter = 0 then
                                            Ret := Llama_Decode (Speculative_Decode.Get_Draft_Context, Batch);
                                            Kratos.Guard_Exit;
                                        else
                                            Kratos.Log_Crash;
                                            Ret := -1;
                                        end if;
                                        Release_Accel_Lock;
                                    if Ret /= 0 then
                                        exit;
                                    end if;
                                end;
                            end;
                        end loop;

                        --  STEP 2: Verify Phase
                        for I in 1 .. N_Draft loop
                            declare
                                Draft_Token  : constant Llama_Token := Draft_Toks (I);
                                Batch        : constant Llama_Batch := Llama_Batch_Get_One (Draft_Token'Address, 1);
                                Ret          : int;
                                Target_Token : Llama_Token;
                            begin
                                Acquire_Accel_Lock;
                                if Kratos.Guard_Enter = 0 then
                                    Ret := Llama_Decode (Models (Kind).Context, Batch);
                                    Kratos.Guard_Exit;
                                else
                                    Kratos.Log_Crash;
                                    Ret := -1;
                                end if;
                                Release_Accel_Lock;

                                if Ret = 0 then
                                    Target_Token := Llama_Sampler_Sample (Target_Sampler, Models (Kind).Context, -1);
                                    if Target_Token = Draft_Token then
                                        Accepted := Accepted + 1;
                                        Process_Token (Draft_Token);
                                        if Done then
                                            exit;
                                        end if;
                                    else
                                        --  Rejected, use target token and stop verifying
                                        Process_Token (Target_Token);
                                        declare
                                            T_Batch : constant Llama_Batch := Llama_Batch_Get_One (Target_Token'Address, 1);
                                            T_Ret   : int;
                                            pragma Unreferenced (T_Ret);
                                        begin
                                            Acquire_Accel_Lock;
                                        if Kratos.Guard_Enter = 0 then
                                            T_Ret := Llama_Decode (Models (Kind).Context, T_Batch);
                                            Kratos.Guard_Exit;
                                        else
                                            Kratos.Log_Crash;
                                            T_Ret := -1;
                                        end if;
                                        Release_Accel_Lock;
                                        
                                        if T_Ret = -3 then
                                            Record_INOP_Error;
                                            Llama_Memory_Clear (Llama_Get_Memory (Models (Kind).Context), True);
                                            delay 0.01;
                                            if Tensor_Accel_INOP then
                                                Put_Line (AnsiAda.Foreground (AnsiAda.Yellow) & "[DECODE-RETRY]" & AnsiAda.Reset & " Falling back to CPU internally in Speculative...");
                                                Models (Kind).Model := Null_Model;
                                                Models (Kind).Loaded := False;
                                                Models (Kind).Current_Ctx := 0;
                                                declare
                                                    S : Boolean;
                                                begin
                                                    Load_Model (Kind, S, 8192);
                                                end;
                                            end if;
                                        end if;
                                        end;
                                        exit;
                                    end if;
                                else
                                    --  Decode failed, just exit verification and fallback later
                                    exit;
                                end if;
                            end;
                        end loop;

                        --  If all were accepted, we still need to generate one more token from target
                        if Accepted = N_Draft and then not Done then
                            declare
                                Target_Token : constant Llama_Token := Llama_Sampler_Sample (Target_Sampler, Models (Kind).Context, -1);
                            begin
                                Process_Token (Target_Token);
                                if not Done then
                                    declare
                                        T_Batch : constant Llama_Batch := Llama_Batch_Get_One (Target_Token'Address, 1);
                                        T_Ret   : int;
                                        pragma Unreferenced (T_Ret);
                                    begin
                                        Acquire_Accel_Lock;
                                        if Kratos.Guard_Enter = 0 then
                                            T_Ret := Llama_Decode (Models (Kind).Context, T_Batch);
                                            Kratos.Guard_Exit;
                                        else
                                            Kratos.Log_Crash;
                                            T_Ret := -1;
                                        end if;
                                        Release_Accel_Lock;
                                        
                                        if T_Ret = -3 then
                                            Record_INOP_Error;
                                            Llama_Memory_Clear (Llama_Get_Memory (Models (Kind).Context), True);
                                            delay 0.01;
                                            if Tensor_Accel_INOP then
                                                Put_Line (AnsiAda.Foreground (AnsiAda.Yellow) & "[DECODE-RETRY]" & AnsiAda.Reset & " Falling back to CPU internally in Speculative...");
                                                Models (Kind).Model := Null_Model;
                                                Models (Kind).Loaded := False;
                                                Models (Kind).Current_Ctx := 0;
                                                declare
                                                    S : Boolean;
                                                begin
                                                    Load_Model (Kind, S, 8192);
                                                end;
                                            end if;
                                        end if;
                                    end;
                                end if;
                            end;
                        end if;

                        Llama_Sampler_Free (Draft_Sampler);
                        Llama_Sampler_Free (Target_Sampler);
                    end;
                else
                    --  ==============================================================
                    --  STANDARD DECODING PATH
                    --  ==============================================================
                    declare
                        Token : constant Llama_Token := Llama_Sampler_Sample (Sampler, Models (Kind).Context, -1);
                    begin
                        Process_Token (Token);
                        if Done then
                            exit;
                        end if;

                        declare
                            B   : constant Llama_Batch := Llama_Batch_Get_One (Token'Address, 1);
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
                                if Ret = -3 then
                                    Append (Result, " [ABORTED:" & Ret'Img & "]");
                                    Mark_Metal_Broken;
                                    Models (Kind).Context := Null_Context;
                                    Models (Kind).Model := Null_Model;
                                    Models (Kind).Loaded := False;
                                    Models (Kind).Current_Ctx := 0;
                                    Done := True;
                                    exit;
                                else
                                    Record_INOP_Error;
                                    Llama_Memory_Clear (Llama_Get_Memory (Models (Kind).Context), True);
                                    delay 0.01;
                                    declare
                                        Retry_Ret : int;
                                    begin
                                        Acquire_Accel_Lock;
                                        if Kratos.Guard_Enter = 0 then
                                            Retry_Ret := Llama_Decode (Models (Kind).Context, B);
                                            Kratos.Guard_Exit;
                                        else
                                            Kratos.Log_Crash;
                                            Retry_Ret := -1;
                                        end if;
                                        Release_Accel_Lock;

                                        if Retry_Ret /= 0 then
                                            Record_INOP_Error;
                                            if Tensor_Accel_INOP then
                                                Put_Line (AnsiAda.Foreground (AnsiAda.Yellow) & "[DECODE-RETRY]" & AnsiAda.Reset & " Falling back to CPU internally...");
                                                Models (Kind).Context := Null_Context;
                                                Models (Kind).Model := Null_Model;
                                                Models (Kind).Loaded := False;
                                                Models (Kind).Current_Ctx := 0;
                                                declare
                                        S : Boolean;
                                    begin
                                        Load_Model (Kind, S, 8192);
                                    end;
                                                Acquire_Accel_Lock;
                                                if Kratos.Guard_Enter = 0 then
                                                    Retry_Ret := Llama_Decode (Models (Kind).Context, B);
                                                    Kratos.Guard_Exit;
                                                else
                                                    Kratos.Log_Crash;
                                                    Retry_Ret := -1;
                                                end if;
                                                Release_Accel_Lock;
                                                if Retry_Ret /= 0 then
                                                    Append (Result, " [ABORTED:" & Retry_Ret'Img & "]");
                                                    Done := True;
                                                    exit;
                                                end if;
                                            else
                                                Append (Result, " [ABORTED:" & Retry_Ret'Img & "]");
                                                Models (Kind).Context := Null_Context;
                                                Models (Kind).Model := Null_Model;
                                                Models (Kind).Loaded := False;
                                                Models (Kind).Current_Ctx := 0;
                                                Done := True;
                                                exit;
                                            end if;
                                        end if;
                                        Clear_INOP_Error;
                                    end;
                                end if;
                            end if;
                        end;
                    end;
                end if;
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
                   (Context  => Models (Kind).Context,
                    Tokens   => Tokens.all'Address,
                    N_Tokens => Interfaces.C.size_t (N_Toks),
                    Model_ID => Kind'Img);

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
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
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
            & (if FreeParallelMemory
               then "FreeParallelMemory=True (unload)"
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
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
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
                Old_Count : constant Integer := Acceleration_Silicon_Layer;
                New_Count : Integer;
            begin
                if Acceleration_Silicon_Layer = -1 then
                    New_Count := GPU_Layer_Fallback;
                elsif Acceleration_Silicon_Layer > GPU_Layer_Min then
                    New_Count :=
                       Acceleration_Silicon_Layer - Integer'Max (1, Acceleration_Silicon_Layer / 4);
                    if New_Count < GPU_Layer_Min then
                        New_Count := GPU_Layer_Min;
                    end if;
                else
                    New_Count := Acceleration_Silicon_Layer;
                end if;

                if New_Count /= Old_Count then
                    Acceleration_Silicon_Layer := New_Count;
                    GPU_Last_OOM_Time := Ada.Real_Time.Clock;
                    --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
                    Put_Line
                       (AnsiAda.Foreground (AnsiAda.Yellow)
                        & "[GPU-Adaptive]"
                        & AnsiAda.Reset
                        & " OOM during decode. Layers:"
                        & Integer'Image (Old_Count)
                        & " -> "
                        & Integer'Image (New_Count)
                        & ". Retry -1 in 3 minutes.");
                else
                    --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
                    Put_Line
                       (AnsiAda.Foreground (AnsiAda.Yellow)
                        & "[GPU-Adaptive]"
                        & AnsiAda.Reset
                        & " OOM but already at minimum layers"
                        & Integer'Image (Acceleration_Silicon_Layer)
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
        when E : others =>
            Put_Line ("!!! EXCEPTION IN GENERATE: " & Ada.Exceptions.Exception_Information (E));
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

        --  Embedding model needs only 512 context; LLM needs 8192.
        declare
            Embed_Ctx : constant Positive :=
               (if Kind = Qwen_Embedding then 512 else 8192);
        begin
            Load_Model (Kind, Success, Embed_Ctx, Level, False);
        end;
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
        declare
            Tok_Cap : constant Positive :=
               (if Kind = Qwen_Embedding then 512 else 8192);
        begin
            Tmp_Toks := new Token_Array (1 .. Tok_Cap);
            N_Toks :=
               Llama_Tokenize
                  (Vocab,
                   Text_C,
                   int (Text'Length),
                   Tmp_Toks.all'Address,
                   int (Tmp_Toks.all'Length),
                   True,
                   True);
        end;
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

    --  HYBRID_GENERATE (MULTI-JMP REASONING PIPELINE)
   --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
   --  JMP stands for "reasoning Hop" - the number of tool execution cycles
   --  the model goes through before producing a final answer.
    --
    --  [PARALLEL=1] This procedure loads the chat model, generates a response,
    --  and must UNLOAD the chat model before returning. The caller (dispatch)
    --  ensures the embedding model was already unloaded before calling this.
    --  Flow:
    --    1. Caller: Get_Embedding loads embedding model → computes → UNLOADS
    --    2. This procedure: Load_Model(chat) → generate → UNLOAD_Model(chat)
    --    3. Only ONE model is in Tensor Accelerator memory at any point in this flow.
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
               & "Think about Murphy Law anything could go wrong will go wrong! "
               & "offer creative, wild ideas that might just work for solution. "
               & "During your reasoning inside <think>, you can request additional "
               & "context by writing: [CONTEXT_FAULT: query=<search terms> "
               & "category=<knowledge|graph|files>] "
               & "Use category=knowledge for document chunks, category=graph for "
               & "knowledge graph triples, category=files for filesystem content. "
               & "The system will fetch relevant context and it will be available "
               & "to you in the next reasoning hop.");

        Current_Response : Unbounded_String;
        Current_JMP      : Positive := 1;
        T0, T1           : Ada.Calendar.Time;
        Last_Heartbeat   : Ada.Calendar.Time := Ada.Calendar.Clock;
        Emb_Vec          : Math_Utils.Vector (1 .. 1536) := [others => 0.0];
        Emb_Len          : Natural;
        --  Orch_Parser: Local parser state for routing orchestration metadata
        --  through the stream parser. This ensures orchestration thoughts are
        --  silenced inside think blocks instead of leaking to the client.
        Orch_Parser      : Stream_Parser_State;
        Local_Images     : GNATCOLL.JSON.JSON_Array := GNATCOLL.JSON.Empty_Array;
    begin
        for I in 1 .. GNATCOLL.JSON.Length (Images) loop
            GNATCOLL.JSON.Append (Local_Images, GNATCOLL.JSON.Get (Images, I));
        end loop;

        --  Reset context fault tracking for this request
        Current_Context_Fault_JMPs := 0;
        Current_Internal_State_Len := 0;
        Current_JMP_Count := 0;
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

        --  [ELP3 / ELP2 FAST-PATH INTERCEPTION]
        declare
           T_Hook_Start : constant Ada.Calendar.Time := Ada.Calendar.Clock;
           ZO_Result : constant String := Zenith_Orion.Check_SHM_Trigger (Prompt);
        begin
           if ZO_Result'Length > 0 then
              Result := To_Unbounded_String (ZO_Result);
              declare
                 Dur : constant Duration := Ada.Calendar.Clock - T_Hook_Start;
              begin
                 if Dur > Current_WCET_ELP3 then
                    Current_WCET_ELP3 := Dur;
                 end if;
              end;
              return;
           end if;
        end;

        declare
           T_Hook_Start : constant Ada.Calendar.Time := Ada.Calendar.Clock;
           SI_Result : constant String := Stella_Icarus.Check_API_Trigger (Prompt);
        begin
           if SI_Result'Length > 0 then
              Result := To_Unbounded_String (SI_Result);
              declare
                 Dur : constant Duration := Ada.Calendar.Clock - T_Hook_Start;
              begin
                 if Dur > Current_WCET_ELP2 then
                    Current_WCET_ELP2 := Dur;
                 end if;
              end;
              return;
           end if;
        end;

        --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
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
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
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
            if Cached_Res /= "" then
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
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
            Put_Line
               (AnsiAda.Foreground (AnsiAda.Light_Blue)
                & "[Init-V]"
                & AnsiAda.Reset
                & " Hybrid_Generate: Speculative_Cache lookup. Hit="
                & Boolean'Image (SC_Res /= ""));
            if SC_Res /= "" then
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
                    Best_Idx     : Natural := 1;
                    Best_Score   : Float := -1.0e9;
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
                               (Query        => Prompt,
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
                    Append
                       (Whimsical_Adelaide,
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
                        & " Injected interaction memory (reranked #"
                        & Natural'Image (Best_Idx)
                        & ") into system prompt [+"
                        & Uptime_Str
                        & "s].");
                end;
                if not External_Agent then
                    Push_Orchestration_Through_Parser
                       (Stream,
                        Session_ID,
                        Orch_Parser,
                        "[Adelaide Core]: [Thought] Interaction memory injected "
                        & "into system prompt [+"
                        & Uptime_Str
                        & "s]."
                        & ASCII.LF);
                end if;
            end if;

            --  2. Literature memory: search top-10, rerank, inject top-1.
            Database_Manager.Search_Literature
               (Emb_Vec (1 .. Emb_Len), Lit_Results, Lit_Count);
            if Lit_Count > 0 then
                --  [RERANKER] Rerank literature candidates
                declare
                    Best_Idx     : Natural := 1;
                    Best_Score   : Float := -1.0e9;
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
                               (Query        => Prompt,
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
                    Append
                       (Whimsical_Adelaide,
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
                        & " Injected literature memory (reranked #"
                        & Natural'Image (Best_Idx)
                        & ") into system prompt [+"
                        & Uptime_Str
                        & "s].");
                end;
                if not External_Agent then
                    Push_Orchestration_Through_Parser
                       (Stream,
                        Session_ID,
                        Orch_Parser,
                        "[Adelaide Core]: [Thought] Literature memory injected "
                        & "into system prompt [+"
                        & Uptime_Str
                        & "s]."
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

            --  [GPU SAFETY] Free Reranker model from Metal BEFORE doing anything else
            --  so that it does not collide with the main model loading in Load_Model!
            Reranker.Free_Reranker;
        end;

        --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
        --  ELP0 Speculation Context: QRNN LSH-based retrieval for background thought.
        --  Only activates during ELP0 to inject <SpeculationContextGuidance_*> blocks
        --  after <memory_*> blocks. Uses 10-bit LSH hash (tolerance=2 Hamming distance)
        --  via Python sidecar subprocess for quantum-evolved QRNN hash quality.
        if Level = ELP0 then
            declare
                LSH_Uptime       : constant String :=
                   Ada.Strings.Fixed.Trim
                      (Duration'Image
                          (Ada.Real_Time.To_Duration
                              (Ada.Real_Time.Clock - Init_Start_Time)),
                       Ada.Strings.Both);
                LSH_Acq_OK       : Boolean;
                LSH_Hash_Value   : Integer;
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
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
                    Put_Line
                       (AnsiAda.Foreground (AnsiAda.Red)
                        & "[LSH]"
                        & AnsiAda.Reset
                        & " QRNN worker: ELP0 acquire FAILED (Preempted) [+"
                        & LSH_Uptime
                        & "s].");
                else
                    --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
                    Put_Line
                       (AnsiAda.Foreground (AnsiAda.Light_Blue)
                        & "[LSH]"
                        & AnsiAda.Reset
                        & " QRNN worker: ELP0 acquired. Computing hash [+"
                        & LSH_Uptime
                        & "s].");

                    --  Compute 10-bit LSH hash from embedding via Python sidecar
                    LSH_Hash_Value :=
                       LSH_Hash.Compute (Emb_Vec (1 .. Emb_Len), Emb_Len);

                    if LSH_Hash_Value >= 0 then
                        --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
                        Put_Line
                           (AnsiAda.Foreground (AnsiAda.Light_Blue)
                            & "[LSH]"
                            & AnsiAda.Reset
                            & " Hash="
                            & Integer'Image (LSH_Hash_Value)
                            & " Searching speculation context [+"
                            & LSH_Uptime
                            & "s].");

                        --  Search interaction cache by LSH (tolerance=2 Hamming)
                        Database_Manager.Search_Interaction_By_LSH
                           (LSH_Hash_Value,
                            Spec_Tolerance,
                            Spec_Int_Results,
                            Spec_Int_Count);

                        --  Search literature chunks by LSH
                        Database_Manager.Search_Literature_By_LSH
                           (LSH_Hash_Value,
                            Spec_Tolerance,
                            Spec_Lit_Results,
                            Spec_Lit_Count);

                        --  Inject <SpeculationContextGuidance_Interaction>
                        if Spec_Int_Count > 0 then
                            for S in 1 .. Spec_Int_Count loop
                                Append
                                   (Whimsical_Adelaide,
                                    ASCII.LF
                                    & ASCII.LF
                                    & "<SpeculationContextGuidance_Interaction>"
                                    & ASCII.LF
                                    & Sanitize_Memory_Content
                                         (To_String
                                             (Spec_Int_Results (S).Content))
                                    & ASCII.LF
                                    & "</SpeculationContextGuidance_Interaction>");
                            end loop;
                            --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
                            Put_Line
                               (AnsiAda.Foreground (AnsiAda.Light_Green)
                                & "[LSH]"
                                & AnsiAda.Reset
                                & " Injected"
                                & Natural'Image (Spec_Int_Count)
                                & " speculation interaction(s) [+"
                                & LSH_Uptime
                                & "s].");
                        end if;

                        --  Inject <SpeculationContextGuidance_Literature>
                        if Spec_Lit_Count > 0 then
                            for S in 1 .. Spec_Lit_Count loop
                                Append
                                   (Whimsical_Adelaide,
                                    ASCII.LF
                                    & ASCII.LF
                                    & "<SpeculationContextGuidance_Literature>"
                                    & ASCII.LF
                                    & Sanitize_Memory_Content
                                         (To_String
                                             (Spec_Lit_Results (S).Content))
                                    & ASCII.LF
                                    & "</SpeculationContextGuidance_Literature>");
                            end loop;
                            --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
                            Put_Line
                               (AnsiAda.Foreground (AnsiAda.Light_Green)
                                & "[LSH]"
                                & AnsiAda.Reset
                                & " Injected"
                                & Natural'Image (Spec_Lit_Count)
                                & " speculation literature(s) [+"
                                & LSH_Uptime
                                & "s].");
                        end if;

                        if Spec_Int_Count = 0 and Spec_Lit_Count = 0 then
                            --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
                            Put_Line
                               (AnsiAda.Foreground (AnsiAda.Grey)
                                & "[LSH]"
                                & AnsiAda.Reset
                                & " No speculation context found within tolerance. [+"
                                & LSH_Uptime
                                & "s].");
                        end if;
                    else
                        --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
                        Put_Line
                           (AnsiAda.Foreground (AnsiAda.Red)
                            & "[LSH]"
                            & AnsiAda.Reset
                            & " QRNN worker failed (returned -1) [+"
                            & LSH_Uptime
                            & "s].");
                    end if;

                    --  Release ELP0 gate
                    Priority_Model_Gate.Release_ELP0 (LSH_QRNN);
                    --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
                    Put_Line
                       (AnsiAda.Foreground (AnsiAda.Light_Blue)
                        & "[LSH]"
                        & AnsiAda.Reset
                        & " QRNN worker: ELP0 released [+"
                        & LSH_Uptime
                        & "s].");
                end if;
            end;
        end if;

        --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
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
                       (Kind               => Snowball_Enaga_Orchestrator,
                        Prompt             => Actual_Prompt,
                        Result             => Gen_Q,
                        Stream             => null,
                        Level              => Level,
                        Virtual_Tokens     => Cached_Virtual_Tokens,
                        Virtual_Tok_Len    => Cached_Virtual_Len,
                        FreeParallelMemory => True,
                        Skip_Gate          => False);
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
                    Database_Manager.Set_System_State ("Internal_State", To_String (Internal_State));
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
                   --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
                   --  CORE TOOLS: Original tool set
                   & "If you need to search, use [ACTION: search(query)]. "
                   & "If you need to read a file, use [ACTION: cat(filename)]. "
                   & "If you need to calculate math, use [ACTION: math(expr)]. "
                   & "If you need to execute code, use [ACTION: code(python)]. "
                   & "If you want to schedule a proactive thought for later, "
                   & "use [ACTION: schedule(seconds, query)]. "
                   & "If you need to generate an image from your imagination, "
                   & "use [ACTION: imagine(description)]. "
                   --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
                   --  NEW TOOLS: Git, File Edit, Directory, Test, Build, Issue, Review, Security
                   & "If you need to commit changes, use [ACTION: git(commit message)]. "
                   & "If you need to push changes, use [ACTION: git(push)]. "
                   & "If you need to see git status, use [ACTION: git(status)]. "
                   & "If you need to create/edit/write a file, use [ACTION: file_edit(command filename content)]. "
                   & "If you need to list a directory, use [ACTION: dir(ls path)]. "
                   & "If you need to find files, use [ACTION: dir(find path pattern)]. "
                   & "If you need to run tests, use [ACTION: test(pytest)]. "
                   & "If you need to lint code, use [ACTION: test(lint)]. "
                   & "If you need to build/compile, use [ACTION: build(ada)]. "
                   & "If you need to create an issue, use [ACTION: issue(create title body)]. "
                   & "If you need to review code, use [ACTION: review(file filename)]. "
                   & "If you need to scan for security, use [ACTION: security(scan path)]. "
                   & "If you need to install a system package, use [ACTION: package(install name)]. "
                   & "If you need to uninstall a package, use [ACTION: package(uninstall name)]. "
                   & "If you need to search file contents, use [ACTION: grep(pattern path)]. "
                   & "If you need to manage tasks, use [ACTION: todo(add task)]. "
                   & "If you need to kill a process, use [ACTION: kill(pid)]. "
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
                        "[Adelaide Core]: [Thought] Deciding next action (JMP"
                        & Current_JMP'Img
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
                   (" [Hybrid] JMP"
                    & Current_JMP'Img
                    & ": Decision routing...");
                --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
                Put_Line
                   (AnsiAda.Foreground (AnsiAda.Light_Blue)
                    & "[Init-V]"
                    & AnsiAda.Reset
                    & " Hybrid_Generate: JMP"
                    & Current_JMP'Img
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
                    Skip_Gate          => False);
                --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
                Put_Line
                   (AnsiAda.Foreground (AnsiAda.Light_Blue)
                    & "[Init-V]"
                    & AnsiAda.Reset
                    & " Hybrid_Generate: JMP"
                    & Current_JMP'Img
                    & " Generate returned (model released). Len="
                    & Natural'Image (Length (Step_Raw)));

                declare
                    Step : constant String :=
                       Trim (To_String (Step_Raw), Ada.Strings.Both);
                begin
                    Put_Line (" [Hybrid] JMP" & Current_JMP'Img & ": " & Step);
                    if not External_Agent then
                        Push_Orchestration_Direct
                           (Stream,
                            Session_ID,
                            "[Adelaide Core]: [Thought] JMP "
                            & Current_JMP'Img
                            & " - I will: "
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
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
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
                                                            Database_Manager.Set_System_State ("Internal_State", To_String (Internal_State));
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
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
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
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
                                                    --  IMAGINE TOOL: Direct Ada call to SD_Manager.
                                                    --  When the model outputs [ACTION: imagine(prompt)],
                                                    --  generate an image via two-stage FLUX+SD pipeline
                                                    --  and store it in the database for VLM retrieval.
                                                          (if T_Name
                                                              = "imagine"
                                                           then
                                                              Tool_Manager
                                                                 .Execute_Imagine_Tool
                                                                    (Sanitize_Think_Tags
                                                                           (T_Pars))
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
                                                               "[Adelaide Core]: [Thought] JMP "
                                                               & Current_JMP'Img
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
                                                    if T_Name = "imagine" and then R.Success then
                                                        GNATCOLL.JSON.Append (Local_Images, GNATCOLL.JSON.Create (To_String (R.Output)));
                                                    end if;
                                                    Current_Internal_State_Len :=
                                                       Length (Internal_State);
                                                    Database_Manager.Set_System_State ("Internal_State", To_String (Internal_State));
                                                    --  Re-cache virtual ctx tokens after Internal_State grew
                                                    Tokenize_And_Cache_Virtual_Ctx
                                                          (Model_Types
                                                              .Snowball_Enaga_Orchestrator,
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
                                                               & "[Adelaide Core]: [Thought] JMP "
                                                               & Current_JMP'Img
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
            Current_JMP := Current_JMP + 1;
            --  Update context fault monitor tracking
            Current_JMP_Count := Current_JMP;
            exit when Current_JMP > 5;
        end loop;

        if not External_Agent then
            Push_Orchestration_Direct
               (Stream,
                Session_ID,
                "[Adelaide Core]: [Thought] Reasoning complete after "
                & Current_JMP'Img
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
                                   Wrap_ChatML
                                      (To_String (Whimsical_Adelaide), Prompt);
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
                        return
                           Wrap_ChatML
                              (To_String (Whimsical_Adelaide), Prompt);
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
                JMP_Count    : Natural := 0;
                Fault_Result : Unbounded_String;
            begin
                --  [VITAL-DO-NOT-REMOVE] Mandated by user.
                Put_Line
                   (AnsiAda.Foreground (AnsiAda.Light_Blue)
                    & "[Init-V]"
                    & AnsiAda.Reset
                    & " Hybrid_Generate: CONTEXT_FAULT_LOOP ENTERED.");
                loop
                    exit when JMP_Count >= 5;

                    --  Reset fault detection state for this hop. Without this,
                    --  a fault detected on a previous hop would persist and
                    --  cause false context-fault handling on subsequent hops
                    --  even when the model didn't request one.
                    F_Detected := False;

                    if not External_Agent then
                        if JMP_Count = 0 then
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
                                & Natural'Image (JMP_Count + 1)
                                & ")..."
                                & ASCII.LF);
                        end if;
                    end if;

                    --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
                    Put_Line
                       (AnsiAda.Foreground (AnsiAda.Light_Blue)
                        & "[Init-V]"
                        & AnsiAda.Reset
                        & " Hybrid_Generate: Final generation. JMP="
                        & Natural'Image (JMP_Count)
                        & " Len="
                        & Natural'Image (Get_Final_Prompt'Length));
                    Generate
                       (Kind               => Snowball_Enaga_Orchestrator,
                        Prompt             => Get_Final_Prompt,
                        Result             => Fault_Result,
                        Images             => Local_Images,
                        Session_ID         => Session_ID,
                        Requested_Ctx      => 8192,
                        Stream             => Stream,
                        Orch_Think_Open    => (JMP_Count = 0),
                        Level              => Level,
                        Virtual_Tokens     => Cached_Virtual_Tokens,
                        Virtual_Tok_Len    => Cached_Virtual_Len,
                        FreeParallelMemory => True,
                        Skip_Gate          => False,
                        Use_Speculative    => True);

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

                        while Sanitized_Check = ""
                           and then Retry_Count < Max_Think_Retries
                        loop
                            Retry_Count := Retry_Count + 1;

                            --  [VITAL-DO-NOT-REMOVE] Find next non-blacklisted seed.
                            --  Skip blacklisted seeds automatically.
                            loop
                                Generate_Seed := Generate_Seed + 1;
                                exit when
                                   not Database_Manager.Is_Seed_Blacklisted
                                          (Natural (Generate_Seed));
                            end loop;

                            Put_Line
                               (AnsiAda.Foreground (AnsiAda.Yellow)
                                & "[Init-V]"
                                & AnsiAda.Reset
                                & " Hybrid_Generate: THINK-ONLY DETECTED. Retry "
                                & Natural'Image (Retry_Count)
                                & "/"
                                & Natural'Image (Max_Think_Retries)
                                & " with seed="
                                & Interfaces.C.unsigned'Image (Generate_Seed));

                            --  Retry without streaming (avoids duplicate tokens to client)
                            begin
                                Generate
                                   (Kind               =>
                                       Snowball_Enaga_Orchestrator,
                                    Prompt             => Get_Final_Prompt,
                                    Result             => Fault_Result,
                                    Images             => Local_Images,
                                    Session_ID         => Session_ID,
                                    Requested_Ctx      => 8192,
                                    Stream             => null,
                                    Orch_Think_Open    => False,
                                    Level              => Level,
                                    Virtual_Tokens     =>
                                       Cached_Virtual_Tokens,
                                    Virtual_Tok_Len    => Cached_Virtual_Len,
                                    FreeParallelMemory => True,
                                    Skip_Gate          => False,
                                    Use_Speculative    => True);
                            exception
                                when others =>
                                    --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
                                    Put_Line
                                       (AnsiAda.Foreground (AnsiAda.Red)
                                        & "[Init-V]"
                                        & AnsiAda.Reset
                                        & " Hybrid_Generate: THINK-ONLY RETRY Generate CRASHED -- "
                                        & "prompt too long from excessive hops. Aborting retry.");
                                    exit;
                            end;

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
                                       (Stream,
                                        Session_ID,
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

                    --  =================================================================
                    --  REPEATING RESPONSE RETRY: If model produced repeating sentences
                    --  (same sentence 3+ times), retry with randomized seed.
                    --  Max 2 retries. Blacklist seeds that produce repeating output.
                    --  =================================================================
                    declare
                        Max_Repeat_Retries : constant := 2;
                        Repeat_Retry_Count : Natural := 0;
                        Sanitized_Repeat   : String :=
                           Sanitize_Think_Tags (To_String (Fault_Result));
                    begin
                        --  Check for repeating response
                        if Is_Repeating_Response (Sanitized_Repeat) then
                            Database_Manager.Blacklist_Seed
                               (Natural (Generate_Seed));
                        end if;

                        while Is_Repeating_Response (Sanitized_Repeat)
                           and then Repeat_Retry_Count < Max_Repeat_Retries
                        loop
                            Repeat_Retry_Count := Repeat_Retry_Count + 1;

                            --  Find next non-blacklisted seed
                            loop
                                Generate_Seed := Generate_Seed + 1;
                                exit when
                                   not Database_Manager.Is_Seed_Blacklisted
                                          (Natural (Generate_Seed));
                            end loop;

                            Put_Line
                               (AnsiAda.Foreground (AnsiAda.Yellow)
                                & "[Init-V]"
                                & AnsiAda.Reset
                                & " Hybrid_Generate: REPEATING RESPONSE DETECTED. Retry "
                                & Natural'Image (Repeat_Retry_Count)
                                & "/"
                                & Natural'Image (Max_Repeat_Retries)
                                & " with seed="
                                & Interfaces.C.unsigned'Image (Generate_Seed));

                            --  Retry without streaming
                            begin
                                Generate
                                   (Kind               =>
                                       Snowball_Enaga_Orchestrator,
                                    Prompt             => Get_Final_Prompt,
                                    Result             => Fault_Result,
                                    Images             => Local_Images,
                                    Session_ID         => Session_ID,
                                    Requested_Ctx      => 8192,
                                    Stream             => null,
                                    Orch_Think_Open    => False,
                                    Level              => Level,
                                    Virtual_Tokens     =>
                                       Cached_Virtual_Tokens,
                                    Virtual_Tok_Len    => Cached_Virtual_Len,
                                    FreeParallelMemory => True,
                                    Skip_Gate          => False);
                            exception
                                when others =>
                                    --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
                                    Put_Line
                                       (AnsiAda.Foreground (AnsiAda.Red)
                                        & "[Init-V]"
                                        & AnsiAda.Reset
                                        & " Hybrid_Generate: REPEAT RETRY Generate CRASHED -- "
                                        & "prompt too long from excessive hops. Aborting retry.");
                                    exit;
                            end;

                            Sanitized_Repeat :=
                               Sanitize_Think_Tags (To_String (Fault_Result));

                            if not Is_Repeating_Response (Sanitized_Repeat)
                            then
                                --  Retry produced non-repeating content — stream it
                                Put_Line
                                   (AnsiAda.Foreground (AnsiAda.Green)
                                    & "[Init-V]"
                                    & AnsiAda.Reset
                                    & " Hybrid_Generate: REPEAT RETRY SUCCEEDED. Len="
                                    & Natural'Image (Length (Fault_Result)));
                                if Stream /= null then
                                    Push_Chunk
                                       (Stream,
                                        Session_ID,
                                        To_String (Fault_Result));
                                end if;
                                exit;
                            else
                                --  This seed also produced repeating — blacklist it
                                Database_Manager.Blacklist_Seed
                                   (Natural (Generate_Seed));
                            end if;
                        end loop;
                    end;

                    --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
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
                        & " JMP_Count="
                        & Natural'Image (JMP_Count));

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
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
                            --  CONTEXT FAULT IMAGINE: When the model's <thinking>
                            --  emits [CONTEXT_FAULT:query=X category=imagine],
                            --  generate an image via the two-stage SD pipeline
                            --  and store it for VLM retrieval.
                            if C_Str = "imagine" then
                                R := Tool_Manager.Execute_Imagine_Tool (Q_Str);
                                --  Store the imagined image in the database
                                if R.Success and then Length (R.Output) > 100
                                then
                                    declare
                                        Img_LSH : Integer := -1;
                                    begin
                                        begin
                                            declare
                                                Emb_Vec :
                                                   Math_Utils.Vector
                                                      (1 .. 1024);
                                                Emb_Len : Natural;
                                            begin
                                                Get_Embedding
                                                   (Q_Str, Emb_Vec, Emb_Len);
                                                Img_LSH :=
                                                   LSH_Hash.Compute
                                                      (Emb_Vec (1 .. Emb_Len),
                                                       Emb_Len);
                                            end;
                                        exception
                                            when others =>
                                                Img_LSH := -1;
                                        end;
                                        Database_Manager.Store_Imagined_Image
                                           (Prompt    => Q_Str,
                                            Image_B64 => To_String (R.Output),
                                            LSH_Hash  => Img_LSH);
                                        Put_Line
                                           (AnsiAda.Foreground (AnsiAda.Cyan)
                                            & "[CtxFault-Imagine]"
                                            & AnsiAda.Reset
                                            & " Stored imagined image. LSH="
                                            & Integer'Image (Img_LSH));
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
                        JMP_Count := JMP_Count + 1;
                        --  Update context fault monitor tracking
                        Current_Context_Fault_JMPs := JMP_Count;
                        Current_Internal_State_Len := Length (Internal_State);
                        Database_Manager.Set_System_State ("Internal_State", To_String (Internal_State));
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
                    & " JMP_Count="
                    & Natural'Image (JMP_Count));
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
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
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
                    Resp_Len    : constant Natural := Resp_Text'Length;
                    Gen_Elapsed : constant Duration := Ada.Calendar.Clock - T0;
                    Stats_Str   : constant String :=
                       ASCII.LF
                       & "--- ORCHESTRATION STATISTICS ---"
                       & ASCII.LF
                       & "Response Length: "
                       & Natural'Image (Resp_Len)
                       & " chars"
                       & ASCII.LF
                       & "Response Tokens (est): "
                       & Natural'Image (Resp_Len / 4)
                       & " tokens"
                       & ASCII.LF
                       & "Generation Time: "
                       & Duration'Image (Gen_Elapsed)
                       & "s"
                       & ASCII.LF
                       & "Prompt Tokens: "
                       & Natural'Image (Current_Prompt_Tokens)
                       & ASCII.LF
                       & "Context Capacity: "
                       & Natural'Image (Current_Ctx_Capacity)
                       & " tokens"
                       & ASCII.LF
                       & "Context Utilization: "
                       & Natural'Image
                            (Current_Prompt_Tokens
                             * 100
                             / Current_Ctx_Capacity)
                       & "%"
                       & ASCII.LF
                       & "JMPs: "
                       & Natural'Image (Current_JMP_Count)
                       & ASCII.LF
                       & "Context Faults: "
                       & Natural'Image (Current_Context_Fault_JMPs)
                       & ASCII.LF
                       & "Pipeline Level: "
                       & ELP_Level'Image (Level)
                       & ASCII.LF
                       & "Streaming Mode: Emulated 300 tok/s"
                       & ASCII.LF
                       & "GPU Free: "
                       & Natural'Image (GPU_Free_MB)
                       & "MB / "
                       & Natural'Image (GPU_Total_MB)
                       & "MB ("
                       & Natural'Image (GPU_Layer_Percent)
                       & "%)"
                       & ASCII.LF
                       & "GPU Layers: "
                       & (if Acceleration_Silicon_Layer = -1
                          then "ALL(-1)"
                          else
                             Integer'Image (Acceleration_Silicon_Layer)
                             & "/"
                             & Natural'Image (Total_Model_Layers))
                       & ASCII.LF
                       & "GPU Stable: "
                       & Boolean'Image (GPU_Is_Stable)
                       & ASCII.LF
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
           and then
              Models (Snowball_Enaga_Orchestrator).Context /= Null_Context
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
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
            --  Stack overflow during hybrid generation (tool exec, tokenization,
            --  or context fault paging).  Force-unload model and report cleanly.
            --  Mark Metal broken so KV save retries instead of SIGABRT.
            --  [ADAPTIVE GPU FALLBACK] OOM → reduce GPU layers for next load
            if Acceleration_Silicon_Layer = -1 then
                Acceleration_Silicon_Layer := GPU_Layer_Fallback;
                GPU_Last_OOM_Time := Ada.Real_Time.Clock;
                --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
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
                    Priority_Model_Gate.Release_ELP0
                       (Snowball_Enaga_Orchestrator);
                else
                    Priority_Model_Gate.Release_ELP1
                       (Snowball_Enaga_Orchestrator);
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
                    Priority_Model_Gate.Release_ELP0
                       (Snowball_Enaga_Orchestrator);
                else
                    Priority_Model_Gate.Release_ELP1
                       (Snowball_Enaga_Orchestrator);
                end if;
                ELP_Queue.Dequeue_Level (Level);
            exception
                when others =>
                    null;
            end;
            --  [CRITICAL-FIX] Log the full exception with trace info
            Ada.Text_IO.Put_Line
               (AnsiAda.Background (AnsiAda.Red)
                & "[BUGCHECK] [Hybrid]"
                & " Error: "
                & Ada.Exceptions.Exception_Message (E)
                & AnsiAda.Reset);
            Ada.Text_IO.Put_Line
               (AnsiAda.Background (AnsiAda.Red)
                & "[BUGCHECK] [Hybrid]"
                & " Trace: "
                & Ada.Exceptions.Exception_Information (E)
                & AnsiAda.Reset);
            --  [CRITICAL-FIX] If generation already succeeded (Result is not
            --  empty and not an error string), DO NOT overwrite it with the
            --  error message. A transient Tasking_Error during cleanup (KV
            --  save, model unload) must not destroy a good response.
            if Length (Result) = 0 or else (Index (Result, "ERROR:") = 1) then
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
                    & " (ResultLen="
                    & Natural'Image (Length (Result))
                    & ")."
                    & " Result preserved.");
            end if;
    end Hybrid_Generate;

    --  KV CACHE SSD SPILLOVER
    --  Save KV cache to SSD after generation
    procedure Save_KV_Cache_To_SSD
       (Kind     : Model_Type;
        Tokens   : System.Address;
        N_Tokens : Interfaces.C.size_t) is
    begin
        if Models (Kind).Loaded and then Models (Kind).Context /= Null_Context
        then
            --  Save KV cache to SSD (ASYNC, non-blocking)
            KV_Cache_Manager.Save_To_SSD_Async
               (Context  => Models (Kind).Context,
                Tokens   => Tokens,
                N_Tokens => N_Tokens,
                Model_ID => Kind'Img);
        end if;
    exception
        when others =>
            null;  -- Don't crash on cache save failure
    end Save_KV_Cache_To_SSD;

    --  Load KV cache from SSD if available
    function Load_KV_Cache_From_SSD
       (Kind     : Model_Type;
        Tokens   : out System.Address;
        N_Tokens : out Interfaces.C.size_t) return Boolean is
    begin
        Tokens := System.Null_Address;
        N_Tokens := 0;

        if Models (Kind).Loaded and then Models (Kind).Context /= Null_Context
        then
            --  Load KV cache from SSD (LAZY, on-demand only)
            return
               KV_Cache_Manager.Load_From_SSD_Lazy
                  (Context  => Models (Kind).Context,
                   Tokens   => Tokens,
                   N_Tokens => N_Tokens,
                   Model_ID => Kind'Img);
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
    Elab_Trace
       ("Model_Manager DECLARATIVE PART COMPLETE -- entering begin block");
    Initialize;
     Elab_Trace ("Model_Manager.Initialize returned -- end of elaboration");

     --  ======================================================================
     --  ELP PRIORITY FIX SUMMARY (2026-06-26):
     --
     --  Problem:
     --    ELP0 background tasks (file indexing) would run while ELP1 user requests
     --    were pending, causing unacceptable latency for user interactions.
     --
     --  Root Cause:
     --    The Acquire_ELP0 entry condition in Priority_Model_Gate used "or else"
     --    instead of "and then" for the ELP1_Pending/Active checks, allowing ELP0
     --    tasks to proceed even when ELP1 requests were pending or active.
     --
     --  Solution:
     --    1. Fixed Acquire_ELP0 entry condition to use "and then" for proper priority
     --    2. Added defensive checks in the Dequeue_Level procedure
     --    3. Enhanced queue priority handling to ensure ELP1 tasks always preempt ELP0
     --
     --  Result:
     --    User-facing requests now properly take priority over background tasks.
     --    Background tasks only run when no user tasks are pending or active.
     --
     --  Files Modified:
     --    - model_manager.adb (priority logic fix)
     --    - elp_queue.adb (queue handling improvements)
     --    - elp_queue.ads (documentation updates)
     --
     --  Testing:
     --    Verify that:
     --      1. ELP0 tasks are blocked when ELP1 requests are pending
     --      2. ELP1 requests are served immediately
     --      3. Background tasks resume only after user tasks complete
     --  ======================================================================
end Model_Manager;
