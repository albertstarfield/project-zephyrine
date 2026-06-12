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
with Interfaces.C; use Interfaces.C;
with Interfaces.C.Strings; use Interfaces.C.Strings;
with Ada.Directories;
with Ada.Real_Time; use Ada.Real_Time;
with Ada.Unchecked_Conversion;
with Ada.Unchecked_Deallocation;
with Ada.Exceptions;
with Watchdog_Manager;
with Kratos;
with ELP_Queue;
with Speculative_Cache;
with System;

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
--  ===========================================================================

package body Model_Manager is
   use Streaming_Queue;

   function Llama_Batch_Get_One
     (T : System.Address; N : int) return Llama_Batch;
   pragma Import (C, Llama_Batch_Get_One, "llama_batch_get_one");

   task type WCET_Printer;
   task body WCET_Printer is
   begin
      loop
         delay 30.0;
         Ada.Text_IO.Put_Line (AnsiAda.Foreground (AnsiAda.Light_Red) &
                               "[WCET]" & AnsiAda.Reset &
                               " Pipeline: " &
                               Long_Long_Integer'Image (Long_Long_Integer (Current_WCET * 1_000_000_000)) & "ns | " &
                               "ELP0: " &
                               Long_Long_Integer'Image (Long_Long_Integer (Current_WCET_ELP0 * 1_000_000_000)) & "ns | " &
                               "ELP1: " &
                               Long_Long_Integer'Image (Long_Long_Integer (Current_WCET_ELP1 * 1_000_000_000)) & "ns | " &
                               "ELP2: " &
                               Long_Long_Integer'Image (Long_Long_Integer (Current_WCET_ELP2 * 1_000_000_000)) & "ns | " &
                               "ELP3: " &
                               Long_Long_Integer'Image (Long_Long_Integer (Current_WCET_ELP3 * 1_000_000_000)) & "ns");
      end loop;
   end WCET_Printer;

   Printer_Task : WCET_Printer;

   type Model_Record is record
      Model       : Llama_Model := Null_Model;
      Context     : Llama_Context := Null_Context;
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
           or else (On_Battery_State and Battery_Level < 80);
      end Should_Abort;

      function Is_ELP0_Owner (Kind : Model_Type) return Boolean is
      begin
         return Owner (Kind) = ELP0;
      end Is_ELP0_Owner;

      --  Barrier: ELP0 tasks block here until all ELP1 requests have completed.
      --  See Wait_For_ELP1_Idle spec in model_manager.ads for full explanation.
      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
      --  Verbose: prints guard state when an ELP0 task arrives.
      entry Wait_For_ELP1_Idle when (ELP1_Pending = 0 and
        ELP1_Active_Count = 0)
        and then (not On_Battery_State or else Battery_Level >= 80) is
      begin
         Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Init-V]" &
                   AnsiAda.Reset & "+" & Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) & "s Wait_For_ELP1_Idle GUARD PASSED" &
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
      Llama_Backend_Init;
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
      Requested_Ctx : Positive := 4096)
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

      --  [QUIRK-M10] GPU Contention & Metal Backend Analysis
      --  ======================================================================
      --  [VITAL-DO-NOT-REMOVE]
      --  ANALYSIS OF llama.cpp METAL FAILURE (Code 5):
      --  On M2 Pro, the Qwen3-Embedding model triggers MTLCommandBufferStatus-
      --  Error (Code 5) when run on the GPU. This is likely due to:
      --    1. Kernel Race: High-frequency indexing calls colliding with ELP1.
      --    2. Quantization Bug: Q8_0 embeddings occasionally trigger out-of-
      --       bounds access in specific llama.cpp Metal kernels.
      --    3. Unified Memory Pressure: Contention during large buffer swaps.
      --
      --  SOLUTION:
      --  By forcing N_Gpu_Layers := 0, we move embedding to the CPU. Since
      --  the model is only ~600MB, the CPU performance penalty is negligible
      --  (< 5ms), but stability is 100%. This preserves the GPU for the
      --  heavy 9B reasoning model.
      --  [NOTE] However, this will result in higher Power Consumption
      --  compared to GPU or ANE execution.
      if Kind = Qwen_Embedding then
         Put_Line ("[VITAL] Using CPU-only for Embedding to prevent " &
                   "Metal trap.");
         M_Params.N_Gpu_Layers := 0;
      else
         M_Params.N_Gpu_Layers := -1;
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
                  begin
                     Models (Kind).Model :=
                       Llama_Model_Load_From_File (Path_C, M_Params);
                  exception
                     when others =>
                        Put_Line ("[!] Exception caught in Ada during " &
                                  "Llama_Model_Load_From_File");
                        Models (Kind).Model := Null_Model;
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
         C_Params.Type_K := GGML_TYPE_F16;
         C_Params.Type_V := GGML_TYPE_F16;

         --  [NOTE] Disabling Flash Attention (Flash_Attn_Type := 0) was tested
         --  but did NOT fix the "WE RAN INTO A PROBLEM" (Trace/BPT trap: 5)
         --  crashes on the Metal backend. CPU-only override is the only fix.
         C_Params.Flash_Attn_Type := 0;

         C_Params.Abort_Callback := Llama_Abort_Callback'Address;
         C_Params.Abort_Callback_Data := Model_Refs (Kind)'Address;
         Models (Kind).Context :=
           Llama_Init_From_Model (Models (Kind).Model, C_Params);
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
         Llama_Free (Models (Kind).Context);
         Llama_Model_Free (Models (Kind).Model);
         Models (Kind).Context := Null_Context;
         Models (Kind).Model := Null_Model;
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
      type Token_Array is array (Positive range <>) of Llama_Token;
      type Token_Array_Access is access Token_Array;
      procedure Free_Tokens is new Ada.Unchecked_Deallocation
        (Token_Array, Token_Array_Access);
      Tokens   : Token_Array_Access;
      N_Toks   : int;
      Clean_P  : constant String := Sanitize_UTF8 (Prompt);
      Prompt_C : chars_ptr := New_String (Clean_P);

      --  Identify source for descriptive logging
      Source   : constant String :=
        (if Level = ELP0 then "Knowledge-Index" else "User-RAG");
   begin
      --  [VITAL-DO-NOT-REMOVE] Mandated by user.
      --  --[Debug] DO NOT REMOVE: Critical for diagnosing Tokenization crashes.
      Put_Line ("[Embedding-Debug] Input (" & Clean_P'Length'Img &
                " chars): " &
                (if Clean_P'Length > 60 then
                   Clean_P (Clean_P'First .. Clean_P'First + 57) & "..."
                 else Clean_P));
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

      Load_Model (Kind, Success, 1024);
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
      declare
         Batch_Size  : constant int :=
           int'Min (256, int (Models (Kind).Current_Ctx));
         Current_Pos : int := 0;
         Tokens_Left : int := N_Toks;
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
                  Put_Line (AnsiAda.Foreground (AnsiAda.Red) &
                            "[FATAL] GPU Backend Error (Code:" &
                            Dec_Result'Img & ")" & AnsiAda.Reset);
                  Put_Line ("[+] Waiting for GPU driver cooldown (2s)...");
                  delay 2.0;
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
               Tokens_Left := Tokens_Left - To_Decode;
               Current_Pos := Current_Pos + To_Decode;
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
      Append (Parser.Sanitize_Buffer, C);
      declare
         Buf : constant String := To_String (Parser.Sanitize_Buffer);
      begin
         if Buf = Think_Tag_A or else Buf = Think_Tag_B then
            Parser.Sanitize_Buffer := Null_Unbounded_String;
            Parser.In_Think_Block := True;
            return;
         elsif Buf = Close_Tag_A or else Buf = Close_Tag_B then
            Parser.Sanitize_Buffer := Null_Unbounded_String;
            Parser.In_Think_Block := False;
            if Parser.Orch_Think_Open then
               Parser.Orch_Think_Open := False;
            end if;
            return;
         elsif Buf = Resp_Tag then
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
         if Parser.In_Think_Block then
            declare
               Fault_Mark : constant String := "[CONTEXT_FAULT:";
               F_Pos      : constant Natural := Index (Buf, Fault_Mark);
            begin
               if F_Pos > 0 then
                  declare
                     Rest      : constant String :=
                       Buf (F_Pos + Fault_Mark'Length .. Buf'Last);
                     Close_Pos : constant Natural := Index (Rest, "]");
                  begin
                     if Close_Pos > 0 then
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
                     end if;
                  end;
               end if;
            end;
         end if;

         -- Stream content out, but SILENCE the think block entirely
         if not Parser.In_Think_Block then
            delay 0.0005;
            Push_Chunk (Stream, Session_ID, Buf);
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

   procedure Flush_Parser
     (Stream     : Streaming_Queue.Queue_Access;
      Session_ID : String;
      Parser     : in out Stream_Parser_State)
   is
   begin
      declare
         S_Str : constant String := To_String (Parser.Sanitize_Buffer);
      begin
         if S_Str /= "" then
            Push_Chunk (Stream, Session_ID, S_Str);
            Parser.Sanitize_Buffer := Null_Unbounded_String;
         end if;
      end;
      if Parser.Orch_Think_Open then
         --  Silently close orchestration thinking; tag is stripped by parser
         Parser.Orch_Think_Open := False;
      end if;
   end Flush_Parser;

   function Sanitize_Think_Tags (Text : String) return String is
      Res : Unbounded_String;
      I   : Positive := Text'First;
   begin
      while I <= Text'Last loop
         if I + 9 <= Text'Last and then Text (I .. I + 9) = "<thinking>" then
            --  Skip everything until closing </thinking>
            I := I + 10;
            while I <= Text'Last loop
               if I + 10 <= Text'Last and then
                 Text (I .. I + 10) = "</thinking>"
               then
                  I := I + 11;
                  exit;
               else
                  I := I + 1;
               end if;
            end loop;
         elsif I + 6 <= Text'Last and then Text (I .. I + 6) = "<think>" then
            --  Skip everything until closing </think>
            I := I + 7;
            while I <= Text'Last loop
               if I + 7 <= Text'Last and then Text (I .. I + 7) = "</think>" then
                  I := I + 8;
                  exit;
               else
                  I := I + 1;
               end if;
            end loop;
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
      Level           : ELP_Level := ELP1)
   is
      Success  : Boolean;
      Vocab    : Llama_Vocab;
      type Token_Array is array (Positive range <>) of Llama_Token;
      type Token_Array_Access is access Token_Array;
      procedure Free_Tokens is new Ada.Unchecked_Deallocation
        (Token_Array, Token_Array_Access);
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

      begin
         if Level = ELP0 then
            declare
               Acq_OK : Boolean;
            begin
               Priority_Model_Gate.Acquire_ELP0 (Kind) (Acq_OK);
               if not Acq_OK then
                  Result := To_Unbounded_String ("ERROR: Preempted");
                  Free (Prompt_C);
                  return;
               end if;
            end;
         else
            Priority_Model_Gate.Request_ELP1;
            Priority_Model_Gate.Acquire_ELP1 (Kind);
         end if;

         Load_Model (Kind, Success, Requested_Ctx);
         if not Success then
            if Level = ELP0 then
               Priority_Model_Gate.Release_ELP0 (Kind);
            else
               Priority_Model_Gate.Release_ELP1 (Kind);
            end if;
            Result := To_Unbounded_String ("ERROR: Load failed");
            Free (Prompt_C);
            return;
         end if;

         Models (Kind).In_Use := True;
         Models (Kind).Last_Used := Clock;

         --  Allocate token array based on actual context size
         Tokens := new Token_Array (1 .. Positive (Models (Kind).Current_Ctx));

         Vocab := Llama_Model_Get_Vocab (Models (Kind).Model);
         N_Toks := Llama_Tokenize
           (Vocab, Prompt_C, int (Clean_P'Length), Tokens.all'Address,
            int (Tokens.all'Length), True, True);
         Put_Line ("[Tokenize-Debug] Model:" & Kind'Img &
                   " Prompt_Len:" & Clean_P'Length'Img &
                   " N_Toks:" & N_Toks'Img);
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

      if Stream /= null then
         Flush_Parser (Stream, Session_ID, Parser);
      end if;

      Llama_Sampler_Free (Sampler);
      Free_Tokens (Tokens);
      Models (Kind).In_Use := False;

      if Level = ELP0 then
         Priority_Model_Gate.Release_ELP0 (Kind);
      else
         Priority_Model_Gate.Release_ELP1 (Kind);
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
         Result := To_Unbounded_String ("ERROR: Decode failed");
   end Generate;

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
   begin
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
         Push_Chunk (Stream, Session_ID,
           "[Adelaide Core]: [Thought] No cached response found, " &
           "starting fresh reasoning chain." & ASCII.LF);
         Push_Chunk (Stream, Session_ID,
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
            Push_Chunk (Stream, Session_ID,
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
                  Push_Chunk (Stream, Session_ID,
                    "[Adelaide Core]: [Thought] I'm still here and " &
                    "processing..." & ASCII.LF);
                  Last_Heartbeat := Now;
               end if;
               Model_Manager.Generate
                 (Kind            => Qwen_9B,
                  Prompt          => Actual_Prompt,
                  Result          => Gen_Q,
                  Stream          => null,
                  Level           => Level);
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
                  Push_Chunk (Stream, Session_ID,
                    "[Adelaide Core]: [Thought] Searching knowledge " &
                    "base for: " & Trim (Final_Q, Ada.Strings.Both) &
                    "..." & ASCII.LF);
                  Push_Chunk (Stream, Session_ID,
                    "[Adelaide Core]: [Thought] Found relevant context " &
                    "from knowledge base." & ASCII.LF);
               end if;
               Append
                 (Internal_State,
                  "[FACTUAL_DATA]: " & To_String (R.Output) & ASCII.LF);
               if not External_Agent then
                  Push_Chunk (Stream, Session_ID,
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
            Paging_Instr : constant String :=
              "Current Data: " & To_String (Internal_State);
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
               Push_Chunk (Stream, Session_ID,
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
                  Push_Chunk (Stream, Session_ID,
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
                null, False, Level);
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
                  Push_Chunk (Stream, Session_ID,
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
                                          Push_Chunk (Stream, Session_ID,
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
                                          Push_Chunk (Stream, Session_ID,
                                            "[Adelaide Core]: [Thought] " &
                                            "Running tool: " &
                                            Sanitize_Orchestration_Output
                                              (T_Name) & ASCII.LF);
                                       end if;
                                       Append
                                         (Internal_State,
                                          "[TOOL (" & T_Name & ")]: " &
                                          To_String (R.Output) & ASCII.LF);
                                       if not External_Agent then
                                          Push_Chunk (Stream, Session_ID,
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
         exit when Current_Hop > 5;
      end loop;

      if not External_Agent then
         Push_Chunk (Stream, Session_ID,
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
                           return Prefix & Sys_Tag & Whimsical_Adelaide &
                             ASCII.LF & "Fact-Check: " &
                             To_String (Internal_State) & ASCII.LF &
                             Prompt (First_Block .. Prompt'Last);
                        else
                           return Prefix & Sys_Tag & Whimsical_Adelaide &
                             ASCII.LF & Prompt (First_Block .. Prompt'Last);
                        end if;
                     end;
                  elsif First_Block = 1 then
                     if Length (Internal_State) > 0 then
                        return Sys_Tag & Whimsical_Adelaide & ASCII.LF &
                          "Fact-Check: " & To_String (Internal_State) &
                          ASCII.LF & Prompt;
                     else
                        return Sys_Tag & Whimsical_Adelaide & ASCII.LF &
                          Prompt;
                     end if;
                  else
                     if Length (Internal_State) > 0 then
                        return Wrap_ChatML (Whimsical_Adelaide,
                          Prompt & ASCII.LF & "Fact-Check: " &
                          To_String (Internal_State));
                     else
                        return Wrap_ChatML (Whimsical_Adelaide, Prompt);
                     end if;
                  end if;
               end;
            else
               if Length (Internal_State) > 0 then
                  return Wrap_ChatML (Whimsical_Adelaide,
                    "User: " & Prompt & ASCII.LF &
                    "Fact-Check: " & To_String (Internal_State));
               else
                  return Wrap_ChatML (Whimsical_Adelaide, Prompt);
               end if;
            end if;
         end Get_Final_Prompt;
      begin
         --  CONTEXT FAULTING LOOP
         declare
            F_Detected   : Boolean := False;
            F_Query      : Unbounded_String;
            F_Category   : Unbounded_String;
            Hop_Count    : Natural := 0;
            Fault_Result : Unbounded_String;
         begin
            loop
               exit when Hop_Count >= 5;

               if not External_Agent then
                  if Hop_Count = 0 then
                     Push_Chunk (Stream, Session_ID,
                       "[Adelaide Core]: [Thought] Starting reasoning " &
                       "chain..." & ASCII.LF);
                  else
                     Push_Chunk (Stream, Session_ID,
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
                   Level           => Level);
                --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
                Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Init-V]" &
                          AnsiAda.Reset & " Hybrid_Generate: Final Generate returned. Len=" &
                          Natural'Image (Length (Fault_Result)));
                F_Detected := False;

               if F_Detected then
                  declare
                     Q_Str : constant String := To_String (F_Query);
                     C_Str : constant String := To_String (F_Category);
                     R     : Tool_Manager.Tool_Result;
                  begin
                     if not External_Agent then
                        Push_Chunk (Stream, Session_ID,
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

                     if not External_Agent then
                        Push_Chunk (Stream, Session_ID,
                          "[Adelaide Core]: [Thought] Context loaded for: " &
                          Q_Str & ASCII.LF);
                     end if;
                  end;
                  Hop_Count := Hop_Count + 1;
               else
                  Current_Response := Fault_Result;
                  exit;
               end if;
            end loop;
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
            Push_Chunk (Stream, Session_ID,
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
            Push_Chunk (Stream, Session_ID,
              "[Adelaide Core]: [Thought] Self-assessment: " &
              Score'Img & "/10" & ASCII.LF);
         end if;
         Ada.Text_IO.Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) &
                               "[Quality Score] " & AnsiAda.Reset &
                               "Score: " & Score'Img & "/10 | " &
                               "Session: " & Session_ID);
      end;

      if not External_Agent then
         declare
            Model_Thinking : constant String :=
              Extract_Think_Content (To_String (Current_Response));
         begin
            if Model_Thinking /= "" then
               Push_Chunk (Stream, Session_ID, Model_Thinking & ASCII.LF);
            end if;
         end;
         Push_Chunk (Stream, Session_ID, ASCII.LF & "</think>" & ASCII.LF);
      end if;

      if not External_Agent and then Stream /= null then
         declare
            Resp_Text    : constant String :=
              Sanitize_Think_Tags (To_String (Current_Response));
            Chunk_Size   : constant Positive := 16;
            Sim_TPS      : constant Float := 300.0;
            Delay_Chunk  : constant Duration :=
              Duration (Float (Chunk_Size) / Sim_TPS);
            Pos          : Natural := 1;
         begin
            Ada.Text_IO.Put_Line
              ("[Adelaide] Simulating ~300 tok/s streaming for " &
               Resp_Text'Length'Img & " chars...");
            while Pos <= Resp_Text'Length loop
               declare
                  Chunk_End : constant Natural :=
                    Natural'Min (Pos + Chunk_Size - 1, Resp_Text'Length);
                  Chunk     : constant String := Resp_Text (Pos .. Chunk_End);
               begin
                  delay Delay_Chunk;
                  Push_Chunk (Stream, Session_ID, Chunk);
                  Pos := Chunk_End + 1;
               end;
            end loop;
            Push_Chunk (Stream, Session_ID, ASCII.LF & "");
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
