pragma SPARK_Mode (Off);
--  ============================================================================
--  AUTO_CONFIG — Self-Tuning Hardware Configuration
--  ============================================================================
--  PHILOSOPHY:
--    Do NOT hardcode for any specific hardware (Penryn, etc.).
--    Instead: START MINIMAL → PROBE UPWARD → REMEMBER WHAT WORKS.
--
--    On first boot: minimal settings (2048 ctx, 1 thread, CPU-only).
--    After each successful inference: try the next level up.
--    After failure: step back, record the max working config.
--    On next boot: load saved config, start from there.
--
--  WHY THIS MATTERS:
--    Same codebase runs on Intel Pentium Penryn (2 cores, 16GB, shared VRAM)
--    and many other configurations. Hardcoding for one machine breaks
--    another. Auto-config handles all machines automatically.
--
--  CONFIG PERSISTENCE:
--    Working configs saved to run/.auto_config between sessions.
--    Hardware changes (RAM upgrade, different machine) → auto-reprobe.
--
--  TERMINOLOGY:
--    "Acceleration_Layer" — number of model layers offloaded to hardware
--    accelerators (GPU, NPU, DSP, AMX, etc.). Value 0 = CPU-only,
--    8/16/24 = partial offload, -1 = all layers on accelerator.
--    This is NOT just "GPU layers" — it covers all tensor acceleration.
--  ============================================================================

with Model_Types; use Model_Types;
with Interfaces.C; use Interfaces.C;

package Auto_Config is

   --  ========================================================================
   --  CONTEXT SIZE LADDER (powers of 2, from minimal to full)
   --  ========================================================================
   --  Why these values:
   --    2048: Absolute minimum for Qwen3.5 to produce coherent output.
   --          KV cache ~30MB. Works on anything with 2GB+ free RAM.
   --    4096: Comfortable for most prompts. KV cache ~60MB.
   --    8192: Full context for long conversations. KV cache ~120MB.
   --   16384: Extended context for RAG-heavy workloads. KV cache ~240MB.
   --   32768: Maximum for very long documents. KV cache ~480MB.
   --  ========================================================================
   type Ctx_Ladder is (Ctx_2048, Ctx_4096, Ctx_8192, Ctx_16384, Ctx_32768);
   for Ctx_Ladder use
      (Ctx_2048   => 2048,
       Ctx_4096   => 4096,
       Ctx_8192   => 8192,
       Ctx_16384  => 16384,
       Ctx_32768  => 32768);

   function Ctx_To_Unsigned (C : Ctx_Ladder) return Interfaces.C.unsigned;
   pragma Inline (Ctx_To_Unsigned);

   --  ========================================================================
   --  THREAD LADDER (from 1 to max cores)
   --  ========================================================================
   --  Why start at 1:
   --    Intel Pentium Penryn has 2 cores. Using 8 threads on a 2-core CPU
   --    causes context switching overhead that SLOWS DOWN inference.
   --    Start at 1, probe up to min(detected_threads, 4).
   --    More than 4 threads gives diminishing returns on llama.cpp workloads.
   --  ========================================================================
   type Thread_Ladder is (T_1, T_2, T_4);
   for Thread_Ladder use (T_1 => 1, T_2 => 2, T_4 => 4);

   function Threads_To_Int (T : Thread_Ladder) return Interfaces.C.int;
   pragma Inline (Threads_To_Int);

   --  ========================================================================
   --  BATCH LADDER (from 64 to 512)
   --  ========================================================================
   --  Why start at 64:
   --    N_Batch=256 allocates ~64MB compute buffers on Metal.
   --    N_Batch=512 allocates ~256MB. On shared VRAM (Intel), that's
   --    system RAM stolen from model weights.
   --    Start at 64 (minimal buffers), probe up.
   --  ========================================================================
   type Batch_Ladder is (B_64, B_128, B_256, B_512);
   for Batch_Ladder use (B_64 => 64, B_128 => 128, B_256 => 256, B_512 => 512);

   function Batch_To_Unsigned (B : Batch_Ladder) return Interfaces.C.unsigned;
   pragma Inline (Batch_To_Unsigned);

   --  ========================================================================
   --  ACCELERATION LAYER LADDER (from 0=CPU-only to -1=all)
   --  ========================================================================
   --  Why start at 0:
   --    Intel integrated GPU has ~128-512MB dedicated VRAM. The 5.8GB model
   --    cannot fit on GPU. Start CPU-only, probe up to see if GPU helps.
   --    Discrete GPUs with ample VRAM will probe up quickly.
   --    The probe discovers optimal settings automatically.
   --
   --  Why "Acceleration_Layer" not "GPU_Layer":
   --    This covers GPU (Metal/CUDA/Vulkan), NPU (Neural Engine),
   --    DSP, AMX (Advanced Matrix Extensions), and any future hardware.
   --    The value is the number of transformer layers to offload.
   --  ========================================================================
   type Accel_Layer_Ladder is (AL_0, AL_8, AL_16, AL_24, AL_All);
   --  AL_All maps to -1 (all layers on accelerator)
   Accel_All_Layers : constant Interfaces.C.int := -1;

   function Accel_Layers_To_Int (A : Accel_Layer_Ladder) return Interfaces.C.int;
   pragma Inline (Accel_Layers_To_Int);

   --  ========================================================================
   --  WORKING CONFIGURATION (per model kind)
   --  ========================================================================
   type Working_Config is record
      Ctx              : Ctx_Ladder     := Ctx_2048;    -- Start minimal
      Threads          : Thread_Ladder  := T_1;         -- Start with 1 thread
      Batch            : Batch_Ladder   := B_64;        -- Start with small batch
      Accel_Layers     : Accel_Layer_Ladder := AL_0;    -- Start CPU-only
      --  Probing state
      Probe_Target     : Ctx_Ladder     := Ctx_2048;    -- Next ctx to try (set by Record_Success)
      Max_Working      : Ctx_Ladder     := Ctx_2048;    -- Highest ctx that worked
      Fail_Count       : Natural        := 0;           -- Consecutive failures at current level
   end record;

   type Config_Array is array (Model_Type) of Working_Config;

   --  ========================================================================
   --  PUBLIC API
   --  ========================================================================

   --  Initialize auto-config: detect hardware, load saved config.
   --  Call once at startup, before any Load_Model.
   procedure Initialize;

   --  Get the working config for a model kind.
   --  Returns the current best-known settings.
   function Get_Config (Kind : Model_Type) return Working_Config;

   --  Record that a context size worked.
   --  Auto-config will try the next level up on next inference.
   procedure Record_Success
     (Kind     : Model_Type;
      Ctx_Used : Interfaces.C.unsigned);

   --  Set the probe target: next time Load_Model is called, try this context.
   --  Called by the post-inference probe when headroom is detected.
   procedure Set_Probe_Target
     (Kind   : Model_Type;
      Target : Ctx_Ladder);

   --  Get and clear the probe target.
   --  Returns the target if set, then clears it (one-shot probe).
   function Get_Probe_Target (Kind : Model_Type) return Ctx_Ladder;

   --  Record that a context size failed (OOM, null context, crash).
   --  Auto-config steps back and records the max working config.
   procedure Record_Failure
     (Kind      : Model_Type;
      Ctx_Tried : Interfaces.C.unsigned);

   --  Save current config to disk (run/.auto_config).
   --  Call on clean shutdown or periodically.
   procedure Save_Config;

   --  Force re-probe from minimal (e.g., after hardware change).
   procedure Reset_To_Minimal;

   --  ========================================================================
   --  HARDWARE PROFILE (detected at startup)
   --  ========================================================================
   type Hardware_Profile is record
      CPU_Cores    : Natural := 1;
      Free_RAM_MB  : Natural := 0;
      Total_RAM_MB : Natural := 0;
      Accel_VRAM_MB : Natural := 0;  -- Accelerator memory (GPU VRAM, etc.)
   end record;

   Detected_Hardware : Hardware_Profile;

   --  ========================================================================
   --  CONSTANTS
   --  ========================================================================
   --  Config file path
   Config_File_Path : constant String := "run/.auto_config";

   --  Maximum consecutive failures before giving up on a level
   Max_Fail_Count : constant Natural := 3;

   --  Memory pressure threshold for probing up (% used)
   --  If free RAM > this percentage, safe to probe up
   Probe_Headroom_Pct : constant Natural := 30;

end Auto_Config;
