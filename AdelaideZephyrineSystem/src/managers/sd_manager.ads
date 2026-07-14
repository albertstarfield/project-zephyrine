pragma SPARK_Mode (Off);
-- c_binding: Stable Diffusion C FFI
--  ============================================================================
--  SD_MANAGER — Two-stage image generation with FreeParallelMemory
--  ============================================================================
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
--
--  ARCHITECTURE (from project-zephyrine imagination_worker.py):
--    STAGE 1: FLUX Schnell Q2_K (sparse, fast, low quality)
--      - Load FLUX context (diffusion + clip_l + t5xxl + VAE)
--      - Generate sparse image (2-4 steps, CFG 1.0)
--      - FreeParallelMemory: Free FLUX context, unload from GPU
--
--    STAGE 2: SD Refinement (img2img upscale, high quality)
--      - Load SD refinement context (SD 1.5 model)
--      - Add noise to Stage 1 output (strength ~0.4)
--      - Refine image (dpmpp2mv2, 8+ steps)
--      - FreeParallelMemory: Free refinement context, unload from GPU
--
--  FREEPARALLELMEMORY PATTERN (LM Studio-style one-component-at-a-time):
--    Each model load MUST call FreeParallelMemory which:
--    1. Frees the C context (Free_SD_Ctx)
--    2. Clears the Ada reference (Ctx := Null_SD_Ctx)
--    3. Logs the unload with [Uptime]+Xs timestamps
--
--    WHY: GPU memory is finite. FLUX Q2_K (~4GB) + SD refinement (~1.9GB)
--    = ~5.9GB total. Cannot have both loaded simultaneously on 9B-class VRAM.
--    Must unload FLUX before loading refinement, and vice versa.
--
--  MEMORY BUDGET:
--    FLUX diffusion:  ~4GB (Q2_K GGUF)
--    FLUX t5xxl:      ~2.9GB (Q4_0 GGUF)
--    FLUX clip_l:     ~0.25GB (safetensors)
--    FLUX VAE:        ~0.34GB (safetensors)
--    SD refinement:   ~1.9GB (Q8_0 GGUF)
--    Total if both loaded: ~9.4GB (TOO MUCH — must serialize)
--  ============================================================================

with SD_Interface; use SD_Interface;
with Ada.Real_Time; use Ada.Real_Time;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;

package SD_Manager is

   --  ============================================================================
   --  GLOBAL STATE
   --  ============================================================================
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
   --  These track the currently loaded SD context.
   --  Only ONE can be loaded at a time (FreeParallelMemory pattern).

   --  Start time for [Uptime]+Xs logging
   Init_Start_Time : Ada.Real_Time.Time;
   --  Set to True during initialization
   Is_Initialized  : Boolean := False;

   --  Currently loaded FLUX context (Stage 1)
   Flux_Ctx        : SD_Ctx := Null_SD_Ctx;
   --  Currently loaded refinement context (Stage 2)
   Refiner_Ctx     : SD_Ctx := Null_SD_Ctx;

   --  Model file paths (set during Initialize)
   Flux_Diffusion_Path : access String := null;
   Flux_Clip_L_Path    : access String := null;
   Flux_T5XXL_Path     : access String := null;
   Flux_VAE_Path       : access String := null;
   Refiner_Model_Path  : access String := null;

   --  ============================================================================
   --  INITIALIZATION
   --  ============================================================================

   --  Initialize SD Manager with model paths
   --  Sets Init_Start_Time, stores paths, does NOT load any models
   procedure Initialize
     (Flux_Diffusion : String;
      Flux_Clip_L    : String;
      Flux_T5XXL     : String;
      Flux_VAE       : String;
      Refiner_Model  : String);

   --  ============================================================================
   --  STAGE 1: FLUX CONTEXT
   --  ============================================================================

   --  Load FLUX context (Stage 1)
   --  FreeParallelMemory: Unloads refinement context first if loaded
   procedure Load_Flux_Context;

   --  Free FLUX context (FreeParallelMemory)
   --  Calls Free_SD_Ctx, clears Flux_Ctx, logs unload
   procedure Free_Flux_Context;

   --  ============================================================================
   --  STAGE 2: REFINEMENT CONTEXT
   --  ============================================================================

   --  Load refinement context (Stage 2)
   --  FreeParallelMemory: Unloads FLUX context first if loaded
   procedure Load_Refiner_Context;

   --  Free refinement context (FreeParallelMemory)
   --  Calls Free_SD_Ctx, clears Refiner_Ctx, logs unload
   procedure Free_Refiner_Context;

   --  ============================================================================
   --  TWO-STAGE GENERATION PIPELINE
   --  ============================================================================

   --  Generate image with two-stage pipeline
   --  Stage 1: FLUX sparse (2-4 steps) → Stage 2: SD refinement (8+ steps)
   --  FreeParallelMemory between stages (unload FLUX, load refinement)
   --  Returns Base64-encoded PNG image data
   procedure Generate_Two_Stage
     (Prompt         : String;
      Width          : Integer := 1024;
      Height         : Integer := 1024;
      Seed           : Long_Long_Integer := -1;
      --  FLUX Stage 1 params
      Flux_Steps     : Integer := 4;
      Flux_Cfg       : Float := 1.0;
      --  Refinement Stage 2 params
      Refine_Enabled : Boolean := True;
      Refine_Steps   : Integer := 8;
      Refine_Strength: Float := 0.4;
      --  Output
      Image_B64      : out Ada.Strings.Unbounded.Unbounded_String;
      Error_Msg      : out Ada.Strings.Unbounded.Unbounded_String);

   --  ============================================================================
   --  CLEANUP
   --  ============================================================================

   --  Free all loaded contexts (shutdown)
   procedure Free_All;

end SD_Manager;
