pragma SPARK_Mode (Off);
--  ============================================================================
--  SD_INTERFACE — Ada FFI bindings for stable-diffusion.cpp
--  ============================================================================
--  This package provides Ada bindings to the stable-diffusion.cpp C library,
--  which implements FLUX, Stable Diffusion 1.5/2.x/SDXL image generation.
--
--  The C library exposes a clean C API via stable-diffusion.h. We map:
--    - Enums (rng_type_t, sample_method_t, etc.) to Ada enumeration types
--    - Structs (sd_image_t, sd_ctx_params_t, etc.) to Ada records
--    - Functions (new_sd_ctx, generate_image, etc.) via pragma Import
--
--  All functions use the C calling convention and link to libstable_diffusion.a
--  built from the stable-diffusion.cpp repository.
--
--  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  ============================================================================

with Interfaces.C; use Interfaces.C;
with Interfaces.C.Strings; use Interfaces.C.Strings;
with System;

package SD_Interface is

   --  ============================================================================
   --  ENUMERATION TYPES (matching C enum definitions exactly)
   --  ============================================================================
   --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
   --  These must match the integer values in the C header exactly.
   --  Ada enums are 0-indexed by default, which matches C enum convention.

   --  RNG type: random number generator backend
   type RNG_Type_T is (
      STD_Default_RNG,   --  0: Standard C++ default RNG
      CUDA_RNG,          --  1: CUDA GPU RNG (for NVIDIA)
      CPU_RNG            --  2: CPU-based RNG
   );
   pragma Convention (C, RNG_Type_T);

   --  Sample method: the denoising algorithm used during generation
   type Sample_Method_T is (
      Euler,              --  0: Euler method (fast, good for FLUX)
      Euler_A,            --  1: Euler Ancestral (more creative)
      Heun,               --  2: Heun's method
      DPM2,               --  3: DPM2
      DPMPP2S_A,          --  4: DPM++ 2S Ancestral
      DPMPP2M,            --  5: DPM++ 2M
      DPMPP2Mv2,          --  6: DPM++ 2M v2 (good for refinement)
      IPNDM,              --  7: IPNDM
      IPNDM_V,            --  8: IPNDM V
      LCM,                --  9: Latent Consistency Model
      DDIM_Trailing,      -- 10: DDIM Trailing
      TCD,                -- 11: TCD
      Res_MultiStep,      -- 12: Res MultiStep
      Res_2S,             -- 13: Res 2S
      ER_SDE,             -- 14: ER SDE
      Euler_CFG_PP,       -- 15: Euler CFG++
      Euler_A_CFG_PP,     -- 16: Euler A CFG++
      Euler_GE            -- 17: Euler GE
   );
   pragma Convention (C, Sample_Method_T);

   --  Scheduler: noise schedule for the diffusion process
   type Scheduler_T is (
      Discrete,           --  0
      Karras,             --  1
      Exponential,        --  2
      AYS,                --  3
      GITS,               --  4
      SGM_Uniform,        --  5
      Simple,             --  6
      Smoothstep,         --  7
      KL_Optimal,         --  8
      LCM,                --  9
      Bong_Tangent,       -- 10
      LTX2,               -- 11
      Logit_Normal        -- 12
   );
   pragma Convention (C, Scheduler_T);

   --  Prediction type: what the model predicts during denoising
   type Prediction_T is (
      EPS_Pred,           --  0: Predict epsilon (noise)
      V_Pred,             --  1: Predict v
      EDM_V_Pred,         --  2: EDM v prediction
      Flow_Pred,          --  3: Flow prediction
      FLUX_Flow_Pred,     --  4: FLUX-specific flow prediction
      FLUX2_Flow_Pred     --  5: FLUX2 flow prediction
   );
   pragma Convention (C, Prediction_T);

   --  Data type for model weights (maps to ggml_type)
   type SD_Type_T is (
      SD_F32,             --  0
      SD_F16,             --  1
      SD_Q4_0,            --  2
      SD_Q4_1,            --  3
      SD_Q5_0,            --  6 (note: 4,5 removed)
      SD_Q5_1,            --  7
      SD_Q8_0,            --  8
      SD_Q8_1,            --  9
      SD_Q2_K,            -- 10
      SD_Q3_K,            -- 11
      SD_Q4_K,            -- 12
      SD_Q5_K,            -- 13
      SD_Q6_K,            -- 14
      SD_Q8_K             -- 15
   );
   pragma Convention (C, SD_Type_T);

   --  Log level for the library's internal logging
   type SD_Log_Level_T is (
      SD_Log_Debug,       --  0
      SD_Log_Info,        --  1
      SD_Log_Warn,        --  2
      SD_Log_Error        --  3
   );
   pragma Convention (C, SD_Log_Level_T);

   --  Preview mode for intermediate image previews
   type Preview_T is (
      Preview_None,       --  0
      Preview_Proj,       --  1
      Preview_TAE,        --  2
      Preview_VAE         --  3
   );
   pragma Convention (C, Preview_T);

   --  LoRA apply mode
   type LoRA_Apply_Mode_T is (
      LoRA_Apply_Auto,        --  0
      LoRA_Apply_Immediately, --  1
      LoRA_Apply_At_Runtime   --  2
   );
   pragma Convention (C, LoRA_Apply_Mode_T);

   --  VAE format type
   type SD_VAE_Format_T is (
      SD_VAE_Format_Auto, -- -1 (but Ada enum starts at 0, we handle this)
      SD_VAE_Format_FLUX, --  0
      SD_VAE_Format_SD3,  --  1
      SD_VAE_Format_FLUX2 --  2
   );
   pragma Convention (C, SD_VAE_Format_T);

   --  ============================================================================
   --  STRUCT TYPES (matching C struct layout exactly)
   --  ============================================================================
   --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
   --  WARNING: Field order MUST match the C header exactly.
   --  Ada records are laid out in declaration order, same as C structs.
   --  Use pragma Convention (C, ...) to ensure C-compatible layout.

   --  SD_Image: Output image data (width x height x channels, RGBA)
   --  This is what generate_image() returns.
   type SD_Image is record
      Width   : unsigned;    -- Image width in pixels
      Height  : unsigned;    -- Image height in pixels
      Channel : unsigned;    -- Number of channels (3=RGB, 4=RGBA)
      Data    : System.Address;  -- Pointer to pixel data (uint8_t*)
   end record;
   pragma Convention (C, SD_Image);

   type SD_Image_Access is access all SD_Image;

   --  SD_Embedding: Named embedding file (e.g., Textual Inversion)
   type SD_Embedding is record
      Name : chars_ptr;   -- Embedding name
      Path : chars_ptr;   -- Path to embedding file
   end record;
   pragma Convention (C, SD_Embedding);

   type SD_Embedding_Access is access all SD_Embedding;

   --  ============================================================================
   --  PARAMETER STRUCTS
   --  ============================================================================
   --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
   --  These are the main parameter structs passed to the C functions.
   --  Field order is CRITICAL — must match C header exactly.

   --  SD_Tiling_Params: Tiled VAE decode settings (for large images)
   type SD_Tiling_Params is record
      Enabled          : int;      -- bool in C (0/1)
      Temporal_Tiling  : int;      -- bool in C (0/1)
      Tile_Size_X      : int;
      Tile_Size_Y      : int;
      Target_Overlap   : interfaces.C.C_float;
      Rel_Size_X       : interfaces.C.C_float;
      Rel_Size_Y       : interfaces.C.C_float;
      Extra_Tiling_Args: chars_ptr;
   end record;
   pragma Convention (C, SD_Tiling_Params);

   --  SD_Sample_Params: Sampling configuration
   type SD_Sample_Params is record
      Txt_Cfg              : interfaces.C.C_float;  -- Text CFG scale
      Img_Cfg              : interfaces.C.C_float;  -- Image CFG scale
      Distilled_Guidance   : interfaces.C.C_float;  -- FLUX distilled guidance
      -- SLG params (skip layer guidance)
      SLG_Layers           : System.Address;   -- int* layers
      SLG_Layer_Count      : size_t;
      SLG_Layer_Start      : interfaces.C.C_float;
      SLG_Layer_End        : interfaces.C.C_float;
      SLG_Scale            : interfaces.C.C_float;
      Scheduler            : Scheduler_T;
      Sample_Method        : Sample_Method_T;
      Sample_Steps         : int;
      Eta                  : interfaces.C.C_float;
      Shifted_Timestep     : int;
      Custom_Sigmas        : System.Address;   -- float*
      Custom_Sigmas_Count  : int;
      Flow_Shift           : interfaces.C.C_float;
      Extra_Sample_Args    : chars_ptr;
   end record;
   pragma Convention (C, SD_Sample_Params);

   --  SD_PM_Params: PhotoMaker parameters
   type SD_PM_Params is record
      ID_Images       : System.Address;  -- sd_image_t*
      ID_Images_Count : int;
      ID_Embed_Path   : chars_ptr;
      Style_Strength  : interfaces.C.C_float;
   end record;
   pragma Convention (C, SD_PM_Params);

   --  SD_PuLID_Params: PuLID face identity parameters
   type SD_PuLID_Params is record
      ID_Embedding_Path : chars_ptr;
      ID_Weight         : interfaces.C.C_float;
   end record;
   pragma Convention (C, SD_PuLID_Params);

   --  SD_Cache_Params: Cache settings for加速生成
   type SD_Cache_Params is record
      Mode                      : int;  -- sd_cache_mode_t
      Reuse_Threshold           : interfaces.C.C_float;
      Start_Percent             : interfaces.C.C_float;
      End_Percent               : interfaces.C.C_float;
      Error_Decay_Rate          : interfaces.C.C_float;
      Use_Relative_Threshold    : int;  -- bool
      Reset_Error_On_Compute    : int;  -- bool
      Fn_Compute_Blocks         : int;
      Bn_Compute_Blocks         : int;
      Residual_Diff_Threshold   : interfaces.C.C_float;
      Max_Warmup_Steps          : int;
      Max_Cached_Steps          : int;
      Max_Continuous_Cached_Steps: int;
      TaylorSeer_N_Derivatives  : int;
      TaylorSeer_Skip_Interval  : int;
      SCM_Mask                  : chars_ptr;
      SCM_Policy_Dynamic        : int;  -- bool
      Spectrum_W                : interfaces.C.C_float;
      Spectrum_M                : int;
      Spectrum_Lam              : interfaces.C.C_float;
      Spectrum_Window_Size      : int;
      Spectrum_Flex_Window      : interfaces.C.C_float;
      Spectrum_Warmup_Steps     : int;
      Spectrum_Stop_Percent     : interfaces.C.C_float;
   end record;
   pragma Convention (C, SD_Cache_Params);

   --  SD_Hires_Params: Hi-res fix / upscale settings
   type SD_Hires_Params is record
      Enabled            : int;  -- bool
      Upscaler           : int;  -- sd_hires_upscaler_t
      Model_Path         : chars_ptr;
      Scale              : interfaces.C.C_float;
      Target_Width       : int;
      Target_Height      : int;
      Steps              : int;
      Denoising_Strength : interfaces.C.C_float;
      Upscale_Tile_Size  : int;
      Custom_Sigmas      : System.Address;  -- float*
      Custom_Sigmas_Count: int;
   end record;
   pragma Convention (C, SD_Hires_Params);

   --  SD_LoRA: LoRA adapter reference
   type SD_LoRA is record
      Is_High_Noise : int;  -- bool
      Multiplier    : interfaces.C.C_float;
      Path          : chars_ptr;
   end record;
   pragma Convention (C, SD_LoRA);

   type SD_LoRA_Access is access all SD_LoRA;

   --  ============================================================================
   --  MAIN PARAMETER STRUCT: sd_img_gen_params_t
   --  ============================================================================
   --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
   --  This is the master struct for image generation.
   --  Field order matches C header: sd_img_gen_params_t exactly.

   type SD_Img_Gen_Params is record
      --  LoRA adapters
      LoRas                : SD_LoRA_Access;
      LoRA_Count           : unsigned;
      --  Prompts
      Prompt               : chars_ptr;
      Negative_Prompt      : chars_ptr;
      Clip_Skip            : int;
      --  Init image (for img2img)
      Init_Image_Width     : unsigned;
      Init_Image_Height    : unsigned;
      Init_Image_Channel   : unsigned;
      Init_Image_Data      : System.Address;  -- uint8_t*
      --  Reference images (for IP-Adapter / PhotoMaker)
      Ref_Images           : System.Address;  -- sd_image_t*
      Ref_Images_Count     : int;
      Auto_Resize_Ref      : int;  -- bool
      Increase_Ref_Index   : int;  -- bool
      --  Mask image (for inpainting)
      Mask_Width           : unsigned;
      Mask_Height          : unsigned;
      Mask_Channel         : unsigned;
      Mask_Data            : System.Address;  -- uint8_t*
      --  Output dimensions
      Width                : int;
      Height               : int;
      --  Sampling
      Sample_Params        : SD_Sample_Params;
      Strength             : interfaces.C.C_float;  -- img2img strength
      Seed                 : Long_Long_Integer;      -- -1 for random
      Batch_Count          : int;
      --  ControlNet
      Control_Width        : unsigned;
      Control_Height       : unsigned;
      Control_Channel      : unsigned;
      Control_Data         : System.Address;  -- uint8_t*
      Control_Strength     : interfaces.C.C_float;
      --  PhotoMaker / PuLID
      PM_Params            : SD_PM_Params;
      PuLID_Params         : SD_PuLID_Params;
      --  Tiling / Cache / HiRes
      VAE_Tiling_Params    : SD_Tiling_Params;
      Cache                : SD_Cache_Params;
      Hires                : SD_Hires_Params;
   end record;
   pragma Convention (C, SD_Img_Gen_Params);

   --  ============================================================================
   --  SD_CTX_PARAMS_T: Context creation parameters
   --  ============================================================================
   --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
   --  This is the master struct for creating an SD context.
   --  Matches sd_ctx_params_t in the C header exactly.

   type SD_Ctx_Params is record
      --  Model paths
      Model_Path                    : chars_ptr;
      Clip_L_Path                   : chars_ptr;
      Clip_G_Path                   : chars_ptr;
      Clip_Vision_Path              : chars_ptr;
      T5XXL_Path                    : chars_ptr;
      LLM_Path                      : chars_ptr;
      LLM_Vision_Path               : chars_ptr;
      Diffusion_Model_Path          : chars_ptr;
      High_Noise_Diffusion_Model_Path : chars_ptr;
      Uncond_Diffusion_Model_Path   : chars_ptr;
      Embeddings_Connectors_Path    : chars_ptr;
      VAE_Path                      : chars_ptr;
      Audio_VAE_Path                : chars_ptr;
      TAESD_Path                    : chars_ptr;
      Control_Net_Path              : chars_ptr;
      --  Embeddings
      Embeddings                    : SD_Embedding_Access;
      Embedding_Count               : unsigned;
      --  Other model paths
      Photo_Maker_Path              : chars_ptr;
      PuLID_Weights_Path            : chars_ptr;
      Tensor_Type_Rules             : chars_ptr;
      --  Runtime settings
      N_Threads                     : int;
      Wtype                         : int;  -- sd_type_t
      RNG_Type                      : int;  -- rng_type_t
      Sampler_RNG_Type              : int;  -- rng_type_t
      Prediction                    : int;  -- prediction_t
      LoRA_Apply_Mode               : int;  -- lora_apply_mode_t
      --  Feature flags
      Enable_MMAP                   : int;  -- bool
      Flash_Attn                    : int;  -- bool
      Diffusion_Flash_Attn          : int;  -- bool
      TAE_Preview_Only              : int;  -- bool
      Diffusion_Conv_Direct         : int;  -- bool
      VAE_Conv_Direct               : int;  -- bool
      Circular_X                    : int;  -- bool
      Circular_Y                    : int;  -- bool
      Force_SDXL_VAE_Conv_Scale     : int;  -- bool
      Chroma_Use_Dit_Mask           : int;  -- bool
      Chroma_Use_T5_Mask            : int;  -- bool
      Chroma_T5_Mask_Pad            : int;
      Qwen_Image_Zero_Cond_T        : int;  -- bool
      --  VAE format
      VAE_Format                    : int;  -- sd_vae_format_t
      --  Memory management
      Max_Vram                      : chars_ptr;
      Stream_Layers                 : int;  -- bool
      Eager_Load                    : int;  -- bool
      --  Backend selection
      Backend                       : chars_ptr;
      Params_Backend                : chars_ptr;
      RPC_Servers                   : chars_ptr;
   end record;
   pragma Convention (C, SD_Ctx_Params);

   --  ============================================================================
   --  OPAQUE POINTER TYPE: sd_ctx_t
   --  ============================================================================
   --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
   --  sd_ctx_t is an opaque struct in C. We treat it as an Address.

   type SD_Ctx is new System.Address;
   Null_SD_Ctx : constant SD_Ctx := SD_Ctx (System.Null_Address);

   --  ============================================================================
   --  CALLBACK TYPES
   --  ============================================================================
   --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
   --  These are C function pointer types for logging and progress callbacks.

   --  Log callback: called by the library for each log message
   type SD_Log_Cb_T is access procedure
      (Level : SD_Log_Level_T;
       Text  : chars_ptr;
       Data  : System.Address);
   pragma Convention (C, SD_Log_Cb_T);

   --  Progress callback: called during generation for each step
   type SD_Progress_Cb_T is access procedure
      (Step  : int;
       Steps : int;
       Time  : interfaces.C.C_float;
       Data  : System.Address);
   pragma Convention (C, SD_Progress_Cb_T);

   --  ============================================================================
   --  C FUNCTION BINDINGS
   --  ============================================================================
   --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
   --  All functions use pragma Import (C, ...) to link to the C library.
   --  The string parameter is the exact symbol name in libstable_diffusion.a.

   --  --- Library info ---
   function SD_Version return chars_ptr;
   pragma Import (C, SD_Version, "sd_version");

   function SD_Commit return chars_ptr;
   pragma Import (C, SD_Commit, "sd_commit");

   function SD_Get_System_Info return chars_ptr;
   pragma Import (C, SD_Get_System_Info, "sd_get_system_info");

   function SD_Get_Num_Physical_Cores return int;
   pragma Import (C, SD_Get_Num_Physical_Cores, "sd_get_num_physical_cores");

   --  --- Logging ---
   procedure SD_Set_Log_Callback (Cb   : SD_Log_Cb_T;
                                   Data : System.Address);
   pragma Import (C, SD_Set_Log_Callback, "sd_set_log_callback");

   --  --- Progress ---
   procedure SD_Set_Progress_Callback (Cb   : SD_Progress_Cb_T;
                                        Data : System.Address);
   pragma Import (C, SD_Set_Progress_Callback, "sd_set_progress_callback");

   --  --- Context lifecycle ---
   function New_SD_Ctx (Params : access SD_Ctx_Params) return SD_Ctx;
   pragma Import (C, New_SD_Ctx, "new_sd_ctx");

   procedure Free_SD_Ctx (Ctx : SD_Ctx);
   pragma Import (C, Free_SD_Ctx, "free_sd_ctx");

   --  --- Context support queries ---
   function SD_Ctx_Supports_Image_Generation (Ctx : SD_Ctx) return int;
   pragma Import (C, SD_Ctx_Supports_Image_Generation, "sd_ctx_supports_image_generation");

   function SD_Ctx_Supports_Video_Generation (Ctx : SD_Ctx) return int;
   pragma Import (C, SD_Ctx_Supports_Video_Generation, "sd_ctx_supports_video_generation");

   --  --- Parameter initialization ---
   procedure SD_Ctx_Params_Init (Params : access SD_Ctx_Params);
   pragma Import (C, SD_Ctx_Params_Init, "sd_ctx_params_init");

   procedure SD_Img_Gen_Params_Init (Params : access SD_Img_Gen_Params);
   pragma Import (C, SD_Img_Gen_Params_Init, "sd_img_gen_params_init");

   procedure SD_Sample_Params_Init (Params : access SD_Sample_Params);
   pragma Import (C, SD_Sample_Params_Init, "sd_sample_params_init");

   --  --- Image generation (the main function) ---
   function Generate_Image
     (Ctx    : SD_Ctx;
      Params : access SD_Img_Gen_Params) return SD_Image_Access;
   pragma Import (C, Generate_Image, "generate_image");

   --  --- Cancel generation ---
   procedure SD_Cancel_Generation (Ctx  : SD_Ctx;
                                    Mode : int);
   pragma Import (C, SD_Cancel_Generation, "sd_cancel_generation");

   --  --- Free results ---
   procedure Free_SD_Images (Images : SD_Image_Access;
                             Count  : int);
   pragma Import (C, Free_SD_Images, "free_sd_images");

   --  --- PNG encoding (miniz/tdefl) ---
   --  Write raw image data to PNG in memory (returns malloc'd buffer, caller must free with mz_free)
   function Tdefl_Write_Image_To_PNG_File_In_Memory
     (PImage   : System.Address;
      W        : int;
      H        : int;
      Num_Chans: int;
      PLen_Out : access size_t) return System.Address;
   pragma Import (C, Tdefl_Write_Image_To_PNG_File_In_Memory, "tdefl_write_image_to_png_file_in_memory");

   --  Free buffer allocated by tdefl_write_image_to_png_file_in_memory
   procedure Mz_Free (P : System.Address);
   pragma Import (C, Mz_Free, "mz_free");

   --  --- Enum name lookups (for verbose logging) ---
   function SD_Type_Name (T : int) return chars_ptr;
   pragma Import (C, SD_Type_Name, "sd_type_name");

   function SD_RNG_Type_Name (T : int) return chars_ptr;
   pragma Import (C, SD_RNG_Type_Name, "sd_rng_type_name");

   function SD_Sample_Method_Name (M : int) return chars_ptr;
   pragma Import (C, SD_Sample_Method_Name, "sd_sample_method_name");

   function SD_Scheduler_Name (S : int) return chars_ptr;
   pragma Import (C, SD_Scheduler_Name, "sd_scheduler_name");

   function SD_Prediction_Name (P : int) return chars_ptr;
   pragma Import (C, SD_Prediction_Name, "sd_prediction_name");

   --  --- Default sample method/scheduler ---
   function SD_Get_Default_Sample_Method (Ctx : SD_Ctx) return int;
   pragma Import (C, SD_Get_Default_Sample_Method, "sd_get_default_sample_method");

   function SD_Get_Default_Scheduler (Ctx    : SD_Ctx;
                                       Method : int) return int;
   pragma Import (C, SD_Get_Default_Scheduler, "sd_get_default_scheduler");

   --  --- String conversion (for verbose logging) ---
   function SD_Ctx_Params_To_Str (Params : access SD_Ctx_Params) return chars_ptr;
   pragma Import (C, SD_Ctx_Params_To_Str, "sd_ctx_params_to_str");

   function SD_Img_Gen_Params_To_Str (Params : access SD_Img_Gen_Params) return chars_ptr;
   pragma Import (C, SD_Img_Gen_Params_To_Str, "sd_img_gen_params_to_str");

   function SD_Sample_Params_To_Str (Params : access SD_Sample_Params) return chars_ptr;
   pragma Import (C, SD_Sample_Params_To_Str, "sd_sample_params_to_str");

   --  ============================================================================
   --  VERBOSE LOGGING PROCEDURES (Ada-side helpers)
   --  ============================================================================
   --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
   --  These are Ada procedures that call the C functions above and format
   --  the output for verbose logging. They are defined in the .adb body.

   --  Print library version and commit hash
   procedure SD_Version_Info;

   --  Print system info (CPU cores, backend, etc.)
   procedure SD_System_Info;

   --  Log all fields of SD_Ctx_Params
   procedure Log_Context_Params (Params : access SD_Ctx_Params);

   --  Log all fields of SD_Img_Gen_Params
   procedure Log_Image_Gen_Params (Params : access SD_Img_Gen_Params);

   --  Log the result of generate_image()
   procedure Log_Generate_Result (Images      : SD_Image_Access;
                                   Count       : int;
                                   Gen_Duration: Duration);

   --  Log all available enum names from the C library
   procedure Log_All_Enum_Names;

end SD_Interface;
