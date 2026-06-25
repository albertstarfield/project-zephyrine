pragma SPARK_Mode (Off);
--  ============================================================================
--  SD_MANAGER — Body implementing two-stage image generation
--  ============================================================================
--  [DO NOT REMOVE, OR YOU WILL BE KILLED]

with Ada.Text_IO; use Ada.Text_IO;
with Ada.Real_Time; use Ada.Real_Time;
with Interfaces.C.Strings;
with System;

package body SD_Manager is

   --  ============================================================================
   --  HELPER: Uptime string for logging
   --  ============================================================================
   --  [DO NOT REMOVE, OR YOU WILL BE KILLED]

   function Uptime_String return String is
      use Ada.Real_Time;
      Elapsed : constant Time_Span := Clock - Init_Start_Time;
      Seconds : constant Integer := To_Duration (Elapsed);
   begin
      return "[" & Integer'Image (Seconds) & "s]";
   end Uptime_String;

   --  ============================================================================
   --  INITIALIZATION
   --  ============================================================================
   --  [DO NOT REMOVE, OR YOU WILL BE KILLED]

   procedure Initialize
     (Flux_Diffusion : String;
      Flux_Clip_L    : String;
      Flux_T5XXL     : String;
      Flux_VAE       : String;
      Refiner_Model  : String)
   is
      --  We need to store copies of the strings as access constants
      --  But since these are access constants, we store them as-is
      --  The caller must ensure the strings outlive this package
   begin
      Init_Start_Time := Clock;
      Is_Initialized := True;

      Put_Line
        (Uptime_String & " [SD-Manager] Initializing..."
         & " FLUX Diffusion=" & Flux_Diffusion
         & " Clip_L=" & Flux_Clip_L
         & " T5XXL=" & Flux_T5XXL
         & " VAE=" & Flux_VAE
         & " Refiner=" & Refiner_Model);

      --  Store paths (caller must ensure these remain valid)
      --  For simplicity, we store them as-is; in production you'd
      --  copy to Unbounded_String and convert back
      Flux_Diffusion_Path := new String'(Flux_Diffusion);
      Flux_Clip_L_Path    := new String'(Flux_Clip_L);
      Flux_T5XXL_Path     := new String'(Flux_T5XXL);
      Flux_VAE_Path       := new String'(Flux_VAE);
      Refiner_Model_Path  := new String'(Refiner_Model);

      Put_Line
        (Uptime_String & " [SD-Manager] Initialization complete.");
   end Initialize;

   --  ============================================================================
   --  STAGE 1: FLUX CONTEXT
   --  ============================================================================
   --  [DO NOT REMOVE, OR YOU WILL BE KILLED]

   procedure Load_Flux_Context is
      use Interfaces.C.Strings;
      Params : aliased SD_Ctx_Params;
   begin
      --  [FREE-PARALLEL-MEMORY] If refinement context is loaded, free it first
      if Refiner_Ctx /= Null_SD_Ctx then
         Put_Line
           (Uptime_String & " [SD-Manager] [FREE-PARALLEL-MEMORY]"
            & " Freeing refinement context before loading FLUX...");
         Free_Refiner_Context;
      end if;

      --  Check if already loaded
      if Flux_Ctx /= Null_SD_Ctx then
         Put_Line
           (Uptime_String & " [SD-Manager] FLUX context already loaded, skipping.");
         return;
      end if;

      Put_Line
        (Uptime_String & " [SD-Manager] [Stage-1] Loading FLUX context...");

      --  Initialize params struct
      SD_Ctx_Params_Init (Params'Access);

      --  Set model paths (C strings — Ada will manage lifetime)
      Params.Model_Path       := New_String (Flux_Diffusion_Path.all);
      Params.Diffusion_Model_Path := New_String (Flux_Diffusion_Path.all);
      Params.Clip_L_Path      := New_String (Flux_Clip_L_Path.all);
      Params.T5XXL_Path       := New_String (Flux_T5XXL_Path.all);
      Params.VAE_Path         := New_String (Flux_VAE_Path.all);

      --  Runtime settings for FLUX Schnell
      Params.N_Threads   := 4;
      Params.Enable_MMAP := 1;  -- True
      Params.Flash_Attn  := 1;  -- True
      Params.Prediction  := 4;  -- FLUX_Flow_Pred

      --  Log params before creation
      Log_Context_Params (Params'Access);

      --  Create context
      Flux_Ctx := New_SD_Ctx (Params'Access);

      if Flux_Ctx = Null_SD_Ctx then
         Put_Line
           (Uptime_String & " [SD-Manager] [ERROR] Failed to create FLUX context!");
         raise Program_Error with "FLUX context creation failed";
      end if;

      --  Free the C strings we allocated
      Free (Params.Model_Path);
      Free (Params.Diffusion_Model_Path);
      Free (Params.Clip_L_Path);
      Free (Params.T5XXL_Path);
      Free (Params.VAE_Path);

      Put_Line
        (Uptime_String & " [SD-Manager] [Stage-1] FLUX context loaded successfully.");
   end Load_Flux_Context;

   --  ============================================================================
   --  FREE FLUX CONTEXT (FreeParallelMemory)
   --  ============================================================================
   --  [DO NOT REMOVE, OR YOU WILL BE KILLED]

   procedure Free_Flux_Context is
   begin
      if Flux_Ctx = Null_SD_Ctx then
         Put_Line
           (Uptime_String & " [SD-Manager] FLUX context already null, nothing to free.");
         return;
      end if;

      Put_Line
        (Uptime_String & " [SD-Manager] [FREE-PARALLEL-MEMORY]"
         & " Freeing FLUX context...");

      Free_SD_Ctx (Flux_Ctx);
      Flux_Ctx := Null_SD_Ctx;

      Put_Line
        (Uptime_String & " [SD-Manager] [FREE-PARALLEL-MEMORY]"
         & " FLUX context freed. GPU memory released.");
   end Free_Flux_Context;

   --  ============================================================================
   --  STAGE 2: REFINEMENT CONTEXT
   --  ============================================================================
   --  [DO NOT REMOVE, OR YOU WILL BE KILLED]

   procedure Load_Refiner_Context is
      use Interfaces.C.Strings;
      Params : aliased SD_Ctx_Params;
   begin
      --  [FREE-PARALLEL-MEMORY] If FLUX context is loaded, free it first
      if Flux_Ctx /= Null_SD_Ctx then
         Put_Line
           (Uptime_String & " [SD-Manager] [FREE-PARALLEL-MEMORY]"
            & " Freeing FLUX context before loading refinement...");
         Free_Flux_Context;
      end if;

      --  Check if already loaded
      if Refiner_Ctx /= Null_SD_Ctx then
         Put_Line
           (Uptime_String & " [SD-Manager] Refinement context already loaded, skipping.");
         return;
      end if;

      Put_Line
        (Uptime_String & " [SD-Manager] [Stage-2] Loading refinement context...");

      --  Initialize params struct
      SD_Ctx_Params_Init (Params'Access);

      --  Set model path (SD 1.5 only needs the main model)
      Params.Model_Path := New_String (Refiner_Model_Path.all);

      --  Runtime settings for SD refinement
      Params.N_Threads   := 4;
      Params.Enable_MMAP := 1;  -- True
      Params.Flash_Attn  := 1;  -- True
      Params.Prediction  := 0;  -- EPS_Pred (standard SD)

      --  Log params before creation
      Log_Context_Params (Params'Access);

      --  Create context
      Refiner_Ctx := New_SD_Ctx (Params'Access);

      if Refiner_Ctx = Null_SD_Ctx then
         Put_Line
           (Uptime_String & " [SD-Manager] [ERROR] Failed to create refinement context!");
         raise Program_Error with "Refinement context creation failed";
      end if;

      --  Free the C string we allocated
      Free (Params.Model_Path);

      Put_Line
        (Uptime_String & " [SD-Manager] [Stage-2] Refinement context loaded successfully.");
   end Load_Refiner_Context;

   --  ============================================================================
   --  FREE REFINEMENT CONTEXT (FreeParallelMemory)
   --  ============================================================================
   --  [DO NOT REMOVE, OR YOU WILL BE KILLED]

   procedure Free_Refiner_Context is
   begin
      if Refiner_Ctx = Null_SD_Ctx then
         Put_Line
           (Uptime_String & " [SD-Manager] Refinement context already null, nothing to free.");
         return;
      end if;

      Put_Line
        (Uptime_String & " [SD-Manager] [FREE-PARALLEL-MEMORY]"
         & " Freeing refinement context...");

      Free_SD_Ctx (Refiner_Ctx);
      Refiner_Ctx := Null_SD_Ctx;

      Put_Line
        (Uptime_String & " [SD-Manager] [FREE-PARALLEL-MEMORY]"
         & " Refinement context freed. GPU memory released.");
   end Free_Refiner_Context;

   --  ============================================================================
   --  TWO-STAGE GENERATION PIPELINE
   --  ============================================================================
   --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
   --  This is the main entry point for image generation.
   --  Implements the two-stage pipeline from project-zephyrine:
   --    Stage 1: FLUX sparse → Stage 2: SD refinement

   procedure Generate_Two_Stage
     (Prompt         : String;
      Width          : Integer := 1024;
      Height         : Integer := 1024;
      Seed           : Long_Long_Integer := -1;
      Flux_Steps     : Integer := 4;
      Flux_Cfg       : Float := 1.0;
      Refine_Enabled : Boolean := True;
      Refine_Steps   : Integer := 8;
      Refine_Strength: Float := 0.4)
   is
      use Interfaces.C.Strings;
      Stage1_Start : Ada.Real_Time.Time;
      Stage1_Duration : Duration;
      Stage2_Start : Ada.Real_Time.Time;
      Stage2_Duration : Duration;
   begin
      Put_Line
        (Uptime_String & " [SD-Manager] === Two-Stage Generation ==="
         & " Prompt='" & Prompt (Prompt'First .. Integer'Min (Prompt'First + 49, Prompt'Last)) & "'"
         & " Size=" & Integer'Image (Width) & "x" & Integer'Image (Height));

      --  ====================================================================
      --  STAGE 1: FLUX Schnell sparse generation
      --  ====================================================================
      Stage1_Start := Clock;

      Put_Line
        (Uptime_String & " [SD-Manager] [Stage-1] Loading FLUX context...");
      Load_Flux_Context;

      --  Build generation params for FLUX
      declare
         Gen_Params : aliased SD_Img_Gen_Params;
         C_Prompt   : chars_ptr := New_String (Prompt);
         Images     : SD_Image_Access;
         Count      : int;
      begin
         SD_Img_Gen_Params_Init (Gen_Params'Access);

         Gen_Params.Prompt          := C_Prompt;
         Gen_Params.Negative_Prompt := Null_Ptr;
         Gen_Params.Width           := int (Width);
         Gen_Params.Height          := int (Height);
         Gen_Params.Seed            := Seed;
         Gen_Params.Batch_Count     := 1;

         --  FLUX Schnell settings
         Gen_Params.Sample_Params.Sample_Method := Euler;
         Gen_Params.Sample_Params.Sample_Steps  := int (Flux_Steps);
         Gen_Params.Sample_Params.Txt_Cfg       := interfaces.C.C_float (Flux_Cfg);
         Gen_Params.Sample_Params.Scheduler     := Simple;

         --  Log params
         Log_Image_Gen_Params (Gen_Params'Access);

         --  Generate
         Put_Line
           (Uptime_String & " [SD-Manager] [Stage-1] Generating with FLUX ("
            & Integer'Image (Flux_Steps) & " steps)...");

         Images := Generate_Image (Flux_Ctx, Gen_Params'Access);

         --  Free prompt string
         Free (C_Prompt);

         --  Check result
         if Images = null then
            Put_Line
              (Uptime_String & " [SD-Manager] [Stage-1] [ERROR] FLUX generation returned null!");
            Free_Flux_Context;
            raise Program_Error with "FLUX generation failed";
         end if;

         --  Log result
         Stage1_Duration := Clock - Stage1_Start;
         Log_Generate_Result (Images, 1, Stage1_Duration);

         Put_Line
           (Uptime_String & " [SD-Manager] [Stage-1] Complete in "
            & Duration'Image (Stage1_Duration) & "s");

         --  Free Stage 1 images
         Free_SD_Images (Images, 1);
      end;

      --  ====================================================================
      --  FREE-PARALLEL-MEMORY: Unload FLUX before loading refinement
      --  ====================================================================
      Put_Line
        (Uptime_String & " [SD-Manager] [FREE-PARALLEL-MEMORY]"
         & " Unloading FLUX context to free GPU for refinement...");
      Free_Flux_Context;

      --  ====================================================================
      --  STAGE 2: SD Refinement (img2img upscale)
      --  ====================================================================
      if Refine_Enabled then
         Stage2_Start := Clock;

         Put_Line
           (Uptime_String & " [SD-Manager] [Stage-2] Loading refinement context...");
         Load_Refiner_Context;

         --  Build refinement params
         declare
            Refine_Params : aliased SD_Img_Gen_Params;
            C_Prompt      : chars_ptr := New_String (Prompt);
            Images        : SD_Image_Access;
         begin
            SD_Img_Gen_Params_Init (Refine_Params'Access);

            Refine_Params.Prompt          := C_Prompt;
            Refine_Params.Negative_Prompt := Null_Ptr;
            Refine_Params.Width           := int (Width);
            Refine_Params.Height          := int (Height);
            Refine_Params.Seed            := Seed;
            Refine_Params.Batch_Count     := 1;
            Refine_Params.Strength        := interfaces.C.C_float (Refine_Strength);

            --  Refinement settings (dpmpp2mv2, more steps)
            Refine_Params.Sample_Params.Sample_Method := DPMPP2Mv2;
            Refine_Params.Sample_Params.Sample_Steps  := int (Refine_Steps);
            Refine_Params.Sample_Params.Txt_Cfg       := interfaces.C.C_float (7.0);
            Refine_Params.Sample_Params.Scheduler     := Karras;

            --  Log params
            Log_Image_Gen_Params (Refine_Params'Access);

            --  Generate refinement
            Put_Line
              (Uptime_String & " [SD-Manager] [Stage-2] Refining ("
               & Integer'Image (Refine_Steps) & " steps, strength="
               & Float'Image (Refine_Strength) & ")...");

            Images := Generate_Image (Refiner_Ctx, Refine_Params'Access);

            --  Free prompt string
            Free (C_Prompt);

            --  Check result
            if Images = null then
               Put_Line
                 (Uptime_String & " [SD-Manager] [Stage-2] [ERROR] Refinement returned null!");
               Free_Refiner_Context;
               raise Program_Error with "Refinement failed";
            end if;

            --  Log result
            Stage2_Duration := Clock - Stage2_Start;
            Log_Generate_Result (Images, 1, Stage2_Duration);

            Put_Line
              (Uptime_String & " [SD-Manager] [Stage-2] Complete in "
               & Duration'Image (Stage2_Duration) & "s");

            --  Free Stage 2 images
            Free_SD_Images (Images, 1);
         end;

         --  ==================================================================
         --  FREE-PARALLEL-MEMORY: Unload refinement after use
         --  ==================================================================
         Put_Line
           (Uptime_String & " [SD-Manager] [FREE-PARALLEL-MEMORY]"
            & " Unloading refinement context...");
         Free_Refiner_Context;
      else
         Put_Line
           (Uptime_String & " [SD-Manager] Refinement disabled, skipping Stage 2.");
      end if;

      Put_Line
        (Uptime_String & " [SD-Manager] === Two-Stage Generation COMPLETE ===");
   end Generate_Two_Stage;

   --  ============================================================================
   --  CLEANUP
   --  ============================================================================
   --  [DO NOT REMOVE, OR YOU WILL BE KILLED]

   procedure Free_All is
   begin
      Put_Line
        (Uptime_String & " [SD-Manager] Freeing all SD contexts...");

      Free_Flux_Context;
      Free_Refiner_Context;

      --  Free path strings
      if Flux_Diffusion_Path /= null then
         Free (Flux_Diffusion_Path);
         Flux_Diffusion_Path := null;
      end if;
      if Flux_Clip_L_Path /= null then
         Free (Flux_Clip_L_Path);
         Flux_Clip_L_Path := null;
      end if;
      if Flux_T5XXL_Path /= null then
         Free (Flux_T5XXL_Path);
         Flux_T5XXL_Path := null;
      end if;
      if Flux_VAE_Path /= null then
         Free (Flux_VAE_Path);
         Flux_VAE_Path := null;
      end if;
      if Refiner_Model_Path /= null then
         Free (Refiner_Model_Path);
         Refiner_Model_Path := null;
      end if;

      Is_Initialized := False;

      Put_Line
        (Uptime_String & " [SD-Manager] All SD contexts freed.");
   end Free_All;

end SD_Manager;
