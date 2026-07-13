pragma SPARK_Mode (Off);
--  ============================================================================
--  SD_MANAGER — Body implementing two-stage image generation
--  ============================================================================
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA

with Ada.Text_IO; use Ada.Text_IO;
with Ada.Real_Time; use Ada.Real_Time;
with Interfaces.C.Strings;
with AnsiAda;
with System;

package body SD_Manager is

   --  ============================================================================
   --  HELPER: Uptime string for logging
   --  ============================================================================
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA

   function Uptime_String return String is
      use Ada.Real_Time;
      Elapsed : constant Time_Span := Clock - Init_Start_Time;
      Seconds : constant Integer := Integer (To_Duration (Elapsed));
   begin
      return "[" & Integer'Image (Seconds) & "s]";
   end Uptime_String;

   --  ============================================================================
   --  INITIALIZATION
   --  ============================================================================
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA

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
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA

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
         Ada.Text_IO.Put_Line
           (AnsiAda.Background (AnsiAda.Red)
            & AnsiAda.Foreground (AnsiAda.Light_Grey)
            & "[BUGCHECK] [SD-Manager] [ERROR]"
            & " Failed to create FLUX context!"
            & " Check GPU memory (need ~4GB free)."
            & " Unload main model before FLUX if VRAM tight."
            & AnsiAda.Reset);
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
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA

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
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA

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
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA

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
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
   --  This is the main entry point for image generation.
   --  Implements the two-stage pipeline from project-zephyrine:
   --    Stage 1: FLUX sparse → Stage 2: SD refinement

   --  FFI to C helper for PNG+Base64 encoding
   function SD_Image_To_Base64_PNG
     (Image_Data : System.Address;
      Width      : Interfaces.C.int;
      Height     : Interfaces.C.int;
      Channels   : Interfaces.C.int) return Interfaces.C.Strings.chars_ptr;
   pragma Import (C, SD_Image_To_Base64_PNG, "sd_image_to_base64_png");

   procedure SD_Free_String (Str : Interfaces.C.Strings.chars_ptr);
   pragma Import (C, SD_Free_String, "sd_free_string");

   procedure Generate_Two_Stage
     (Prompt         : String;
      Width          : Integer := 1024;
      Height         : Integer := 1024;
      Seed           : Long_Long_Integer := -1;
      Flux_Steps     : Integer := 4;
      Flux_Cfg       : Float := 1.0;
      Refine_Enabled : Boolean := True;
      Refine_Steps   : Integer := 8;
      Refine_Strength: Float := 0.4;
      Image_B64      : out Ada.Strings.Unbounded.Unbounded_String;
      Error_Msg      : out Ada.Strings.Unbounded.Unbounded_String)
   is
      use Interfaces.C;
      use Interfaces.C.Strings;
      use Ada.Strings.Unbounded;
      Stage1_Start    : Ada.Real_Time.Time;
      Stage1_Elapsed  : Ada.Real_Time.Time_Span;
      Stage2_Start    : Ada.Real_Time.Time;
      Stage2_Elapsed  : Ada.Real_Time.Time_Span;
      Last_Image      : SD_Image_Access := null;
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
         Gen_Params.Sample_Params.Txt_Cfg       := C_float (Flux_Cfg);
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
         Stage1_Elapsed := Clock - Stage1_Start;
         Log_Generate_Result (Images, 1, To_Duration (Stage1_Elapsed));

         Put_Line
           (Uptime_String & " [SD-Manager] [Stage-1] Complete in "
            & Duration'Image (To_Duration (Stage1_Elapsed)) & "s");

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
            Refine_Params.Strength        := C_float (Refine_Strength);

            --  Refinement settings (dpmpp2mv2, more steps)
            Refine_Params.Sample_Params.Sample_Method := DPMPP2Mv2;
            Refine_Params.Sample_Params.Sample_Steps  := int (Refine_Steps);
            Refine_Params.Sample_Params.Txt_Cfg       := C_float (7.0);
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
            Stage2_Elapsed := Clock - Stage2_Start;
            Log_Generate_Result (Images, 1, To_Duration (Stage2_Elapsed));

            Put_Line
              (Uptime_String & " [SD-Manager] [Stage-2] Complete in "
               & Duration'Image (To_Duration (Stage2_Elapsed)) & "s");

            --  Keep reference for Base64 conversion after freeing images
            Last_Image := Images;
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

      --  ====================================================================
      --  CONVERT TO BASE64 PNG via C helper
      --  ====================================================================
      if Last_Image /= null then
         declare
            Img     : constant SD_Image := Last_Image.all;
            C_Result: chars_ptr;
         begin
            Put_Line
              (Uptime_String & " [SD-Manager] Converting to Base64 PNG..."
               & " W=" & unsigned'Image (Img.Width)
               & " H=" & unsigned'Image (Img.Height)
               & " Ch=" & unsigned'Image (Img.Channel));

            C_Result := SD_Image_To_Base64_PNG
              (Img.Data,
               int (Img.Width),
               int (Img.Height),
               int (Img.Channel));

            if C_Result /= Null_Ptr then
               Image_B64 := To_Unbounded_String (Value (C_Result));
               Put_Line
                 (Uptime_String & " [SD-Manager] Base64 conversion complete."
                  & " Length=" & Integer'Image (Length (Image_B64)));
               SD_Free_String (C_Result);
            else
               Error_Msg := To_Unbounded_String ("Base64 conversion failed");
               Put_Line
                 (Uptime_String & " [SD-Manager] [ERROR] Base64 conversion returned null!");
            end if;
         end;

         --  Free the SD images after conversion
         Free_SD_Images (Last_Image, 1);
      end if;

      Put_Line
        (Uptime_String & " [SD-Manager] === Two-Stage Generation COMPLETE ===");
   end Generate_Two_Stage;

   --  ============================================================================
   --  CLEANUP
   --  ============================================================================
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA

   procedure Free_All is
   begin
      Put_Line
        (Uptime_String & " [SD-Manager] Freeing all SD contexts...");

      Free_Flux_Context;
      Free_Refiner_Context;

      --  Free path strings
      Flux_Diffusion_Path := null;
      Flux_Clip_L_Path := null;
      Flux_T5XXL_Path := null;
      Flux_VAE_Path := null;
      Refiner_Model_Path := null;

      Is_Initialized := False;

      Put_Line
        (Uptime_String & " [SD-Manager] All SD contexts freed.");
   end Free_All;

end SD_Manager;
