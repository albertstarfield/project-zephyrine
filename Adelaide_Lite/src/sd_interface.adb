pragma SPARK_Mode (Off);
--  ============================================================================
--  SD_INTERFACE — Body with verbose logging helpers
--  ============================================================================
--  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
--  This body provides helper procedures for verbose logging of SD operations.
--  The actual C functions are imported in the spec via pragma Import.
--  This body only contains Ada-side helper logic.
--  ============================================================================

with Ada.Text_IO; use Ada.Text_IO;
with Ada.Real_Time; use Ada.Real_Time;
with Interfaces.C.Strings;

package body SD_Interface is

   --  ============================================================================
   --  VERBOSE LOGGING: SD_Version_Info
   --  ============================================================================
   --  Prints the library version and commit hash.
   --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA

   procedure SD_Version_Info is
      Version : constant chars_ptr := SD_Version;
      Commit  : constant chars_ptr := SD_Commit;
   begin
      if Version /= Null_Ptr then
         Put_Line ("[SD-Interface] Library version: "
                   & Interfaces.C.Strings.Value (Version));
      end if;
      if Commit /= Null_Ptr then
         Put_Line ("[SD-Interface] Library commit:   "
                   & Interfaces.C.Strings.Value (Commit));
      end if;
   end SD_Version_Info;

   --  ============================================================================
   --  VERBOSE LOGGING: SD_System_Info
   --  ============================================================================
   --  Prints system info (CPU cores, backend, etc.)
   --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA

   procedure SD_System_Info is
      Info : constant chars_ptr := SD_Get_System_Info;
      Cores : constant int := SD_Get_Num_Physical_Cores;
   begin
      if Info /= Null_Ptr then
         Put_Line ("[SD-Interface] System info: "
                   & Interfaces.C.Strings.Value (Info));
      end if;
      Put_Line ("[SD-Interface] Physical cores: " & int'Image (Cores));
   end SD_System_Info;

   --  ============================================================================
   --  VERBOSE LOGGING: Log_Context_Params
   --  ============================================================================
   --  Logs all fields of SD_Ctx_Params for debugging.
   --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA

   procedure Log_Context_Params (Params : access SD_Ctx_Params) is
      use Interfaces.C.Strings;
   begin
      Put_Line ("[SD-Interface] === Context Parameters ===");
      if Params /= null then
         if Params.Model_Path /= Null_Ptr then
            Put_Line ("  Model_Path:       " & Value (Params.Model_Path));
         end if;
         if Params.Clip_L_Path /= Null_Ptr then
            Put_Line ("  Clip_L_Path:      " & Value (Params.Clip_L_Path));
         end if;
         if Params.T5XXL_Path /= Null_Ptr then
            Put_Line ("  T5XXL_Path:       " & Value (Params.T5XXL_Path));
         end if;
         if Params.VAE_Path /= Null_Ptr then
            Put_Line ("  VAE_Path:         " & Value (Params.VAE_Path));
         end if;
         Put_Line ("  N_Threads:        " & int'Image (Params.N_Threads));
         Put_Line ("  RNG_Type:         " & int'Image (Params.RNG_Type));
         Put_Line ("  Prediction:       " & int'Image (Params.Prediction));
         Put_Line ("  Flash_Attn:       " & int'Image (Params.Flash_Attn));
         Put_Line ("  Enable_MMAP:      " & int'Image (Params.Enable_MMAP));
         if Params.Backend /= Null_Ptr then
            Put_Line ("  Backend:          " & Value (Params.Backend));
         end if;
      end if;
      Put_Line ("[SD-Interface] === End Context Parameters ===");
   end Log_Context_Params;

   --  ============================================================================
   --  VERBOSE LOGGING: Log_Image_Gen_Params
   --  ============================================================================
   --  Logs all fields of SD_Img_Gen_Params for debugging.
   --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA

   procedure Log_Image_Gen_Params (Params : access SD_Img_Gen_Params) is
      use Interfaces.C.Strings;
   begin
      Put_Line ("[SD-Interface] === Image Generation Parameters ===");
      if Params /= null then
         if Params.Prompt /= Null_Ptr then
            Put_Line ("  Prompt:           " & Value (Params.Prompt));
         end if;
         if Params.Negative_Prompt /= Null_Ptr then
            Put_Line ("  Negative_Prompt:  " & Value (Params.Negative_Prompt));
         end if;
         Put_Line ("  Width:            " & int'Image (Params.Width));
         Put_Line ("  Height:           " & int'Image (Params.Height));
         Put_Line ("  Seed:             " & Long_Long_Integer'Image (Params.Seed));
         Put_Line ("  Batch_Count:      " & int'Image (Params.Batch_Count));
         Put_Line ("  Strength:         " &
                   Interfaces.C.C_float'Image (Params.Strength));
         Put_Line ("  Clip_Skip:        " & int'Image (Params.Clip_Skip));
         --  Sample params
         Put_Line ("  -- Sample Parameters --");
         Put_Line ("    Sample_Method:  " & int'Image (int (Sample_Method_T'Pos (Params.Sample_Params.Sample_Method))));
         Put_Line ("    Sample_Steps:   " & int'Image (Params.Sample_Params.Sample_Steps));
         Put_Line ("    Txt_Cfg:        " &
                   Interfaces.C.C_float'Image (Params.Sample_Params.Txt_Cfg));
         Put_Line ("    Scheduler:      " & int'Image (int (Scheduler_T'Pos (Params.Sample_Params.Scheduler))));
      end if;
      Put_Line ("[SD-Interface] === End Image Generation Parameters ===");
   end Log_Image_Gen_Params;

   --  ============================================================================
   --  VERBOSE LOGGING: Log_Generate_Result
   --  ============================================================================
   --  Logs the result of generate_image().
   --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA

   procedure Log_Generate_Result (Images      : SD_Image_Access;
                                   Count       : int;
                                   Gen_Duration: Duration) is
   begin
      Put_Line ("[SD-Interface] === Generation Result ===");
      Put_Line ("  Images generated: " & int'Image (Count));
      Put_Line ("  Generation time:  " & Duration'Image (Gen_Duration) & "s");
      if Images /= null then
         for I in 0 .. Count - 1 loop
            declare
               --  Access the I-th image in the array
               Img : constant SD_Image := Images.all;
            begin
               Put_Line ("  Image" & int'Image (I) & ":"
                         & " Width=" & unsigned'Image (Img.Width)
                         & " Height=" & unsigned'Image (Img.Height)
                         & " Channels=" & unsigned'Image (Img.Channel));
            end;
         end loop;
      end if;
      Put_Line ("[SD-Interface] === End Generation Result ===");
   end Log_Generate_Result;

   --  ============================================================================
   --  VERBOSE LOGGING: Log_All_Enum_Names
   --  ============================================================================
   --  Logs all available enum names from the C library.
   --  Useful for debugging which enums are supported.
   --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA

   procedure Log_All_Enum_Names is
      use Interfaces.C.Strings;
      Name_Ptr : chars_ptr;
   begin
      Put_Line ("[SD-Interface] === Available Sample Methods ===");
      for I in 0 .. 17 loop  -- 0..17 = Euler through Euler_GE
         Name_Ptr := SD_Sample_Method_Name (int (I));
         if Name_Ptr /= Null_Ptr then
            Put_Line ("  " & int'Image (int (I)) & ": " & Value (Name_Ptr));
         end if;
      end loop;

      Put_Line ("[SD-Interface] === Available Schedulers ===");
      for I in 0 .. 12 loop  -- 0..12 = Discrete through Logit_Normal
         Name_Ptr := SD_Scheduler_Name (int (I));
         if Name_Ptr /= Null_Ptr then
            Put_Line ("  " & int'Image (int (I)) & ": " & Value (Name_Ptr));
         end if;
      end loop;

      Put_Line ("[SD-Interface] === Available Data Types ===");
      for I in 0 .. 15 loop  -- 0..15 = F32 through Q8_K
         Name_Ptr := SD_Type_Name (int (I));
         if Name_Ptr /= Null_Ptr then
            Put_Line ("  " & int'Image (int (I)) & ": " & Value (Name_Ptr));
         end if;
      end loop;

      Put_Line ("[SD-Interface] === Available RNG Types ===");
      for I in 0 .. 2 loop  -- 0..2 = STD_DEFAULT, CUDA, CPU
         Name_Ptr := SD_RNG_Type_Name (int (I));
         if Name_Ptr /= Null_Ptr then
            Put_Line ("  " & int'Image (int (I)) & ": " & Value (Name_Ptr));
         end if;
      end loop;

      Put_Line ("[SD-Interface] === Available Predictions ===");
      for I in 0 .. 5 loop  -- 0..5 = EPS through FLUX2_FLOW_PRED
         Name_Ptr := SD_Prediction_Name (int (I));
         if Name_Ptr /= Null_Ptr then
            Put_Line ("  " & int'Image (int (I)) & ": " & Value (Name_Ptr));
         end if;
      end loop;
   end Log_All_Enum_Names;

end SD_Interface;
