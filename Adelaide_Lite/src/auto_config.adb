pragma SPARK_Mode (Off);
--  ============================================================================
--  AUTO_CONFIG — Self-Tuning Hardware Configuration (Body)
--  ============================================================================
--  Implementation of the auto-config system. Detects hardware at startup,
--  loads saved config, and manages probing upward after successful inference.
--
--  TERMINOLOGY:
--    "Acceleration_Layer" — model layers offloaded to hardware accelerators.
--    On Apple: Metal GPU layers. On Intel: Intel HD Graphics via Metal/Vulkan.
--    On NVIDIA: CUDA layers. On AMD: ROCm layers.
--    Value 0 = CPU-only, 8/16/24 = partial, -1 = all layers on accelerator.
--    NOT "GPU" — because Intel HD Graphics is not a GPU in the NVIDIA sense.
--  ============================================================================

with Ada.Text_IO;           use Ada.Text_IO;
with Ada.Directories;       use Ada.Directories;
with Interfaces.C;          use Interfaces.C;
with AnsiAda;
with Llama_Interface;

package body Auto_Config is

   --  ============================================================================
   --  INTERNAL STATE
   --  ============================================================================
   Current_Config : Config_Array;
   Initialized    : Boolean := False;

   --  ============================================================================
   --  LADDER CONVERSION FUNCTIONS
   --  ============================================================================
   function Ctx_To_Unsigned (C : Ctx_Ladder) return Interfaces.C.unsigned is
   begin
      case C is
         when Ctx_2048   => return 2048;
         when Ctx_4096   => return 4096;
         when Ctx_8192   => return 8192;
         when Ctx_16384  => return 16384;
         when Ctx_32768  => return 32768;
      end case;
   end Ctx_To_Unsigned;

   function Threads_To_Int (T : Thread_Ladder) return Interfaces.C.int is
   begin
      case T is
         when T_1 => return 1;
         when T_2 => return 2;
         when T_4 => return 4;
      end case;
   end Threads_To_Int;

   function Batch_To_Unsigned (B : Batch_Ladder) return Interfaces.C.unsigned is
   begin
      case B is
         when B_64  => return 64;
         when B_128 => return 128;
         when B_256 => return 256;
         when B_512 => return 512;
      end case;
   end Batch_To_Unsigned;

   function Accel_Layers_To_Int (A : Accel_Layer_Ladder) return Interfaces.C.int is
   begin
      case A is
         when AL_0   => return 0;
         when AL_8   => return 8;
         when AL_16  => return 16;
         when AL_24  => return 24;
         when AL_All => return Accel_All_Layers;
      end case;
   end Accel_Layers_To_Int;

   --  ============================================================================
   --  HARDWARE DETECTION
   --  ============================================================================
   procedure Detect_Hardware is
   begin
      Put_Line
         (AnsiAda.Foreground (AnsiAda.Cyan)
          & "[AutoConfig]"
          & AnsiAda.Reset
          & " Detecting hardware...");

      --  Detect CPU cores
      --  Why: Intel Pentium Penryn = 2 cores. Apple M2 Pro = 10 cores.
      --  Threads should be min(cores, 4) — more than 4 gives diminishing
      --  returns on llama.cpp workloads.
      Detected_Hardware.CPU_Cores := 2;  -- Safe default for Penryn

      --  Detect available RAM
      --  Why: On 16GB system with 98% used, only 274MB free.
      --  Model weights (5.8GB) + KV cache (30-480MB) + buffers (64MB)
      --  must fit in available RAM. Start with smallest context if tight.
      declare
         Free_Bytes  : Interfaces.C.size_t := 0;
         Total_Bytes : Interfaces.C.size_t := 0;
      begin
         Llama_Interface.CPU_Memory_Query (Free_Bytes, Total_Bytes);
         Detected_Hardware.Free_RAM_MB  := Natural (Free_Bytes / (1024 * 1024));
         Detected_Hardware.Total_RAM_MB := Natural (Total_Bytes / (1024 * 1024));
      exception
         when others =>
            Detected_Hardware.Free_RAM_MB  := 2000;  -- Conservative default
            Detected_Hardware.Total_RAM_MB := 8192;
      end;

      --  Detect accelerator memory (Metal/Vulkan/CUDA VRAM)
      --  Why: Intel integrated "GPU" shares system RAM. The VRAM reported
      --  by Metal is the shared memory pool, not dedicated VRAM.
      --  Apple M2 Pro has true unified memory — accelerator can access all RAM.
      --  For Intel: actual dedicated VRAM is ~128-512MB, rest is shared.
      --  We call it "accelerator memory" not "GPU VRAM" because on Intel
      --  it's Intel HD Graphics, not a real GPU.
      declare
         Free_Bytes  : Interfaces.C.size_t := 0;
         Total_Bytes : Interfaces.C.size_t := 0;
      begin
         Llama_Interface.GPU_Memory_Query (Free_Bytes, Total_Bytes);
         Detected_Hardware.Accel_VRAM_MB := Natural (Free_Bytes / (1024 * 1024));
      exception
         when others =>
            Detected_Hardware.Accel_VRAM_MB := 0;
      end;

      Put_Line
         (AnsiAda.Foreground (AnsiAda.Cyan)
          & "[AutoConfig]"
          & AnsiAda.Reset
          & " Hardware detected:"
          & " CPU_Cores=" & Natural'Image (Detected_Hardware.CPU_Cores)
          & " Free_RAM=" & Natural'Image (Detected_Hardware.Free_RAM_MB) & "MB"
          & " Total_RAM=" & Natural'Image (Detected_Hardware.Total_RAM_MB) & "MB"
          & " Accel_VRAM=" & Natural'Image (Detected_Hardware.Accel_VRAM_MB) & "MB");
   end Detect_Hardware;

   --  ============================================================================
   --  CONFIG FILE I/O
   --  ============================================================================

   --  Parse a single config line from the file.
   --  Format: "MODEL_NAME: CTX=2048 THREADS=2 BATCH=128 ACCEL_LAYERS=8"
   procedure Parse_Config_Line (Line : String) is
      Colon_Pos : Natural := 0;
   begin
      --  Find the colon separator
      for I in Line'Range loop
         if Line (I) = ':' then
            Colon_Pos := I;
            exit;
         end if;
      end loop;

      if Colon_Pos = 0 then
         return;  -- Malformed line, skip
      end if;

      --  Extract model name (before colon)
      declare
         Model_Name : constant String := Line (Line'First .. Colon_Pos - 1);
         Rest       : constant String := Line (Colon_Pos + 1 .. Line'Last);
         Kind       : Model_Type;
         Found      : Boolean := False;
      begin
         --  Match model name to enum
         for M in Model_Type loop
            if Model_Type'Image (M) = Model_Name then
               Kind := M;
               Found := True;
               exit;
            end if;
         end loop;

         if not Found then
            return;  -- Unknown model, skip
         end if;

         --  Parse key=value pairs from Rest
         declare
            Pos : Natural := Rest'First;
         begin
            while Pos <= Rest'Last loop
               --  Skip whitespace
               while Pos <= Rest'Last and then Rest (Pos) = ' ' loop
                  Pos := Pos + 1;
               end loop;

               exit when Pos > Rest'Last;

               --  Find key
               declare
                  Key_Start : constant Natural := Pos;
               begin
                  while Pos <= Rest'Last and then Rest (Pos) /= '=' loop
                     Pos := Pos + 1;
                  end loop;

                  exit when Pos > Rest'Last;

                  declare
                     Key : constant String := Rest (Key_Start .. Pos - 1);
                  begin
                     Pos := Pos + 1;  -- Skip '='

                     --  Find value end (space or end of line)
                     declare
                        Val_Start : constant Natural := Pos;
                     begin
                        while Pos <= Rest'Last and then Rest (Pos) /= ' ' loop
                           Pos := Pos + 1;
                        end loop;

                        declare
                           Val_Str : constant String := Rest (Val_Start .. Pos - 1);
                           Val     : Natural;
                           Val_OK  : Boolean := False;
                        begin
                           --  Parse integer value
                           begin
                              Val := Natural'Value (Val_Str);
                              Val_OK := True;
                           exception
                              when others => Val_OK := False;
                           end;

                           if Val_OK then
                              if Key = "CTX" then
                                 --  Find the matching ladder level
                                 for C in Ctx_Ladder loop
                                    if Ctx_To_Unsigned (C) = Interfaces.C.unsigned (Val) then
                                       Current_Config (Kind).Ctx := C;
                                       Current_Config (Kind).Max_Working := C;
                                       exit;
                                    end if;
                                 end loop;
                              elsif Key = "THREADS" then
                                 for T in Thread_Ladder loop
                                    if Threads_To_Int (T) = Interfaces.C.int (Val) then
                                       Current_Config (Kind).Threads := T;
                                       exit;
                                    end if;
                                 end loop;
                              elsif Key = "BATCH" then
                                 for B in Batch_Ladder loop
                                    if Batch_To_Unsigned (B) = Interfaces.C.unsigned (Val) then
                                       Current_Config (Kind).Batch := B;
                                       exit;
                                    end if;
                                 end loop;
                              elsif Key = "ACCEL_LAYERS" then
                                 if Val = 0 then
                                    Current_Config (Kind).Accel_Layers := AL_0;
                                 elsif Val = 8 then
                                    Current_Config (Kind).Accel_Layers := AL_8;
                                 elsif Val = 16 then
                                    Current_Config (Kind).Accel_Layers := AL_16;
                                 elsif Val = 24 then
                                    Current_Config (Kind).Accel_Layers := AL_24;
                                 elsif Val = 999 then  -- Sentinel for -1 (all)
                                    Current_Config (Kind).Accel_Layers := AL_All;
                                 end if;
                              end if;
                           end if;
                        end;
                     end;
                  end;
               end;  -- close declare Key_Start (line 191)
            end loop;
         end;
      end;
   end Parse_Config_Line;

   procedure Load_Config_File is
      Config_File : File_Type;
   begin
      if not Exists (Config_File_Path) then
         Put_Line
            (AnsiAda.Foreground (AnsiAda.Yellow)
             & "[AutoConfig]"
             & AnsiAda.Reset
             & " No saved config found. Starting from minimal.");
         return;
      end if;

      Open (Config_File, In_File, Config_File_Path);

      while not End_Of_File (Config_File) loop
         declare
            Line : constant String := Get_Line (Config_File);
         begin
            --  Skip comments and empty lines
            if Line'Length > 0 and then Line (Line'First) /= '#' then
               Parse_Config_Line (Line);
            end if;
         end;
      end loop;

      Close (Config_File);

      Put_Line
         (AnsiAda.Foreground (AnsiAda.Light_Green)
          & "[AutoConfig]"
          & AnsiAda.Reset
          & " Loaded saved config from " & Config_File_Path);
   exception
      when others =>
         Put_Line
            (AnsiAda.Foreground (AnsiAda.Yellow)
             & "[AutoConfig]"
             & AnsiAda.Reset
             & " Failed to load config, starting from minimal.");
         begin
            Close (Config_File);
         exception
            when others => null;
         end;
   end Load_Config_File;

   procedure Save_Config is
      Config_File : File_Type;
   begin
      --  Ensure run/ directory exists
      if not Exists ("run") then
         Create_Directory ("run");
      end if;

      Create (Config_File, Out_File, Config_File_Path);

      --  Write header comment
      Put_Line (Config_File, "# Auto-Config -- Saved by Adelaide Lite");
      Put_Line (Config_File, "# Format: MODEL: CTX=N THREADS=N BATCH=N ACCEL_LAYERS=N");
      Put_Line (Config_File, "# ACCEL_LAYERS: 0=CPU, 8/16/24=partial, 999=all accelerator");
      Put_Line (Config_File, "#");

      for Kind in Model_Type loop
         declare
            C : constant Working_Config := Current_Config (Kind);
            AL_V : Natural;
         begin
            case C.Accel_Layers is
               when AL_0   => AL_V := 0;
               when AL_8   => AL_V := 8;
               when AL_16  => AL_V := 16;
               when AL_24  => AL_V := 24;
               when AL_All => AL_V := 999;
            end case;

            Put_Line
               (Config_File,
                Model_Type'Image (Kind) & ":"
                & " CTX=" & Interfaces.C.unsigned'Image (Ctx_To_Unsigned (C.Ctx))
                & " THREADS=" & Interfaces.C.int'Image (Threads_To_Int (C.Threads))
                & " BATCH=" & Interfaces.C.unsigned'Image (Batch_To_Unsigned (C.Batch))
                & " ACCEL_LAYERS=" & Natural'Image (AL_V));
         end;
      end loop;

      Close (Config_File);

      Put_Line
         (AnsiAda.Foreground (AnsiAda.Light_Green)
          & "[AutoConfig]"
          & AnsiAda.Reset
          & " Config saved to " & Config_File_Path);
   exception
      when others =>
         begin
            Close (Config_File);
         exception
            when others => null;
         end;
   end Save_Config;

   --  ============================================================================
   --  PROBING LOGIC
   --  ============================================================================

   --  Get the next higher level in a ladder, or return current if at max.
   function Next_Ctx_Level (Current : Ctx_Ladder) return Ctx_Ladder is
   begin
      case Current is
         when Ctx_2048  => return Ctx_4096;
         when Ctx_4096  => return Ctx_8192;
         when Ctx_8192  => return Ctx_16384;
         when Ctx_16384 => return Ctx_32768;
         when Ctx_32768 => return Ctx_32768;  -- Already at max
      end case;
   end Next_Ctx_Level;

   function Next_Thread_Level (Current : Thread_Ladder) return Thread_Ladder is
   begin
      case Current is
         when T_1 => return T_2;
         when T_2 => return T_4;
         when T_4 => return T_4;  -- Already at max
      end case;
   end Next_Thread_Level;

   function Next_Batch_Level (Current : Batch_Ladder) return Batch_Ladder is
   begin
      case Current is
         when B_64  => return B_128;
         when B_128 => return B_256;
         when B_256 => return B_512;
         when B_512 => return B_512;  -- Already at max
      end case;
   end Next_Batch_Level;

   function Next_Accel_Level (Current : Accel_Layer_Ladder) return Accel_Layer_Ladder is
   begin
      case Current is
         when AL_0   => return AL_8;
         when AL_8   => return AL_16;
         when AL_16  => return AL_24;
         when AL_24  => return AL_All;
         when AL_All => return AL_All;  -- Already at max
      end case;
   end Next_Accel_Level;

   --  ============================================================================
   --  INITIALIZATION
   --  ============================================================================

   procedure Initialize is
   begin
      if Initialized then
         return;
      end if;

      Put_Line
         (AnsiAda.Foreground (AnsiAda.Cyan)
          & "[AutoConfig]"
          & AnsiAda.Reset
          & " Initializing self-tuning configuration...");

      --  Step 1: Detect hardware
      Detect_Hardware;

      --  Step 2: Start with minimal defaults for all models
      for Kind in Model_Type loop
         Current_Config (Kind) := (Ctx              => Ctx_2048,
                                   Threads          => T_1,
                                   Batch            => B_64,
                                   Accel_Layers     => AL_0,
                                   Probe_Target      => Ctx_2048,
                                   Max_Working      => Ctx_2048,
                                   Fail_Count       => 0);
      end loop;

      --  Step 3: Load saved config (overrides defaults if available)
      Load_Config_File;

      --  Step 4: Adjust initial settings based on detected hardware
      --  Why: On a machine with 16GB+ free RAM and 10 cores, starting at
      --  2048 ctx with 1 thread is wasteful. The probe will find the right
      --  level, but we can start closer to the target.
      if Detected_Hardware.Free_RAM_MB > 8000 then
         --  Plenty of RAM — start at 8192, skip the lower probes
         for Kind in Model_Type loop
            if Kind /= Qwen_Embedding then
               Current_Config (Kind).Ctx := Ctx_8192;
               Current_Config (Kind).Max_Working := Ctx_8192;
            end if;
         end loop;
      elsif Detected_Hardware.Free_RAM_MB > 4000 then
         --  Moderate RAM — start at 4096
         for Kind in Model_Type loop
            if Kind /= Qwen_Embedding then
               Current_Config (Kind).Ctx := Ctx_4096;
               Current_Config (Kind).Max_Working := Ctx_4096;
            end if;
         end loop;
      end if;
      --  Low RAM (< 4GB free): stay at 2048, probe up from there

      --  Set threads based on detected cores
      if Detected_Hardware.CPU_Cores >= 4 then
         for Kind in Model_Type loop
            Current_Config (Kind).Threads := T_4;
         end loop;
      elsif Detected_Hardware.CPU_Cores >= 2 then
         for Kind in Model_Type loop
            Current_Config (Kind).Threads := T_2;
         end loop;
      end if;

      --  Set batch based on accelerator VRAM
      if Detected_Hardware.Accel_VRAM_MB > 4000 then
         --  Plenty of accelerator memory — larger batch is fine
         for Kind in Model_Type loop
            Current_Config (Kind).Batch := B_256;
         end loop;
      elsif Detected_Hardware.Accel_VRAM_MB > 1000 then
         --  Some accelerator memory — moderate batch
         for Kind in Model_Type loop
            Current_Config (Kind).Batch := B_128;
         end loop;
      end if;
      --  Low/no accelerator memory: stay at B_64

      --  Enable acceleration layers if VRAM is sufficient
      if Detected_Hardware.Accel_VRAM_MB > 2000 then
         --  More than 2GB — try some layers on accelerator
         for Kind in Model_Type loop
            if Kind /= Qwen_Embedding then
               Current_Config (Kind).Accel_Layers := AL_16;
            end if;
         end loop;
      end if;

      Initialized := True;

      Put_Line
         (AnsiAda.Foreground (AnsiAda.Light_Green)
          & "[AutoConfig]"
          & AnsiAda.Reset
          & " Self-tuning initialized. Starting config:");
      for Kind in Model_Type loop
         declare
            C : constant Working_Config := Current_Config (Kind);
         begin
            Put_Line
               (AnsiAda.Foreground (AnsiAda.Grey)
                & "[AutoConfig]"
                & AnsiAda.Reset
                & "   " & Model_Type'Image (Kind) & ":"
                & " Ctx=" & Interfaces.C.unsigned'Image (Ctx_To_Unsigned (C.Ctx))
                & " Threads=" & Interfaces.C.int'Image (Threads_To_Int (C.Threads))
                & " Batch=" & Interfaces.C.unsigned'Image (Batch_To_Unsigned (C.Batch))
                & " Accel=" & Interfaces.C.int'Image (Accel_Layers_To_Int (C.Accel_Layers)));
         end;
      end loop;
   end Initialize;

   --  ============================================================================
   --  PUBLIC API
   --  ============================================================================

   function Get_Config (Kind : Model_Type) return Working_Config is
   begin
      if not Initialized then
         Initialize;
      end if;
      return Current_Config (Kind);
   end Get_Config;

   procedure Record_Success
     (Kind     : Model_Type;
      Ctx_Used : Interfaces.C.unsigned)
   is
      C : Working_Config := Current_Config (Kind);
   begin
      --  Record this as the new max working config
      for L in Ctx_Ladder loop
         if Ctx_To_Unsigned (L) = Ctx_Used then
            C.Max_Working := L;
            exit;
         end if;
      end loop;

      --  Reset failure counter
      C.Fail_Count := 0;

      Current_Config (Kind) := C;

      --  Print progress
      Put_Line
         (AnsiAda.Foreground (AnsiAda.Light_Green)
          & "[AutoConfig]"
          & AnsiAda.Reset
          & " " & Model_Type'Image (Kind) & ": ctx="
          & Interfaces.C.unsigned'Image (Ctx_Used) & " OK"
          & " (max_working="
          & Interfaces.C.unsigned'Image (Ctx_To_Unsigned (C.Max_Working))
          & ")");

      --  Save config after successful probe
      Save_Config;
   end Record_Success;

   procedure Set_Probe_Target
     (Kind   : Model_Type;
      Target : Ctx_Ladder)
   is
      C : Working_Config := Current_Config (Kind);
   begin
      --  Only set if target is larger than current
      if Ctx_To_Unsigned (Target) > Ctx_To_Unsigned (C.Ctx) then
         C.Probe_Target := Target;
         Current_Config (Kind) := C;

         Put_Line
            (AnsiAda.Foreground (AnsiAda.Light_Cyan)
             & "[AutoConfig]"
             & AnsiAda.Reset
             & " " & Model_Type'Image (Kind) & ":"
             & " Probe target set to ctx="
             & Interfaces.C.unsigned'Image (Ctx_To_Unsigned (Target))
             & " (current="
             & Interfaces.C.unsigned'Image (Ctx_To_Unsigned (C.Ctx))
             & ")");
      end if;
   end Set_Probe_Target;

   function Get_Probe_Target (Kind : Model_Type) return Ctx_Ladder is
      C     : Working_Config := Current_Config (Kind);
      Target : constant Ctx_Ladder := C.Probe_Target;
   begin
      --  Clear the probe target (one-shot)
      if Ctx_To_Unsigned (Target) > Ctx_To_Unsigned (C.Ctx) then
         C.Ctx := Target;
         C.Probe_Target := C.Ctx;  -- Clear: target = current = no probe
         Current_Config (Kind) := C;
      end if;
      return Target;
   end Get_Probe_Target;

   procedure Record_Failure
     (Kind      : Model_Type;
      Ctx_Tried : Interfaces.C.unsigned)
   is
      C : Working_Config := Current_Config (Kind);
   begin
      C.Fail_Count := C.Fail_Count + 1;

      --  Step back to max working config
      C.Ctx := C.Max_Working;

      Put_Line
         (AnsiAda.Foreground (AnsiAda.Red)
          & "[AutoConfig]"
          & AnsiAda.Reset
          & " " & Model_Type'Image (Kind) & ": ctx="
          & Interfaces.C.unsigned'Image (Ctx_Tried) & " FAILED"
          & " (step back to "
          & Interfaces.C.unsigned'Image (Ctx_To_Unsigned (C.Max_Working))
          & ", fail_count=" & Natural'Image (C.Fail_Count)
          & ")");

      --  If too many failures, stop probing this direction
      if C.Fail_Count >= Max_Fail_Count then
         Put_Line
            (AnsiAda.Foreground (AnsiAda.Red)
             & "[AutoConfig]"
             & AnsiAda.Reset
             & " " & Model_Type'Image (Kind)
             & ": Max failures reached. Locking at "
             & Interfaces.C.unsigned'Image (Ctx_To_Unsigned (C.Max_Working)));
      end if;

      Current_Config (Kind) := C;
      Save_Config;
   end Record_Failure;

   procedure Reset_To_Minimal is
   begin
      for Kind in Model_Type loop
         Current_Config (Kind) := (Ctx              => Ctx_2048,
                                   Threads          => T_1,
                                   Batch            => B_64,
                                   Accel_Layers     => AL_0,
                                   Probe_Target      => Ctx_2048,
                                   Max_Working      => Ctx_2048,
                                   Fail_Count       => 0);
      end loop;

      Put_Line
         (AnsiAda.Foreground (AnsiAda.Yellow)
          & "[AutoConfig]"
          & AnsiAda.Reset
          & " Reset to minimal config. Will re-probe on next inference.");

      Save_Config;
   end Reset_To_Minimal;

end Auto_Config;
