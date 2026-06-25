pragma SPARK_Mode (Off);

--  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  ===========================================================================
--  Adelaide Intelligence Server — Main Entry Point
--  ===========================================================================
--  This is the primary Ada procedure that boots the entire Adelaide backend.
--  It initializes the LLM engine, knowledge base, priority queue (ELP),
--  and binds the HTTP API on port 11420.
--
--  Startup ordering is critical.  The sequence is:
--    1. LLM backend + DB + ELP queue init  (Model_Manager.Initialize)
--    2. Knowledge base init                (Knowledge_Manager.Initialize)
--    3. Scheduler init                     (Scheduler_Manager.Initialize)
--    4. Watchdog IPC init                   (Watchdog_IPC.Init)
--    5. Background tasks start             (Knowledge_Manager.Start_Tasks)
--       — ELP0 producers (Native_Crawl, Proactive_Cache) begin immediately
--    6. HTTP server bind                   (AWS.Server.Start on port 11420)
--    7. Startup health ping loop           (3s interval, 60s deadline)
--    8. Moonshine STT model load           (non-blocking after bind)
--    9. Main heartbeat loop
--
--  If any step 1-6 hangs or the health ping fails for 60 seconds, the
--  process exits with code 69 so run.sh displays the error banner.
--  ===========================================================================

with AnsiAda;
with Ada.Text_IO; use Ada.Text_IO;
with Ada.Exceptions;
with Ada.Command_Line;
with Ada.Strings; use Ada.Strings;
with Ada.Strings.Fixed; use Ada.Strings.Fixed;
with Ada.Real_Time; use Ada.Real_Time;
with Ada.Environment_Variables;
with Ada.Directories;
with Adelaide_Server_Pkg;
with Model_Manager;
with Knowledge_Manager;
with Scheduler_Manager;
with Watchdog_Manager;
with Watchdog_IPC;
with Shutdown_Manager;
with AWS.Config;
with AWS.Config.Set;
with AWS.Server;
with AWS.Client;
with AWS.Response;
with AWS.Messages; use AWS.Messages;
with Ada.Calendar;
with Ada.Streams.Stream_IO;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Moonshine_Interface;
with ELP_Queue;
with KV_Cache_Manager;
with Interfaces.C; use Interfaces.C;
with SD_Manager;

--  ===========================================================================
--  SERVER QUIRKS & DISCOVERED WORKAROUNDS
--  ===========================================================================
--  [QUIRK-S01] [ALL] Pre-existing unload crash (exit code -1) — FIXED
--  After QWEN_0_8B processes a request and ELP0 releases the model, the
--  server could crash with exit code -1 (kratos signal isolation).  Root
--  cause: Idle_Monitor unloaded QWEN_0_8B via Llama_Free after 30s
--  inactivity, triggering a ggml-metal GPU buffer race.  FIXED [macOS]:
--  QWEN_0_8B is now exempt from Idle_Monitor unloading and kept permanently
--  loaded.  LINUX-COMPAT / Android-Termux: On Linux (no ggml-metal) remove
--  the Snowball_Enaga_ShortNetworkAnswer exemption in Idle_Monitor to allow aggressive unloading.
--  For smartphone / Termux targets, also consider lowering Idle_Monitor
--  timeout from 30s to 10-15s for tighter memory pressure response.
--  See QUIRK-M03 for details.
--
--  [QUIRK-S02] [ALL] Port 11420 binding with retry
--  The server tries to bind to port 11420 with up to 3 retries (2s apart).
--  If the port is still in use (e.g., stale server from a previous crash),
--  a [BUGCHECK] message is printed with the port conflict details.
--  Kill the stale process: kill $(lsof -ti:11420)
--
--  [QUIRK-S03] [ALL] ELP Queue utilization logging
--  Every 5 seconds, the main loop logs ELP queue depth and utilization.
--  This is the primary diagnostic for whether background tasks (ELP0) are
--  keeping up or the system is overloaded with user requests (ELP1).
--
--  [QUIRK-S04] [macOS] Moonshine model path
--  The Moonshine STT model is loaded from a hardcoded relative path:
--    "../moonshine/models/download.moonshine.ai/model/medium-streaming-en/" &
--    "quantized"
--  LINUX-COMPAT: This path is the same on Linux, but the model files may
--  reside at a different location.  The path is relative to the CWD at
--  startup (Adelaide_Lite/ when run via run.py).
--
--  [QUIRK-S05] [ALL] Startup health ping watchdog
--  After binding port 11420, the server pings its own /api/power endpoint
--  every 3 seconds.  If no response is received for 60 seconds total,
--  the process exits with code 69.  A warning is printed every 3 seconds
--  before the final bugcheck so the user can see what's happening.
--  ===========================================================================

procedure Adelaide_Server is

   function Get_Port return Natural;
   function Get_Host return String;

   --  [DO NOT REMOVE] C FFI for graceful shutdown (SIGINT/SIGTERM)
   procedure Install_Shutdown_Handlers;
   pragma Import (C, Install_Shutdown_Handlers, "install_shutdown_handlers");
   function Is_Shutdown_Requested return Interfaces.C.int;
   pragma Import (C, Is_Shutdown_Requested, "is_shutdown_requested");

   --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
   --  C import to force unbuffered stdout/stderr.  When run.py launches this
   --  server via subprocess.Popen(), stdout becomes a pipe (not a terminal).
   --  C stdio defaults to full buffering (8KB) on pipes, so Ada.Text_IO.Put_Line
   --  output sits in the C buffer and is never flushed.  The server runs fine
   --  but is completely invisible — no banner, no init logs, no API responses.
   --  Call these as the VERY FIRST thing in main(), before any Put_Line.
   procedure Force_Stdout_Unbuffered;
   pragma Import (C, Force_Stdout_Unbuffered, "force_stdout_unbuffered");
   procedure Force_Stderr_Unbuffered;
   pragma Import (C, Force_Stderr_Unbuffered, "force_stderr_unbuffered");

   --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
   --  ==================================================================
   --  INIT PHASE EPOCH CLOCK
   --  ==================================================================
   --  This task prints the current epoch time (in seconds) every 10
   --  seconds during the initialization phase.  It starts immediately
   --  when the binary launches and stops once the health ping succeeds.
   --
   --  WHY THIS EXISTS: During init, the server may hang silently
   --  (e.g., Moonshine model load, SQLite lock, protected object
   --  deadlock).  Without this clock, there's no way to tell if the
   --  process is alive or stuck.  The epoch timestamps let you:
   --    a) Confirm the process is still running (timestamps keep going)
   --    b) Measure exactly how long each init step takes
   --    c) Correlate with system logs (syslog, dmesg use epoch time)
   --
   --  The clock prints in a distinct color (yellow) so it's easy to
   --  spot in scrollback.  Format: [InitClock] EPOCH: <seconds>
   --  ----------------------------------------------------------------
   protected Init_Clock_Control is
      procedure Stop_Clock;
      function Is_Running return Boolean;
   private
      Running : Boolean := True;
   end Init_Clock_Control;

   protected body Init_Clock_Control is
      procedure Stop_Clock is
      begin
         Running := False;
      end Stop_Clock;

      function Is_Running return Boolean is
      begin
         return Running;
      end Is_Running;
   end Init_Clock_Control;

   task Init_Clock_Task is
      entry Start;
   end Init_Clock_Task;

   task body Init_Clock_Task is
      Epoch_Sec : Ada.Calendar.Time;
   begin
      accept Start;
      while Init_Clock_Control.Is_Running loop
         Epoch_Sec := Ada.Calendar.Clock;
         Put_Line (Character'Val (27) & "[33m" &
                   "[ClockLogAnchor]" & AnsiAda.Reset &
                   " EPOCH:" &
                   Ada.Calendar.Seconds (Epoch_Sec)'Img);
         delay 10.0;
      end loop;
   end Init_Clock_Task;

   --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
   --  AWS HTTP server handle and configuration object.
   --  Conf is populated from AWS defaults then overridden with our port/host.
   WS   : AWS.Server.HTTP;
   Conf : AWS.Config.Object := AWS.Config.Get_Current;

   --  Port/Host resolution: args > env vars > defaults
   function Get_Port return Natural is
   begin
      for I in 1 .. Ada.Command_Line.Argument_Count loop
         if Ada.Command_Line.Argument (I) = "--port"
           and then I < Ada.Command_Line.Argument_Count
         then
            return Natural'Value (Ada.Command_Line.Argument (I + 1));
         end if;
      end loop;
      if Ada.Environment_Variables.Exists ("ADLAIDE_SERVER_PORT") then
         return Natural'Value
           (Ada.Environment_Variables.Value ("ADLAIDE_SERVER_PORT"));
      end if;
      return 11420;
   end Get_Port;

   function Get_Host return String is
   begin
      for I in 1 .. Ada.Command_Line.Argument_Count loop
         if Ada.Command_Line.Argument (I) = "--host"
           and then I < Ada.Command_Line.Argument_Count
         then
            return Ada.Command_Line.Argument (I + 1);
         end if;
      end loop;
      if Ada.Environment_Variables.Exists ("ADLAIDE_SERVER_HOST") then
         return Ada.Environment_Variables.Value ("ADLAIDE_SERVER_HOST");
      end if;
      return "0.0.0.0";
   end Get_Host;

   --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
   --  Max_Retries: How many times we attempt to bind port 11420 before
   --  giving up with a [BUGCHECK].  Each retry waits 2 seconds.
   --  Retry_Count: Tracks the current attempt number.
   --  Started: Set to True once AWS.Server.Start succeeds.
   Max_Retries : constant := 3;
   Retry_Count : Natural := 0;
   Started     : Boolean := False;
begin
   --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
   --  CRITICAL: Force unbuffered stdout/stderr BEFORE any Put_Line.
   --  When run.py launches via subprocess.Popen(), stdout becomes a pipe.
   --  C stdio defaults to full buffering (8KB) on pipes, so Ada.Text_IO
   --  output is stuck in the C buffer and never flushed.  This makes the
   --  server invisible — no banner, no init logs, no API responses.
   --  This MUST be the absolute first statement in the main block.
   Force_Stdout_Unbuffered;
   Force_Stderr_Unbuffered;

   declare
      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
      --  Start_Time: Captured at the very first line of the main block.
      --  All [Init-V] verbose prints compute uptime relative to this.
      Start_Time : constant Ada.Real_Time.Time := Ada.Real_Time.Clock;
   begin
      --  ==================================================================
      --  ENFORCE ENVIRONMENT
      --  ==================================================================
      Ada.Environment_Variables.Set 
        ("HF_HOME", Ada.Directories.Current_Directory & "/model");

      --  ==================================================================
      --  SINGLE-INSTANCE LOCK (FIRST CHECK)
      --  ==================================================================
      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
      --  Before ANYTHING else (before banner, before init), check if
      --  another adelaide_server is already running.  This prevents
      --  port conflicts, database locks, and split-brain state.
      --  ----------------------------------------------------------------
      if Watchdog_IPC.Check_Single_Instance then
         Put_Line (Character'Val (27) & "[31m" &
                   "[FATAL] Another adelaide_server instance is " &
                   "already running!" &
                   Character'Val (27) & "[0m");
         Put_Line (Character'Val (27) & "[31m" &
                   " Kill the existing instance: kill $(cat " &
                   "run/adelaide_server.pid)" &
                   Character'Val (27) & "[0m");
         Ada.Command_Line.Set_Exit_Status (Ada.Command_Line.Failure);
         return;
      end if;

      --  [DO NOT REMOVE] Install SIGINT/SIGTERM handlers for
      --  graceful shutdown.
      --  Without this, Ctrl+C kills the process immediately, leaving stale
      --  PID/heartbeat files, and the watchdog restarts the server.
      Install_Shutdown_Handlers;

      --  ==================================================================
      --  STEP 1-4: Core subsystem initialization
      --  ==================================================================
      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
      --  The ASCII art banner is printed first for visual confirmation that
      --  the binary started.  If you don't see this, the Ada runtime or
      --  dynamic linker failed before main().
      Put_Line ("    ___       __     __      _     __ ");
      Put_Line ("   /   | ____/ /__  / /___ _(_)___/ /__ ");
      Put_Line ("  / /| |/ __  / _ \/ / __ `/ / __  / _ \ ");
      Put_Line (" / ___ / /_/ /  __/ / /_/ / / /_/ /  __/ ");
      Put_Line ("/_/  |_\__,_/\___/_/\__,_/_/\__,_/\___/ ");
      Put_Line ("");

      --  ==================================================================
      --  STEP 0: Disk Read Benchmark (1GB sequential from GGUF)
      --  ==================================================================
      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
      --  Reads 1GB sequentially from the largest GGUF model file to
      --  measure storage throughput. This tells us how fast model loading
      --  will be and categorizes the hardware for the user.
      --
      --  Categories:
      --    <1 MB/s   = Apollo Mission Computer (Unusable)
      --    <=5 MB/s  = USB Powered? (Almost Unusable)
      --    <=30 MB/s = Potato (Why Bother)
      --    <=100 MB/s = HDD Spinning Drive (Ultra Low)
      --    <=500 MB/s = SSHD Drive (Very Low)
      --    <=3000 MB/s = SATA SSD (Mildly Low)
      --    <=8000 MB/s = NVMe SSD (Low)
      --    <=14000 MB/s = NVMe SSD (Medium/Standard) RECOMMENDED MIN
      --    <=18000 MB/s = NVMe SSD (Medium to High)
      --    <=25000 MB/s = NVMe SSD (High)
      --    <=90000 MB/s = NVMe SSD (Mildly High)
      --    <=200000 MB/s = Next Generation Drive (Very High)
      --    >200000 MB/s = Next Generation Drive (Ultra High)
      --  ----------------------------------------------------------------
      declare
         GGUF_Path : constant String :=
           "model/Mythos9bHybridq4.gguf";
         --  Target: 1GB = 1024 * 1024 * 1024 bytes
         Target_Bytes : constant Long_Long_Integer := 1024 * 1024 * 1024;
         --  Read in 1MB chunks to avoid huge stack allocations
         Chunk_Size   : constant := 1024 * 1024;
         Buffer       : Ada.Streams.Stream_Element_Array
           (1 .. Ada.Streams.Stream_Element_Offset (Chunk_Size));
         Bytes_Read   : Long_Long_Integer := 0;
         F            : Ada.Streams.Stream_IO.File_Type;
         Last         : Ada.Streams.Stream_Element_Offset;
         T_Start      : Ada.Calendar.Time;
         T_End        : Ada.Calendar.Time;
         Elapsed      : Duration;
         MB_per_sec   : Long_Long_Integer;
         Category     : Unbounded_String;
         Warning      : Unbounded_String;
      begin
         --  Verbose: announce benchmark start
         Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Init-V]" &
                   AnsiAda.Reset & "+" &
                   Trim (Duration'Image (Ada.Real_Time.To_Duration
                     (Ada.Real_Time.Clock - Start_Time)), Both) &
                   "s STEP 0: Starting disk read benchmark (1GB from " &
                   GGUF_Path & ")...");

         --  Try to open the GGUF file; skip if not found
         begin
            Ada.Streams.Stream_IO.Open (F, Ada.Streams.Stream_IO.In_File,
                                        GGUF_Path);
         exception
            when others =>
               Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Init-V]" &
                         AnsiAda.Reset & "+" &
                         Trim (Duration'Image (Ada.Real_Time.To_Duration
                           (Ada.Real_Time.Clock - Start_Time)), Both) &
                         "s STEP 0 SKIPPED: GGUF file not found at " &
                         GGUF_Path);
               goto Skip_Disk_Bench;
         end;

         --  Read 1GB sequentially, timing the whole operation
         T_Start := Ada.Calendar.Clock;
         while not Ada.Streams.Stream_IO.End_Of_File (F)
               and then Bytes_Read < Target_Bytes
         loop
            Ada.Streams.Stream_IO.Read (F, Buffer, Last);
            exit when Integer (Last) = 0;
            Bytes_Read := Bytes_Read +
              Long_Long_Integer (Last);
         end loop;
         T_End := Ada.Calendar.Clock;
         Ada.Streams.Stream_IO.Close (F);

         --  Calculate throughput in MB/s
         Elapsed := Ada.Calendar."-" (T_End, T_Start);
         if Elapsed > 0.0 then
            MB_per_sec := Long_Long_Integer (
              Long_Long_Float (Bytes_Read) / (1024.0 * 1024.0) /
              Long_Long_Float (Elapsed));
         else
            MB_per_sec := 0;
         end if;

         --  ----------------------------------------------------------------
         --  Categorize throughput and assign label + warning
         --  ----------------------------------------------------------------
         if MB_per_sec < 1 then
            --  Apollo Mission Computer: slower than a 1960s spacecraft
            Category := To_Unbounded_String ("Apollo Mission Computer");
            Warning  := To_Unbounded_String ("(Unusable)");
         elsif MB_per_sec <= 5 then
            --  USB Powered?: likely a slow USB stick or network mount
            Category := To_Unbounded_String ("USB Powered?");
            Warning  := To_Unbounded_String ("(Almost Unusable)");
         elsif MB_per_sec <= 30 then
            --  Potato: the machine is actively fighting you
            Category := To_Unbounded_String ("Potato");
            Warning  := To_Unbounded_String ("(Why Bother)");
         elsif MB_per_sec <= 100 then
            --  HDD Spinning Drive: mechanical platters, 5400/7200 RPM
            Category := To_Unbounded_String ("HDD Spinning Drive");
            Warning  := To_Unbounded_String ("(Ultra Low)");
         elsif MB_per_sec <= 500 then
            --  SSHD Drive: hybrid or slow SATA SSD
            Category := To_Unbounded_String ("SSHD Drive");
            Warning  := To_Unbounded_String ("(Very Low)");
         elsif MB_per_sec <= 3000 then
            --  SATA SSD: typical 2.5" SSD, capped at SATA III ~550 MB/s
            --  (benchmark reads cached/OS-buffered, so can appear higher)
            Category := To_Unbounded_String ("SATA SSD");
            Warning  := To_Unbounded_String ("(Mildly Low)");
         elsif MB_per_sec <= 8000 then
            --  NVMe SSD: entry-level NVMe, PCIe Gen3 x2 or similar
            Category := To_Unbounded_String ("NVMe SSD");
            Warning  := To_Unbounded_String ("(Low)");
         elsif MB_per_sec <= 14000 then
            --  NVMe SSD: standard NVMe, PCIe Gen3 x4 / Gen4 x2
            --  THIS IS THE RECOMMENDED BARE MINIMUM for model loading
            Category := To_Unbounded_String ("NVMe SSD");
            Warning  := To_Unbounded_String
              ("(Medium/Standard) RECOMMENDED BARE MINIMUM");
         elsif MB_per_sec <= 18000 then
            --  NVMe SSD: good NVMe, PCIe Gen4 x4
            Category := To_Unbounded_String ("NVMe SSD");
            Warning  := To_Unbounded_String ("(Medium to High)");
         elsif MB_per_sec <= 25000 then
            --  NVMe SSD: fast NVMe, PCIe Gen4 x4 / Gen5 x2
            Category := To_Unbounded_String ("NVMe SSD");
            Warning  := To_Unbounded_String ("(High)");
         elsif MB_per_sec <= 90000 then
            --  NVMe SSD: top-tier consumer NVMe, PCIe Gen5 x4
            Category := To_Unbounded_String ("NVMe SSD");
            Warning  := To_Unbounded_String ("(Mildly High)");
         elsif MB_per_sec <= 200000 then
            --  Next Generation Drive: unified memory, extreme bandwidth
            Category := To_Unbounded_String ("Next Generation Drive");
            Warning  := To_Unbounded_String ("(Very High)");
         else
            --  Next Generation Drive: extreme bandwidth tier
            Category := To_Unbounded_String ("Next Generation Drive");
            Warning  := To_Unbounded_String ("(Ultra High)");
         end if;

         --  Print the result with category
         Put_Line (AnsiAda.Foreground (AnsiAda.Yellow) & "[Hardware]" &
                   AnsiAda.Reset & " Storage: " &
                   Long_Long_Integer'Image (MB_per_sec) & " MB/s " &
                   To_String (Category) & " " & To_String (Warning));
         Put_Line (AnsiAda.Foreground (AnsiAda.Yellow) & "[Hardware]" &
                   AnsiAda.Reset & " Read " &
                   Long_Long_Integer'Image (Bytes_Read / (1024 * 1024)) &
                   " MB in" & Duration'Image (Elapsed) & "s from " &
                   GGUF_Path);

         --  Print warning if below recommended minimum
         if MB_per_sec < 8001 then
            Put_Line (AnsiAda.Foreground (AnsiAda.Red) & "[Hardware]" &
                      AnsiAda.Reset &
                      " WARNING: Storage below recommended minimum" &
                      " (8001 MB/s). Model loading will be slow.");
         end if;

         --  Verbose: announce benchmark complete
         Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Init-V]" &
                   AnsiAda.Reset & "+" &
                   Trim (Duration'Image (Ada.Real_Time.To_Duration
                     (Ada.Real_Time.Clock - Start_Time)), Both) &
                   "s STEP 0 DONE: Disk benchmark complete.");

         <<Skip_Disk_Bench>>
      end;

      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
      --  Start the init phase epoch clock.  This prints the current
      --  epoch time every 10 seconds so we can see if the process is
      --  alive during long init steps (Moonshine load, SQLite lock, etc).
      Init_Clock_Task.Start;

      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
      --  Model_Manager.Initialize does three things:
      --    a) Calls Llama_Backend_Init — loads the ggml-metal/CPU backends
      --       for hardware-accelerated inference on Apple Silicon.
      --    b) Calls Database_Manager.Initialize — opens SQLite databases
      --       (adelaide_memory.db, literatureRefIndex.db) and creates
      --       tables if they don't exist.
      --    c) Calls ELP_Queue.Initialize — starts the ELP monitor task
      --       that prints queue depth every 5 seconds.
      --    d) Starts the Idle_Monitor task that unloads idle models after
      --       30 seconds to free GPU memory.
      --  This MUST complete before any other init because Knowledge_Manager
      --  and Scheduler_Manager depend on the database and ELP queue being
      --  ready.
      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
      --  Verbose: wraps Model_Manager.Initialize so we can see timing.
      Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Init-V]" &
                AnsiAda.Reset & "+" &
                Trim (Duration'Image (Ada.Real_Time.To_Duration
                  (Ada.Real_Time.Clock - Start_Time)), Both) &
                "s STEP 1: Calling Model_Manager.Initialize...");
      Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Main]" &
                AnsiAda.Reset &
                " Initializing Adelaide Intelligence Backend...");
      Model_Manager.Initialize;
      Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Init-V]" &
                AnsiAda.Reset & "+" &
                Trim (Duration'Image (Ada.Real_Time.To_Duration
                  (Ada.Real_Time.Clock - Start_Time)), Both) &
                "s STEP 1 DONE: Model_Manager.Initialize returned.");

      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
      --  Verbose: wraps Knowledge_Manager.Initialize.
      Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Init-V]" &
                AnsiAda.Reset & "+" &
                Trim (Duration'Image (Ada.Real_Time.To_Duration
                  (Ada.Real_Time.Clock - Start_Time)), Both) &
                "s STEP 2: Calling Knowledge_Manager.Initialize...");
      Knowledge_Manager.Initialize;
      Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Init-V]" &
                AnsiAda.Reset & "+" &
                Trim (Duration'Image (Ada.Real_Time.To_Duration
                  (Ada.Real_Time.Clock - Start_Time)), Both) &
                "s STEP 2 DONE: Knowledge_Manager.Initialize returned.");

      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
      --  Verbose: wraps Scheduler_Manager.Initialize.
      Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Init-V]" &
                AnsiAda.Reset & "+" &
                Trim (Duration'Image (Ada.Real_Time.To_Duration
                  (Ada.Real_Time.Clock - Start_Time)), Both) &
                "s STEP 3: Calling Scheduler_Manager.Initialize...");
      Scheduler_Manager.Initialize;
      Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Init-V]" &
                AnsiAda.Reset & "+" &
                Trim (Duration'Image (Ada.Real_Time.To_Duration
                  (Ada.Real_Time.Clock - Start_Time)), Both) &
                "s STEP 3 DONE: Scheduler_Manager.Initialize returned.");

      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
      --  Verbose: wraps Watchdog_IPC.Init.
      Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Init-V]" &
                AnsiAda.Reset & "+" &
                Trim (Duration'Image (Ada.Real_Time.To_Duration
                  (Ada.Real_Time.Clock - Start_Time)), Both) &
                "s STEP 4: Calling Watchdog_IPC.Init...");
      Watchdog_IPC.Init;
      Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Init-V]" &
                AnsiAda.Reset & "+" &
                Trim (Duration'Image (Ada.Real_Time.To_Duration
                  (Ada.Real_Time.Clock - Start_Time)), Both) &
                "s STEP 4 DONE: Watchdog_IPC.Init returned.");

      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
      --  Verbose: wraps SD_Manager.Initialize for image generation.
      Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Init-V]" &
                AnsiAda.Reset & "+" &
                Trim (Duration'Image (Ada.Real_Time.To_Duration
                  (Ada.Real_Time.Clock - Start_Time)), Both) &
                "s STEP 4.5: Calling SD_Manager.Initialize...");
      SD_Manager.Initialize
        (Flux_Diffusion => "model/flux1-schnell.gguf",
         Flux_Clip_L    => "model/clip_l.safetensors",
         Flux_T5XXL     => "model/flux1-t5xxl.gguf",
         Flux_VAE       => "model/ae.safetensors",
         Refiner_Model  => "model/sd-refinement.gguf");
      Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Init-V]" &
                AnsiAda.Reset & "+" &
                Trim (Duration'Image (Ada.Real_Time.To_Duration
                  (Ada.Real_Time.Clock - Start_Time)), Both) &
                "s STEP 4.5 DONE: SD_Manager.Initialize returned.");

      --  ==================================================================
      --  STEP 5: Start ELP0 background tasks BEFORE HTTP bind
      --  ==================================================================
      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
      --  CRITICAL FIX: Start Knowledge_Manager.Start_Tasks here, BEFORE
      --  binding the HTTP server.  This starts:
      --    - Indexing_Task: Parses references.bib and indexes chunks
      --    - Native_Crawl_Task: Crawls the filesystem for .adb/.c/.md
      --      files and generates embeddings at ELP0 priority
      --    - Proactive_Cache_Task: Predicts follow-up questions and
      --      pre-caches answers at ELP0 priority
      --    - Salience_Maintenance_Task, Telemetry_Sync_Task, etc.
      --
      --  WHY THIS MATTERS: Previously, Start_Tasks was called AFTER
      --  AWS.Server.Start.  This meant ELP0 stayed at 0 until the first
      --  HTTP request arrived.  Worse, if Moonshine init hung, the
      --  server never bound at all — so Start_Tasks was never called,
      --  and ELP0 producers never started.  The system appeared alive
      --  (ELP monitor printing zeros) but was functionally dead.
      --
      --  By starting tasks here, ELP0 producers begin enqueuing
      --  immediately while the HTTP server binds in the next step.
      --  This eliminates the chicken-and-egg dependency.
      --  ----------------------------------------------------------------
      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
      --  Verbose: wraps Knowledge_Manager.Start_Tasks.
      Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Init-V]" &
                AnsiAda.Reset & "+" &
                Trim (Duration'Image (Ada.Real_Time.To_Duration
                  (Ada.Real_Time.Clock - Start_Time)), Both) &
                "s STEP 5: Calling Knowledge_Manager.Start_Tasks...");
      Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Main]" &
                AnsiAda.Reset &
                " Starting background tasks (ELP0 producers)...");
      Knowledge_Manager.Start_Tasks;
      Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Init-V]" &
                AnsiAda.Reset & "+" &
                Trim (Duration'Image (Ada.Real_Time.To_Duration
                  (Ada.Real_Time.Clock - Start_Time)), Both) &
                "s STEP 5 DONE: Knowledge_Manager.Start_Tasks returned.");

      --  ==================================================================
      --  STEP 6: Bind HTTP server
      --  ==================================================================
      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
      --  Configure AWS to listen on all interfaces (0.0.0.0).
      --  Port is read from ADLAIDE_SERVER_PORT env var, default 11420.
      --  Reuse_Address allows rebinding even if a previous server is in
      --  TIME_WAIT state.  This is critical for crash-recovery scenarios
      --  where the old process was killed but the socket hasn't fully
      --  released yet.
      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
      --  Verbose: wraps HTTP server config and bind.
      declare
         Port : constant Natural := Get_Port;
         Host : constant String := Get_Host;
      begin
         Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Init-V]" &
                   AnsiAda.Reset & "+" &
                   Trim (Duration'Image (Ada.Real_Time.To_Duration
                     (Ada.Real_Time.Clock - Start_Time)), Both) &
                   "s STEP 6: Configuring AWS HTTP server on " & Host & ":" &
                   Natural'Image (Port) & "...");
         AWS.Config.Set.Server_Port (Conf, Port);
         AWS.Config.Set.Server_Host (Conf, Host);
         AWS.Config.Set.Reuse_Address (Conf, True);
         --  [VITAL-DO-NOT-REMOVE] Mandated by user.
         --  Set long timeouts so AWS doesn't drop connections while waiting
         --  for the large QWEN_9B model to load from disk!
         AWS.Config.Set.Send_Timeout (Conf, 600.0);
         AWS.Config.Set.Receive_Timeout (Conf, 600.0);
      end;
      Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Init-V]" &
                AnsiAda.Reset & "+" &
                Trim (Duration'Image (Ada.Real_Time.To_Duration
                  (Ada.Real_Time.Clock - Start_Time)), Both) &
                "s STEP 6: AWS config set. Entering bind retry loop...");

      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
      --  Read back the configured port so we can use it in the health
      --  ping URL and log messages.  This ensures we always ping the
      --  correct port, even if the default changes in the future.
      declare
         Server_Port : constant Natural := AWS.Config.Server_Port (Conf);
      begin
         Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Main]" &
                   AnsiAda.Reset &
                   " Adelaide-Lite Server starting on port" &
                   Natural'Image (Server_Port) & "...");

         --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
         --  Port binding with retry loop.
         --  AWS.Server.Start can fail if:
         --    a) Another process holds the port (stale server from crash)
         --    b) The OS hasn't released the socket yet (TIME_WAIT)
         --    c) A firewall or SELinux policy blocks the bind
         --  We retry up to Max_Retries times with a 2-second delay between
         --  attempts.  If all retries fail, we print [BUGCHECK] and exit.
         while not Started and then Retry_Count < Max_Retries loop
            begin
               AWS.Server.Start
                 (Web_Server => WS,
                  Callback   => Adelaide_Server_Pkg.Dispatch'Access,
                  Config     => Conf);
               Started := True;
               --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
               --  Verbose: confirms port bind succeeded.
               Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Init-V]" &
                         AnsiAda.Reset & "+" &
                         Trim (Duration'Image (Ada.Real_Time.To_Duration
                           (Ada.Real_Time.Clock - Start_Time)), Both) &
                         "s STEP 6 DONE: AWS.Server.Start SUCCEEDED on port" &
                         Natural'Image (Server_Port) & ".");
            exception
               when E : others =>
                  Retry_Count := Retry_Count + 1;
                  if Retry_Count < Max_Retries then
                     Put_Line
                       ("[Warning] Port" & Natural'Image (Server_Port) &
                        " might be bound. Retrying in 2 seconds (" &
                        Natural'Image (Retry_Count) & "/" &
                        Natural'Image (Max_Retries) & ")...");
                     delay 2.0;
                  else
                     --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
                     --  Final retry failed.  Print the [BUGCHECK] banner in
                     --  red so it's visible in scrollback.
                     Put_Line (Character'Val (27) & "[31m" &
                               "[BUGCHECK] Failed to bind to port" &
                               Natural'Image (Server_Port) & ": " &
                               Ada.Exceptions.Exception_Message (E) &
                               Character'Val (27) & "[0m");
                     Ada.Command_Line.Set_Exit_Status
                       (Ada.Command_Line.Failure);
                     return;
                  end if;
            end;
         end loop;

         --  ==================================================================
         --  STEP 7: Startup health ping watchdog
         --  ==================================================================
         --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
         --  After the HTTP server is bound, we ping our own /api/power
         --  endpoint every 3 seconds to verify the server is actually
         --  responding to requests.  This catches two failure modes:
         --
         --    a) The server bound the port but the dispatch callback is
         --       broken (e.g., null access, constraint error on first
         --       request).  The port is open but requests hang or crash.
         --
         --    b) The server bound the port but the main loop is stuck
         --       (e.g., a protected object deadlock, or a blocking FFI
         --       call that never returns).
         --
         --  If no response is received within 60 seconds of startup, the
         --  process exits with code 69 so run.sh displays the error banner.
         --
         --  The 3-second interval is chosen to be:
         --    - Fast enough to catch hangs quickly
         --    - Slow enough to not spam the log during normal startup
         --    - Aligned with the Python power monitor's check interval
         --
         --  WARNING messages are printed every 3 seconds BEFORE the final
         --  bugcheck so the user can see the watchdog counting down.
         --  ----------------------------------------------------------------
         declare
            --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
            --  Startup_Deadline: Absolute time when the watchdog gives up.
            Startup_Deadline : constant Ada.Real_Time.Time :=
              Ada.Real_Time.Clock + Ada.Real_Time.Seconds (60);

            --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
            --  Ping_Count: How many pings we've sent so far.
            Ping_Count    : Natural := 0;

            --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
            --  Health_URL: The endpoint we ping.  /api/power is lightweight.
            Health_URL : constant String :=
              "http://127.0.0.1:" &
              Trim (Natural'Image (Server_Port), Ada.Strings.Both) &
              "/api/power";
         begin
            --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
            --  Verbose: confirms we entered the health ping loop.
            Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Init-V]" &
                      AnsiAda.Reset & "+" &
                      Trim (Duration'Image (Ada.Real_Time.To_Duration
                        (Ada.Real_Time.Clock - Start_Time)), Both) &
                      "s STEP 7: Entering health ping loop. URL=" &
                      Health_URL);
            --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
            --  Ping loop: Send GET requests to /api/power every 3 seconds.
            loop
               Ping_Count := Ping_Count + 1;

               --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
               --  Attempt the health ping.
               declare
                  Response : AWS.Response.Data;
               begin
                  Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) &
                            "[Init-V]" & AnsiAda.Reset & "+" &
                            Trim (Duration'Image (Ada.Real_Time.To_Duration
                              (Ada.Real_Time.Clock - Start_Time)), Both) &
                            "s STEP 7: Sending health ping #" &
                            Natural'Image (Ping_Count) & " to " & Health_URL);
                  Response := AWS.Client.Get (Health_URL);
                  --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
                  --  Check the HTTP status code.
                  if AWS.Response.Status_Code (Response) in Success then
                     --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
                     --  Stop the init phase epoch clock.
                     Init_Clock_Control.Stop_Clock;
                     --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
                     --  Verbose: confirms health ping passed.
                     Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) &
                               "[Init-V]" & AnsiAda.Reset & "+" &
                               Trim (Duration'Image (Ada.Real_Time.To_Duration
                                 (Ada.Real_Time.Clock - Start_Time)), Both) &
                               "s STEP 7 DONE: Health ping OK on ping" &
                               Natural'Image (Ping_Count) & ". Status=" &
                               AWS.Response.Status_Code (Response)'Img);
                     Put_Line (AnsiAda.Foreground (AnsiAda.Green) &
                              "[Watchdog]" & AnsiAda.Reset &
                               " Health ping OK -- server responding on port" &
                               Natural'Image (Server_Port) & ".");
                     exit;
                  end if;
               exception
                  when E : others =>
                     --  Connection refused, timeout, or any other error.
                     Put_Line (AnsiAda.Foreground (AnsiAda.Yellow) &
                               "[Init-V]" & AnsiAda.Reset & "+" &
                               Trim (Duration'Image (Ada.Real_Time.To_Duration
                                 (Ada.Real_Time.Clock - Start_Time)), Both) &
                               "s STEP 7: Health ping #" &
                               Natural'Image (Ping_Count) & " FAILED: " &
                               Ada.Exceptions.Exception_Message (E));
               end;

               --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
               --  Exceeded the 60-second deadline.
               if Ada.Real_Time.Clock > Startup_Deadline then
                  Init_Clock_Control.Stop_Clock;
                  Put_Line ("");
                  Put_Line (Character'Val (27) & "[31m" &
                            "[WARNING BEFORE BUGCHECK]: No Response from " &
                            "Adelaide server after" & Ping_Count'Img &
                            " health pings (" & Health_URL & ")." &
                            Character'Val (27) & "[0m");
                  Put_Line (Character'Val (27) & "[31m" &
                            "[BUGCHECK] Startup watchdog: server did not " &
                            "respond to health ping within 60 seconds.  " &
                            "Exiting with code 69." &
                            Character'Val (27) & "[0m");
                  Ada.Command_Line.Set_Exit_Status (69);
                  return;
               end if;

               --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
               --  Print a warning every 3 seconds so the user can see the
               --  watchdog counting down.
               Put_Line (Character'Val (27) & "[33m" &
                         "[Watchdog]" & AnsiAda.Reset &
                         " WARNING: No response from Adelaide server yet. " &
                         "Ping" & Ping_Count'Img &
                         " -- retrying in 3s (deadline in" &
                         Natural'Image (60 - (Ping_Count * 3)) & "s)...");

               delay 3.0;
            end loop;
         end;
      end;

      --  ==================================================================
      --  STEP 8: Moonshine STT model load (after HTTP bind)
      --  ==================================================================
      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
      --  CRITICAL FIX: Moonshine init moved AFTER the HTTP server is live.
      --  Previously, Init_Moonshine was called before AWS.Server.Start,
      --  which blocked the port bind for 30-60 seconds while the 500MB+
      --  ONNX model files were loaded into memory.
      Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Main]" &
                AnsiAda.Reset & " Initializing Moonshine (STT)...");
      Moonshine_Interface.Init_Moonshine
        ("../moonshine/models/download.moonshine.ai/model/" &
         "medium-streaming-en/quantized");

      --  ==================================================================
      --  STEP 9: Server is fully up
      --  ==================================================================
      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
      --  This message confirms all initialization steps completed.
      Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Main]" &
                AnsiAda.Reset &
                " Server is UP. Press Q to shutdown (or kill if background).");

      --  ==================================================================
      --  STEP 10: Main heartbeat loop
      --  ==================================================================
      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
      --  The main loop runs forever, updating the watchdog heartbeat
      --  every 1 second and printing ELP queue stats every 5 seconds.
      --  ----------------------------------------------------------------
      declare
         Heartbeat_Count : Natural := 0;
         Alive_Count     : Natural := 0;
      begin
         loop
            --  [VITAL-DO-NOT-REMOVE] Catch-all exception handler for heartbeat loop.
            --  If ANY unknown/uncategorized exception occurs in the main loop,
            --  dump the full exception info with a red banner and RETRY after 10s.
            --  Server stays alive and continues serving.
            begin
               --  [DO NOT REMOVE] Graceful shutdown check (SIGINT/SIGTERM).
               if Is_Shutdown_Requested /= 0 then
                  Put_Line (AnsiAda.Foreground (AnsiAda.Yellow) &
                            "[Shutdown]" & AnsiAda.Reset &
                            " SIGINT/SIGTERM received. Cleaning up...");

                  --  Signal all Ada tasks to stop
                  Shutdown_Manager.Shutdown_Status.Request;
                  Watchdog_Manager.AWS_Server_Monitor.Deactivate;

                  --  KV Cache: No blocking save at shutdown
                  --  WHY: Background async saves will complete or die with process
                  --  This ensures instant shutdown (no waiting for disk I/O)
                  Put_Line (AnsiAda.Foreground (AnsiAda.Yellow) &
                            "[Shutdown]" & AnsiAda.Reset &
                            " KV Cache: async saves will complete in background...");

                  --  Write clean exit reason (not a crash)
                  Watchdog_IPC.Write_Exit_Reason
                    ("Clean Shutdown (SIGINT/SIGTERM)", 0);
                  --  Delete PID file so watchdog doesn't try to restart
                  if Ada.Directories.Exists ("run/adelaide_server.pid") then
                     Ada.Directories.Delete_File ("run/adelaide_server.pid");
                  end if;
                  if Ada.Directories.Exists ("run/adelaide_server.heartbeat") then
                     Ada.Directories.Delete_File
                       ("run/adelaide_server.heartbeat");
                  end if;
                  Put_Line (AnsiAda.Foreground (AnsiAda.Yellow) &
                            "[Shutdown]" & AnsiAda.Reset &
                            " Clean shutdown complete.");
                  return;
               end if;

               Watchdog_Manager.AWS_Server_Monitor.Heartbeat (Clock);
               Watchdog_IPC.Write_Heartbeat;
               Heartbeat_Count := Heartbeat_Count + 1;
               Alive_Count     := Alive_Count + 1;
               if Alive_Count >= 3 then
                  Alive_Count := 0;
                  Ada.Text_IO.Put_Line
                    (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Heartbeat]" &
                      AnsiAda.Reset & " Server alive - uptime " &
                     Trim (Duration'Image (Ada.Real_Time.To_Duration
                       (Ada.Real_Time.Clock - Start_Time)), Both) & "s" &
                     " | API: " & Adelaide_Server_Pkg.Get_Last_API);
               end if;
               if Heartbeat_Count >= 5 then
                  Heartbeat_Count := 0;
                  Ada.Text_IO.Put_Line
                    (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Main]" &
                     AnsiAda.Reset & " ELP Queue: " &
                     ELP_Queue.Utilization'Img & "% full (" &
                     ELP_Queue.Depth'Img & " pending)");
               end if;
               delay 1.0;
            exception
               when E : others =>
                  --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
                  --  UNKNOWN/CATEGORIZED ERROR: Full exception dump + red banner.
                  --  Server RETRIES after 10s delay. Never exits.
                  Ada.Text_IO.Put_Line
                     (AnsiAda.Foreground (AnsiAda.Red)
                      & "=========================================================="
                      & AnsiAda.Reset);
                  Ada.Text_IO.Put_Line
                     (AnsiAda.Foreground (AnsiAda.Red)
                      & "  !!! UNKNOWN ERROR / UNCATEGORIZED EXCEPTION !!!"
                      & AnsiAda.Reset);
                  Ada.Text_IO.Put_Line
                     (AnsiAda.Foreground (AnsiAda.Red)
                      & "  Exception: "
                      & Ada.Exceptions.Exception_Name (E)
                      & AnsiAda.Reset);
                  Ada.Text_IO.Put_Line
                     (AnsiAda.Foreground (AnsiAda.Red)
                      & "  Message: "
                      & Ada.Exceptions.Exception_Message (E)
                      & AnsiAda.Reset);
                  Ada.Text_IO.Put_Line
                     (AnsiAda.Foreground (AnsiAda.Red)
                      & "  Full Trace:"
                      & AnsiAda.Reset);
                  Ada.Text_IO.Put_Line
                     (Ada.Exceptions.Exception_Information (E));
                  Ada.Text_IO.Put_Line
                     (AnsiAda.Foreground (AnsiAda.Red)
                      & "=========================================================="
                      & AnsiAda.Reset);
                  Ada.Text_IO.Put_Line
                     (AnsiAda.Foreground (AnsiAda.Red)
                      & "  REPORT TO DEVELOPER! Retrying in 10s..."
                      & AnsiAda.Reset);
                  Ada.Text_IO.Put_Line
                     (AnsiAda.Foreground (AnsiAda.Red)
                      & "=========================================================="
                      & AnsiAda.Reset);
                  --  Retry with 10s delay — server stays alive
                  delay 10.0;
            end;
         end loop;
      end;

   --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
   --  ==================================================================
   --  Top-level exception handler
   --  ==================================================================
   exception
      when E : others =>
         Put_Line (AnsiAda.Foreground (AnsiAda.Red) & "[FATAL]" &
                   AnsiAda.Reset & " Server Error: " &
                   Ada.Exceptions.Exception_Message (E));
         Watchdog_IPC.Write_Exit_Reason
           ("Exception: " & Ada.Exceptions.Exception_Message (E), -1);
         Ada.Command_Line.Set_Exit_Status (Ada.Command_Line.Failure);
   end;

   --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
   Watchdog_IPC.Write_Exit_Reason ("Clean Shutdown", 0);
end Adelaide_Server;
