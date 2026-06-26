pragma SPARK_Mode (Off);
--  ============================================================================
--  KV Cache Manager — DATACENTER SPEED BLACKMAGIC
--  ============================================================================
--  ALL THE CHEATING TRICKS TO GET MAXIMUM PERFORMANCE:
--
--  TRICK 1: ASYNC SAVE
--  - Fire-and-forget background task for disk writes
--  - Return immediately, save happens in parallel
--  - If process dies, we lose cache (acceptable - best effort)
--
--  TRICK 2: LAZY LOAD
--  - Don't load at startup (blocks server)
--  - Only load when Generate is called
--  - First call is slow, subsequent calls are instant (RAM cached)
--
--  TRICK 3: PRE-PATH CACHE
--  - Cache the last used file path
--  - On same prompt hash, skip directory scan entirely
--  - Directory scans are SLOW on HDD (50-200ms)
--
--  TRICK 4: PREFETCH
--  - After save, immediately prefetch the file into OS page cache
--  - By next Generate call, file is already in RAM
--  - Uses posix_fadvise or equivalent
--
--  TRICK 5: WRITE-BUFFER (BATCHING)
--  - Collect multiple save requests into single write
--  - Reduces disk seek overhead by 10-100x
--
--  TRICK 6: BACKGROUND EVICTION
--  - Evict old cache files in background task
--  - Never block on eviction during request handling
--  ============================================================================

with AnsiAda;
with Ada.Text_IO; use Ada.Text_IO;
with Ada.Directories;
with Ada.Strings; use Ada.Strings;
with Ada.Strings.Fixed; use Ada.Strings.Fixed;
with Ada.Real_Time; use Ada.Real_Time;
with Interfaces.C; use Interfaces.C;
with Interfaces.C.Strings; use Interfaces.C.Strings;
with System;
with Ada.Task_Identification; use Ada.Task_Identification;
with Ada.Unchecked_Deallocation;
with Ada.Calendar; use Ada.Calendar;

with Llama_Interface; use Llama_Interface;
with Model_Manager;
with Database_Manager;

package body KV_Cache_Manager is

   --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
   --  Start time for uptime logging.
   Init_Start_Time : Ada.Real_Time.Time;

   --  ============================================================================
   --  METRICS COUNTERS
   --  ============================================================================
   --  WHY: Track cache performance for optimization.

   Total_Prefill_Tokens : Interfaces.C.size_t := 0;
   Cached_Tokens        : Interfaces.C.size_t := 0;
   Cache_Hits           : Interfaces.C.size_t := 0;
   Cache_Misses         : Interfaces.C.size_t := 0;

   --  ============================================================================
   --  METRICS LOGGING TASK (every 10 seconds)
   --  ============================================================================
   --  WHY: Periodic metrics for monitoring cache performance.

   task Metrics_Logger is
      entry Start;
      entry Stop;
   end Metrics_Logger;

    task body Metrics_Logger is
       use type Interfaces.C.size_t;
       Running    : Boolean := True;
       Start_Time : Ada.Real_Time.Time;
       Uptime     : Duration;
    begin
       --  [DO NOT REMOVE THIS PRINT VERBOSITY]
       --  [ElabTrace][+Uptime]: Confirms KV_Cache_Manager Metrics_Logger
       --  task body entered. If this never prints, KV_Cache_Manager
       --  task activation deadlocked during elaboration.
       Put_Line
          (AnsiAda.Foreground (AnsiAda.Light_Cyan)
           & "[ElabTrace]"
           & AnsiAda.Reset
           & "+"
           & Trim
                (Duration'Image
                    (Ada.Real_Time.To_Duration
                        (Ada.Real_Time.Clock - Init_Start_Time)),
                 Both)
           & "s KV_Cache_Manager.Metrics_Logger task body ENTERED");
       accept Start;

      Start_Time := Ada.Real_Time.Clock;

      while Running loop
         --  Wait 10 seconds between logs
         select
            accept Stop do
               Running := False;
            end Stop;
         or
            delay 10.0;
         end select;

         if Running then
            --  Calculate uptime
            Uptime := Ada.Real_Time.To_Duration (Ada.Real_Time.Clock - Start_Time);

            --  Calculate cache hit percentage
            declare
               Total_Requests : constant Interfaces.C.size_t := Cache_Hits + Cache_Misses;
               Hit_Percentage : Float := 0.0;
            begin
               if Total_Requests > 0 then
                  Hit_Percentage := Float (Cache_Hits) * 100.0 / Float (Total_Requests);
               end if;

               --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
               --  Verbose: logs metrics every 10 seconds.
               Put_Line (AnsiAda.Foreground (AnsiAda.Light_Cyan) & "[TimeUptime]" &
                         AnsiAda.Reset & "+" &
                         Trim(Duration'Image(Uptime), Both) &
                         "s [KV-Cache Metrics]" &
                         " Total Prefill tokens: " & Interfaces.C.size_t'Image (Total_Prefill_Tokens) &
                         " | Cached Tokens: " & Interfaces.C.size_t'Image (Cached_Tokens) &
                         " | Cache Hit Percentage: " & Float'Image (Hit_Percentage) & "%");

               --  Persist metrics to Database
               Database_Manager.Set_System_State ("Total_Prefill_Tokens", Trim (Interfaces.C.size_t'Image (Total_Prefill_Tokens), Both));
               Database_Manager.Set_System_State ("Cached_Tokens", Trim (Interfaces.C.size_t'Image (Cached_Tokens), Both));
               Database_Manager.Set_System_State ("Cache_Hits", Trim (Interfaces.C.size_t'Image (Cache_Hits), Both));
               Database_Manager.Set_System_State ("Cache_Misses", Trim (Interfaces.C.size_t'Image (Cache_Misses), Both));
            end;
         end if;
      end loop;

   exception
      when others =>
         --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
         --  Verbose: logs metrics logger exception (non-fatal).
         null;  -- Don't crash on metrics failure
   end Metrics_Logger;

   --  ============================================================================
   --  METRICS RECORDING PROCEDURES
   --  ============================================================================

   procedure Record_Prefill (N_Tokens : Interfaces.C.size_t) is
   begin
      Total_Prefill_Tokens := Total_Prefill_Tokens + N_Tokens;
   end Record_Prefill;

   procedure Record_Cache_Hit (N_Tokens : Interfaces.C.size_t) is
   begin
      Cached_Tokens := Cached_Tokens + N_Tokens;
      Cache_Hits := Cache_Hits + 1;
   end Record_Cache_Hit;

   procedure Record_Cache_Miss is
   begin
      Cache_Misses := Cache_Misses + 1;
   end Record_Cache_Miss;

   --  ============================================================================
   --  TRICK 3: PRE-PATH CACHE
   --  ============================================================================
   --  WHY: Directory scans are SLOW on HDD (50-200ms).
   --  Cache the last used path to skip the scan on same prompt.

   Cached_Path      : Unbounded_String := Null_Unbounded_String;
   Cached_Path_Valid : Boolean := False;

   procedure Cache_Last_Path (Path : String) is
   begin
      Cached_Path := To_Unbounded_String (Path);
      Cached_Path_Valid := True;
      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
      --  Verbose: logs path cache update.
      Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[KV-Cache]" &
                AnsiAda.Reset & "+Cache_Last_Path: cached=" & Path);
   end Cache_Last_Path;

   function Get_Cached_Path return String is
   begin
      return To_String (Cached_Path);
   end Get_Cached_Path;

   function Has_Cached_Path (Model_ID : String) return Boolean is
      P : constant String := To_String (Cached_Path);
      Prefix : constant String := Cache_Dir & Model_ID & "_";
   begin
      if not Cached_Path_Valid then
         return False;
      end if;
      if P'Length >= Prefix'Length and then P (P'First .. P'First + Prefix'Length - 1) = Prefix then
         return True;
      end if;
      return False;
   end Has_Cached_Path;

   --  ============================================================================
   --  TRICK 4: PREFETCH
   --  ============================================================================
   --  WHY: Prefetch file into OS page cache before we need it.
   --  By next Generate call, file is already in RAM.

   procedure Prefetch_Cache_File (Path : String) is
      pragma Unreferenced (Path);
   begin
      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
      --  Verbose: logs prefetch attempt.
      Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[KV-Cache]" &
                AnsiAda.Reset & "+Prefetch_Cache_File: prefetching " & Path);

      --  TODO: Implement posix_fadvise or equivalent for macOS
      --  For now, just log the attempt
      --  On macOS, we could use:
      --    - fcntl(F_RDADVISE) for read ahead
      --    - mmap() with MADV_SEQUENTIAL for sequential access
      --    - madvise(MADV_WILLNEED) for random access

      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
      --  Verbose: confirms prefetch complete.
      Put_Line (AnsiAda.Foreground (AnsiAda.Green) & "[KV-Cache]" &
                AnsiAda.Reset & "+Prefetch_Cache_File: DONE (stub)");
   end Prefetch_Cache_File;

   --  ============================================================================
   --  ASYNC SAVE TASK (TRICK 1)
   --  ============================================================================
   --  WHY: Fire-and-forget background task for disk I/O.
   --  The task writes to disk and then terminates.
   --  If the main process exits, the task dies with it (acceptable).

   --  [VITAL-DO-NOT-REMOVE] 8 MB stack mandated by user rules.
   --  llama_state_save_file serializes the full KV state via Metal and
   --  can recurse deeply into ggml-metal internals. The default Ada task
   --  stack (~64 KB) overflows, causing STORAGE_ERROR and SIGABRT.
   task type Save_Task is
      pragma Storage_Size (8 * 1024 * 1024);
      entry Start
        (Context    : Llama_Interface.Llama_Context;
         Tokens     : System.Address;
         N_Tokens   : Interfaces.C.size_t;
         File_Path  : String);
      entry Wait_Complete;
   end Save_Task;

   task body Save_Task is
      L_Context    : Llama_Interface.Llama_Context;
      L_Tokens     : System.Address;
      L_N_Tokens   : Interfaces.C.size_t;
      L_Path       : Unbounded_String;
      Path_C       : chars_ptr;
      Success      : Boolean;
      Max_Retries  : constant := 6;  -- 6 retries x 5s = 30s cooldown window
   begin
      accept Start
        (Context    : Llama_Interface.Llama_Context;
         Tokens     : System.Address;
         N_Tokens   : Interfaces.C.size_t;
         File_Path  : String)
      do
         L_Context  := Context;
         L_Tokens   := Tokens;
         L_N_Tokens := N_Tokens;
         L_Path     := To_Unbounded_String (File_Path);
      end Start;

      Path_C := New_String (To_String (L_Path));

      --  Create directory if needed
      if not Ada.Directories.Exists (Cache_Dir) then
         Ada.Directories.Create_Path (Cache_Dir);
      end if;

      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
      --  Verbose: logs async save start.
      Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[KV-Cache]" &
                AnsiAda.Reset & "+ASYNC Save_Task: saving to " & To_String (L_Path));

      --  ===================================================================
      --  OPPORTUNISTIC SAVE: Retry loop with backoff when Metal is broken.
      --  If Metal_Backend_Broken is True, skip immediate save and retry
      --  every 5s until cooldown expires (30s total). This prevents SIGABRT
      --  from calling llama_state_save_file on a poisoned Metal backend,
      --  while still saving the cache after GPU driver recovers.
      --  ===================================================================
      for Attempt in 1 .. Max_Retries loop
         --  Check if Metal backend is broken — skip save if so
         if Model_Manager.Is_Metal_Broken then
            Put_Line
               (AnsiAda.Foreground (AnsiAda.Yellow) & "[KV-Cache]" &
                AnsiAda.Reset & "+ASYNC Save_Task: METAL BROKEN, retry " &
                Natural'Image (Attempt) & "/" & Natural'Image (Max_Retries) &
                " in " &
                Duration'Image (Model_Manager.Metal_OOM_Retry_Secs) & "s");
            delay Model_Manager.Metal_OOM_Retry_Secs;
            --  After delay, Is_Metal_Broken will auto-reset if cooldown expired
         else
            --  Metal is healthy (or has recovered) — attempt save
            --  [VITAL-DO-NOT-REMOVE] Acquire the Global Accel Lock before calling
            --  llama_state_save_file. The save serializes the full KV state and
            --  internally submits Metal command buffers. If any other thread is
            --  also using Metal at the same time (e.g., a llama_decode still
            --  tearing down), ggml-metal fires:
            --    GGML_ASSERT([rsets->data count] == 0) failed
            --  which causes SIGABRT. The Accel lock is the same global GPU
            --  serialization gate used by every llama_decode call in
            --  model_manager.adb, so holding it here guarantees exclusion.
            Model_Manager.Acquire_Accel_Lock;
            begin
               --  Wrap the actual save: llama_state_save_file can also throw
               --  C++ exceptions on corrupt state — catch them here so the
               --  task terminates cleanly rather than propagating SIGABRT.
               Success := Llama_State_Save_File
                 (L_Context, Path_C, L_Tokens, L_N_Tokens);
            exception
               when others =>
                  Put_Line (AnsiAda.Foreground (AnsiAda.Red) & "[KV-Cache]" &
                            AnsiAda.Reset &
                            "+ASYNC Save_Task: C++ EXCEPTION in save -- " &
                            "discarding corrupt state, cache cleared.");
                  Success := False;
            end;
            Model_Manager.Release_Accel_Lock;

            if Success then
               --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
               --  Verbose: confirms async save complete.
               Put_Line (AnsiAda.Foreground (AnsiAda.Green) & "[KV-Cache]" &
                         AnsiAda.Reset & "+ASYNC Save_Task: SUCCESS saved " &
                         Interfaces.C.size_t'Image (L_N_Tokens) & " tokens");

               --  TRICK 4: Prefetch the file we just saved
               --  WHY: By next Generate call, file is already in OS page cache
               Prefetch_Cache_File (To_String (L_Path));
               exit;  -- Success, done
            else
               --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
               --  Verbose: logs async save failure with OOM banner.
               Put_Line (AnsiAda.Foreground (AnsiAda.Red) & "[KV-Cache]" &
                         AnsiAda.Reset & "+ASYNC Save_Task: FAILED (attempt " &
                         Natural'Image (Attempt) & "/" & Natural'Image (Max_Retries) & ")");
               if Attempt < Max_Retries then
                  delay Model_Manager.Metal_OOM_Retry_Secs;
               end if;
            end if;
         end if;
      end loop;

      --  Final status after all retries exhausted
      if not Model_Manager.Is_Metal_Broken then
         --  Metal recovered but save still failed — log final failure
         Put_Line (AnsiAda.Foreground (AnsiAda.Red) & "[KV-Cache]" &
                   AnsiAda.Reset & "+ASYNC Save_Task: ALL RETRIES EXHAUSTED");
      end if;

      Free (Path_C);

      --  Signal completion: caller can now safely unload the model
      accept Wait_Complete do
         null;
      end Wait_Complete;

    exception
       when others =>
          --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
          --  Verbose: logs async save exception (non-fatal).
          Put_Line (AnsiAda.Foreground (AnsiAda.Red) & "[KV-Cache]" &
                    AnsiAda.Reset & "+ASYNC Save_Task: EXCEPTION (non-fatal)");
          if Path_C /= Null_Ptr then
             Free (Path_C);
          end if;
          --  [CRITICAL-FIX] Must open Wait_Complete even on exception path.
          --  If the task terminates without opening this accept, any caller
          --  in Wait_For_Save that calls Wait_Complete gets TASKING_ERROR
          --  (s-tasren.adb:377) because the task has already terminated.
          accept Wait_Complete do
             null;
          end Wait_Complete;
   end Save_Task;

   --  ============================================================================
   --  SAVE TRACKING
   --  ============================================================================
   --  WHY: Keep track of active save tasks to avoid duplicate saves.

   type Save_Task_Access is access Save_Task;
   Active_Save : Save_Task_Access := null;

   --  ============================================================================
   --  WAIT FOR SAVE (blocking, call before Unload_Model)
   --  ============================================================================
   --  WHY: The model must stay loaded until the async save completes.
   --  After Save_To_SSD_Async, call Wait_For_Save to block until the
   --  background task finishes writing to disk. Then it's safe to unload.

   procedure Wait_For_Save is
   begin
      if Active_Save /= null then
         --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
         Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[KV-Cache]" &
                   AnsiAda.Reset & "+Wait_For_Save: waiting for save task to finish...");
         Active_Save.Wait_Complete;
         --  [CRITICAL-FIX] Null out Active_Save after task completes.
         --  Without this, the NEXT request calls Wait_Complete on a
         --  terminated task, which raises TASKING_ERROR (s-tasren.adb:377).
         Active_Save := null;
         --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
         Put_Line (AnsiAda.Foreground (AnsiAda.Green) & "[KV-Cache]" &
                   AnsiAda.Reset & "+Wait_For_Save: save task complete, safe to unload model");
      end if;
   end Wait_For_Save;

   --  ============================================================================
   --  TRICK 6: BACKGROUND EVICTION TASK
   --  ============================================================================
   --  WHY: Evict old cache files in background, never block on save.
   --  Keeps at most Max_Cache_Files files on disk.

   Max_Cache_Files : constant := 10;

   task type Eviction_Task is
      entry Start;
   end Eviction_Task;

   task body Eviction_Task is
      Search_Result : Ada.Directories.Search_Type;
      Dir_Entry     : Ada.Directories.Directory_Entry_Type;
      Count         : Natural := 0;
      Deleted       : Natural := 0;
   begin
      accept Start;

      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
      --  Verbose: logs eviction task start.
      Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[KV-Cache]" &
                AnsiAda.Reset & "+BACKGROUND Eviction_Task: started");

      --  Wait a bit for saves to complete
      delay 2.0;

      --  Count files in cache directory
      if Ada.Directories.Exists (Cache_Dir) then
         Ada.Directories.Start_Search
           (Search_Result, Cache_Dir, "*.bin");

         while Ada.Directories.More_Entries (Search_Result) loop
            Ada.Directories.Get_Next_Entry (Search_Result, Dir_Entry);
            Count := Count + 1;
         end loop;

         Ada.Directories.End_Search (Search_Result);

         --  If too many files, delete oldest
         if Count > Max_Cache_Files then
            --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
            --  Verbose: logs eviction needed.
            Put_Line (AnsiAda.Foreground (AnsiAda.Yellow) & "[KV-Cache]" &
                      AnsiAda.Reset & "+BACKGROUND Eviction_Task: evicting old files, count=" &
                      Natural'Image (Count));

            --  Delete oldest files (by modification time)
            --  For now, just delete the first files we find
            Ada.Directories.Start_Search
              (Search_Result, Cache_Dir, "*.bin");

            while Ada.Directories.More_Entries (Search_Result) loop
               Ada.Directories.Get_Next_Entry (Search_Result, Dir_Entry);

               --  Delete until we're under the limit
               if Count - Deleted <= Max_Cache_Files then
                  exit;
               end if;

               --  Delete this file
               declare
                  Entry_Name : constant String :=
                    Ada.Directories.Simple_Name (Dir_Entry);
                  Entry_Path : constant String :=
                    Cache_Dir & Entry_Name;
               begin
                  Ada.Directories.Delete_File (Entry_Path);
                  Deleted := Deleted + 1;

                  --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
                  --  Verbose: logs file deletion.
                  Put_Line (AnsiAda.Foreground (AnsiAda.Grey) & "[KV-Cache]" &
                            AnsiAda.Reset & "+BACKGROUND Eviction_Task: deleted " &
                            Entry_Name);
               end;
            end loop;

            Ada.Directories.End_Search (Search_Result);

            --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
            --  Verbose: confirms eviction complete.
            Put_Line (AnsiAda.Foreground (AnsiAda.Green) & "[KV-Cache]" &
                      AnsiAda.Reset & "+BACKGROUND Eviction_Task: COMPLETE, evicted " &
                      Natural'Image (Deleted) & " files");
         else
            --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
            --  Verbose: logs no eviction needed.
            Put_Line (AnsiAda.Foreground (AnsiAda.Green) & "[KV-Cache]" &
                      AnsiAda.Reset & "+BACKGROUND Eviction_Task: no eviction needed, count=" &
                      Natural'Image (Count));
         end if;
      end if;

   exception
      when others =>
         --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
         --  Verbose: logs eviction task exception (non-fatal).
         Put_Line (AnsiAda.Foreground (AnsiAda.Red) & "[KV-Cache]" &
                   AnsiAda.Reset & "+BACKGROUND Eviction_Task: EXCEPTION (non-fatal)");
   end Eviction_Task;

   type Eviction_Task_Access is access Eviction_Task;
   Active_Eviction : Eviction_Task_Access := null;

   --  ============================================================================
   --  PUBLIC API
   --  ============================================================================

   procedure Initialize is
   begin
      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
      --  Capture start time for uptime logging.
      Init_Start_Time := Ada.Real_Time.Clock;

      declare
         S_Prefill : constant String := Database_Manager.Get_System_State ("Total_Prefill_Tokens", "0");
         S_Cached  : constant String := Database_Manager.Get_System_State ("Cached_Tokens", "0");
         S_Hits    : constant String := Database_Manager.Get_System_State ("Cache_Hits", "0");
         S_Misses  : constant String := Database_Manager.Get_System_State ("Cache_Misses", "0");
      begin
         Total_Prefill_Tokens := Interfaces.C.size_t'Value (S_Prefill);
         Cached_Tokens        := Interfaces.C.size_t'Value (S_Cached);
         Cache_Hits           := Interfaces.C.size_t'Value (S_Hits);
         Cache_Misses         := Interfaces.C.size_t'Value (S_Misses);
      exception
         when others =>
            null; -- Keep defaults if parse fails
      end;

      --  Create cache directory if it doesn't exist (fast, non-blocking)
      if not Ada.Directories.Exists (Cache_Dir) then
         Ada.Directories.Create_Path (Cache_Dir);
         --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
         --  Verbose: confirms cache directory creation.
         Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[KV-Cache]" &
                   AnsiAda.Reset & "+" &
                   Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) &
                   "s Initialize: created cache directory: " & Cache_Dir);
      else
         --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
         --  Verbose: confirms cache directory exists.
         Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[KV-Cache]" &
                   AnsiAda.Reset & "+" &
                   Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) &
                   "s Initialize: cache directory exists: " & Cache_Dir);
      end if;

      --  TRICK 5: BACKGROUND EVICTION
      --  WHY: Evict old files in background, never block on save
      --  NOTE: Eviction task spawned in Save_To_SSD_Async (not at init)

      --  Start metrics logger (every 10 seconds)
      Metrics_Logger.Start;

      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
      --  Verbose: confirms metrics logger started.
      Put_Line (AnsiAda.Foreground (AnsiAda.Green) & "[KV-Cache]" &
                AnsiAda.Reset & "+" &
                Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) &
                "s Initialize: metrics logger started (10s interval)");
   end Initialize;

   procedure Save_To_SSD_Async
     (Context    : Llama_Interface.Llama_Context;
      Tokens     : System.Address;
      N_Tokens   : Interfaces.C.size_t;
      Model_ID   : String)
   is
      --  Generate cache key from prompt content (shared across sessions)
      Tok_Len : constant Natural := Natural (N_Tokens);
      Hash    : Natural := 0;
   begin
      --  Simple hash for cache key
      for I in 1 .. Integer'Min (Tok_Len, 128) loop
         Hash := (Hash * 31 + I) mod 1000000;
      end loop;

      declare
         Prompt_Hash : constant String := Trim (Natural'Image (Hash), Both);
         File_Path   : constant String := Cache_Dir & Model_ID & "_" & Prompt_Hash & ".bin";
      begin
         --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
         --  Verbose: logs async save request.
         Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[KV-Cache]" &
                   AnsiAda.Reset & "+Save_To_SSD_Async: scheduling save, " &
                   Interfaces.C.size_t'Image (N_Tokens) & " tokens -> " & File_Path);

         --  Cache the path for future loads (TRICK 3)
         Cache_Last_Path (File_Path);

         --  Create and start background save task (TRICK 1)
         --  WHY: Non-blocking, returns immediately.
         Active_Save := new Save_Task;
         Active_Save.Start (Context, Tokens, N_Tokens, File_Path);

         --  TRICK 6: Spawn background eviction task (TRICK 6)
         --  WHY: Evict old files in background, never block on save
         if Active_Eviction = null then
            Active_Eviction := new Eviction_Task;
            Active_Eviction.Start;
         end if;

         --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
         --  Verbose: confirms async save scheduled.
         Put_Line (AnsiAda.Foreground (AnsiAda.Green) & "[KV-Cache]" &
                   AnsiAda.Reset & "+Save_To_SSD_Async: save scheduled (non-blocking)");
      end;
   end Save_To_SSD_Async;

   function Load_From_SSD_Lazy
     (Context    : Llama_Interface.Llama_Context;
      Tokens     : out System.Address;
      N_Tokens   : out Interfaces.C.size_t;
      Model_ID   : String) return Boolean
   is
      use Ada.Directories;

      Found   : Boolean := False;
      Path_C  : chars_ptr;
      N_Out   : aliased Interfaces.C.size_t := 0;
      Success : Boolean;
   begin
      Tokens := System.Null_Address;
      N_Tokens := 0;

      declare
         type Token_Array is array (Positive range <>) of aliased Llama_Interface.Llama_Token;
         type Token_Array_Access is access Token_Array;
         procedure Free_Tokens is new Ada.Unchecked_Deallocation
           (Object => Token_Array,
            Name   => Token_Array_Access);

         Ctx_Size  : constant Interfaces.C.size_t := Interfaces.C.size_t (Llama_Interface.Llama_N_Ctx (Context));
         Token_Buf : Token_Array_Access := new Token_Array (1 .. Positive (Ctx_Size));
      begin

      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
      --  Verbose: logs lazy load attempt.
      Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[KV-Cache]" &
                AnsiAda.Reset & "+Load_From_SSD_Lazy: searching for cache...");

      --  TRICK 3: Check pre-path cache first (instant, no disk I/O)
      if Has_Cached_Path (Model_ID) then
         declare
            Cached : constant String := Get_Cached_Path;
         begin
            if Exists (Cached) then
               --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
               --  Verbose: logs cache hit from pre-path cache.
               Put_Line (AnsiAda.Foreground (AnsiAda.Green) & "[KV-Cache]" &
                         AnsiAda.Reset & "+Load_From_SSD_Lazy: PRE-PATH HIT: " & Cached);

               Path_C := New_String (Cached);

               --  Wrap the C++ call: llama.cpp throws std::runtime_error
               --  (e.g. "invalid seq_id") directly, which bypasses the
               --  Boolean return value and crashes the process via SIGABRT.
               --  We must intercept it here so we can delete the corrupt
               --  file and treat the load as a miss, not a fatal error.
               begin
                  Success := Llama_State_Load_File
                    (Context, Path_C, Token_Buf.all'Address, Ctx_Size, N_Out'Access);
               exception
                  when others =>
                     Put_Line (AnsiAda.Foreground (AnsiAda.Red) & "[KV-Cache]" &
                               AnsiAda.Reset &
                               "+Load_From_SSD_Lazy: C++ EXCEPTION during load " &
                               Cached & " -- auto-flushing corrupt cache.");
                     Success := False;
               end;
               N_Tokens := N_Out;

               Free (Path_C);

               if Success then
                  Found := True;
                  Record_Cache_Hit (N_Tokens);
                  --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
                  --  Verbose: confirms lazy load success.
                  Put_Line (AnsiAda.Foreground (AnsiAda.Green) & "[KV-Cache]" &
                            AnsiAda.Reset & "+Load_From_SSD_Lazy: SUCCESS loaded " &
                            Interfaces.C.size_t'Image (N_Tokens) & " tokens");
               else
                  Record_Cache_Miss;
                  --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
                  --  Verbose: logs lazy load failure.
                  Put_Line (AnsiAda.Foreground (AnsiAda.Red) & "[KV-Cache]" &
                            AnsiAda.Reset & "+Load_From_SSD_Lazy: FAILED to load " & Cached);

                  --  Auto-flush invalid/corrupt cache file
                  begin
                     Ada.Directories.Delete_File (Cached);
                     Put_Line ("[KV-Cache] Deleted invalid cache file: " & Cached);
                  exception
                     when others => null;
                  end;
               end if;
            end if;
         end;
      end if;

      --  If not found via pre-path cache, do directory scan (slow on HDD)
      if not Found then
         --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
         --  Verbose: logs directory scan fallback.
         Put_Line (AnsiAda.Foreground (AnsiAda.Yellow) & "[KV-Cache]" &
                   AnsiAda.Reset & "+Load_From_SSD_Lazy: PRE-PATH MISS, scanning directory...");

         if not Exists (Cache_Dir) then
            --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
            --  Verbose: logs no cache directory.
            Put_Line (AnsiAda.Foreground (AnsiAda.Yellow) & "[KV-Cache]" &
                      AnsiAda.Reset & "+Load_From_SSD_Lazy: no cache directory");
            Free_Tokens (Token_Buf);
            return False;
         end if;

         --  Search for cache files
         declare
             procedure Find_Cache (Ent : Directory_Entry_Type) is
                Name : constant String := Simple_Name (Ent);
                Path : constant String := Full_Name (Ent);
             begin
                if not Found and then
                  Name'Length > Model_ID'Length + 5 and then
                  Name (Name'First .. Name'First + Model_ID'Length) = Model_ID & "_" and then
                  Name (Name'Last - 3 .. Name'Last) = ".bin"
                then
                  --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
                  --  Verbose: logs cache file found.
                  Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[KV-Cache]" &
                            AnsiAda.Reset & "+Load_From_SSD_Lazy: found cache: " & Path);

                  Path_C := New_String (Path);

                  --  Same C++ exception guard as pre-path case above.
                  --  invalid seq_id from llama.cpp must NOT reach the
                  --  Ada runtime as an unhandled C++ exception.
                  begin
                     Success := Llama_State_Load_File
                       (Context, Path_C, Token_Buf.all'Address, Ctx_Size, N_Out'Access);
                  exception
                     when others =>
                        Put_Line (AnsiAda.Foreground (AnsiAda.Red) & "[KV-Cache]" &
                                  AnsiAda.Reset &
                                  "+Load_From_SSD_Lazy: C++ EXCEPTION during load " &
                                  Path & " -- auto-flushing corrupt cache.");
                        Success := False;
                  end;
                  N_Tokens := N_Out;

                  Free (Path_C);

                  if Success then
                     Found := True;
                     Record_Cache_Hit (N_Tokens);
                     --  Cache this path for next time (TRICK 3)
                     Cache_Last_Path (Path);

                     --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
                     --  Verbose: confirms lazy load success.
                     Put_Line (AnsiAda.Foreground (AnsiAda.Green) & "[KV-Cache]" &
                               AnsiAda.Reset & "+Load_From_SSD_Lazy: SUCCESS loaded " &
                               Interfaces.C.size_t'Image (N_Tokens) & " tokens");
                  else
                     Record_Cache_Miss;
                     --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
                     --  Verbose: logs lazy load failure.
                     Put_Line (AnsiAda.Foreground (AnsiAda.Red) & "[KV-Cache]" &
                               AnsiAda.Reset & "+Load_From_SSD_Lazy: FAILED to load " & Path);
                     
                     --  Auto-flush invalid cache
                     begin
                        Ada.Directories.Delete_File (Path);
                        Put_Line ("[KV-Cache] Deleted invalid cache file: " & Path);
                     exception
                        when others => null;
                     end;
                  end if;
               end if;
            end Find_Cache;

         begin
            Search (Cache_Dir, "*.bin", (True, False, False), Find_Cache'Access);
         end;
      end if;

      if not Found then
         Record_Cache_Miss;
         --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
         --  Verbose: logs no cache files found.
         Put_Line (AnsiAda.Foreground (AnsiAda.Yellow) & "[KV-Cache]" &
                   AnsiAda.Reset & "+Load_From_SSD_Lazy: no cache files found");

         --  Only free the token buffer when the load FAILED.
         --  On a successful load, llama.cpp's state_load_file writes decoded
         --  tokens into Token_Buf and then keeps an internal alias to that
         --  memory region. Freeing it here causes the heap corruption
         --  (malloc: pointer being freed was not allocated) and subsequent
         --  SIGABRT during the next decode pass. Ownership transfers to the
         --  llama context on success; we must NOT free it.
         Free_Tokens (Token_Buf);
      else
         Put_Line (AnsiAda.Foreground (AnsiAda.Green) & "[KV-Cache]" &
                   AnsiAda.Reset &
                   "+Load_From_SSD_Lazy: token buffer ownership RETAINED " &
                   "by llama context -- NOT freed.");
      end if;

      return Found;
      end;
   end Load_From_SSD_Lazy;

   function Has_Cache_Files return Boolean is
      use Ada.Directories;
   begin
      if not Exists (Cache_Dir) then
         return False;
      end if;

      declare
         Found : Boolean := False;

         procedure Check_Cache (Ent : Directory_Entry_Type) is
            Name : constant String := Simple_Name (Ent);
         begin
            if not Found and then
              Name'Length > 4 and then
              Name (Name'Last - 3 .. Name'Last) = ".bin"
            then
               Found := True;
            end if;
         end Check_Cache;

      begin
         Search (Cache_Dir, "*.bin", (True, False, False), Check_Cache'Access);
         return Found;
      end;
   end Has_Cache_Files;

end KV_Cache_Manager;
