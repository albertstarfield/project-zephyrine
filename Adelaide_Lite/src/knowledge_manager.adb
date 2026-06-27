pragma SPARK_Mode (Off);
with AnsiAda;
with Ada.Text_IO; use Ada.Text_IO;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Ada.Strings; use Ada.Strings;
with Ada.Strings.Fixed; use Ada.Strings.Fixed;
with Model_Manager;
with Model_Types; use Model_Types;
with Database_Manager;
with Math_Utils;
with Zenith_Manager;
with Ada.Directories; use Ada.Directories;
with Ada.Exceptions;
with Ada.Calendar;
with Ada.Real_Time; use Ada.Real_Time;
with Embedding_Batcher;
with Speculative_Cache;
with Cronia_Scheduler; use Cronia_Scheduler;
with Proactive_Engine;
with Ada.Calendar.Formatting; use Ada.Calendar.Formatting;
with Interfaces.C;
with Interfaces.C.Strings;
with Ada.Streams;
with Ada.Streams.Stream_IO;

package body Knowledge_Manager is

   --  ELP levels hierarchy:
   --  ELP0: Background Literature Indexing (Lowest Priority) (Self reflecting)
   --  ELP1: Active RAG / Memory Retrieval (User Interaction)
   --  ELP2: StellaIcarus Hooks (Deterministic API Logic)
   --  ELP3: ZenithOrion (Deterministic 1ms Pacing Lock)

   --  [QUIRK-M08] Task Stack Expansion
   --  ========================================================================
   --  [VITAL-DO-NOT-REMOVE] Mandated by user for stability.
   --  REASONING:
   --  Default macOS thread stacks (512KB-1MB) are insufficient for deep
   --  filesystem recursion and large local Math_Utils.Vector buffers.
   --  Exceeding stack triggers Trace/BPT trap: 5. We expand to 128MB minimum.
   task Indexing_Task is
      pragma Storage_Size (128 * 1024 * 1024);
      entry Start;
   end Indexing_Task;

   task Thought_Task is
      pragma Storage_Size (128 * 1024 * 1024);
      entry Start;
   end Thought_Task;

   --  [VITAL-DO-NOT-REMOVE] Native_Crawl_Task calls Load_Model →
   --  Llama_Init_From_Model, which logs to stderr via llama_log_internal.
   --  fprintf() uses deep C stack for format buffering. macOS default
   --  pthread stacks (~512KB) overflow during this call chain, causing:
   --    STORAGE_ERROR : stack overflow → signal → jorvik_handler → exit
   --  CRASH LOG (2026-06-22, llama.cpp b9757):
   --    llama_init_from_model → llama_log_internal → _vsnprintf → __vfprintf
   --    → SIGSEGV (stack overflow in fprintf's formatting buffer)
   --  Storage_Size controls HEAP, not stack. Task_Stack_Size controls the
   --  pthread stack. Without it, the task uses macOS default (~512KB).
   --  16 MB is sufficient for llama_init_from_model + Metal device init.
   --  Also set gnatbind -D16384 (in adelaide_lite.gpr) for default stack.
   task Native_Crawl_Task is
      pragma Storage_Size (128 * 1024 * 1024);
      pragma Task_Stack_Size (16 * 1024 * 1024);
      entry Start;
   end Native_Crawl_Task;

   task Salience_Maintenance_Task is
      pragma Storage_Size (128 * 1024 * 1024);
      entry Start;
   end Salience_Maintenance_Task;

   task Telemetry_Sync_Task is
      pragma Storage_Size (128 * 1024 * 1024);
      entry Start;
   end Telemetry_Sync_Task;

   task Zenith_Prover_Task is
      pragma Storage_Size (128 * 1024 * 1024);
      entry Start;
   end Zenith_Prover_Task;

   task Proactive_Cache_Task is
      pragma Storage_Size (128 * 1024 * 1024);
      pragma Task_Stack_Size (16 * 1024 * 1024);  --  16 MB thread stack (llama.cpp tokenize needs deep C stack)
      entry Start;
   end Proactive_Cache_Task;

   --  CRONIA TASK: Manages scheduled cron jobs and prints status every 10s
   --  Named after Cronia (Ancient Greek: "Time" / "The Right Moment").
   task Cronia_Task is
      pragma Storage_Size (128 * 1024 * 1024);
      entry Start;
   end Cronia_Task;

   --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
   --  Init_Start_Time: Captured when Knowledge_Manager.Initialize is called.
   --  All [Init-V] verbose prints in this package compute uptime relative
   --  to this timestamp.
   Init_Start_Time : Ada.Real_Time.Time;

   procedure Initialize is
   begin
      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
      --  Capture start time for uptime calculation.
      Init_Start_Time := Ada.Real_Time.Clock;
      --  Verbose: confirms Knowledge_Manager.Initialize was entered.
      Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Init-V]" &
                AnsiAda.Reset & "+" & Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) & "s +" & Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) & "s Knowledge_Manager.Initialize ENTERED.");
      Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Knowledge]" &
                AnsiAda.Reset & " Initializing Knowledge Base...");
      Database_Manager.Initialize;
      Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Init-V]" &
                AnsiAda.Reset & "+" & Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) & "s  Knowledge_Manager.Initialize COMPLETE.");
   end Initialize;

   procedure Start_Tasks is
   begin
      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
      --  Verbose: prints each task start so we can see which one hangs.
      Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Init-V]" &
                AnsiAda.Reset & "+" & Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) & "s  Knowledge_Manager.Start_Tasks ENTERED.");

      Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Init-V]" &
                AnsiAda.Reset & "+" & Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) & "s  Starting Indexing_Task...");
      Indexing_Task.Start;
      Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Init-V]" &
                AnsiAda.Reset & "+" & Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) & "s  Indexing_Task.Start DONE.");

      Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Init-V]" &
                AnsiAda.Reset & "+" & Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) & "s  Starting Thought_Task...");
      Thought_Task.Start;
      Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Init-V]" &
                AnsiAda.Reset & "+" & Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) & "s  Thought_Task.Start DONE.");

      Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Init-V]" &
                AnsiAda.Reset & "+" & Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) & "s  Starting Native_Crawl_Task...");
      Native_Crawl_Task.Start;
      Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Init-V]" &
                AnsiAda.Reset & "+" & Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) & "s  Native_Crawl_Task.Start DONE.");

      Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Init-V]" &
                AnsiAda.Reset & "+" & Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) & "s  Starting Salience_Maintenance_Task...");
      Salience_Maintenance_Task.Start;
      Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Init-V]" &
                AnsiAda.Reset & "+" & Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) & "s  Salience_Maintenance_Task.Start DONE.");

      Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Init-V]" &
                AnsiAda.Reset & "+" & Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) & "s  Starting Zenith_Orion_Task...");
      Zenith_Manager.Zenith_Orion_Task.Start;
      Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Init-V]" &
                AnsiAda.Reset & "+" & Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) & "s  Zenith_Orion_Task.Start DONE.");

      Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Init-V]" &
                AnsiAda.Reset & "+" & Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) & "s  Starting Telemetry_Sync_Task...");
      Telemetry_Sync_Task.Start;
      Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Init-V]" &
                AnsiAda.Reset & "+" & Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) & "s  Telemetry_Sync_Task.Start DONE.");

      Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Init-V]" &
                AnsiAda.Reset & "+" & Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) & "s  Starting Zenith_Prover_Task...");
      Zenith_Prover_Task.Start;
      Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Init-V]" &
                AnsiAda.Reset & "+" & Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) & "s  Zenith_Prover_Task.Start DONE.");

      Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Init-V]" &
                AnsiAda.Reset & "+" & Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) & "s  Starting Proactive_Cache_Task...");
      Proactive_Cache_Task.Start;
      Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Init-V]" &
                AnsiAda.Reset & "+" & Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) & "s  Proactive_Cache_Task.Start DONE.");

      Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Init-V]" &
                AnsiAda.Reset & "+" & Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) & "s  Starting Cronia_Task...");
      Cronia_Task.Start;
      Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Init-V]" &
                AnsiAda.Reset & "+" & Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) & "s  Cronia_Task.Start DONE.");

      Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Init-V]" &
                AnsiAda.Reset & "+" & Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) & "s  Knowledge_Manager.Start_Tasks COMPLETE.");
   end Start_Tasks;

    --  Retrieve the current user's home directory from the environment
    function Get_Home_Directory return String is
       use Interfaces.C.Strings;
       function Get_Env (Name : chars_ptr) return chars_ptr;
       pragma Import (C, Get_Env, "getenv");
       
       C_Name  : chars_ptr;
       Env_Ptr : chars_ptr;
    begin
       C_Name := New_String ("HOME");
       Env_Ptr := Get_Env (C_Name);
       Free (C_Name);
       
       if Env_Ptr = Null_Ptr then
          return "/tmp";
       else
          return Value (Env_Ptr);
       end if;
    end Get_Home_Directory;

     --  Check if a file is readable text by scanning for binary markers (null bytes)
     function Is_Readable_Text (FilePath : String) return Boolean is
         use type Ada.Streams.Stream_Element;
         use type Ada.Streams.Stream_Element_Offset;
        File_S : Ada.Streams.Stream_IO.File_Type;
        Buffer : Ada.Streams.Stream_Element_Array (1 .. 4096);
        Last   : Ada.Streams.Stream_Element_Offset;
        Non_Printable : Natural := 0;
     begin
        begin
           Ada.Streams.Stream_IO.Open (File_S, Ada.Streams.Stream_IO.In_File, FilePath);
           Ada.Streams.Stream_IO.Read (File_S, Buffer, Last);
           Ada.Streams.Stream_IO.Close (File_S);

           for I in 1 .. Last loop
              if Buffer (I) = 0 then
                 return False; -- Null byte is a definitive binary marker
              end if;

              -- Count non-printable characters (excluding tab, LF, CR)
              if not (Buffer (I) in 32 .. 126 or else
                      Buffer (I) = 9 or else
                      Buffer (I) = 10 or else
                      Buffer (I) = 13) then
                 Non_Printable := Non_Printable + 1;
              end if;
           end loop;

           -- If more than 10% of the first 4KB are non-printable, consider it binary
           if Last > 0 and then (Non_Printable * 100 / Natural (Last)) > 10 then
              return False;
           end if;

           return True;
        exception
           when others =>
              if Ada.Streams.Stream_IO.Is_Open (File_S) then
                 Ada.Streams.Stream_IO.Close (File_S);
              end if;
              return False;
        end;
     end Is_Readable_Text;

     procedure Wait_For_ELP1_Cooldown is
        Timer_Done : Boolean := False;
     begin
        Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Knowledge]" &
                  AnsiAda.Reset & " Indexing HALTED due to ELP1 request.");
        while not Timer_Done loop
           Model_Manager.Wait_For_ELP1_Idle;
           Timer_Done := True;
           for I in 1 .. 600 loop
              delay 1.0;
              if Model_Manager.Should_Abort_ELP0 then
                 Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Knowledge]" &
                           AnsiAda.Reset & " ELP1 interrupted cooldown, timer reset to 600s!");
                 Timer_Done := False;
                 exit;
              end if;
           end loop;
        end loop;
        Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Knowledge]" &
                  AnsiAda.Reset & " 600s cooldown finished. RESUMING indexing.");
     end Wait_For_ELP1_Cooldown;

   procedure Index_References is
      File          : File_Type;
      Opened        : Boolean := False;
      Current_Entry : Unbounded_String;
      Line          : Unbounded_String;
   begin
      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
      --  Verbose: prints every path tried for references.bib.
      Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Init-V]" &
                AnsiAda.Reset & "+" & Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) & "s  Index_References ENTERED.");
      begin
         Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Init-V]" &
                   AnsiAda.Reset & "+" & Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) & "s Index_References: trying ../legacyPython/references.bib...");
         Open (File, In_File, "../legacyPython/references.bib");
         Opened := True;
         Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Init-V]" &
                   AnsiAda.Reset & "+" & Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) & "s Index_References: OPENED ../legacyPython/references.bib");
      exception
         when others =>
            Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Init-V]" &
                      AnsiAda.Reset & "+" & Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) & "s Index_References: ../legacyPython/references.bib NOT FOUND.");
            begin
               Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Init-V]" &
                         AnsiAda.Reset & "+" & Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) & "s Index_References: trying legacyPython/references.bib...");
               Open (File, In_File, "legacyPython/references.bib");
               Opened := True;
               Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Init-V]" &
                         AnsiAda.Reset & "+" & Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) & "s Index_References: OPENED legacyPython/references.bib");
            exception
               when others =>
                  Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Init-V]" &
                            AnsiAda.Reset & "+" & Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) & "s Index_References: legacyPython/references.bib NOT FOUND.");
                  Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Knowledge]" &
                            AnsiAda.Reset & " references.bib not found.");
            end;
      end;

      if not Opened then
         return;
      end if;

      Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Knowledge]" &
                AnsiAda.Reset & " Parsing and indexing references.bib...");

      while not End_Of_File (File) loop
         if Model_Manager.Should_Abort_ELP0 then
            Wait_For_ELP1_Cooldown;
         end if;

         Line := To_Unbounded_String (Get_Line (File));
         if Index (To_String (Line), "@") = 1 then
            if Length (Current_Entry) > 0 then
               declare
                  Raw_Content : constant String := To_String (Current_Entry);
                  Content     : constant String :=
                    Model_Manager.Sanitize_Think_Tags (Raw_Content);
                  Vec         : Math_Utils.Vector (1 .. 4096) :=
                    [others => 0.0];
                  Len         : Natural := 0;
               begin
                  Model_Manager.Get_Embedding (Content, Vec, Len, ELP0);
                  if Len > 0 then
                     Database_Manager.Add_Literature_Chunk
                       ("references.bib", Content, Vec (1 .. Len), "hash");
                  end if;
               end;
               Current_Entry := Null_Unbounded_String;
            end if;
         end if;
         Append (Current_Entry, To_String (Line) & ASCII.LF);
      end loop;

      if Length (Current_Entry) > 0 then
         declare
            Raw_Content : constant String := To_String (Current_Entry);
            Content     : constant String :=
              Model_Manager.Sanitize_Think_Tags (Raw_Content);
            Vec         : Math_Utils.Vector (1 .. 4096) := [others => 0.0];
            Len         : Natural := 0;
         begin
            Model_Manager.Get_Embedding (Content, Vec, Len, ELP0);
            if Len > 0 then
               Database_Manager.Add_Literature_Chunk
                 ("references.bib", Content, Vec (1 .. Len), "hash");
            end if;
         end;
      end if;

      Close (File);
   end Index_References;

   task body Indexing_Task is
   begin
      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
      --  Verbose: confirms task accepted Start and is running.
      Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Init-V]" &
                AnsiAda.Reset & "+" & Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) & "s  Indexing_Task waiting for Start...");
      accept Start;
      Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Init-V]" &
                AnsiAda.Reset & "+" & Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) & "s  Indexing_Task ACCEPTED Start, calling Index_References...");
      Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Knowledge]" &
                AnsiAda.Reset & " Indexing Task Active.");
       Index_References;
      Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Init-V]" &
                AnsiAda.Reset & "+" & Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) & "s  Indexing_Task Index_References DONE.");
   exception
      when E : others =>
         Put_Line (AnsiAda.Foreground (AnsiAda.Red) & "[FATAL]" &
                   AnsiAda.Reset & " Indexing_Task Error: " &
                   Ada.Exceptions.Exception_Message (E));
   end Indexing_Task;

   task body Thought_Task is
   begin
      accept Start;
      Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Knowledge]" &
                AnsiAda.Reset & " Thought Task Active.");
   exception
      when E : others =>
         Put_Line (AnsiAda.Foreground (AnsiAda.Red) & "[FATAL]" &
                   AnsiAda.Reset & " Thought_Task Error: " &
                   Ada.Exceptions.Exception_Message (E));
   end Thought_Task;

    procedure Crawl_Directory (Path : String) is
       Search  : Ada.Directories.Search_Type;
       Entry_D : Ada.Directories.Directory_Entry_Type;
       Files_Scanned : Natural := 0;
    begin
       --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
       Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Init-V]" &
                 AnsiAda.Reset & "+" & Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) & "s  Crawl_Directory ENTERED: Path=" & Path);
       Ada.Directories.Start_Search (Search, Path, "");
       while Ada.Directories.More_Entries (Search) loop
          if Model_Manager.Should_Abort_ELP0 then
             Wait_For_ELP1_Cooldown;
          end if;

          Ada.Directories.Get_Next_Entry (Search, Entry_D);
          declare
             Name : constant String := Simple_Name (Entry_D);
             Full : constant String := Full_Name (Entry_D);
          begin
             if Name /= "." and then Name /= ".." and then
                Name /= "node_modules" and then Name /= ".git" and then
                Name /= "dist" and then Name /= "build" and then
                Name /= "venv" and then Name /= "__pycache__" and then
                Name /= ".svn" and then Name /= ".idea" and then
                Name /= ".vscode"
             then
                if Kind (Entry_D) = Directory then
                   Crawl_Directory (Full);
                else
                   --  [DO NOT REMOVE COMMENT EXPLANATION]
                   --  FIX 4: Pre-Tokenization Filtering (QUIRK-M10)
                   --  Skip web and code files from natural language embedding.
                   --  This stops the compute kernel from faulting on hyper-dense symbol clusters.
                   if Index (Name, ".css") > 0 or else Index (Name, ".js") > 0 or else 
                      Index (Name, ".html") > 0 or else Index (Name, ".json") > 0 or else 
                      Index (Name, ".xml") > 0 or else Index (Name, ".ts") > 0 then
                      null;
                   elsif Is_Readable_Text (Full) then
                      declare
                         File_Content : Unbounded_String;
                         File_H       : File_Type;
                      begin
                         begin
                            Open (File_H, In_File, Full);
                            while not End_Of_File (File_H) loop
                               Append (File_Content,
                                       Get_Line (File_H) & ASCII.LF);
                            end loop;
                            Close (File_H);

                            Files_Scanned := Files_Scanned + 1;
                            declare
                               Raw_Content : constant String := To_String (File_Content);
                               Content     : constant String := Model_Manager.Sanitize_Think_Tags (Raw_Content);
                            begin
                               if Content'Length > 0 then
                                  Embedding_Batcher.Add_To_Batch (Content, Name, ELP0);
                               end if;
                            end;
                         exception
                            when others =>
                               if Is_Open (File_H) then
                                  Close (File_H);
                               end if;
                          end;
                       end;
                       end if;
                    end if;
                 end if;
                 --  Yield to OS scheduler to prevent starvation (framebuffer glitching)
                 delay 0.001;
              end;
           end loop;
           Ada.Directories.End_Search (Search);
           -- Flush batch at end of directory to maintain granularity
           Embedding_Batcher.Flush_Batch (ELP0);

       Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Init-V]" &
                 AnsiAda.Reset & "+" & Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) & "s  Crawl_Directory COMPLETE: Path=" & Path &
                 " Files_Scanned=" & Natural'Image (Files_Scanned));
    end Crawl_Directory;


   task body Native_Crawl_Task is
   begin
      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
      --  Verbose: confirms task accepted Start and is running.
      Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Init-V]" &
                AnsiAda.Reset & "+" & Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) & "s  Native_Crawl_Task waiting for Start...");
      accept Start;
      Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Init-V]" &
                AnsiAda.Reset & "+" & Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) & "s  Native_Crawl_Task ACCEPTED Start, entering main loop.");
      Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Knowledge]" &
                AnsiAda.Reset & " Native Crawl Task Active.");
       loop
          --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
          --  Verbose: prints BEFORE blocking on Wait_For_ELP1_Idle so we can
          --  see if the task is stuck on the guard condition.
          Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Init-V]" &
                    AnsiAda.Reset & "+" & Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) & "s Native_Crawl_Task: calling Wait_For_ELP1_Idle...");
          Model_Manager.Wait_For_ELP1_Idle;
          --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
          --  Verbose: prints AFTER guard passes so we know it unblocked.
          Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Init-V]" &
                    AnsiAda.Reset & "+" & Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) & "s Native_Crawl_Task: Wait_For_ELP1_Idle PASSED, starting systemic crawl...");
          
          --  [SYSTEMIC INDEXING] Expanding scope to all readable system areas.
          --  This is mandated to prevent "blind spots" in the knowledge base.
          
          --  1. Crawl the root filesystem (covers system files, config, and apps)
          Crawl_Directory("/");
          
           --  2. Explicitly crawl the user home directory
           Crawl_Directory (Get_Home_Directory);

          
          --  3. [DYNAMIC MOUNT DETECTION] Cross-platform mount discovery.
          --  Scans common mount points for Darwin (/Volumes) and Linux (/mnt, /media).
          declare
             type Mount_Path_List is array (1 .. 3) of String (1 .. 10);
             Mount_Points : Mount_Path_List := ("/Volumes  ", "/mnt      ", "/media    ");
             
             procedure Scan_Mount_Point (Path : String) is
                Search  : Ada.Directories.Search_Type;
                Entry_D : Ada.Directories.Directory_Entry_Type;
             begin
                begin
                   Ada.Directories.Start_Search (Search, Trim (Path, Both), "");
                   while Ada.Directories.More_Entries (Search) loop
                      Ada.Directories.Get_Next_Entry (Search, Entry_D);
                      if Kind (Entry_D) = Directory then
                         Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Dynamic-Mount]" & AnsiAda.Reset & " Found mount point: " & Full_Name (Entry_D));
                         Crawl_Directory (Full_Name (Entry_D));
                      end if;
                   end loop;
                   Ada.Directories.End_Search (Search);
                exception
                   when others => null; -- Path might not exist on this OS
                end;
             end Scan_Mount_Point;
          begin
             for I in Mount_Points'Range loop
                Scan_Mount_Point (Mount_Points (I));
             end loop;
          end;
          
          Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Init-V]" &
                    AnsiAda.Reset & "+" & Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) & "s Native_Crawl_Task: systemic crawl DONE, sleeping 3600s...");
          delay 3600.0; -- Crawl every hour
       end loop;

   exception
      when E : others =>
         Put_Line (AnsiAda.Foreground (AnsiAda.Red) & "[FATAL]" &
                   AnsiAda.Reset & " Native_Crawl_Task Error: " &
                   Ada.Exceptions.Exception_Message (E));
   end Native_Crawl_Task;

   task body Salience_Maintenance_Task is
   begin
      accept Start;
      Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Knowledge]" &
                AnsiAda.Reset & " Salience Maintenance Task Active.");
   exception
      when E : others =>
         Put_Line (AnsiAda.Foreground (AnsiAda.Red) & "[FATAL]" &
                   AnsiAda.Reset & " Salience_Maintenance_Task Error: " &
                   Ada.Exceptions.Exception_Message (E));
   end Salience_Maintenance_Task;

   task body Telemetry_Sync_Task is
   begin
      accept Start;
      Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Knowledge]" &
                AnsiAda.Reset & " Telemetry Sync Task Active.");
   exception
      when E : others =>
         Put_Line (AnsiAda.Foreground (AnsiAda.Red) & "[FATAL]" &
                   AnsiAda.Reset & " Telemetry_Sync_Task Error: " &
                   Ada.Exceptions.Exception_Message (E));
   end Telemetry_Sync_Task;

   task body Zenith_Prover_Task is
   begin
      accept Start;
      Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Knowledge]" &
                AnsiAda.Reset & " Zenith Prover Task Active.");
   exception
      when E : others =>
         Put_Line (AnsiAda.Foreground (AnsiAda.Red) & "[FATAL]" &
                   AnsiAda.Reset & " Zenith_Prover_Task Error: " &
                   Ada.Exceptions.Exception_Message (E));
   end Zenith_Prover_Task;

   task body Proactive_Cache_Task is
   begin
      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
      --  Verbose: confirms task accepted Start and is running.
      Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Init-V]" &
                AnsiAda.Reset & "+" & Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) & "s  Proactive_Cache_Task waiting for Start...");
      accept Start;
      Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Init-V]" &
                AnsiAda.Reset & "+" & Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) & "s  Proactive_Cache_Task ACCEPTED Start, entering main loop.");
      Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Knowledge]" &
                AnsiAda.Reset & " Proactive Cache Task Active.");
      loop
         declare
            Last_Prompt : constant String :=
              To_String (Model_Manager.Last_User_Prompt);
         begin
            if Last_Prompt /= "" then
               --  Predict follow-up questions
               declare
                  Prediction_Prompt : constant String :=
                    "User just asked: """ & Last_Prompt & """. " &
                    "Predict the MOST LIKELY follow-up technical " &
                    "question they will ask next. " &
                    "Output ONLY the question text. NO PREAMBLE.";
                  Predicted_Q : Unbounded_String;
                  Result      : Unbounded_String;
               begin
                  Model_Manager.Generate
                    (Kind   => Snowball_Enaga_ShortNetworkAnswer,
                     Prompt => Prediction_Prompt,
                     Result => Predicted_Q,
                     Level  => ELP0);

                  if To_String (Predicted_Q) /= "" and then
                    To_String (Predicted_Q) /= "ERROR: Preempted"
                  then
                     Put_Line (AnsiAda.Foreground (AnsiAda.Light_Magenta) &
                               "[Proactive]" & AnsiAda.Reset &
                               " Speculating answer for: " &
                               To_String (Predicted_Q));

                     Model_Manager.Hybrid_Generate
                       (Prompt => To_String (Predicted_Q),
                        Result => Result,
                        Level  => ELP0);

                     if To_String (Result) /= "" then
                        Speculative_Cache.Proactive_Cache.Store
                          (To_String (Predicted_Q), To_String (Result));
                     end if;
                  end if;
               end;
            end if;
         end;

         declare
            use Speculative_Cache;
         begin
            if Proactive_Cache.Count > 0 then
               Put_Line (AnsiAda.Foreground (AnsiAda.Light_Magenta) &
                         "[Proactive]" & AnsiAda.Reset &
                         " Cache entries: " &
                         Natural'Image (Proactive_Cache.Count));
            end if;
         end;
         delay 10.0;
      end loop;
   end Proactive_Cache_Task;

   --  ============================================================================
   --  CRONIA TASK BODY
   --  ============================================================================
   --  Fires scheduled cron jobs and prints active schedules every 10s.
   --  This is the heartbeat of the timed_cronia_answer system.
   --  ============================================================================
   task body Cronia_Task is
      Last_Print : Ada.Real_Time.Time := Ada.Real_Time.Time_First;
      Print_Interval : constant Duration := 10.0;
   begin
      Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Init-V]" &
                AnsiAda.Reset & "+" & Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) & "s  Cronia_Task waiting for Start...");
      accept Start;
      Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Init-V]" &
                AnsiAda.Reset & "+" & Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) & "s  Cronia_Task ACCEPTED Start, entering main loop.");
      Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Cronia]" &
                AnsiAda.Reset & " Cronia Task Active.");

      --  Initialize the scheduler and proactive engine
      Cronia_Scheduler.Initialize;
      Proactive_Engine.Initialize;

      loop
         --  Fire any pending cron jobs
         Cronia_Scheduler.Tick;

         --  Fire any pending proactive questions
         Proactive_Engine.Tick;

         --  Print cron status every 10 seconds
         declare
            Now : constant Ada.Real_Time.Time := Ada.Real_Time.Clock;
         begin
            if Ada.Real_Time.To_Duration (Now - Last_Print) >= Print_Interval then
               Last_Print := Now;
               declare
                  Active_Count : constant Natural := Cronia_Scheduler.Active_Job_Count;
               begin
                  Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Cronia]" &
                            AnsiAda.Reset & " === CRON STATUS (10s) ===");
                  Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Cronia]" &
                            AnsiAda.Reset & " Active jobs: " & Natural'Image (Active_Count));

                  --  Print each active job
                  for I in 1 .. Active_Count loop
                     declare
                        J : constant Cronia_Scheduler.Cron_Job := Cronia_Scheduler.Get_Job (I);
                     begin
                        if J.State = Cronia_Scheduler.Scheduled then
                           Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Cronia]" &
                                     AnsiAda.Reset & "   [" & Natural'Image (I) & "] " &
                                     To_String (J.Name) &
                                     " | Next: " & Ada.Calendar.Formatting.Image (J.Scheduled_Time) &
                                     (if J.Repeat_Interval > 0.0
                                      then " | Repeat: " & Duration'Image (J.Repeat_Interval) & "s"
                                      else ""));
                        end if;
                     end;
                  end loop;

                  --  Print handless mode status
                  if Proactive_Engine.Is_Handless_Mode_Active then
                     Put_Line (AnsiAda.Foreground (AnsiAda.Green) & "[Cronia]" &
                               AnsiAda.Reset & " Handless Mode: ACTIVE");
                  else
                     Put_Line (AnsiAda.Foreground (AnsiAda.Grey) & "[Cronia]" &
                               AnsiAda.Reset & " Handless Mode: OFF");
                  end if;

                  Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Cronia]" &
                            AnsiAda.Reset & " ==========================");
               end;
            end if;
         end;

         delay 1.0;  --  Tick every second, print every 10s
      end loop;
   end Cronia_Task;

end Knowledge_Manager;
