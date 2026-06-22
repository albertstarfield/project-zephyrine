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
with Speculative_Cache;

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

   task Native_Crawl_Task is
      pragma Storage_Size (128 * 1024 * 1024);
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

   --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
   --  Init_Start_Time: Captured when Knowledge_Manager.Initialize is called.
   --  All [Init-V] verbose prints in this package compute uptime relative
   --  to this timestamp.
   Init_Start_Time : Ada.Real_Time.Time;

   procedure Initialize is
   begin
      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
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
                AnsiAda.Reset & "+" & Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) & "s  Knowledge_Manager.Start_Tasks COMPLETE.");
   end Start_Tasks;

   --  Helper to index references.bib
   procedure Index_References is
      File          : File_Type;
      Opened        : Boolean := False;
      Current_Entry : Unbounded_String;
      Line          : Unbounded_String;
   begin
      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
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
            Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Knowledge]" &
                      AnsiAda.Reset & " Indexing aborted by ELP1.");
            Close (File);
            return;
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
      Files_Indexed : Natural := 0;
   begin
      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
      --  Verbose: prints crawl start and stats so we can see if it runs.
      Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Init-V]" &
                AnsiAda.Reset & "+" & Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) & "s  Crawl_Directory ENTERED: Path=" & Path);
      Ada.Directories.Start_Search (Search, Path, "");
      while Ada.Directories.More_Entries (Search) loop
         --  [VITAL-DO-NOT-REMOVE] Halt ELP0 crawl when ELP1 is pending.
         --  This is the critical guard: if a user chat arrives while we are
         --  indexing background files, we MUST stop immediately and wait
         --  until the user request is fully served before resuming.
         --  Without this, ELP0 keeps calling Get_Embedding which spins on
         --  Acquire_ELP0/DENIED in a tight loop, starving ELP1 of the GPU.
         if Model_Manager.Should_Abort_ELP0 then
            Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Init-V]" &
                      AnsiAda.Reset &
                      " Crawl_Directory: ELP1 pending, HALTING crawl...");
            Model_Manager.Wait_For_ELP1_Idle;
            Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Init-V]" &
                      AnsiAda.Reset &
                      " Crawl_Directory: ELP1 idle, RESUMING crawl...");
         end if;

         Ada.Directories.Get_Next_Entry (Search, Entry_D);
         declare
            Name : constant String := Simple_Name (Entry_D);
            Full : constant String := Full_Name (Entry_D);
         begin
            if Name /= "." and then Name /= ".." then
               if Kind (Entry_D) = Directory then
                  Crawl_Directory (Full);
               else
                   --  Check if it is a text file or C/Ada source
                   --  FIX: Use exact suffix check (Ada.Strings fixed-length).
                   --  Previously used Index (substring search) which matched
                   --  .css for .c, .html for .h, .js for .s, etc. causing
                   --  the crawler to index compiled frontend dist/ bundles.
                   declare
                      N_Len : constant Natural := Name'Length;
                      function Has_Suffix (Suf : String) return Boolean is
                        (N_Len > Suf'Length and then
                         Name (Name'Last - Suf'Length + 1 .. Name'Last) = Suf);
                   begin
                   if Has_Suffix (".adb") or else
                     Has_Suffix (".ads") or else
                     Has_Suffix (".c") or else
                     Has_Suffix (".h") or else
                     Has_Suffix (".txt") or else
                     Has_Suffix (".md") or else
                     Has_Suffix (".py") or else
                     Has_Suffix (".json")
                   then
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

                           --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
                           --  Verbose: print every file being indexed so we
                           --  can see if the crawl actually finds files.
                           Files_Scanned := Files_Scanned + 1;
                           Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Init-V]" &
                                     AnsiAda.Reset & "+" & Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) & "s Crawl: indexing file " &
                                     Full & " (" &
                                     Natural'Image (Files_Scanned) & " scanned)");

                           declare
                              Raw_Content : constant String :=
                                To_String (File_Content);
                              Content     : constant String :=
                                Model_Manager.Sanitize_Think_Tags (Raw_Content);
                              Vec         : Math_Utils.Vector (1 .. 4096) :=
                                [others => 0.0];
                              Len         : Natural := 0;
                           begin
                              if Content'Length > 0 then
                                 --  [VITAL-DO-NOT-REMOVE] Skip embedding if
                                 --  ELP1 arrived mid-file. Prevents wasted
                                 --  ELP0 acquire/deny cycles during crawl.
                                 if Model_Manager.Should_Abort_ELP0 then
                                    Put_Line
                                      (AnsiAda.Foreground (AnsiAda.Cyan) &
                                       "[Init-V]" & AnsiAda.Reset &
                                       " Crawl: ELP1 pending, SKIPPING " &
                                       Name);
                                    Model_Manager.Wait_For_ELP1_Idle;
                                 end if;
                                 Model_Manager.Get_Embedding
                                   (Content, Vec, Len, ELP0);
                                 if Len > 0 then
                                    Database_Manager.Add_Literature_Chunk
                                      (Name, Content, Vec (1 .. Len), "hash");
                                 end if;
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
                   end; -- Has_Suffix declare block
                end if;
            end if;
         end;
      end loop;
      Ada.Directories.End_Search (Search);
      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
      --  Verbose: prints crawl completion stats.
      Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Init-V]" &
                AnsiAda.Reset & "+" & Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) & "s  Crawl_Directory COMPLETE: Path=" & Path &
                " Files_Scanned=" & Natural'Image (Files_Scanned));
   end Crawl_Directory;

   task body Native_Crawl_Task is
   begin
      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
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
         --  Verbose: prints BEFORE blocking on Wait_For_ELP1_Idle so we can
         --  see if the task is stuck on the guard condition.
         Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Init-V]" &
                   AnsiAda.Reset & "+" & Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) & "s Native_Crawl_Task: calling Wait_For_ELP1_Idle...");
         Model_Manager.Wait_For_ELP1_Idle;
         --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
         --  Verbose: prints AFTER guard passes so we know it unblocked.
         Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Init-V]" &
                   AnsiAda.Reset & "+" & Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) & "s Native_Crawl_Task: Wait_For_ELP1_Idle PASSED, starting crawl...");
         Crawl_Directory (".");
         Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Init-V]" &
                   AnsiAda.Reset & "+" & Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) & "s Native_Crawl_Task: crawl DONE, sleeping 3600s...");
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
                    (Kind   => Qwen_0_8B,
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

end Knowledge_Manager;
