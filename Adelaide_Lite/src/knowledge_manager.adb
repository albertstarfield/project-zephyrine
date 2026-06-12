pragma SPARK_Mode (Off);
with AnsiAda;
with Ada.Text_IO; use Ada.Text_IO;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Ada.Strings.Fixed; use Ada.Strings.Fixed;
with Model_Manager;
with Model_Types; use Model_Types;
with Database_Manager;
with Math_Utils;
with Zenith_Manager;
with Ada.Directories; use Ada.Directories;
with Ada.Exceptions;
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
      entry Start;
   end Proactive_Cache_Task;

   procedure Initialize is
   begin
      Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Knowledge]" &
                AnsiAda.Reset & " Initializing Knowledge Base...");
      Database_Manager.Initialize;
   end Initialize;

   procedure Start_Tasks is
   begin
      Indexing_Task.Start;
      Thought_Task.Start;
      Native_Crawl_Task.Start;
      Salience_Maintenance_Task.Start;
      Zenith_Manager.Zenith_Orion_Task.Start;
      Telemetry_Sync_Task.Start;
      Zenith_Prover_Task.Start;
      Proactive_Cache_Task.Start;
   end Start_Tasks;

   --  Helper to index references.bib
   procedure Index_References is
      File          : File_Type;
      Opened        : Boolean := False;
      Current_Entry : Unbounded_String;
      Line          : Unbounded_String;
   begin
      begin
         Open (File, In_File, "../legacyPython/references.bib");
         Opened := True;
      exception
         when others =>
            begin
               Open (File, In_File, "legacyPython/references.bib");
               Opened := True;
            exception
               when others =>
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
      accept Start;
      Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Knowledge]" &
                AnsiAda.Reset & " Indexing Task Active.");
      Index_References;
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
   begin
      Ada.Directories.Start_Search (Search, Path, "");
      while Ada.Directories.More_Entries (Search) loop
         if Model_Manager.Should_Abort_ELP0 then
            return;
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
                  if Index (Name, ".adb") > 0 or else
                    Index (Name, ".ads") > 0 or else
                    Index (Name, ".c") > 0 or else
                    Index (Name, ".h") > 0 or else
                    Index (Name, ".txt") > 0 or else
                    Index (Name, ".md") > 0
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
               end if;
            end if;
         end;
      end loop;
      Ada.Directories.End_Search (Search);
   end Crawl_Directory;

   task body Native_Crawl_Task is
   begin
      accept Start;
      Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Knowledge]" &
                AnsiAda.Reset & " Native Crawl Task Active.");
      loop
         Model_Manager.Wait_For_ELP1_Idle;
         Crawl_Directory (".");
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
      accept Start;
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
