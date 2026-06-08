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
with Ada.Directories;
with Ada.Exceptions;
with Ada.Streams;
with Ada.Streams.Stream_IO;
with GNAT.Expect;
with GNAT.OS_Lib;
with Integrity_Utils;
with Interfaces;

package body Knowledge_Manager is

   --  ELP levels hierarchy:
   --  ELP0: Background Literature Indexing (Lowest Priority) (Self reflecting)
   --  ELP1: Active RAG / Memory Retrieval (User Interaction)
   --  ELP2: StellaIcarus Hooks (Deterministic API Logic)
   --  ELP3: ZenithOrion (Deterministic 1ms Pacing Lock)

   task Indexing_Task is
      entry Start;
   end Indexing_Task;

   task Thought_Task is
      entry Start;
   end Thought_Task;

   task Native_Crawl_Task is
      entry Start;
   end Native_Crawl_Task;

   task Salience_Maintenance_Task is
      entry Start;
   end Salience_Maintenance_Task;

   task Telemetry_Sync_Task is
      entry Start;
   end Telemetry_Sync_Task;

   task Zenith_Prover_Task is
      entry Start;
   end Zenith_Prover_Task;

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
                   Content : constant String :=
                     Model_Manager.Sanitize_Think_Tags (Raw_Content);
                   Vec     : Math_Utils.Vector (1 .. 4096) := [others => 0.0];
                   Len     : Natural := 0;
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
            Content : constant String :=
              Model_Manager.Sanitize_Think_Tags (Raw_Content);
            Vec     : Math_Utils.Vector (1 .. 4096) := [others => 0.0];
            Len     : Natural := 0;
         begin
            Model_Manager.Get_Embedding (Content, Vec, Len, ELP0);
            if Len > 0 then
               Database_Manager.Add_Literature_Chunk
                 ("references.bib", Content, Vec (1 .. Len), "hash");
            end if;
         end;
      end if;

      Close (File);
   exception
      when others =>
         if Is_Open (File) then
            Close (File);
         end if;
   end Index_References;

   task body Indexing_Task is
      Done : Boolean := False;
   begin
      accept Start;
      loop
         if Model_Manager.Should_Abort_ELP0 then
            delay 1.0;
         elsif not Done then
            Index_References;
            Done := True;
         else
            delay 10.0;
         end if;
      end loop;
   end Indexing_Task;

   task body Native_Crawl_Task is
      procedure Index_File (Path : String) is
         File    : File_Type;
         Content : Unbounded_String;
         Line    : Unbounded_String;
         Vec     : Math_Utils.Vector (1 .. 4096) := [others => 0.0];
         Len     : Natural := 0;
         use type Ada.Directories.File_Size;

         procedure Process_Text (Text : String) is
            Local_Content : Unbounded_String;
            Start_Idx     : Positive := Text'First;
            End_Idx       : Natural;
         begin
            while Start_Idx <= Text'Last loop
               End_Idx := Ada.Strings.Fixed.Index
                 (Text (Start_Idx .. Text'Last), [1 => ASCII.LF]);
               if End_Idx = 0 then
                  End_Idx := Text'Last + 1;
               end if;
                Append (Local_Content, Text (Start_Idx .. End_Idx - 1) & ASCII.LF);
                if Length (Local_Content) > 1000 then
                   declare
                      Raw_C : constant String := To_String (Local_Content);
                      Clean_C : constant String :=
                        Model_Manager.Sanitize_Think_Tags (Raw_C);
                   begin
                      Model_Manager.Get_Embedding (Clean_C, Vec, Len, ELP0);
                      if Len > 0 then
                         Database_Manager.Add_Literature_Chunk
                           (Path, Clean_C, Vec (1 .. Len), "hash");
                      end if;
                   end;
                   Local_Content := Null_Unbounded_String;
                end if;
               Start_Idx := End_Idx + 1;
            end loop;
         end Process_Text;

      begin
         if Ada.Directories.Size (Path) > 5_000_000 then
            return;
         end if;

         -- Binary detection check
         declare
            use Interfaces;
            use Ada.Streams;
            use Ada.Streams.Stream_IO;
            Stream_File : Ada.Streams.Stream_IO.File_Type;
            Header      : Stream_Element_Array (1 .. 1024);
            Last        : Stream_Element_Offset;
         begin
            Open (Stream_File, In_File, Path);
            Read (Stream_File, Header, Last);
            Close (Stream_File);
            
            if Last > 0 then
               declare
                  Buffer : Integrity_Utils.Byte_Array (1 .. Positive (Last));
               begin
                  for I in Buffer'Range loop
                     Buffer (I) := Unsigned_8 (Header (Stream_Element_Offset (I)));
                  end loop;
                  if Integrity_Utils.Is_Binary (Buffer) then
                     Put_Line (AnsiAda.Foreground (AnsiAda.Yellow) & "[Knowledge]" &
                               AnsiAda.Reset & " Skipping binary file: " & Path);
                     return;
                  end if;
               end;
            end if;
         exception
            when others =>
               if Is_Open (Stream_File) then
                  Close (Stream_File);
               end if;
         end;

         if Ada.Strings.Fixed.Index (Path, ".pdf") > 0 then
            declare
               Args : GNAT.OS_Lib.Argument_List (1 .. 1);
               Status : aliased Integer;
            begin
               Args (1) := new String'(Path);
               Process_Text (GNAT.Expect.Get_Command_Output
                 ("python/extract_pdf.py", Args, "", Status'Access));
            end;
            return;
         end if;

         Open (File, In_File, Path);
         while not End_Of_File (File) loop
            Line := To_Unbounded_String (Get_Line (File));
            Append (Content, To_String (Line) & ASCII.LF);
            if Length (Content) > 1000 then
               declare
                  Raw_C : constant String := To_String (Content);
                  Clean_C : constant String :=
                    Model_Manager.Sanitize_Think_Tags (Raw_C);
               begin
                  Model_Manager.Get_Embedding (Clean_C, Vec, Len, ELP0);
                  if Len > 0 then
                     Database_Manager.Add_Literature_Chunk
                       (Path, Clean_C, Vec (1 .. Len), "hash");
                  end if;
               end;
               Content := Null_Unbounded_String;
            end if;
         end loop;
         Close (File);
      exception
         when others =>
            if Is_Open (File) then
               Close (File);
            end if;
      end Index_File;

      procedure Walk_Directory (Dir : String) is
         Search : Ada.Directories.Search_Type;
         Ent    : Ada.Directories.Directory_Entry_Type;
         use Ada.Directories;

         function Should_Skip_Dir (Name : String) return Boolean is
         begin
            return Name = "node_modules" or else
                   Name = ".git" or else
                   Name = ".svn" or else
                   Name = "__pycache__" or else
                   Name = "venv" or else
                   Name = ".venv" or else
                   Name = "env" or else
                   Name = ".env" or else
                   Name = ".cache" or else
                   Name = "Caches" or else
                   Name = "Trash" or else
                   Name = ".Trash" or else
                   Name = "tmp" or else
                   Name = "Temp" or else
                   Name = "logs" or else
                   Name = "Logs" or else
                   Name = ".npm" or else
                   Name = ".yarn" or else
                   Name = ".cargo" or else
                   Name = ".rustup" or else
                   Name = ".gem" or else
                   Name = ".m2" or else
                   Name = ".ivy2" or else
                   Name = ".sbt" or else
                   Name = ".vagrant" or else
                   Name = ".docker" or else
                   Name = ".gitlab" or else
                   Name = ".github" or else
                   Name = ".circleci" or else
                   Name = "build" or else
                   Name = "dist" or else
                   Name = "target" or else
                   Name = "_build" or else
                   Name = ".terraform" or else
                   Name = ".serverless" or else
                   Name = "miniconda3" or else
                   Name = "miniconda" or else
                   Name = "anaconda3" or else
                   Name = ".opam" or else
                   Name = "_opam";
         end Should_Skip_Dir;

      begin
         Start_Search (Search, Dir, "");
         while More_Entries (Search) loop
            if Model_Manager.Should_Abort_ELP0 then
               return;
            end if;
            Get_Next_Entry (Search, Ent);
            declare
               N : constant String := Simple_Name (Ent);
               P : constant String := Full_Name (Ent);
            begin
               if N /= "." and then N /= ".." then
                  if Kind (Ent) = Directory then
                     if not Should_Skip_Dir (N) then
                        Walk_Directory (P);
                     end if;
                  elsif Kind (Ent) = Ordinary_File then
                     declare
                        function Ends_With (S, Ext : String) return Boolean is
                        begin
                           return S'Length >= Ext'Length and then
                                  S (S'Last - Ext'Length + 1 .. S'Last) = Ext;
                        end Ends_With;
                     begin
                        if Ends_With (N, ".txt") or else
                           Ends_With (N, ".md") or else
                           Ends_With (N, ".adb") or else
                           Ends_With (N, ".ads") or else
                           Ends_With (N, ".py") or else
                           Ends_With (N, ".pdf") or else
                           Ends_With (N, ".json") or else
                           Ends_With (N, ".yaml") or else
                           Ends_With (N, ".yml") or else
                           Ends_With (N, ".xml") or else
                           Ends_With (N, ".csv") or else
                           Ends_With (N, ".html") or else
                           Ends_With (N, ".htm") or else
                           Ends_With (N, ".css") or else
                           Ends_With (N, ".js") or else
                           Ends_With (N, ".ts") or else
                           Ends_With (N, ".rs") or else
                           Ends_With (N, ".go") or else
                           Ends_With (N, ".c") or else
                           Ends_With (N, ".h") or else
                           Ends_With (N, ".cpp") or else
                           Ends_With (N, ".hpp") or else
                           Ends_With (N, ".lua") or else
                           Ends_With (N, ".toml") or else
                           Ends_With (N, ".ini") or else
                           Ends_With (N, ".cfg") or else
                           Ends_With (N, ".conf") or else
                           Ends_With (N, ".log")
                        then
                           Index_File (P);
                        end if;
                     end;
                  end if;
               end if;
            end;
         end loop;
      exception
         when others => null;
      end Walk_Directory;

    begin
       accept Start;
       loop
          if Model_Manager.Should_Abort_ELP0 then
             delay 1.0;
          else
             Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Knowledge]" &
                       AnsiAda.Reset & " Starting filesystem crawl...");
             --  Crawl common cross-computer knowledge directories
             if Ada.Directories.Exists ("/Users") then
                Walk_Directory ("/Users");
             end if;
             if Ada.Directories.Exists ("/opt") then
                Walk_Directory ("/opt");
             end if;
             if Ada.Directories.Exists ("/usr/local") then
                Walk_Directory ("/usr/local");
             end if;

             delay 3600.0;
         end if;
      end loop;
   end Native_Crawl_Task;

   task body Thought_Task is
      Success          : Boolean;
      Chunk_Content    : Unbounded_String;
      Target_Text      : Unbounded_String;
      Prompt           : Unbounded_String;
      Res_Val          : Unbounded_String;
      Extraction_Count : Natural := 0;
      GraphML_Interval : constant Natural := 10;

      --  Parse "Source | Relation | Target" lines from LLM output
      procedure Store_Triples (Output : String; Source_Text : String) is
         Line_Start : Positive := Output'First;
         Line_End   : Natural;
         Src, Rel, Tgt : Unbounded_String;
         Sep_Pos1, Sep_Pos2 : Natural;
      begin
         while Line_Start <= Output'Last loop
             Line_End := Index (Output (Line_Start .. Output'Last), (1 => ASCII.LF));
            if Line_End = 0 then
               Line_End := Output'Last + 1;
            end if;
            declare
               Line : constant String := Trim (Output (Line_Start .. Line_End - 1), Ada.Strings.Both);
            begin
               if Line'Length > 0 then
                  Sep_Pos1 := Index (Line, " | ");
                  if Sep_Pos1 > 0 then
                     Sep_Pos2 := Index (Line (Sep_Pos1 + 3 .. Line'Last), " | ");
                     if Sep_Pos2 > 0 then
                        Sep_Pos2 := Sep_Pos2 + Sep_Pos1 + 2;
                        Src := To_Unbounded_String (Trim (Line (Line'First .. Sep_Pos1 - 1), Ada.Strings.Both));
                        Rel := To_Unbounded_String (Trim (Line (Sep_Pos1 + 3 .. Sep_Pos2 - 1), Ada.Strings.Both));
                        Tgt := To_Unbounded_String (Trim (Line (Sep_Pos2 + 3 .. Line'Last), Ada.Strings.Both));
                        if Length (Src) > 0 and then Length (Rel) > 0 and then Length (Tgt) > 0 then
                           Database_Manager.Add_Graph_Relation
                             (Source   => To_String (Src),
                              Relation => To_String (Rel),
                              Target   => To_String (Tgt),
                              Weight   => 1.0,
                              Context  => Source_Text);
                           Extraction_Count := Extraction_Count + 1;

                           --  Periodic GraphML export
                           if Extraction_Count mod GraphML_Interval = 0 then
                              begin
                                 Database_Manager.Export_GraphML ("knowledge_graph.graphml");
                                 Put_Line (AnsiAda.Foreground (AnsiAda.Green) & "[Thought]" &
                                   AnsiAda.Reset & " Exported knowledge_graph.graphml (" &
                                   Extraction_Count'Img & " relations)");
                              exception
                                 when others => null;
                              end;
                           end if;
                        end if;
                     end if;
                  end if;
               end if;
            end;
            Line_Start := Line_End + 1;
         end loop;
      end Store_Triples;

   begin
      accept Start;
      loop
         if Model_Manager.Should_Abort_ELP0 then
            delay 1.0;
         else
            Database_Manager.Get_Random_Literature_Chunk (Chunk_Content, Success);
            if Success and then Length (Chunk_Content) > 0 then
               Target_Text := Chunk_Content;
            else
               Target_Text := To_Unbounded_String ("Fallback Knowledge");
            end if;

            Prompt := To_Unbounded_String
              ("Extract relations: Source | Relation | Target from: " &
               To_String (Target_Text));

            begin
               Model_Manager.Generate
                 (Kind => Qwen_0_8B, Prompt => To_String (Prompt),
                  Result => Res_Val, Level => ELP0);
               --  Store extracted triples
               if Length (Res_Val) > 0 then
                  Store_Triples (To_String (Res_Val), To_String (Target_Text));
               end if;
            exception
               when others => null;
            end;
            delay 15.0;
         end if;
      end loop;
   end Thought_Task;

   task body Salience_Maintenance_Task is
      Latency_Threshold : constant Duration := 300.0;
      Check_Interval    : constant Duration := 300.0;
      Chunk_Size        : constant Positive := 100;
   begin
      accept Start;
      loop
         if Model_Manager.Current_WCET > Latency_Threshold then
            Database_Manager.Evict_Low_Salience (Chunk_Size);
         end if;
         delay Check_Interval;
      end loop;
   exception
      when E : others =>
         Put_Line (AnsiAda.Foreground (AnsiAda.Red) & "[Salience] Error: " &
           Ada.Exceptions.Exception_Message (E));
   end Salience_Maintenance_Task;

   task body Telemetry_Sync_Task is
   begin
      accept Start;
      loop
         Model_Manager.Current_WCET_ELP3 := Zenith_Manager.Telemetry_Store.Get_Timing;
         Model_Manager.Current_Jitter_Max := Zenith_Manager.Telemetry_Store.Get_Jitter_Max;
         Model_Manager.Current_Jitter_Avg := Zenith_Manager.Telemetry_Store.Get_Jitter_Avg;
         delay 1.0;
      end loop;
   end Telemetry_Sync_Task;

   task body Zenith_Prover_Task is
   begin
      accept Start;
      loop
         Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[ZenithOrion]" &
                   AnsiAda.Reset & " Auto-scanning and Proving SPARK core...");

         declare
            use GNAT.OS_Lib;
            Ret     : Integer;
            Cmd     : constant String := "gnatprove";
            Args    : Argument_List := [
               1 => new String'("-P"),
               2 => new String'("ZenithOrion/zenith_orion.gpr"),
               3 => new String'("--level=2"),
               4 => new String'("--report=all"),
               5 => new String'("-j0")
            ];
         begin
            Ret := Spawn (Cmd, Args);
            for I in Args'Range loop Free (Args (I)); end loop;

            if Ret = 0 then
               Put_Line (AnsiAda.Foreground (AnsiAda.Green) & "[ZenithOrion]" &
                         AnsiAda.Reset & " SPARK Proof Level 2: SUCCESS");
            else
               Put_Line (AnsiAda.Foreground (AnsiAda.Red) & "[ZenithOrion]" &
                         AnsiAda.Reset & " SPARK Proof FAILED (Ret:" &
                         Ret'Img & ")");
            end if;
         end;

         delay 3600.0;
      end loop;
   end Zenith_Prover_Task;

end Knowledge_Manager;
