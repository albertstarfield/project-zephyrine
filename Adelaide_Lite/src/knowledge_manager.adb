with AnsiAda;
with Ada.Text_IO; use Ada.Text_IO;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Ada.Strings.Fixed; use Ada.Strings.Fixed;
with Model_Manager;
with Database_Manager;
with Math_Utils;
with Ada.Directories;
with Ada.Environment_Variables;
with Ada.Exceptions;
with GNAT.Expect;
with GNAT.OS_Lib;

package body Knowledge_Manager is

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

   procedure Initialize is
   begin
      Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Knowledge]" & AnsiAda.Reset & " Initializing Knowledge Base...");
      Database_Manager.Initialize;
   end Initialize;

   procedure Start_Tasks is
   begin
      Indexing_Task.Start;
      Thought_Task.Start;
      Native_Crawl_Task.Start;
      Salience_Maintenance_Task.Start;
   end Start_Tasks;

   --  Helper to index references.bib
   procedure Index_References is
      File          : File_Type;
      Opened        : Boolean := False;
      Current_Entry : Unbounded_String;
      Line          : Unbounded_String;
   begin
      --  Try different relative paths
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
                  Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Knowledge]" & AnsiAda.Reset & " references.bib not found. Skipping.");
            end;
      end;

      if not Opened then
         return;
      end if;

      Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Knowledge]" & AnsiAda.Reset & " Parsing and indexing references.bib...");

      while not End_Of_File (File) loop
         if Model_Manager.Should_Abort_ELP0 then
            Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Knowledge]" & AnsiAda.Reset & " Indexing aborted due to ELP1 preemption.");
            Close (File);
            return;
         end if;

         Line := To_Unbounded_String (Get_Line (File));
         if Index (To_String (Line), "@") = 1 then
            if Length (Current_Entry) > 0 then
               declare
                  Content : constant String := To_String (Current_Entry);
                  Vec     : Math_Utils.Vector (1 .. 4096) := (others => 0.0);
                  Len     : Natural := 0;
               begin
                  Model_Manager.Get_Embedding (Content, Vec, Len);
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
            Content : constant String := To_String (Current_Entry);
            Vec     : Math_Utils.Vector (1 .. 4096) := (others => 0.0);
            Len     : Natural := 0;
         begin
            Model_Manager.Get_Embedding (Content, Vec, Len);
            if Len > 0 then
               Database_Manager.Add_Literature_Chunk
                 ("references.bib", Content, Vec (1 .. Len), "hash");
            end if;
         end;
      end if;

      Close (File);
      Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Knowledge]" & AnsiAda.Reset & " references.bib indexing completed.");
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
         Vec     : Math_Utils.Vector (1 .. 4096) := (others => 0.0);
         Len     : Natural := 0;
         use type Ada.Directories.File_Size;

         procedure Process_Text (Text : String) is
            Local_Content : Unbounded_String;
            Start_Idx     : Positive := Text'First;
            End_Idx       : Natural;
         begin
            while Start_Idx <= Text'Last loop
               End_Idx := Ada.Strings.Fixed.Index (Text (Start_Idx .. Text'Last), (1 => ASCII.LF));
               if End_Idx = 0 then
                  End_Idx := Text'Last + 1;
               end if;
               Append (Local_Content, Text (Start_Idx .. End_Idx - 1) & ASCII.LF);
               if Length (Local_Content) > 1000 then
                  Model_Manager.Get_Embedding (To_String (Local_Content), Vec, Len);
                  if Len > 0 then
                     Database_Manager.Add_Literature_Chunk
                       (Path, To_String (Local_Content), Vec (1 .. Len), "hash");
                  end if;
                  Local_Content := Null_Unbounded_String;
               end if;
               Start_Idx := End_Idx + 1;
            end loop;
            if Length (Local_Content) > 0 then
               Model_Manager.Get_Embedding (To_String (Local_Content), Vec, Len);
               if Len > 0 then
                  Database_Manager.Add_Literature_Chunk
                    (Path, To_String (Local_Content), Vec (1 .. Len), "hash");
               end if;
            end if;
         end Process_Text;

      begin
         if Ada.Directories.Size (Path) > 5_000_000 then
            return;
         end if;
         
         if Ada.Strings.Fixed.Index (Path, ".pdf") > 0 then
            declare
               Args : GNAT.OS_Lib.Argument_List (1 .. 1);
               Status : aliased Integer;
            begin
               Args (1) := new String'(Path);
               Process_Text (GNAT.Expect.Get_Command_Output ("python/extract_pdf.py", Args, "", Status'Access));
            end;
            return;
         end if;

         Open (File, In_File, Path);
         while not End_Of_File (File) loop
            Line := To_Unbounded_String (Get_Line (File));
            Append (Content, To_String (Line) & ASCII.LF);
            if Length (Content) > 1000 then
               Model_Manager.Get_Embedding (To_String (Content), Vec, Len);
               if Len > 0 then
                  Database_Manager.Add_Literature_Chunk
                    (Path, To_String (Content), Vec (1 .. Len), "hash");
               end if;
               Content := Null_Unbounded_String;
            end if;
         end loop;
         if Length (Content) > 0 then
            Model_Manager.Get_Embedding (To_String (Content), Vec, Len);
            if Len > 0 then
               Database_Manager.Add_Literature_Chunk
                 (Path, To_String (Content), Vec (1 .. Len), "hash");
            end if;
         end if;
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
                     Walk_Directory (P);
                  elsif Kind (Ent) = Ordinary_File then
                     if Index (N, ".txt") > 0 or else 
                        Index (N, ".md") > 0 or else
                        Index (N, ".adb") > 0 or else 
                        Index (N, ".ads") > 0 or else
                        Index (N, ".py") > 0 or else
                        Index (N, ".pdf") > 0
                     then
                        Index_File (P);
                     end if;
                  end if;
               end if;
            end;
         end loop;
      exception
         when others => null;
      end Walk_Directory;
      
      Home_Dir : constant String :=
        (if Ada.Environment_Variables.Exists ("HOME")
         then Ada.Environment_Variables.Value ("HOME")
         else ".");
   begin
      accept Start;
      loop
         if Model_Manager.Should_Abort_ELP0 then
            delay 1.0;
         else
            Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Knowledge]" & 
                      AnsiAda.Reset & " Starting native Ada filesystem crawl...");
            Walk_Directory (Home_Dir);
            Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Knowledge]" & 
                      AnsiAda.Reset & " Native crawl finished.");
            delay 3600.0;
         end if;
      end loop;
   end Native_Crawl_Task;

   task body Thought_Task is
      Fallback_Text : constant String :=
        "Adelaide Zephyrine Charlotte is a senior engineer. " &
        "She designs the Adelaide-Lite orchestration platform in Ada. " &
        "Ada is a structured, statically typed programming language.";
      Success       : Boolean;
      Chunk_Content : Unbounded_String;
      Target_Text   : Unbounded_String;
      Prompt        : Unbounded_String;
      Res_Val       : Unbounded_String;
   begin
      accept Start;
      loop
         if Model_Manager.Should_Abort_ELP0 then
            delay 1.0;
         else
            Put_Line (AnsiAda.Foreground (AnsiAda.Light_Yellow) & "[Thought]" & AnsiAda.Reset & " Background thinker active. Processing...");
            Database_Manager.Get_Random_Literature_Chunk
              (Chunk_Content, Success);
            if Success and then Length (Chunk_Content) > 0 then
               Target_Text := Chunk_Content;
            else
               Target_Text := To_Unbounded_String (Fallback_Text);
            end if;

            Prompt := To_Unbounded_String
              ("Extract key entities and their relationships from this " &
               "text: """ & To_String (Target_Text) & """. " &
               "Format the output strictly as a list of pipe-separated " &
               "relations: Source | Relation | Target. " &
               "Return nothing else, no explanation.");

            begin
               Model_Manager.Generate
                 (Kind          => Model_Manager.Qwen_0_8B,
                  Prompt        => To_String (Prompt),
                  Result        => Res_Val,
                  Level         => Model_Manager.ELP0);

               if To_String (Res_Val) /= "ERROR: Preempted" and then
                  Length (Res_Val) > 0
               then
                  declare
                     Res_Str : constant String := To_String (Res_Val);
                     Start_Idx : Positive := 1;
                     Next_Idx  : Natural;
                  begin
                     while Start_Idx <= Res_Str'Last loop
                        Next_Idx := Index
                          (Res_Str (Start_Idx .. Res_Str'Last),
                           (1 => ASCII.LF));
                        declare
                           Line_End : constant Natural :=
                             (if Next_Idx = 0 then Res_Str'Last
                              else Next_Idx - 1);
                           Line     : constant String :=
                             Trim
                               (Res_Str (Start_Idx .. Line_End),
                                Ada.Strings.Both);
                           Pipe1    : constant Natural := Index (Line, "|");
                        begin
                           if Pipe1 > 0 then
                              declare
                                 Pipe2 : constant Natural :=
                                   Index
                                     (Line (Pipe1 + 1 .. Line'Last), "|");
                              begin
                                 if Pipe2 > 0 then
                                    declare
                                       Src : constant String :=
                                         Trim
                                           (Line (Line'First .. Pipe1 - 1),
                                            Ada.Strings.Both);
                                       Rel : constant String :=
                                         Trim
                                           (Line (Pipe1 + 1 .. Pipe2 - 1),
                                            Ada.Strings.Both);
                                       Tgt : constant String :=
                                         Trim
                                           (Line (Pipe2 + 1 .. Line'Last),
                                            Ada.Strings.Both);
                                    begin
                                       if Src'Length > 0 and then
                                          Tgt'Length > 0
                                       then
                                          Database_Manager.Add_Graph_Relation
                                            (Src, Rel, Tgt);
                                       end if;
                                    end;
                                 end if;
                              end;
                           end if;
                        end;

                        if Next_Idx = 0 then
                           exit;
                        end if;
                        Start_Idx := Next_Idx + 1;
                     end loop;
                  end;

                  --  Export current knowledge graph to GraphML format
                  Database_Manager.Export_GraphML ("literature.graphml");
               end if;
            exception
               when others =>
                  Put_Line (AnsiAda.Foreground (AnsiAda.Light_Yellow) & "[Thought]" & AnsiAda.Reset & " Error in background thought extraction.");
            end;

            delay 15.0;
         end if;
      end loop;
   end Thought_Task;

   task body Salience_Maintenance_Task is
      Latency_Threshold : constant Duration := 300.0;
      Check_Interval    : constant Duration := 300.0; -- 5 Minutes
      Chunk_Size        : constant Positive := 100;
   begin
      accept Start;
      loop
         if Model_Manager.Current_WCET > Latency_Threshold then
            --  Perform eviction
            Database_Manager.Evict_Low_Salience (Chunk_Size);
         end if;
         delay Check_Interval;
      end loop;
   exception
      when E : others =>
         Put_Line (AnsiAda.Foreground (AnsiAda.Red) & "[Salience]" & AnsiAda.Reset & " Task Error: " &
           Ada.Exceptions.Exception_Message (E));
   end Salience_Maintenance_Task;

end Knowledge_Manager;
