with Ada.Text_IO; use Ada.Text_IO;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Ada.Strings.Fixed; use Ada.Strings.Fixed;
with Model_Manager;
with Database_Manager;
with Math_Utils;

package body Knowledge_Manager is

   task Indexing_Task is
      entry Start;
   end Indexing_Task;

   task Thought_Task is
      entry Start;
   end Thought_Task;

   procedure Initialize is
   begin
      Put_Line ("[Knowledge] Initializing Knowledge Base...");
      Database_Manager.Initialize;
   end Initialize;

   procedure Start_Tasks is
   begin
      Indexing_Task.Start;
      Thought_Task.Start;
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
                  Put_Line ("[Knowledge] references.bib not found. Skipping.");
            end;
      end;

      if not Opened then
         return;
      end if;

      Put_Line ("[Knowledge] Parsing and indexing references.bib...");

      while not End_Of_File (File) loop
         if Model_Manager.Should_Abort_ELP0 then
            Put_Line ("[Knowledge] Indexing aborted due to ELP1 preemption.");
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
      Put_Line ("[Knowledge] references.bib indexing completed.");
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

   task body Thought_Task is
      Fallback_Text : constant String :=
        "Adelaide Zephyrine Charlotte is a senior engineer. " &
        "She designs the Adelaide-Lite orchestration platform in Ada. " &
        "Ada is a structured, statically typed programming language.";
   begin
      accept Start;
      loop
         if Model_Manager.Should_Abort_ELP0 then
            delay 1.0;
         else
            Put_Line ("[Thought] Background thinker active. Processing...");
            
            declare
                  while Start <= Res_Str'Last loop
                     Next := Index (Res_Str (Start .. Res_Str'Last), (1 => ASCII.LF));
                     declare
                        Line_End : constant Natural :=
                          (if Next = 0 then Res_Str'Last else Next - 1);
                        Line     : constant String :=
                          Trim (Res_Str (Start .. Line_End), Ada.Strings.Both);
                        Pipe1    : constant Natural := Index (Line, "|");
                     begin
                        if Pipe1 > 0 then
                           declare
                              Pipe2 : constant Natural :=
                                Index (Line (Pipe1 + 1 .. Line'Last), "|");
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

                     if Next = 0 then
                        exit;
                     end if;
                     Start := Next + 1;
                  end loop;
               end;

               --  Export current knowledge graph to GraphML format
               Database_Manager.Export_GraphML ("literature.graphml");
            exception
               when others =>
                  Put_Line ("[Thought] Error in background thought extraction.");
            end;

            delay 15.0;
         end if;
      end loop;
   end Thought_Task;

end Knowledge_Manager;
