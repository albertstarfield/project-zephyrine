with Ada.Text_IO; use Ada.Text_IO;
with Ada.Directories;
with Ada.Exceptions;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Ada.Strings.Fixed; use Ada.Strings.Fixed;
with Ada.Strings;
with Ada.Calendar; use Ada.Calendar;
with Database_Manager;
with Model_Manager;
with Math_Utils;
with AWS.Client;
with AWS.Response;
with AWS.Messages; use AWS.Messages;
with GNATCOLL.JSON;

package body Knowledge_Manager is

   ORCHESTRATOR_URL : constant String := 
     "http://localhost:11435/api/adelaide/extract";
   
   First_Index_Done : Boolean := False;

   --  Task to traverse and index files
   task Indexing_Task is
      entry Start;
   end Indexing_Task;

   --  Task for proactive background thinking
   task Thought_Task is
      entry Start;
   end Thought_Task;

   ----------------
   -- Initialize --
   ----------------
   procedure Initialize is
   begin
      Database_Manager.Initialize;
   end Initialize;

   -----------------
   -- Start_Tasks --
   -----------------
   procedure Start_Tasks is
   begin
      Indexing_Task.Start;
      Thought_Task.Start;
   end Start_Tasks;

   -------------------
   -- Indexing_Task --
   -------------------
   task body Indexing_Task is
      procedure Process_File (Path : String) is
         use GNATCOLL.JSON;
         Request_Body : constant JSON_Value := Create_Object;
         Resp : AWS.Response.Data;
      begin
         --  Check preemption
         if Model_Manager.Should_Abort_ELP0 then
            return;
         end if;

         Put_Line ("[Indexer] Processing: " & Path);
         Set_Field (Request_Body, "path", Path);
         
         begin
            Resp := AWS.Client.Post (ORCHESTRATOR_URL, Write (Request_Body));
            if AWS.Response.Status_Code (Resp) = S200 then
               declare
                  B_Str : constant String := AWS.Response.Message_Body (Resp);
                  Val : constant JSON_Value := Read (B_Str).Value;
                  Content : constant String := Get (Val, "content");
               begin
                  if Content'Length > 0 then
                     --  Chunking (simplified for now)
                        declare
                           C_Size : constant Positive := 1000;
                           Pos    : Positive := Content'First;
                           V      : Math_Utils.Vector (1 .. 16384);
                           V_Len  : Natural;
                        begin
                           while Pos <= Content'Last loop
                              declare
                                 Last : constant Positive := 
                                   Positive'Min (Pos + C_Size - 1, Content'Last);
                                 Chunk : constant String := Content (Pos .. Last);
                              begin
                                 Model_Manager.Get_Embedding (Chunk, V, V_Len);
                                 if V_Len > 0 then
                                    Database_Manager.Add_Literature_Chunk
                                      (Path, Chunk, V (1 .. V_Len), "hash_placeholder");
                                 end if;
                                 Pos := Pos + C_Size;
                                 
                                 exit when Model_Manager.Should_Abort_ELP0;
                              end;
                           end loop;
                        end;
                  end if;
               end;
            end if;
         exception
            when others =>
               Put_Line ("[Indexer] Extraction bridge failed for " & Path);
         end;
      end Process_File;

      procedure Scan_Dir (Dir : String) is
         use Ada.Directories;
         Filter : constant Filter_Type := 
           (Ordinary_File => True, others => False);
         Ent    : Directory_Entry_Type;
         Search : Search_Type;
      begin
         Start_Search (Search, Dir, "*", Filter);
         while Ada.Directories.More_Entries (Search) loop
            Ada.Directories.Get_Next_Entry (Search, Ent);
            Process_File (Ada.Directories.Full_Name (Ent));
            exit when Model_Manager.Should_Abort_ELP0;
         end loop;
         Ada.Directories.End_Search (Search);
      end Scan_Dir;

   begin
      accept Start;
      loop
         begin
            Put_Line ("[Indexer] Starting background crawl...");
            Scan_Dir ("legacyPython");
            Scan_Dir ("Adelaide_Lite/src");
            
            First_Index_Done := True;
            Put_Line ("[Indexer] Crawl complete. Sleeping...");
            delay 300.0; --  Wait 5 mins before re-scan
         exception
            when E : others =>
               Put_Line ("[Indexer] Fatal Error in loop: " & 
                         Ada.Exceptions.Exception_Message (E));
               delay 60.0;
         end;
      end loop;
   end Indexing_Task;

   ------------------
   -- Thought_Task --
   ------------------
   task body Thought_Task is
      Res : Unbounded_String;
      Prompt : constant String := 
        "Synthesize a new research hypothesis based on the existing " &
        "literature. Focus on cross-domain connections. Output in JSON: " &
        "{""subject"": ""..."", ""relation"": ""...""," &
        " ""target"": ""..."", ""thought"": ""...""}";
   begin
      accept Start;
      loop
         begin
            if First_Index_Done then
               if not Model_Manager.Should_Abort_ELP0 then
                  Put_Line ("[Proactive] Initiating background thought cycle...");
                  
                  --  Use ELP0 for background thinking
                  Model_Manager.Hybrid_Generate
                    (Prompt, Res, "background-thought-loop", 
                     null, Model_Manager.ELP0);
                  
                  declare
                     use GNATCOLL.JSON;
                     Raw : constant String := To_String (Res);
                     --  Attempt to parse JSON from thought
                     Start_Idx : constant Natural := Index (Raw, "{");
                     End_Idx   : Natural := 0;
                  begin
                     if Start_Idx > 0 then
                        for K in reverse Raw'Range loop
                           if Raw (K) = '}' then
                              End_Idx := K;
                              exit;
                           end if;
                        end loop;
                     end if;

                     if Start_Idx > 0 and then End_Idx > Start_Idx then
                        declare
                           JSON_Raw : constant String := 
                             Raw (Start_Idx .. End_Idx);
                           Val : constant Read_Result := Read (JSON_Raw);
                        begin
                           if Val.Success then
                              Database_Manager.Add_Graph_Relation
                                (Get (Val.Value, "subject"),
                                 Get (Val.Value, "relation"),
                                 Get (Val.Value, "target"));
                              Put_Line ("[Proactive] Knowledge Graph updated: " &
                                        Get (Val.Value, "subject") & " -> " &
                                        Get (Val.Value, "relation") & " -> " &
                                        Get (Val.Value, "target"));
                           end if;
                        end;
                     end if;
                  end;
               end if;
            end if;
         exception
            when E : others =>
               Put_Line ("[Proactive] Error in thought cycle: " &
                         Ada.Exceptions.Exception_Message (E));
         end;
         delay 60.0; --  Wait 1 min between thoughts
      end loop;
   end Thought_Task;

end Knowledge_Manager;
