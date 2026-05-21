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

   task Indexing_Task is
      entry Start;
   end Indexing_Task;

   task Thought_Task is
      entry Start;
   end Thought_Task;

   procedure Initialize is
   begin
      Database_Manager.Initialize;
   end Initialize;

   procedure Start_Tasks is
   begin
      Indexing_Task.Start;
      Thought_Task.Start;
   end Start_Tasks;

   task body Indexing_Task is
      procedure Process_File (Path : String) is
         use GNATCOLL.JSON;
         Body_Obj : constant JSON_Value := Create_Object;
         Resp     : AWS.Response.Data;
      begin
         if Model_Manager.Should_Abort_ELP0 then return; end if;
         Put_Line ("[Indexer] Processing: " & Path);
         Set_Field (Body_Obj, "path", Path);
         begin
            Resp := AWS.Client.Post (ORCHESTRATOR_URL, Write (Body_Obj));
            if AWS.Response.Status_Code (Resp) = S200 then
               declare
                  B_Str : constant String := AWS.Response.Message_Body (Resp);
                  R_Res : constant Read_Result := Read (B_Str);
               begin
                  if R_Res.Success and then R_Res.Value.Kind = JSON_Object_Type then
                     if Has_Field (R_Res.Value, "content") then
                        declare
                           C_Text : constant String := Get (R_Res.Value, "content");
                        begin
                           if C_Text'Length > 0 then
                              declare
                                 V : Math_Utils.Vector (1 .. 16384);
                                 VL : Natural;
                                 Pos : Positive := C_Text'First;
                              begin
                                 while Pos <= C_Text'Last loop
                                    declare
                                       L : constant Positive := Positive'Min (Pos + 999, C_Text'Last);
                                       Chunk : constant String := C_Text (Pos .. L);
                                    begin
                                       Model_Manager.Get_Embedding (Chunk, V, VL);
                                       if VL > 0 then
                                          Database_Manager.Add_Literature_Chunk (Path, Chunk, V (1 .. VL), "hash");
                                       end if;
                                       Pos := L + 1;
                                       exit when Model_Manager.Should_Abort_ELP0;
                                    end;
                                 end loop;
                              end;
                           end if;
                        end;
                     end if;
                  end if;
               end;
            end if;
         exception when others => null; end;
      end Process_File;

      procedure Scan (Dir : String) is
         use Ada.Directories;
         Search : Search_Type;
         Ent : Directory_Entry_Type;
      begin
         Start_Search (Search, Dir, "*");
         while More_Entries (Search) loop
            Get_Next_Entry (Search, Ent);
            if Kind (Ent) = Ordinary_File then Process_File (Full_Name (Ent)); end if;
            exit when Model_Manager.Should_Abort_ELP0;
         end loop;
         End_Search (Search);
      exception when others => null; end Scan;
   begin
      accept Start;
      loop
         begin
            Put_Line ("[Indexer] Crawl started.");
            Scan ("legacyPython");
            Scan ("Adelaide_Lite/src");
            First_Index_Done := True;
            Put_Line ("[Indexer] Crawl done.");
            delay 300.0;
         exception when E : others => delay 60.0; end;
      end loop;
   end Indexing_Task;

   task body Thought_Task is
      Res : Unbounded_String;
      Prompt : constant String := "Synthesize knowledge relation in JSON: {""subject"": ""..."",""relation"": ""..."",""target"": ""...""}";
   begin
      accept Start;
      loop
         begin
            if First_Index_Done and then not Model_Manager.Should_Abort_ELP0 then
               Model_Manager.Hybrid_Generate (Prompt, Res, GNATCOLL.JSON.Empty_Array, "thought-loop", null, Model_Manager.ELP0);
               declare
                  use GNATCOLL.JSON;
                  S : constant String := To_String (Res);
                  P1 : constant Natural := Index (S, "{");
                  P2 : Natural := 0;
               begin
                  if P1 > 0 then
                     for K in reverse S'Range loop if S (K) = '}' then P2 := K; exit; end if; end loop;
                     if P2 > P1 then
                        declare
                           R : constant Read_Result := Read (S (P1 .. P2));
                        begin
                           if R.Success then
                              Database_Manager.Add_Graph_Relation (Get (R.Value, "subject"), Get (R.Value, "relation"), Get (R.Value, "target"));
                           end if;
                        end;
                     end if;
                  end if;
               end;
            end if;
         exception when others => null; end;
         delay 60.0;
      end loop;
   end Thought_Task;

end Knowledge_Manager;
