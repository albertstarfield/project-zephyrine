with AnsiAda;
with AWS.Response.Set;
with AWS.Messages;
with GNATCOLL.JSON;
with Ada.Text_IO;
with Model_Manager;
with Database_Manager;
with Ada.Strings.Unbounded;
with Ada.Exceptions;
with Math_Utils;
with Ada.Calendar;
with Ada.Calendar.Formatting;
with Ada.Strings.Fixed;
with GNAT.OS_Lib;

package body Adelaide_Server_Pkg is

   procedure Register (ID : String; Q : Streaming_Queue.Queue_Access) is
      pragma Unreferenced (Q);
   begin
      Ada.Text_IO.Put_Line (AnsiAda.Foreground (AnsiAda.Green) & "[Server]" & AnsiAda.Reset & " Registered: " & ID);
   end Register;

   procedure Unregister (ID : String) is
   begin
      Ada.Text_IO.Put_Line (AnsiAda.Foreground (AnsiAda.Green) & "[Server]" & AnsiAda.Reset & " Unregistered: " & ID);
   end Unregister;

   procedure Push_Log (ID : String; Log : String) is
   begin
      Ada.Text_IO.Put_Line (AnsiAda.Foreground (AnsiAda.Light_Grey) & "[Log]" & AnsiAda.Reset & " [" & ID & "] " & Log);
   end Push_Log;

   --  CORS and Header Helper
   function Build_Response
     (Content : String;
      Status  : AWS.Messages.Status_Code := AWS.Messages.S200;
      Type_Str : String := "application/json") return AWS.Response.Data
   is
      Resp : AWS.Response.Data := AWS.Response.Build
        (Content_Type => Type_Str,
         Message_Body => Content,
         Status_Code  => Status);
   begin
      Ada.Text_IO.Put_Line (AnsiAda.Foreground (AnsiAda.Green) & "[Server]" & AnsiAda.Reset & " Status: " & AWS.Messages.Image (Status));

      AWS.Response.Set.Add_Header (Resp, "Access-Control-Allow-Origin", "*");
      AWS.Response.Set.Add_Header
        (Resp, "Access-Control-Allow-Methods", "GET, POST, OPTIONS");
      AWS.Response.Set.Add_Header
        (Resp, "Access-Control-Allow-Headers", "Content-Type, Authorization");
      return Resp;
   end Build_Response;

   --  ASYNCHRONOUS GENERATION TASK
   task type Generator_Task is
      entry Start
        (Prompt     : String;
         Model_Name : String;
         Format     : Streaming_Queue.Format_Type;
         Q          : Streaming_Queue.Queue_Access;
         Agentic    : Boolean;
         Raw_Prompt : Boolean);
   end Generator_Task;

   type Generator_Task_Access is access Generator_Task;

   task body Generator_Task is
      use Ada.Strings.Unbounded;
      Local_Prompt : Unbounded_String;
      Local_Model  : Unbounded_String;
      Local_Format : Streaming_Queue.Format_Type;
      Queue        : Streaming_Queue.Queue_Access;
      Local_Agentic : Boolean;
      Local_Raw_Prompt : Boolean;
      Result       : Unbounded_String;
   begin
      accept Start
        (Prompt     : String;
         Model_Name : String;
         Format     : Streaming_Queue.Format_Type;
         Q          : Streaming_Queue.Queue_Access;
         Agentic    : Boolean;
         Raw_Prompt : Boolean)
      do
         Local_Prompt := To_Unbounded_String (Prompt);
         Local_Model := To_Unbounded_String (Model_Name);
         Local_Format := Format;
         Queue := Q;
         Local_Agentic := Agentic;
         Local_Raw_Prompt := Raw_Prompt;
      end Start;

      Ada.Text_IO.Put_Line (AnsiAda.Foreground (AnsiAda.Yellow) & "[Async]" & AnsiAda.Reset & " Generator Task Started.");
      Queue.Set_Format (Local_Format, To_String (Local_Model));
      Model_Manager.Hybrid_Generate
        (Prompt     => To_String (Local_Prompt),
         Result     => Result,
         Stream     => Queue,
         Session_ID => "async-stream",
         Agentic    => Local_Agentic,
         Raw_Prompt => Local_Raw_Prompt);
      Queue.Close;
      Ada.Text_IO.Put_Line (AnsiAda.Foreground (AnsiAda.Yellow) & "[Async]" & AnsiAda.Reset & " Generator Task Finished.");
   exception
      when E : others =>
         Ada.Text_IO.Put_Line (AnsiAda.Foreground (AnsiAda.Yellow) & "[Async]" & AnsiAda.Reset & " Error: " &
                               Ada.Exceptions.Exception_Message (E));
         Queue.Close;
   end Generator_Task;

   function Dispatch (Request : AWS.Status.Data) return AWS.Response.Data is
      use Ada.Strings.Unbounded;
      URI    : constant String := AWS.Status.URI (Request);
      Method : constant String := AWS.Status.Method (Request);
      Raw_S  : constant String := AWS.Status.Payload (Request);
      Raw_B  : constant Unbounded_String := AWS.Status.Binary_Data (Request);
   begin
      Ada.Text_IO.Put_Line (AnsiAda.Foreground (AnsiAda.Green) & "[Server]" & AnsiAda.Reset & " >>> Incoming: " & Method & " " & URI);

      if Method = "OPTIONS" then
         return Build_Response ("", AWS.Messages.S204);
      end if;

      if (URI'Length >= 10 and then URI (1 .. 10) = "/v1/audio/") or else
         (URI'Length >= 11 and then URI (1 .. 11) = "/v1/images/")
      then
         return Build_Response ("{""error"": ""For Adelaide-Lite it is unavailable for now, please visit the project-zephyrine full instead""}", AWS.Messages.S501);
      end if;

      if URI = "/" then
         if Method = "HEAD" then
            return AWS.Response.Build ("text/plain", "");
         else
            return Build_Response ("Ollama is running");
         end if;
      end if;

      if URI = "/api/version" then
         return Build_Response ("{""version"": ""Project-Zephyrine-0.27""}");
      end if;

      if URI = "/api/ps" then
         return Build_Response ("{""models"": [{""name"": ""metamodel-ELP0"", ""size"": 0, ""size_vram"": 0}, {""name"": ""metamodel-ELP1"", ""size"": 0, ""size_vram"": 0}]}");
      end if;

      if URI = "/api/pull" or else URI = "/api/create" or else URI = "/api/push" or else URI = "/api/copy" or else URI = "/api/delete" then
         return Build_Response ("{""status"": ""success""}");
      end if;

      if URI = "/v1/files" and then Method = "POST" then
         declare
            use GNATCOLL.JSON;
            Req_CT : constant String := AWS.Status.Content_Type (Request);
            B_Idx  : constant Natural := Ada.Strings.Fixed.Index (Req_CT, "boundary=");
            Boundary : Unbounded_String;
            Raw    : constant String := AWS.Status.Payload (Request);
            Content : Unbounded_String;
            Resp   : constant JSON_Value := Create_Object;
         begin
            if B_Idx > 0 then
               Boundary := To_Unbounded_String ("--" & Req_CT (B_Idx + 9 .. Req_CT'Last));
            end if;
            
            declare
               Start_Sig : constant String := "name=""file""";
               F_Idx     : constant Natural := Ada.Strings.Fixed.Index (Raw, Start_Sig);
            begin
               if F_Idx > 0 then
                  declare
                     H_End : constant Natural := Ada.Strings.Fixed.Index (Raw (F_Idx .. Raw'Last), ASCII.CR & ASCII.LF & ASCII.CR & ASCII.LF);
                  begin
                     if H_End > 0 then
                        declare
                           C_Start : constant Natural := H_End + 4;
                           B_End   : constant Natural := Ada.Strings.Fixed.Index (Raw (C_Start .. Raw'Last), To_String (Boundary));
                        begin
                           if B_End > C_Start then
                              Content := To_Unbounded_String (Raw (C_Start .. B_End - 3));
                           end if;
                        end;
                     end if;
                  end;
               end if;
            end;
            
            declare
               TS_Str  : constant String := "123456";
               File_ID : constant String := "file-" & TS_Str;
               Emb_Vec : Math_Utils.Vector (1 .. 1536) := (others => 0.0);
               Emb_Len : Natural := 1536;
            begin
               if Length (Content) > 0 then
                  Model_Manager.Get_Embedding (To_String (Content), Emb_Vec, Emb_Len);
                  Database_Manager.Add_Literature_Chunk
                    (File_Path => ".state/uploads/" & File_ID & ".txt",
                     Content   => To_String (Content),
                     Embedding => Emb_Vec (1 .. Emb_Len),
                     Doc_Hash  => File_ID);
               end if;

               Set_Field (Resp, "id", File_ID);
               Set_Field (Resp, "object", "file");
               Set_Field (Resp, "bytes", Length (Content));
               Set_Field (Resp, "created_at", Long_Integer'(1686935002));
               Set_Field (Resp, "filename", File_ID & ".txt");
               Set_Field (Resp, "purpose", "fine-tune");
               
               return Build_Response (Write (Resp));
            end;
         end;

      elsif URI = "/v1/fine_tuning/jobs" or else URI = "/v1/batches" then
         declare
            use GNATCOLL.JSON;
            TS_Str : constant String := "123456";
            Resp : constant JSON_Value := Create_Object;
            Job_ID : constant String := (if URI = "/v1/batches" then "batch-" else "ftjob-") & TS_Str;
         begin
            Database_Manager.Export_GraphML ("literature.graphml");
            
            Set_Field (Resp, "id", Job_ID);
            Set_Field (Resp, "object", (if URI = "/v1/batches" then "batch" else "fine_tuning.job"));
            Set_Field (Resp, "status", "validating_files");
            Set_Field (Resp, "created_at", Long_Integer'(1686935002));
            
            return Build_Response (Write (Resp));
         end;

      elsif URI = "/v1/assistants" and then Method = "POST" then
         declare
            use GNATCOLL.JSON;
            TS_Str : constant String := "123456";
            Resp : constant JSON_Value := Create_Object;
         begin
            Set_Field (Resp, "id", "asst-" & TS_Str);
            Set_Field (Resp, "object", "assistant");
            Set_Field (Resp, "created_at", Long_Integer'(1686935002));
            Set_Field (Resp, "name", "Zephyrine");
            Set_Field (Resp, "model", "Project-Zephyrine-0.27");
            return Build_Response (Write (Resp));
         end;

      elsif URI = "/v1/threads" and then Method = "POST" then
         declare
            use GNATCOLL.JSON;
            TS_Str : constant String := "123456";
            Resp : constant JSON_Value := Create_Object;
         begin
            Set_Field (Resp, "id", "thread-" & TS_Str);
            Set_Field (Resp, "object", "thread");
            Set_Field (Resp, "created_at", Long_Integer'(1686935002));
            return Build_Response (Write (Resp));
         end;

      elsif URI'Length > 12 and then URI (1 .. 12) = "/v1/threads/" then
         declare
            Rest  : constant String := URI (13 .. URI'Last);
            S_Idx : constant Natural := Ada.Strings.Fixed.Index (Rest, "/");
            use GNATCOLL.JSON;
         begin
            if S_Idx > 0 then
               declare
                  Thread_ID : constant String := Rest (Rest'First .. S_Idx - 1);
                  Action    : constant String := Rest (S_Idx + 1 .. Rest'Last);
                  TS_Str    : constant String := "123456";
                  Resp      : constant JSON_Value := Create_Object;
               begin
                  if Action = "messages" then
                     Set_Field (Resp, "id", "msg-" & TS_Str);
                     Set_Field (Resp, "object", "thread.message");
                     Set_Field (Resp, "created_at", Long_Integer'(1686935002));
                     Set_Field (Resp, "thread_id", Thread_ID);
                     Set_Field (Resp, "role", "user");
                     return Build_Response (Write (Resp));
                  elsif Action = "runs" then
                     Set_Field (Resp, "id", "run-" & TS_Str);
                     Set_Field (Resp, "object", "thread.run");
                     Set_Field (Resp, "created_at", Long_Integer'(1686935002));
                     Set_Field (Resp, "thread_id", Thread_ID);
                     Set_Field (Resp, "status", "completed");
                     return Build_Response (Write (Resp));
                  end if;
                  return Build_Response ("{""status"": ""not_implemented""}", AWS.Messages.S404);
               end;
            end if;
            return Build_Response ("{""status"": ""not_found""}", AWS.Messages.S404);
         end;

      elsif URI = "/v1/models" or else URI = "/api/tags" then
         declare
            Resp   : constant GNATCOLL.JSON.JSON_Value :=
              GNATCOLL.JSON.Create_Object;
            Models : GNATCOLL.JSON.JSON_Array :=
              GNATCOLL.JSON.Empty_Array;
            procedure Add_Model (Id : String; Family : String) is
               M : constant GNATCOLL.JSON.JSON_Value :=
                 GNATCOLL.JSON.Create_Object;
               D : constant GNATCOLL.JSON.JSON_Value :=
                 GNATCOLL.JSON.Create_Object;
            begin
               GNATCOLL.JSON.Set_Field (M, "name", Id);
               GNATCOLL.JSON.Set_Field (M, "model", Id);
               GNATCOLL.JSON.Set_Field (M, "id", Id);
               GNATCOLL.JSON.Set_Field (M, "object", "model");
               GNATCOLL.JSON.Set_Field (M, "created", Long_Integer'(1686935002));
               GNATCOLL.JSON.Set_Field (M, "owned_by", "adelaide");
               GNATCOLL.JSON.Set_Field (M, "modified_at", "2026-05-22T00:00:00Z");
               GNATCOLL.JSON.Set_Field (M, "size", Long_Integer'(4000000000));
               GNATCOLL.JSON.Set_Field (M, "digest", "sha256:adelaide" & Id);
               GNATCOLL.JSON.Set_Field (D, "format", "gguf");
               GNATCOLL.JSON.Set_Field (D, "family", Family);
               GNATCOLL.JSON.Set_Field (M, "details", D);
               GNATCOLL.JSON.Append (Models, M);
            end Add_Model;
         begin
            Add_Model ("Snowball-Enaga", "qwen2");
            Add_Model ("Snowball-Enaga-Embedding", "bert");
            GNATCOLL.JSON.Set_Field (Resp, "object", "list");
            GNATCOLL.JSON.Set_Field (Resp, "data", Models);
            GNATCOLL.JSON.Set_Field (Resp, "models", Models);
            return Build_Response (GNATCOLL.JSON.Write (Resp));
         end;

      elsif URI'Length > 11 and then URI (1 .. 11) = "/v1/models/" then
         declare
            Model_Id : constant String := URI (12 .. URI'Last);
            M : constant GNATCOLL.JSON.JSON_Value :=
              GNATCOLL.JSON.Create_Object;
         begin
            GNATCOLL.JSON.Set_Field (M, "id", Model_Id);
            GNATCOLL.JSON.Set_Field (M, "object", "model");
            GNATCOLL.JSON.Set_Field (M, "created", Long_Integer'(1686935002));
            GNATCOLL.JSON.Set_Field (M, "owned_by", "adelaide");
            return Build_Response (GNATCOLL.JSON.Write (M));
         end;

      elsif URI = "/api/show" then
         declare
            Payload : Unbounded_String :=
              (if Raw_S /= "" then To_Unbounded_String (Raw_S) else Raw_B);
            Model_Name : Unbounded_String :=
              To_Unbounded_String ("Snowball-Enaga");
         begin
            if Length (Payload) > 0 then
               declare
                  Parser_Result : constant GNATCOLL.JSON.Read_Result :=
                    GNATCOLL.JSON.Read (To_String (Payload));
               begin
                  if Parser_Result.Success then
                     declare
                        Val : constant GNATCOLL.JSON.JSON_Value :=
                          Parser_Result.Value;
                     begin
                        if GNATCOLL.JSON.Has_Field (Val, "name") then
                           Model_Name := To_Unbounded_String
                             (String'(GNATCOLL.JSON.Get (Val, "name")));
                        elsif GNATCOLL.JSON.Has_Field (Val, "model") then
                           Model_Name := To_Unbounded_String
                             (String'(GNATCOLL.JSON.Get (Val, "model")));
                        end if;
                     end;
                  end if;
               end;
            end if;
            declare
               Resp : constant GNATCOLL.JSON.JSON_Value :=
                 GNATCOLL.JSON.Create_Object;
               Details : constant GNATCOLL.JSON.JSON_Value :=
                 GNATCOLL.JSON.Create_Object;
               Families : GNATCOLL.JSON.JSON_Array :=
                 GNATCOLL.JSON.Empty_Array;
               Name_Str : constant String := To_String (Model_Name);
            begin
               if Name_Str = "Snowball-Enaga-Embedding" then
                  GNATCOLL.JSON.Append (Families,
                                        GNATCOLL.JSON.Create ("bert"));
                  GNATCOLL.JSON.Set_Field (Details, "family", "bert");
               else
                  GNATCOLL.JSON.Append (Families,
                                        GNATCOLL.JSON.Create ("qwen2"));
                  GNATCOLL.JSON.Set_Field (Details, "family", "qwen2");
               end if;
               GNATCOLL.JSON.Set_Field (Details, "families", Families);
               GNATCOLL.JSON.Set_Field (Resp, "details", Details);
               return Build_Response (GNATCOLL.JSON.Write (Resp));
            end;
         end;

      elsif URI = "/api/chat" or else URI = "/v1/chat/completions" or else
        URI = "/v1/completions" or else URI = "/api/generate"
      then
         declare
            Payload : Unbounded_String :=
              (if Raw_S /= "" then To_Unbounded_String (Raw_S) else Raw_B);
            Val     : GNATCOLL.JSON.JSON_Value;
            Prompt  : Unbounded_String := To_Unbounded_String ("No payload");
            Result  : Unbounded_String;
            Resp    : constant GNATCOLL.JSON.JSON_Value :=
              GNATCOLL.JSON.Create_Object;
            Req_Model : Unbounded_String :=
              To_Unbounded_String ("Snowball-Enaga");
            Is_Streaming : Boolean := False;
            Is_Agentic   : Boolean := False;
            Is_Raw_Prompt : Boolean := False;
            Now      : constant Ada.Calendar.Time := Ada.Calendar.Clock;
            TS_Str   : String := Ada.Calendar.Formatting.Image (Now);
         begin
            if TS_Str'Length >= 11 then
               TS_Str (11) := 'T';
            end if;
            if Length (Payload) > 0 then
               begin
                  declare
                     Parser_Result : constant GNATCOLL.JSON.Read_Result :=
                       GNATCOLL.JSON.Read (To_String (Payload));
                  begin
                     if Parser_Result.Success then
                        Val := Parser_Result.Value;
                        if GNATCOLL.JSON.Has_Field (Val, "model") then
                           Req_Model := To_Unbounded_String
                             (String'(GNATCOLL.JSON.Get (Val, "model")));
                        end if;
                        if GNATCOLL.JSON.Has_Field (Val, "stream") then
                           Is_Streaming := GNATCOLL.JSON.Get (Val, "stream");
                        end if;
                        if GNATCOLL.JSON.Has_Field (Val, "tools") then
                           Is_Agentic := True;
                        end if;
                        if GNATCOLL.JSON.Has_Field (Val, "messages") then
                           declare
                              Msgs : constant GNATCOLL.JSON.JSON_Array :=
                                GNATCOLL.JSON.Get (Val, "messages");
                              Built_Prompt : Unbounded_String := Null_Unbounded_String;
                           begin
                              if GNATCOLL.JSON.Length (Msgs) > 0 then
                                 for I in 1 .. GNATCOLL.JSON.Length (Msgs) loop
                                    declare
                                       Msg : constant GNATCOLL.JSON.JSON_Value :=
                                         GNATCOLL.JSON.Get (Msgs, I);
                                       Role : constant String :=
                                         GNATCOLL.JSON.Get (Msg, "role");
                                       Content : constant String :=
                                         GNATCOLL.JSON.Get (Msg, "content");
                                    begin
                                       Append (Built_Prompt, "<|im_start|>" & Role & ASCII.LF & Content & "<|im_end|>" & ASCII.LF);
                                    end;
                                 end loop;
                                 Append (Built_Prompt, "<|im_start|>assistant" & ASCII.LF);
                                 Prompt := Built_Prompt;
                                 Is_Raw_Prompt := True;
                              end if;
                           end;
                        elsif GNATCOLL.JSON.Has_Field (Val, "prompt") then
                           Prompt := To_Unbounded_String
                             (String'(GNATCOLL.JSON.Get (Val, "prompt")));
                        end if;
                     else
                        return Build_Response ("{""error"": ""Malformed JSON""}", AWS.Messages.S400);
                     end if;
                  end;
               exception
                  when others =>
                     return Build_Response ("{""error"": ""Payload processing error""}", AWS.Messages.S400);
               end;
            end if;

            if Prompt = "No payload" then
               return Build_Response ("{""error"": ""Invalid request: missing prompt or messages""}", AWS.Messages.S400);
            end if;

            Ada.Text_IO.Put_Line (AnsiAda.Foreground (AnsiAda.Green) & "[Server]" & AnsiAda.Reset & " Extracted Prompt: " & To_String (Prompt));

            if Length (Prompt) = 0 then
               Ada.Text_IO.Put_Line (AnsiAda.Foreground (AnsiAda.Green) & "[Server]" & AnsiAda.Reset & " Empty prompt detected, bypassing generation.");
               if Is_Streaming then
                  declare
                     Q : constant Streaming_Queue.Queue_Access := new Streaming_Queue.Queue;
                     S : constant Streaming_Queue.Response_Stream_Access := new Streaming_Queue.Response_Stream;
                  begin
                     S.Q := Q;
                     Q.Set_Format ((if URI = "/v1/chat/completions" or else URI = "/v1/completions" then Streaming_Queue.OpenAI else Streaming_Queue.Ollama), To_String (Req_Model));
                     Q.Close;
                     declare
                        Resp : AWS.Response.Data := AWS.Response.Stream
                          (Content_Type => (if URI = "/v1/chat/completions" or else URI = "/v1/completions" then "text/event-stream" else "application/x-ndjson"),
                           Handle => S);
                     begin
                        AWS.Response.Set.Add_Header (Resp, "Access-Control-Allow-Origin", "*");
                        AWS.Response.Set.Add_Header (Resp, "Access-Control-Allow-Methods", "GET, POST, OPTIONS");
                        AWS.Response.Set.Add_Header (Resp, "Access-Control-Allow-Headers", "Content-Type, Authorization");
                        return Resp;
                     end;
                  end;
               else
                  declare
                     R : constant GNATCOLL.JSON.JSON_Value := GNATCOLL.JSON.Create_Object;
                  begin
                     GNATCOLL.JSON.Set_Field (R, "model", To_String (Req_Model));
                     GNATCOLL.JSON.Set_Field (R, "response", "");
                     GNATCOLL.JSON.Set_Field (R, "done", True);
                     return Build_Response (GNATCOLL.JSON.Write (R));
                  end;
               end if;
            end if;

            -- StellaIcarus Hook Check --
            declare
               use GNAT.OS_Lib;
               Temp_File   : constant String := "stella_cap.tmp";
               Python_Path : GNAT.OS_Lib.String_Access := GNAT.OS_Lib.Locate_Exec_On_Path ("python3");
               Args        : Argument_List (1 .. 2);
               Success     : Boolean;
               Ret_Code    : Integer;
               Hook_Result : Unbounded_String := Null_Unbounded_String;
            begin
               if Python_Path /= null then
                  Args (1) := new String'("python/stellaicarus_bridge.py");
                  Args (2) := new String'(To_String (Prompt));
                  Spawn (Python_Path.all, Args, Temp_File, Success, Ret_Code);
                  Free (Python_Path);
                  for I in Args'Range loop Free (Args (I)); end loop;
                  
                  if Success then
                     declare
                        File : Ada.Text_IO.File_Type;
                        Line : Unbounded_String;
                        Is_Match : Boolean := False;
                     begin
                        Ada.Text_IO.Open (File, Ada.Text_IO.In_File, Temp_File);
                        while not Ada.Text_IO.End_Of_File (File) loop
                           Line := To_Unbounded_String (Ada.Text_IO.Get_Line (File));
                           if Line = "__STELLA_MATCH__" then
                              Is_Match := True;
                           elsif Is_Match then
                              Append (Hook_Result, Line & ASCII.LF);
                           end if;
                        end loop;
                        Ada.Text_IO.Close (File);
                     exception
                        when others => null;
                     end;
                  end if;
                  
                  if Length (Hook_Result) > 0 then
                     Ada.Text_IO.Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[StellaIcarus]" & AnsiAda.Reset & " Hook intercepted generation!");
                     if Is_Streaming then
                        -- For streaming, we need to send the entire string as a chunk
                        declare
                           Q : constant Streaming_Queue.Queue_Access := new Streaming_Queue.Queue;
                           S : constant Streaming_Queue.Response_Stream_Access := new Streaming_Queue.Response_Stream;
                        begin
                           S.Q := Q;
                           Q.Set_Format ((if URI = "/v1/chat/completions" or else URI = "/v1/completions" then Streaming_Queue.OpenAI else Streaming_Queue.Ollama), To_String (Req_Model));
                           Q.Push (To_String (Hook_Result));
                           Q.Close;
                           declare
                              Resp : AWS.Response.Data := AWS.Response.Stream
                                (Content_Type => (if URI = "/v1/chat/completions" or else URI = "/v1/completions" then "text/event-stream" else "application/x-ndjson"),
                                 Handle => S);
                           begin
                              AWS.Response.Set.Add_Header (Resp, "Access-Control-Allow-Origin", "*");
                              AWS.Response.Set.Add_Header (Resp, "Access-Control-Allow-Methods", "GET, POST, OPTIONS");
                              AWS.Response.Set.Add_Header (Resp, "Access-Control-Allow-Headers", "Content-Type, Authorization");
                              return Resp;
                           end;
                        end;
                     else
                        declare
                           R : constant GNATCOLL.JSON.JSON_Value := GNATCOLL.JSON.Create_Object;
                        begin
                           GNATCOLL.JSON.Set_Field (R, "model", To_String (Req_Model));
                           if URI = "/v1/chat/completions" then
                              declare
                                 Choices : GNATCOLL.JSON.JSON_Array := GNATCOLL.JSON.Empty_Array;
                                 Choice  : constant GNATCOLL.JSON.JSON_Value := GNATCOLL.JSON.Create_Object;
                                 Msg     : constant GNATCOLL.JSON.JSON_Value := GNATCOLL.JSON.Create_Object;
                              begin
                                 GNATCOLL.JSON.Set_Field (Msg, "role", "assistant");
                                 GNATCOLL.JSON.Set_Field (Msg, "content", To_String (Hook_Result));
                                 GNATCOLL.JSON.Set_Field (Choice, "message", Msg);
                                 GNATCOLL.JSON.Append (Choices, Choice);
                                 GNATCOLL.JSON.Set_Field (R, "choices", Choices);
                              end;
                           else
                              GNATCOLL.JSON.Set_Field (R, "response", To_String (Hook_Result));
                           end if;
                           GNATCOLL.JSON.Set_Field (R, "done", True);
                           return Build_Response (GNATCOLL.JSON.Write (R));
                        end;
                     end if;
                  end if;
               end if;
            end;

            if Is_Streaming then
               declare
                  Q : constant Streaming_Queue.Queue_Access :=
                    new Streaming_Queue.Queue;
                  T : constant Generator_Task_Access := new Generator_Task;
                  S : constant Streaming_Queue.Response_Stream_Access :=
                    new Streaming_Queue.Response_Stream;
               begin
                  S.Q := Q;
                  T.Start (To_String (Prompt), To_String (Req_Model),
                           (if URI = "/v1/chat/completions" or else URI = "/v1/completions"
                            then Streaming_Queue.OpenAI
                            else Streaming_Queue.Ollama), Q,
                           Is_Agentic, Is_Raw_Prompt);
                  declare
                     Resp : AWS.Response.Data := AWS.Response.Stream
                       (Content_Type => (if URI = "/v1/chat/completions" or else URI = "/v1/completions"
                                         then "text/event-stream"
                                         else "application/x-ndjson"),
                        Handle => S);
                  begin
                     AWS.Response.Set.Add_Header (Resp, "Access-Control-Allow-Origin", "*");
                     AWS.Response.Set.Add_Header (Resp, "Access-Control-Allow-Methods", "GET, POST, OPTIONS");
                     AWS.Response.Set.Add_Header (Resp, "Access-Control-Allow-Headers", "Content-Type, Authorization");
                     return Resp;
                  end;
               end;
            else
               Model_Manager.Hybrid_Generate
                 (Prompt     => To_String (Prompt),
                  Result     => Result,
                  Session_ID => "server-sync",
                  Agentic    => Is_Agentic,
                  Raw_Prompt => Is_Raw_Prompt);
               
               if URI = "/api/generate" then
                  GNATCOLL.JSON.Set_Field (Resp, "model", To_String (Req_Model));
                  GNATCOLL.JSON.Set_Field
                    (Resp, "response", To_String (Result));
                  GNATCOLL.JSON.Set_Field (Resp, "done", True);
                  GNATCOLL.JSON.Set_Field (Resp, "created_at", TS_Str & "Z");
                  GNATCOLL.JSON.Set_Field (Resp, "eval_count", Integer'(0));
                  GNATCOLL.JSON.Set_Field (Resp, "prompt_eval_count", Integer'(0));
                  GNATCOLL.JSON.Set_Field (Resp, "total_duration", Integer'(0));
                  GNATCOLL.JSON.Set_Field (Resp, "load_duration", Integer'(0));
                  GNATCOLL.JSON.Set_Field (Resp, "prompt_eval_duration", Integer'(0));
               else
                  declare
                     Msg_Out : constant GNATCOLL.JSON.JSON_Value :=
                       GNATCOLL.JSON.Create_Object;
                     Choice  : constant GNATCOLL.JSON.JSON_Value :=
                       GNATCOLL.JSON.Create_Object;
                     Choices : GNATCOLL.JSON.JSON_Array :=
                       GNATCOLL.JSON.Empty_Array;
                     Usage   : constant GNATCOLL.JSON.JSON_Value :=
                       GNATCOLL.JSON.Create_Object;
                     Res_Str : constant String := To_String (Result);
                  begin
                     if Is_Agentic and then Res_Str'Length > 12 and then
                        Res_Str (Res_Str'First .. Res_Str'First + 11) =
                          "[TOOL_CALL: "
                     then
                        declare
                           E_Pos  : constant Natural :=
                             Ada.Strings.Fixed.Index
                               (Res_Str, "]", Res_Str'First + 12);
                           A_Full : constant String :=
                             Res_Str (Res_Str'First + 12 .. E_Pos - 1);
                           P_Pos  : constant Natural :=
                             Ada.Strings.Fixed.Index (A_Full, "(");
                           EP_Pos : constant Natural :=
                             Ada.Strings.Fixed.Index (A_Full, ")", P_Pos);
                           T_Name : constant String :=
                             Ada.Strings.Fixed.Trim
                               (A_Full (A_Full'First .. P_Pos - 1),
                                Ada.Strings.Both);
                           T_Pars : constant String :=
                             Ada.Strings.Fixed.Trim
                               (A_Full (P_Pos + 1 .. EP_Pos - 1),
                                Ada.Strings.Both);
                           
                           Tool_Call  : constant GNATCOLL.JSON.JSON_Value :=
                             GNATCOLL.JSON.Create_Object;
                           Func_Obj   : constant GNATCOLL.JSON.JSON_Value :=
                             GNATCOLL.JSON.Create_Object;
                           Tool_Calls : GNATCOLL.JSON.JSON_Array :=
                             GNATCOLL.JSON.Empty_Array;
                        begin
                           GNATCOLL.JSON.Set_Field (Func_Obj, "name", T_Name);
                           GNATCOLL.JSON.Set_Field
                             (Func_Obj, "arguments",
                              "{""query"": """ & T_Pars & """}");
                           
                           GNATCOLL.JSON.Set_Field
                             (Tool_Call, "id", "call_" & TS_Str);
                           GNATCOLL.JSON.Set_Field
                             (Tool_Call, "type", "function");
                           GNATCOLL.JSON.Set_Field
                             (Tool_Call, "function", Func_Obj);
                           
                           GNATCOLL.JSON.Append (Tool_Calls, Tool_Call);
                           
                           GNATCOLL.JSON.Set_Field
                             (Msg_Out, "role", "assistant");
                           GNATCOLL.JSON.Set_Field
                             (Msg_Out, "content", GNATCOLL.JSON.JSON_Null);
                           GNATCOLL.JSON.Set_Field
                             (Msg_Out, "tool_calls", Tool_Calls);
                           
                           GNATCOLL.JSON.Set_Field
                             (Choice, "finish_reason", "tool_calls");
                        end;
                     else
                        GNATCOLL.JSON.Set_Field
                          (Msg_Out, "role", "assistant");
                        GNATCOLL.JSON.Set_Field
                          (Msg_Out, "content", Res_Str);
                        GNATCOLL.JSON.Set_Field
                          (Choice, "finish_reason", "stop");
                     end if;
                     
                     if URI = "/v1/completions" then
                        GNATCOLL.JSON.Set_Field (Choice, "text", Res_Str);
                     else
                        GNATCOLL.JSON.Set_Field (Choice, "message", Msg_Out);
                     end if;
                     GNATCOLL.JSON.Append (Choices, Choice);
                     GNATCOLL.JSON.Set_Field
                       (Resp, "id", "chatcmpl-adelaide-" & TS_Str);
                     GNATCOLL.JSON.Set_Field (Resp, "object",
                                              (if URI = "/v1/completions" then "text_completion" else "chat.completion"));
                     GNATCOLL.JSON.Set_Field (Resp, "created",
                                              Long_Integer'(1686935002));
                     GNATCOLL.JSON.Set_Field (Resp, "created_at", TS_Str & "Z");
                     GNATCOLL.JSON.Set_Field (Resp, "eval_count", Integer'(0));
                     GNATCOLL.JSON.Set_Field (Resp, "prompt_eval_count", Integer'(0));
                     GNATCOLL.JSON.Set_Field (Resp, "total_duration", Integer'(0));
                     GNATCOLL.JSON.Set_Field (Resp, "load_duration", Integer'(0));
                     GNATCOLL.JSON.Set_Field (Resp, "prompt_eval_duration", Integer'(0));
                     GNATCOLL.JSON.Set_Field (Resp, "model",
                                              To_String (Req_Model));
                     GNATCOLL.JSON.Set_Field (Resp, "choices", Choices);
                     GNATCOLL.JSON.Set_Field (Resp, "message", Msg_Out);
                     GNATCOLL.JSON.Set_Field (Resp, "done", True);
                     GNATCOLL.JSON.Set_Field (Usage, "prompt_tokens",
                                              Integer'(0));
                     GNATCOLL.JSON.Set_Field (Usage, "completion_tokens",
                                              Integer'(0));
                     GNATCOLL.JSON.Set_Field (Usage, "total_tokens",
                                              Integer'(0));
                     GNATCOLL.JSON.Set_Field (Resp, "usage", Usage);
                  end;
               end if;
               return Build_Response (GNATCOLL.JSON.Write (Resp));
            end if;
         end;

      elsif URI = "/api/embeddings" or else URI = "/v1/embeddings" or else URI = "/api/embed" then
         declare
            Payload : Unbounded_String :=
              (if Raw_S /= "" then To_Unbounded_String (Raw_S) else Raw_B);
            Prompt  : Unbounded_String := Null_Unbounded_String;
            Resp    : constant GNATCOLL.JSON.JSON_Value :=
              GNATCOLL.JSON.Create_Object;
         begin
            if Length (Payload) > 0 then
               declare
                  Parser_Result : constant GNATCOLL.JSON.Read_Result :=
                    GNATCOLL.JSON.Read (To_String (Payload));
               begin
                  if Parser_Result.Success then
                     declare
                        Val : constant GNATCOLL.JSON.JSON_Value :=
                          Parser_Result.Value;
                     begin
                        if GNATCOLL.JSON.Has_Field (Val, "prompt") then
                           Prompt := To_Unbounded_String
                             (String'(GNATCOLL.JSON.Get (Val, "prompt")));
                        elsif GNATCOLL.JSON.Has_Field (Val, "input") then
                           begin
                              Prompt := To_Unbounded_String
                                (String'(GNATCOLL.JSON.Get (Val, "input")));
                           exception
                              when others => null;
                           end;
                        end if;
                     end;
                  end if;
               end;
            end if;
            declare
               Vec     : Math_Utils.Vector (1 .. 4096) := [others => 0.0];
               Len     : Natural := 0;
               Emb_Arr : GNATCOLL.JSON.JSON_Array := GNATCOLL.JSON.Empty_Array;
            begin
               Model_Manager.Get_Embedding (To_String (Prompt), Vec, Len);
               if Len = 0 then
                  for I in 1 .. 128 loop
                     GNATCOLL.JSON.Append
                       (Emb_Arr, GNATCOLL.JSON.Create (Long_Float (0.1)));
                  end loop;
               else
                  for I in 1 .. Len loop
                     GNATCOLL.JSON.Append
                       (Emb_Arr, GNATCOLL.JSON.Create (Long_Float (Vec (I))));
                  end loop;
               end if;
               if URI = "/api/embeddings" or else URI = "/api/embed" then
                  GNATCOLL.JSON.Set_Field (Resp, "embedding", Emb_Arr);
               else
                  declare
                     Data_Arr  : GNATCOLL.JSON.JSON_Array :=
                       GNATCOLL.JSON.Empty_Array;
                     Data_Obj  : constant GNATCOLL.JSON.JSON_Value :=
                       GNATCOLL.JSON.Create_Object;
                  begin
                     GNATCOLL.JSON.Set_Field (Data_Obj, "object", "embedding");
                     GNATCOLL.JSON.Set_Field (Data_Obj, "index", Integer'(0));
                     GNATCOLL.JSON.Set_Field (Data_Obj, "embedding", Emb_Arr);
                     GNATCOLL.JSON.Append (Data_Arr, Data_Obj);
                     GNATCOLL.JSON.Set_Field (Resp, "object", "list");
                     GNATCOLL.JSON.Set_Field (Resp, "data", Data_Arr);
                     GNATCOLL.JSON.Set_Field (Resp, "model",
                                              "adelaide-embedding");
                  end;
               end if;
               return Build_Response (GNATCOLL.JSON.Write (Resp));
            end;
         end;
      else
         return Build_Response ("Adelaide API Endpoint",
                                AWS.Messages.S404, "text/plain");
      end if;
   exception
      when E : others =>
         Ada.Text_IO.Put_Line (AnsiAda.Foreground (AnsiAda.Green) & "[Server]" & AnsiAda.Reset & " Error: " &
                               Ada.Exceptions.Exception_Message (E));
         return Build_Response ("{}", AWS.Messages.S500);
   end Dispatch;

end Adelaide_Server_Pkg;
