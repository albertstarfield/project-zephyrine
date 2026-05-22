with AWS.Response.Set;
with AWS.Messages;
with GNATCOLL.JSON;
with Ada.Text_IO;
with Model_Manager;
with Ada.Strings.Unbounded;
with Ada.Exceptions;
with Math_Utils;
with Ada.Calendar;
with Ada.Calendar.Formatting;
with Ada.Real_Time;
with Ada.Unchecked_Deallocation;
with Streaming_Queue;
with AWS.Resources.Streams;
with AWS.Response;

package body Adelaide_Server_Pkg is

   procedure Register (ID : String; Q : Streaming_Queue.Queue_Access) is
      pragma Unreferenced (Q);
   begin
      Ada.Text_IO.Put_Line ("[Server] Registered: " & ID);
   end Register;

   procedure Unregister (ID : String) is
   begin
      Ada.Text_IO.Put_Line ("[Server] Unregistered: " & ID);
   end Unregister;

   procedure Push_Log (ID : String; Log : String) is
   begin
      Ada.Text_IO.Put_Line ("[Log] [" & ID & "] " & Log);
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
      Ada.Text_IO.Put_Line ("[Server] Output JSON:");
      Ada.Text_IO.Put_Line (Content);
      Ada.Text_IO.Put_Line ("[Server] Status: " & AWS.Messages.Image (Status));

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
         Q          : Streaming_Queue.Queue_Access);
   end Generator_Task;

   type Generator_Task_Access is access Generator_Task;

   task body Generator_Task is
      Local_Prompt : Ada.Strings.Unbounded.Unbounded_String;
      Local_Model  : Ada.Strings.Unbounded.Unbounded_String;
      Local_Format : Streaming_Queue.Format_Type;
      Queue        : Streaming_Queue.Queue_Access;
      Result       : Ada.Strings.Unbounded.Unbounded_String;
   begin
      accept Start
        (Prompt     : String;
         Model_Name : String;
         Format     : Streaming_Queue.Format_Type;
         Q          : Streaming_Queue.Queue_Access)
      do
         Local_Prompt := Ada.Strings.Unbounded.To_Unbounded_String (Prompt);
         Local_Model := Ada.Strings.Unbounded.To_Unbounded_String (Model_Name);
         Local_Format := Format;
         Queue := Q;
      end Start;

      Ada.Text_IO.Put_Line
        ("[Async] Generator Task Started for " &
         Ada.Strings.Unbounded.To_String (Local_Model));
      
      Queue.Set_Format
        (Local_Format,
         Ada.Strings.Unbounded.To_String (Local_Model));
      
      Model_Manager.Hybrid_Generate
        (Prompt     => Ada.Strings.Unbounded.To_String (Local_Prompt),
         Result     => Result,
         Session_ID => "async-stream",
         Stream     => Queue);

      Queue.Close;
      Ada.Text_IO.Put_Line ("[Async] Generator Task Finished.");
   exception
      when E : others =>
         Ada.Text_IO.Put_Line
           ("[Async] Error: " & Ada.Exceptions.Exception_Message (E));
         Queue.Close;
   end Generator_Task;

   procedure Free is new Ada.Unchecked_Deallocation
     (Streaming_Queue.Queue, Streaming_Queue.Queue_Access);
   procedure Free is new Ada.Unchecked_Deallocation
     (Generator_Task, Generator_Task_Access);

   function Dispatch (Request : AWS.Status.Data) return AWS.Response.Data is
      URI    : constant String := AWS.Status.URI (Request);
      Method : constant String := AWS.Status.Method (Request);
      Raw_S  : constant String := AWS.Status.Payload (Request);
      Raw_B  : constant Ada.Strings.Unbounded.Unbounded_String :=
        AWS.Status.Binary_Data (Request);
      Is_OpenAI : constant Boolean := (URI = "/v1/chat/completions");
   begin
      Ada.Text_IO.Put_Line ("[Server] >>> Incoming Request: " & Method & " " & URI);

      if Method = "OPTIONS" then
         Ada.Text_IO.Put_Line ("[Server] Handling Preflight OPTIONS request.");
         return Build_Response ("", AWS.Messages.S204);
      end if;

      if URI = "/v1/models" or else URI = "/api/tags" then
         Ada.Text_IO.Put_Line ("[Server] Processing Model List request...");
         declare
            Resp   : constant GNATCOLL.JSON.JSON_Value :=
              GNATCOLL.JSON.Create_Object;
            Models : GNATCOLL.JSON.JSON_Array := GNATCOLL.JSON.Empty_Array;
            procedure Add_Model (Id, Family : String) is
               M : constant GNATCOLL.JSON.JSON_Value :=
                 GNATCOLL.JSON.Create_Object;
               D : constant GNATCOLL.JSON.JSON_Value :=
                 GNATCOLL.JSON.Create_Object;
            begin
               GNATCOLL.JSON.Set_Field (M, "id", Id);
               GNATCOLL.JSON.Set_Field (M, "object", "model");
               GNATCOLL.JSON.Set_Field (M, "created", Long_Integer'(1686935002));
               GNATCOLL.JSON.Set_Field (M, "owned_by", "adelaide");
               GNATCOLL.JSON.Set_Field (M, "name", Id);
               GNATCOLL.JSON.Set_Field (M, "model", Id);
               GNATCOLL.JSON.Set_Field (M, "modified_at", "2024-05-21T15:00:00Z");
               GNATCOLL.JSON.Set_Field (M, "size", Long_Integer'(4000000000));
               GNATCOLL.JSON.Set_Field (M, "digest", "sha256:adelaide" & Id);
               GNATCOLL.JSON.Set_Field (D, "format", "gguf");
               GNATCOLL.JSON.Set_Field (D, "family", Family);
               GNATCOLL.JSON.Set_Field (M, "details", D);
               GNATCOLL.JSON.Append (Models, M);
            end Add_Model;
         begin
            Add_Model ("adelaide-hybrid", "qwen2");
            Add_Model ("adelaide-embedding", "bert");
            Add_Model ("metamodel", "qwen2");
            Add_Model ("adelaide-metamodel", "qwen2");
            GNATCOLL.JSON.Set_Field (Resp, "object", "list");
            GNATCOLL.JSON.Set_Field (Resp, "data", Models);
            GNATCOLL.JSON.Set_Field (Resp, "models", Models);
            return Build_Response (GNATCOLL.JSON.Write (Resp));
         end;

      elsif URI = "/api/show" then
         declare
            Payload : Ada.Strings.Unbounded.Unbounded_String := 
              (if Raw_S /= "" then
                 Ada.Strings.Unbounded.To_Unbounded_String (Raw_S)
               else Raw_B);
            Model_Name : Ada.Strings.Unbounded.Unbounded_String := 
              Ada.Strings.Unbounded.To_Unbounded_String ("adelaide-hybrid");
         begin
            if Ada.Strings.Unbounded.Length (Payload) > 0 then
               declare
                  Parser_Result : constant GNATCOLL.JSON.Read_Result :=
                    GNATCOLL.JSON.Read (Ada.Strings.Unbounded.To_String (Payload));
               begin
                  if Parser_Result.Success then
                     declare
                        Val : constant GNATCOLL.JSON.JSON_Value :=
                          Parser_Result.Value;
                     begin
                        if GNATCOLL.JSON.Has_Field (Val, "name") then
                           Model_Name := Ada.Strings.Unbounded.To_Unbounded_String
                             (String'(GNATCOLL.JSON.Get (Val, "name")));
                        elsif GNATCOLL.JSON.Has_Field (Val, "model") then
                           Model_Name := Ada.Strings.Unbounded.To_Unbounded_String
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
               Families : GNATCOLL.JSON.JSON_Array := GNATCOLL.JSON.Empty_Array;
               Name_Str : constant String :=
                 Ada.Strings.Unbounded.To_String (Model_Name);
            begin
               if Name_Str = "adelaide-embedding" then
                  GNATCOLL.JSON.Append (Families, GNATCOLL.JSON.Create ("bert"));
                  GNATCOLL.JSON.Set_Field (Details, "family", "bert");
               else
                  GNATCOLL.JSON.Append (Families, GNATCOLL.JSON.Create ("qwen2"));
                  GNATCOLL.JSON.Set_Field (Details, "family", "qwen2");
               end if;
               GNATCOLL.JSON.Set_Field (Details, "families", Families);
               GNATCOLL.JSON.Set_Field (Resp, "details", Details);
               return Build_Response (GNATCOLL.JSON.Write (Resp));
            end;
         end;

      elsif URI = "/api/chat" or else URI = "/v1/chat/completions" then
         declare
            Payload : Ada.Strings.Unbounded.Unbounded_String := 
              (if Raw_S /= "" then
                 Ada.Strings.Unbounded.To_Unbounded_String (Raw_S)
               else Raw_B);
            Val     : GNATCOLL.JSON.JSON_Value;
            Prompt  : Ada.Strings.Unbounded.Unbounded_String :=
              Ada.Strings.Unbounded.To_Unbounded_String ("No payload");
            Images  : GNATCOLL.JSON.JSON_Array := GNATCOLL.JSON.Empty_Array;
            Result  : Ada.Strings.Unbounded.Unbounded_String;
            Resp    : constant GNATCOLL.JSON.JSON_Value :=
              GNATCOLL.JSON.Create_Object;
            Req_Model : Ada.Strings.Unbounded.Unbounded_String :=
              Ada.Strings.Unbounded.To_Unbounded_String ("adelaide-hybrid");
            Is_Streaming : Boolean := False;

            procedure Parse_Content_Array (C_Arr : GNATCOLL.JSON.JSON_Array) is
            begin
               Prompt := Ada.Strings.Unbounded.Null_Unbounded_String;
               for I in 1 .. GNATCOLL.JSON.Length (C_Arr) loop
                  declare
                     Item : constant GNATCOLL.JSON.JSON_Value :=
                       GNATCOLL.JSON.Get (C_Arr, I);
                  begin
                     if GNATCOLL.JSON.Has_Field (Item, "type") then
                        if GNATCOLL.JSON.Get (Item, "type") = "text"
                          and then GNATCOLL.JSON.Has_Field (Item, "text")
                        then
                           Ada.Strings.Unbounded.Append
                             (Prompt, String'(GNATCOLL.JSON.Get (Item, "text")));
                        elsif GNATCOLL.JSON.Get (Item, "type") = "image_url" then
                           GNATCOLL.JSON.Append
                             (Images,
                              GNATCOLL.JSON.Get
                                (GNATCOLL.JSON.Get (Item, "image_url"), "url"));
                        end if;
                     end if;
                  end;
               end loop;
            end Parse_Content_Array;
         begin
            if Ada.Strings.Unbounded.Length (Payload) > 0 then
               declare
                  Parser_Result : constant GNATCOLL.JSON.Read_Result :=
                    GNATCOLL.JSON.Read (Ada.Strings.Unbounded.To_String (Payload));
               begin
                  if Parser_Result.Success then
                     Val := Parser_Result.Value;
                     if GNATCOLL.JSON.Has_Field (Val, "model") then
                        Req_Model := Ada.Strings.Unbounded.To_Unbounded_String
                          (String'(GNATCOLL.JSON.Get (Val, "model")));
                     end if;
                     if GNATCOLL.JSON.Has_Field (Val, "stream") then
                        Is_Streaming := GNATCOLL.JSON.Get (Val, "stream");
                     end if;
                     
                     if GNATCOLL.JSON.Has_Field (Val, "messages") then
                        declare
                           Msgs : constant GNATCOLL.JSON.JSON_Array :=
                             GNATCOLL.JSON.Get (Val, "messages");
                           Last : constant GNATCOLL.JSON.JSON_Value :=
                             GNATCOLL.JSON.Get (Msgs, GNATCOLL.JSON.Length (Msgs));
                        begin
                           if GNATCOLL.JSON.Has_Field (Last, "content") then
                              begin
                                 Prompt := Ada.Strings.Unbounded.To_Unbounded_String
                                   (String'(GNATCOLL.JSON.Get (Last, "content")));
                              exception
                                 when others =>
                                    Parse_Content_Array (GNATCOLL.JSON.Get (Last, "content"));
                              end;
                           end if;
                        end;
                     elsif GNATCOLL.JSON.Has_Field (Val, "prompt") then
                        Prompt := Ada.Strings.Unbounded.To_Unbounded_String
                          (String'(GNATCOLL.JSON.Get (Val, "prompt")));
                     end if;
                  end if;
               end;
            end if;

            if Is_Streaming then
               declare
                  Q : constant Streaming_Queue.Queue_Access :=
                    new Streaming_Queue.Queue;
                  T : constant Generator_Task_Access :=
                    new Generator_Task;
                  S : constant Streaming_Queue.Response_Stream_Access :=
                    new Streaming_Queue.Response_Stream;
               begin
                  S.Q := Q;
                  T.Start (Ada.Strings.Unbounded.To_String (Prompt),
                           Ada.Strings.Unbounded.To_String (Req_Model), 
                           (if Is_OpenAI then Streaming_Queue.OpenAI
                            else Streaming_Queue.Ollama), Q);
                  declare
                     R : AWS.Response.Data := AWS.Response.Stream
                       (Content_Type => (if Is_OpenAI then "text/event-stream"
                                         else "application/x-ndjson"),
                        Handle       => S);
                  begin
                     AWS.Response.Set.Add_Header (R, "Access-Control-Allow-Origin", "*");
                     return R;
                  end;
               end;
            else
               declare
                  Now      : constant Ada.Calendar.Time := Ada.Calendar.Clock;
                  TS_Str   : String := Ada.Calendar.Formatting.Image (Now);
               begin
                  if TS_Str'Length >= 11 then TS_Str (11) := 'T'; end if;
                  Model_Manager.Hybrid_Generate
                    (Ada.Strings.Unbounded.To_String (Prompt), Result, Images, "web-api");
                  
                  declare
                     Msg_Out : constant GNATCOLL.JSON.JSON_Value :=
                       GNATCOLL.JSON.Create_Object;
                     Choice  : constant GNATCOLL.JSON.JSON_Value :=
                       GNATCOLL.JSON.Create_Object;
                     Choices : GNATCOLL.JSON.JSON_Array :=
                       GNATCOLL.JSON.Empty_Array;
                  begin
                     GNATCOLL.JSON.Set_Field (Msg_Out, "role", "assistant");
                     GNATCOLL.JSON.Set_Field (Msg_Out, "content",
                                              Ada.Strings.Unbounded.To_String (Result));
                     GNATCOLL.JSON.Set_Field (Choice, "message", Msg_Out);
                     GNATCOLL.JSON.Append (Choices, Choice);
                     GNATCOLL.JSON.Set_Field (Resp, "model",
                                              Ada.Strings.Unbounded.To_String (Req_Model));
                     GNATCOLL.JSON.Set_Field (Resp, "choices", Choices);
                     GNATCOLL.JSON.Set_Field (Resp, "message", Msg_Out);
                     GNATCOLL.JSON.Set_Field (Resp, "done", True);
                     GNATCOLL.JSON.Set_Field (Resp, "created", Long_Integer'(1686935002));
                     GNATCOLL.JSON.Set_Field (Resp, "created_at", TS_Str & "Z");
                     return Build_Response (GNATCOLL.JSON.Write (Resp));
                  end;
               end;
            end if;
         end;

      elsif URI = "/api/embeddings" or else URI = "/v1/embeddings" then
         declare
            Payload : Ada.Strings.Unbounded.Unbounded_String := 
              (if Raw_S /= "" then
                 Ada.Strings.Unbounded.To_Unbounded_String (Raw_S)
               else Raw_B);
            Prompt  : Ada.Strings.Unbounded.Unbounded_String :=
              Ada.Strings.Unbounded.Null_Unbounded_String;
            Resp    : constant GNATCOLL.JSON.JSON_Value :=
              GNATCOLL.JSON.Create_Object;
         begin
            if Ada.Strings.Unbounded.Length (Payload) > 0 then
               declare
                  Parser_Result : constant GNATCOLL.JSON.Read_Result :=
                    GNATCOLL.JSON.Read (Ada.Strings.Unbounded.To_String (Payload));
               begin
                  if Parser_Result.Success then
                     declare
                        Val : constant GNATCOLL.JSON.JSON_Value :=
                          Parser_Result.Value;
                     begin
                        if GNATCOLL.JSON.Has_Field (Val, "prompt") then
                           Prompt := Ada.Strings.Unbounded.To_Unbounded_String
                             (String'(GNATCOLL.JSON.Get (Val, "prompt")));
                        elsif GNATCOLL.JSON.Has_Field (Val, "input") then
                           begin
                              Prompt := Ada.Strings.Unbounded.To_Unbounded_String
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
               Vec     : Math_Utils.Vector (1 .. 4096) := (others => 0.0);
               Len     : Natural := 0;
               Emb_Arr : GNATCOLL.JSON.JSON_Array := GNATCOLL.JSON.Empty_Array;
            begin
               Model_Manager.Get_Embedding
                 (Ada.Strings.Unbounded.To_String (Prompt), Vec, Len);
               for I in 1 .. Len loop
                  GNATCOLL.JSON.Append (Emb_Arr, GNATCOLL.JSON.Create (Long_Float (Vec (I))));
               end loop;

               if URI = "/api/embeddings" then
                  GNATCOLL.JSON.Set_Field (Resp, "embedding", Emb_Arr);
               else
                  declare
                     Data_Arr  : GNATCOLL.JSON.JSON_Array := GNATCOLL.JSON.Empty_Array;
                     Data_Obj  : constant GNATCOLL.JSON.JSON_Value :=
                       GNATCOLL.JSON.Create_Object;
                  begin
                     GNATCOLL.JSON.Set_Field (Data_Obj, "object", "embedding");
                     GNATCOLL.JSON.Set_Field (Data_Obj, "index", Integer'(0));
                     GNATCOLL.JSON.Set_Field (Data_Obj, "embedding", Emb_Arr);
                     GNATCOLL.JSON.Append (Data_Arr, Data_Obj);
                     GNATCOLL.JSON.Set_Field (Resp, "object", "list");
                     GNATCOLL.JSON.Set_Field (Resp, "data", Data_Arr);
                     GNATCOLL.JSON.Set_Field (Resp, "model", "adelaide-embedding");
                  end;
               end if;
               return Build_Response (GNATCOLL.JSON.Write (Resp));
            end;
         end;
      else
         return Build_Response ("Adelaide API Endpoint", AWS.Messages.S404, "text/plain");
      end if;
   exception
      when E : others =>
         Ada.Text_IO.Put_Line ("[Server] Error: " & Ada.Exceptions.Exception_Message (E));
         return Build_Response ("{}", AWS.Messages.S500);
   end Dispatch;

end Adelaide_Server_Pkg;
