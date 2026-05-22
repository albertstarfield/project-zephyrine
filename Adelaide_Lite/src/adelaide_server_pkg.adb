with AWS.Response.Set;
with AWS.Messages;
with GNATCOLL.JSON; use GNATCOLL.JSON;
with Ada.Text_IO; use Ada.Text_IO;
with Model_Manager;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Ada.Exceptions;
with Math_Utils;
with Ada.Strings.Fixed; use Ada.Strings.Fixed;
with Ada.Calendar;
with Ada.Calendar.Formatting;
with Ada.Real_Time;
with Ada.Unchecked_Deallocation;
with Streaming_Queue;
with AWS.Resources;

package body Adelaide_Server_Pkg is

   procedure Register (ID : String; Q : Streaming_Queue.Queue_Access) is
      pragma Unreferenced (Q);
   begin
      Put_Line ("[Server] Registered: " & ID);
   end Register;

   procedure Unregister (ID : String) is
   begin
      Put_Line ("[Server] Unregistered: " & ID);
   end Unregister;

   procedure Push_Log (ID : String; Log : String) is
   begin
      Put_Line ("[Log] [" & ID & "] " & Log);
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
      Put_Line ("[Server] Output JSON:");
      Put_Line (Content);
      Put_Line ("[Server] Status: " & AWS.Messages.Image (Status));

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
      Local_Prompt : Unbounded_String;
      Local_Model  : Unbounded_String;
      Local_Format : Streaming_Queue.Format_Type;
      Queue        : Streaming_Queue.Queue_Access;
      Result       : Unbounded_String;
   begin
      accept Start
        (Prompt     : String;
         Model_Name : String;
         Format     : Streaming_Queue.Format_Type;
         Q          : Streaming_Queue.Queue_Access)
      do
         Local_Prompt := To_Unbounded_String (Prompt);
         Local_Model := To_Unbounded_String (Model_Name);
         Local_Format := Format;
         Queue := Q;
      end Start;

      Put_Line ("[Async] Generator Task Started for " & To_String (Local_Model));
      
      Queue.Set_Format (Local_Format, To_String (Local_Model));
      
      Model_Manager.Hybrid_Generate
        (Prompt     => To_String (Local_Prompt),
         Result     => Result,
         Session_ID => "async-stream",
         Stream     => Queue);

      Queue.Close;
      Put_Line ("[Async] Generator Task Finished.");
   exception
      when E : others =>
         Put_Line ("[Async] Error: " & Ada.Exceptions.Exception_Message (E));
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
      Raw_B  : constant Unbounded_String := AWS.Status.Binary_Data (Request);
      Is_OpenAI : constant Boolean := (URI = "/v1/chat/completions");
   begin
      Put_Line ("[Server] >>> Incoming Request: " & Method & " " & URI);

      if Method = "OPTIONS" then
         Put_Line ("[Server] Handling Preflight OPTIONS request.");
         return Build_Response ("", AWS.Messages.S204);
      end if;

      if URI = "/v1/models" or else URI = "/api/tags" then
         Put_Line ("[Server] Processing Model List request...");
         declare
            Resp   : constant JSON_Value := Create_Object;
            Models : JSON_Array := Empty_Array;
            procedure Add_Model (Id, Family : String) is
               M : constant JSON_Value := Create_Object;
               D : constant JSON_Value := Create_Object;
            begin
               Set_Field (M, "id", Id);
               Set_Field (M, "object", "model");
               Set_Field (M, "created", Long_Integer'(1686935002));
               Set_Field (M, "owned_by", "adelaide");
               Set_Field (M, "name", Id);
               Set_Field (M, "model", Id);
               Set_Field (M, "modified_at", "2024-05-21T15:00:00Z");
               Set_Field (M, "size", Long_Integer'(4000000000));
               Set_Field (M, "digest", "sha256:adelaide" & Id);
               Set_Field (D, "format", "gguf");
               Set_Field (D, "family", Family);
               Set_Field (M, "details", D);
               Append (Models, M);
            end Add_Model;
         begin
            Add_Model ("adelaide-hybrid", "qwen2");
            Add_Model ("adelaide-embedding", "bert");
            Add_Model ("metamodel", "qwen2");
            Add_Model ("adelaide-metamodel", "qwen2");
            Set_Field (Resp, "object", "list");
            Set_Field (Resp, "data", Models);
            Set_Field (Resp, "models", Models);
            return Build_Response (Write (Resp));
         end;

      elsif URI = "/api/show" then
         declare
            Payload : Unbounded_String := (if Raw_S'Length > 0 then To_Unbounded_String (Raw_S) else Raw_B);
            Model_Name : Unbounded_String := To_Unbounded_String ("adelaide-hybrid");
         begin
            if Length (Payload) > 0 then
               declare
                  Parser_Result : constant Read_Result := Read (To_String (Payload));
               begin
                  if Parser_Result.Success then
                     declare
                        Val : constant JSON_Value := Parser_Result.Value;
                     begin
                        if Has_Field (Val, "name") then
                           Model_Name := To_Unbounded_String (String'(Get (Val, "name")));
                        elsif Has_Field (Val, "model") then
                           Model_Name := To_Unbounded_String (String'(Get (Val, "model")));
                        end if;
                     end;
                  end if;
               end;
            end if;

            declare
               Resp : constant JSON_Value := Create_Object;
               Details : constant JSON_Value := Create_Object;
               Families : JSON_Array := Empty_Array;
               Name_Str : constant String := To_String (Model_Name);
            begin
               if Name_Str = "adelaide-embedding" then
                  Append (Families, Create ("bert"));
                  Set_Field (Details, "family", "bert");
               else
                  Append (Families, Create ("qwen2"));
                  Set_Field (Details, "family", "qwen2");
               end if;
               Set_Field (Details, "families", Families);
               Set_Field (Resp, "details", Details);
               return Build_Response (Write (Resp));
            end;
         end;

      elsif URI = "/api/chat" or else URI = "/v1/chat/completions" then
         declare
            Payload : Unbounded_String := (if Raw_S'Length > 0 then To_Unbounded_String (Raw_S) else Raw_B);
            Val     : JSON_Value;
            Prompt  : Unbounded_String := To_Unbounded_String ("No payload");
            Images  : JSON_Array := Empty_Array;
            Result  : Unbounded_String;
            Resp    : constant JSON_Value := Create_Object;
            Req_Model : Unbounded_String := To_Unbounded_String ("adelaide-hybrid");
            Is_Streaming : Boolean := False;

            procedure Parse_Content_Array (C_Arr : JSON_Array) is
            begin
               Prompt := Null_Unbounded_String;
               for I in 1 .. Length (C_Arr) loop
                  declare
                     Item : constant JSON_Value := Get (C_Arr, I);
                  begin
                     if Has_Field (Item, "type") then
                        if Get (Item, "type") = "text" and then Has_Field (Item, "text") then
                           Append (Prompt, String'(Get (Item, "text")));
                        elsif Get (Item, "type") = "image_url" then
                           Append (Images, Get (Get (Item, "image_url"), "url"));
                        end if;
                     end if;
                  end;
               end loop;
            end Parse_Content_Array;
         begin
            if Length (Payload) > 0 then
               declare
                  Parser_Result : constant Read_Result := Read (To_String (Payload));
               begin
                  if Parser_Result.Success then
                     Val := Parser_Result.Value;
                     if Has_Field (Val, "model") then
                        Req_Model := To_Unbounded_String (String'(Get (Val, "model")));
                     end if;
                     if Has_Field (Val, "stream") then
                        Is_Streaming := Get (Val, "stream");
                     end if;
                     
                     if Has_Field (Val, "messages") then
                        declare
                           Msgs : constant JSON_Array := Get (Val, "messages");
                           Last : constant JSON_Value := Get (Msgs, Msgs.Length);
                        begin
                           if Has_Field (Last, "content") then
                              begin
                                 Prompt := To_Unbounded_String (String'(Get (Last, "content")));
                              exception
                                 when others =>
                                    Parse_Content_Array (Get (Last, "content"));
                              end;
                           end if;
                        end;
                     elsif Has_Field (Val, "prompt") then
                        Prompt := To_Unbounded_String (String'(Get (Val, "prompt")));
                     end if;
                  end if;
               end;
            end if;

            if Is_Streaming then
               declare
                  Q : constant Streaming_Queue.Queue_Access := new Streaming_Queue.Queue;
                  T : constant Generator_Task_Access := new Generator_Task;
                  S : constant Streaming_Queue.Response_Stream_Access := new Streaming_Queue.Response_Stream;
               begin
                  S.Q := Q;
                  T.Start (To_String (Prompt), To_String (Req_Model), 
                           (if Is_OpenAI then Streaming_Queue.OpenAI else Streaming_Queue.Ollama), Q);
                  declare
                     R : AWS.Response.Data := AWS.Response.Build
                       (Content_Type => (if Is_OpenAI then "text/event-stream" else "application/x-ndjson"),
                        Stream       => AWS.Resources.Streams.Stream_Type_Access (S));
                  begin
                     AWS.Response.Set.Add_Header (R, "Access-Control-Allow-Origin", "*");
                     return R;
                  end;
               end;
            else
               declare
                  use Ada.Real_Time;
                  T_Start  : constant Time := Clock;
                  Now      : constant Ada.Calendar.Time := Ada.Calendar.Clock;
                  TS_Str   : String := Ada.Calendar.Formatting.Image (Now);
               begin
                  if TS_Str'Length >= 11 then TS_Str (11) := 'T'; end if;
                  Model_Manager.Hybrid_Generate (To_String (Prompt), Result, Images, "web-api");
                  
                  declare
                     Msg_Out : constant JSON_Value := Create_Object;
                     Choice  : constant JSON_Value := Create_Object;
                     Choices : JSON_Array := Empty_Array;
                  begin
                     Set_Field (Msg_Out, "role", "assistant");
                     Set_Field (Msg_Out, "content", To_String (Result));
                     Set_Field (Choice, "message", Msg_Out);
                     Append (Choices, Choice);
                     Set_Field (Resp, "model", To_String (Req_Model));
                     Set_Field (Resp, "choices", Choices);
                     Set_Field (Resp, "message", Msg_Out);
                     Set_Field (Resp, "done", True);
                     Set_Field (Resp, "created", Long_Integer'(1686935002));
                     Set_Field (Resp, "created_at", TS_Str & "Z");
                     return Build_Response (Write (Resp));
                  end;
               end;
            end if;
         end;

      elsif URI = "/api/embeddings" or else URI = "/v1/embeddings" then
         declare
            Payload : Unbounded_String := (if Raw_S'Length > 0 then To_Unbounded_String (Raw_S) else Raw_B);
            Prompt  : Unbounded_String := Null_Unbounded_String;
            Resp    : constant JSON_Value := Create_Object;
         begin
            if Length (Payload) > 0 then
               declare
                  Parser_Result : constant Read_Result := Read (To_String (Payload));
               begin
                  if Parser_Result.Success then
                     declare
                        Val : constant JSON_Value := Parser_Result.Value;
                     begin
                        if Has_Field (Val, "prompt") then
                           Prompt := To_Unbounded_String (String'(Get (Val, "prompt")));
                        elsif Has_Field (Val, "input") then
                           begin
                              Prompt := To_Unbounded_String (String'(Get (Val, "input")));
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
               Emb_Arr : JSON_Array := Empty_Array;
            begin
               Model_Manager.Get_Embedding (To_String (Prompt), Vec, Len);
               for I in 1 .. Len loop
                  Append (Emb_Arr, Create (Long_Float (Vec (I))));
               end loop;

               if URI = "/api/embeddings" then
                  Set_Field (Resp, "embedding", Emb_Arr);
               else
                  declare
                     Data_Arr  : JSON_Array := Empty_Array;
                     Data_Obj  : constant JSON_Value := Create_Object;
                  begin
                     Set_Field (Data_Obj, "object", "embedding");
                     Set_Field (Data_Obj, "index", Integer'(0));
                     Set_Field (Data_Obj, "embedding", Emb_Arr);
                     Append (Data_Arr, Data_Obj);
                     Set_Field (Resp, "object", "list");
                     Set_Field (Resp, "data", Data_Arr);
                     Set_Field (Resp, "model", "adelaide-embedding");
                  end;
               end if;
               return Build_Response (Write (Resp));
            end;
         end;
      else
         return Build_Response ("Adelaide API Endpoint", AWS.Messages.S404, "text/plain");
      end if;
   exception
      when E : others =>
         Put_Line ("[Server] Error: " & Ada.Exceptions.Exception_Message (E));
         return Build_Response ("{}", AWS.Messages.S500);
   end Dispatch;

end Adelaide_Server_Pkg;
