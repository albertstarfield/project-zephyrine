with Ada.Text_IO;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Ada.Exceptions;
with AWS.Client;
with AWS.Headers;
with AWS.Messages;
with AWS.Response.Set;
with AWS.Resources.Streams;
with GNATCOLL.JSON;
with Math_Utils;
with Model_Manager; use Model_Manager;
with Database_Manager;
with Ada.Strings.Fixed;
with Ada.Calendar; use Ada.Calendar;
with Streaming_Queue;

package body Adelaide_Server_Pkg is

   OLLAMA_PORT : constant String := "11435";
   OLLAMA_URL  : constant String := "http://localhost:" & OLLAMA_PORT;

   subtype ID_Type is String (1 .. 64);
   type Entry_Rec is record
      ID  : ID_Type;
      Len : Natural;
      Q   : Streaming_Queue.Queue_Access;
   end record;
   type Map_Type is array (1 .. 100) of Entry_Rec;

   --  Registry to track active streaming queues by Session ID
   --  for cross-component log streaming (e.g. from Python).
   protected Stream_Registry is
      procedure Register (ID : String; Q : Streaming_Queue.Queue_Access);
      procedure Unregister (ID : String);
      procedure Push_Log (ID : String; Log : String);
   private
      Map   : Map_Type;
      Count : Natural := 0;
   end Stream_Registry;

   protected body Stream_Registry is
      procedure Register (ID : String; Q : Streaming_Queue.Queue_Access) is
         S_ID : ID_Type := (others => ' ');
      begin
         if Count < 100 then
            Count := Count + 1;
            if ID'Length <= 64 then
               S_ID (1 .. ID'Length) := ID;
            else
               S_ID := ID (ID'First .. ID'First + 63);
            end if;
            Map (Count).ID  := S_ID;
            Map (Count).Len := (if ID'Length > 64 then 64 else ID'Length);
            Map (Count).Q   := Q;
         end if;
      end Register;

      procedure Unregister (ID : String) is
      begin
         for I in 1 .. Count loop
            if Map (I).ID (1 .. Map (I).Len) = ID then
               Map (I .. Count - 1) := Map (I + 1 .. Count);
               Count := Count - 1;
               return;
            end if;
         end loop;
      end Unregister;

      procedure Push_Log (ID : String; Log : String) is
      begin
         for I in 1 .. Count loop
            if Map (I).ID (1 .. Map (I).Len) = ID then
               --  Push to the specific queue
               Model_Manager.Push_Chunk (Map (I).Q, ID, Log);
               return;
            end if;
         end loop;
         --  Fallback: if no specific session, log to console
         Ada.Text_IO.Put_Line (Ada.Text_IO.Standard_Error, "[Orchestrator Log] " & ID & ": " & Log);
         Ada.Text_IO.Flush (Ada.Text_IO.Standard_Error);
      end Push_Log;
   end Stream_Registry;

   --  Generator Task
   task type Generator_Task is
      entry Start
        (Stream_Ptr     : Streaming_Queue.Queue_Access;
         Prompt_Val     : String;
         Images_Val     : GNATCOLL.JSON.JSON_Array;
         Session_ID_Val : String;
         URI_Str_Val    : String;
         Start_Time_Val : Ada.Calendar.Time;
         Level_Val      : Model_Manager.ELP_Level := Model_Manager.ELP1);
   end Generator_Task;

   type Generator_Task_Access is access Generator_Task;

   task body Generator_Task is
      Stream     : Streaming_Queue.Queue_Access;
      Prompt     : Unbounded_String;
      Images     : GNATCOLL.JSON.JSON_Array;
      Session_ID : Unbounded_String;
      Level      : Model_Manager.ELP_Level;
   begin
      accept Start
        (Stream_Ptr     : Streaming_Queue.Queue_Access;
         Prompt_Val     : String;
         Images_Val     : GNATCOLL.JSON.JSON_Array;
         Session_ID_Val : String;
         URI_Str_Val    : String;
         Start_Time_Val : Ada.Calendar.Time;
         Level_Val      : Model_Manager.ELP_Level := Model_Manager.ELP1) do
         Stream     := Stream_Ptr;
         Prompt     := To_Unbounded_String (Prompt_Val);
         Images     := Images_Val;
         Session_ID := To_Unbounded_String (Session_ID_Val);
         Level      := Level_Val;
      end Start;

      declare
         Res : Unbounded_String;
      begin
         Model_Manager.Hybrid_Generate
             (To_String (Prompt), Res, Images, To_String (Session_ID), Stream, Level);
         Stream_Registry.Unregister (To_String (Session_ID));
      exception
         when E : others =>
            Ada.Text_IO.Put_Line (Ada.Text_IO.Standard_Error, "[FATAL] Generator Task Error: " & Ada.Exceptions.Exception_Message (E));
            Stream_Registry.Unregister (To_String (Session_ID));
      end;
   end Generator_Task;

   --  Extract prompt string from request body (Universal format)
   function Extract_Prompt (Body_Str : String) return String is
      use GNATCOLL.JSON;
      Res : constant Read_Result := Read (Body_Str);
   begin
      if not Res.Success then return ""; end if;
      declare
         Val : constant JSON_Value := Res.Value;
      begin
         if Val.Kind /= JSON_Object_Type then return ""; end if;
         if Has_Field (Val, "prompt") then
            return Get (Get (Val, "prompt"));
         elsif Has_Field (Val, "messages") then
            declare
               Arr : constant JSON_Array := Get (Get (Val, "messages"));
               Len : constant Natural := Length (Arr);
            begin
               if Len > 0 then
                  declare
                     Last_Msg : constant JSON_Value := Get (Arr, Len);
                  begin
                     if Has_Field (Last_Msg, "content") then
                        declare
                           Cont : constant JSON_Value := Get (Last_Msg, "content");
                        begin
                           if Cont.Kind = JSON_String_Type then
                              return Get (Cont);
                           elsif Cont.Kind = JSON_Array_Type then
                              declare
                                 C_Arr : constant JSON_Array := Get (Cont);
                                 P_Acc : Unbounded_String;
                              begin
                                 for I in 1 .. Length (C_Arr) loop
                                    declare
                                       Item : constant JSON_Value := Get (C_Arr, I);
                                    begin
                                       if Get (Item, "type") = "text" then
                                          Append (P_Acc, String'(Get (Get (Item, "text"))));
                                       end if;
                                    end;
                                 end loop;
                                 return To_String (P_Acc);
                              end;
                           end if;
                        end;
                     end if;
                  end;
               end if;
            end;
         end if;
      end;
      return "";
   exception
      when others => return "";
   end Extract_Prompt;

   function Extract_Images (Body_Str : String) return GNATCOLL.JSON.JSON_Array is
      use GNATCOLL.JSON;
      Res : constant Read_Result := Read (Body_Str);
      Empty : JSON_Array := Empty_Array;
   begin
      if not Res.Success then return Empty; end if;
      declare
         Val : constant JSON_Value := Res.Value;
      begin
         if Has_Field (Val, "images") then
            return Get (Get (Val, "images"));
         elsif Has_Field (Val, "messages") then
            declare
               Arr : constant JSON_Array := Get (Get (Val, "messages"));
               Len : constant Natural := Length (Arr);
            begin
               if Len > 0 then
                  declare
                     Last_Msg : constant JSON_Value := Get (Arr, Len);
                  begin
                     if Has_Field (Last_Msg, "content") and then Get (Last_Msg, "content").Kind = JSON_Array_Type then
                        declare
                           C_Arr : constant JSON_Array := Get (Get (Last_Msg, "content"));
                           Res_Arr : JSON_Array := Empty_Array;
                        begin
                           for I in 1 .. Length (C_Arr) loop
                              declare
                                 Item : constant JSON_Value := Get (C_Arr, I);
                              begin
                                 if Get (Item, "type") = "image_url" then
                                    Append (Res_Arr, Get (Get (Item, "image_url"), "url"));
                                 end if;
                              end;
                           end loop;
                           return Res_Arr;
                        end;
                     end if;
                  end;
               end if;
            end;
         end if;
      end;
      return Empty;
   exception
      when others => return Empty;
   end Extract_Images;

   function Extract_Stream (Body_Str : String) return Boolean is
      use GNATCOLL.JSON;
      Res : constant Read_Result := Read (Body_Str);
   begin
      if not Res.Success then return False; end if;
      if Has_Field (Res.Value, "stream") then
         return Get (Get (Res.Value, "stream"));
      end if;
      return False;
   exception
      when others => return False;
   end Extract_Stream;

   function Format_Universal_Response
     (URI_Str : String;
      Text    : String;
      Similarity : Float := 0.0;
      Duration_Ns : Long_Integer := 0) return String
   is
      use GNATCOLL.JSON;
      Res_Obj : constant JSON_Value := Create_Object;
   begin
      Set_Field (Res_Obj, "model", String'("adelaide-hybrid"));
      Set_Field (Res_Obj, "done", True);
      if URI_Str = "/api/chat" or else URI_Str = "/v1/chat/completions" then
         declare
            Msg_Obj : constant JSON_Value := Create_Object;
         begin
            Set_Field (Msg_Obj, "role", String'("assistant"));
            Set_Field (Msg_Obj, "content", Text);
            if URI_Str = "/v1/chat/completions" then
               declare
                  Choice_Arr : JSON_Array := Empty_Array;
                  Choice_Obj : constant JSON_Value := Create_Object;
               begin
                  Set_Field (Choice_Obj, "message", Msg_Obj);
                  Append (Choice_Arr, Choice_Obj);
                  Set_Field (Res_Obj, "choices", Choice_Arr);
               end;
            else
               Set_Field (Res_Obj, "message", Msg_Obj);
            end if;
         end;
      else
         Set_Field (Res_Obj, "response", Text);
      end if;
      return Write (Res_Obj);
   end Format_Universal_Response;

   function Dispatch (Request : AWS.Status.Data) return AWS.Response.Data is
      use AWS.Status;
      URI_Str : constant String := URI (Request);
      Method_Val : constant Request_Method := Method (Request);
      Start_Time : constant Ada.Calendar.Time := Ada.Calendar.Clock;

      procedure Set_CORS (Resp : in out AWS.Response.Data) is
      begin
         AWS.Response.Set.Add_Header (Resp, "Access-Control-Allow-Origin", "*");
         AWS.Response.Set.Add_Header (Resp, "Access-Control-Allow-Headers", "Content-Type, Authorization, Session-ID");
      end Set_CORS;
   begin
      begin
         Ada.Text_IO.Put_Line (Ada.Text_IO.Standard_Error, "[Request] " & Method_Val'Img & " " & URI_Str);
         Ada.Text_IO.Flush (Ada.Text_IO.Standard_Error);

         if Method_Val = OPTIONS then
            declare
               Resp : AWS.Response.Data := AWS.Response.Acknowledge (AWS.Messages.S200);
            begin
               Set_CORS (Resp);
               return Resp;
            end;
         end if;

         if Method_Val = GET then
            if URI_Str = "/v1/models" or else URI_Str = "/api/tags" then
               declare
                  use GNATCOLL.JSON;
                  Res_Obj    : constant JSON_Value := Create_Object;
                  Models_Arr : JSON_Array := Empty_Array;
                  M : constant JSON_Value := Create_Object;
                  D : constant JSON_Value := Create_Object;
               begin
                  Set_Field (M, "name", String'("adelaide-hybrid"));
                  Set_Field (M, "id", String'("adelaide-hybrid"));
                  Set_Field (D, "format", String'("gguf"));
                  Set_Field (D, "family", String'("qwen"));
                  --  METAMODEL ADVERTISEMENT
                  Set_Field (D, "context_length", Create (Long_Long_Integer (9_223_372_036_854_775_807)));
                  Set_Field (D, "embedding_length", Create (Long_Long_Integer (4_294_967_295)));
                  Set_Field (M, "details", D);
                  Append (Models_Arr, M);
                  if URI_Str = "/v1/models" then Set_Field (Res_Obj, "data", Models_Arr); else Set_Field (Res_Obj, "models", Models_Arr); end if;
                  declare
                     Resp : AWS.Response.Data := AWS.Response.Build (Content_Type => "application/json", Message_Body => Write (Res_Obj));
                  begin
                     Set_CORS (Resp);
                     return Resp;
                  end;
               end;
            elsif URI_Str = "/api/version" then
               return AWS.Response.Build (Content_Type => "application/json", Message_Body => "{""version"":""0.1.48""}");
            end if;
         end if;

         if Method_Val = POST then
            declare
               Body_Str : constant String := To_String (Binary_Data (Request));
               Prompt_Raw : constant String := Extract_Prompt (Body_Str);
               Prompt : Unbounded_String := To_Unbounded_String (Prompt_Raw);
               Images : constant GNATCOLL.JSON.JSON_Array := Extract_Images (Body_Str);
               Session_H : constant String := AWS.Headers.Get (AWS.Status.Header (Request), "Session-ID");
               Session_ID : constant String := (if Session_H /= "" then Session_H else AWS.Status.Peername (Request));
            begin
               if URI_Str = "/api/chat" or else URI_Str = "/v1/chat/completions" or else URI_Str = "/api/generate" then
                  if Prompt_Raw /= "" then
                     declare
                        use Database_Manager;
                        V : Math_Utils.Vector (1 .. 16384);
                        V_Len : Natural;
                        Results : Chunk_Array (1 .. 3);
                        N_Res : Natural;
                        RAG_Context : Unbounded_String;
                     begin
                        Model_Manager.Get_Embedding (Prompt_Raw, V, V_Len);
                        if V_Len > 0 then
                           Search_Literature (V (1 .. V_Len), Results, N_Res);
                           if N_Res > 0 then
                              Append (RAG_Context, "[LOCAL LITERATURE CONTEXT]" & ASCII.LF);
                              for J in 1 .. N_Res loop
                                 Append (RAG_Context, "Source: " & To_String (Results (J).File_Path) & ASCII.LF);
                                 Append (RAG_Context, To_String (Results (J).Content) & ASCII.LF & "---" & ASCII.LF);
                              end loop;
                              Prompt := RAG_Context & Prompt;
                           end if;
                        end if;
                     end;

                     if Extract_Stream (Body_Str) then
                        declare
                           Q : constant Streaming_Queue.Queue_Access := new Streaming_Queue.Queue;
                           T : constant Generator_Task_Access := new Generator_Task;
                           RS : constant Streaming_Queue.Response_Stream_Access := 
                             new Streaming_Queue.Response_Stream'(AWS.Resources.Streams.Stream_Type with Q => Q);
                        begin
                           Stream_Registry.Register (Session_ID, Q);
                           T.Start (Q, To_String (Prompt), Images, Session_ID, URI_Str, Start_Time, Model_Manager.ELP1);
                           return AWS.Response.Stream (Content_Type => "text/event-stream", Handle => RS);
                        end;
                     else
                        declare
                           Gen_Res : Unbounded_String;
                        begin
                           Model_Manager.Hybrid_Generate (To_String (Prompt), Gen_Res, Images, Session_ID, null, Model_Manager.ELP1);
                           return AWS.Response.Build (Content_Type => "application/json", Message_Body => Format_Universal_Response (URI_Str, To_String (Gen_Res)));
                        end;
                     end if;
                  end if;
               elsif URI_Str = "/api/embeddings" or else URI_Str = "/v1/embeddings" then
                  declare
                     use GNATCOLL.JSON;
                     V : Math_Utils.Vector (1 .. 16384);
                     V_Len : Natural;
                     Res_Obj : constant JSON_Value := Create_Object;
                     Arr : JSON_Array := Empty_Array;
                  begin
                     Model_Manager.Get_Embedding (Prompt_Raw, V, V_Len);
                     for I in 1 .. V_Len loop Append (Arr, Create (V (I))); end loop;
                     Set_Field (Res_Obj, "embedding", Arr);
                     return AWS.Response.Build (Content_Type => "application/json", Message_Body => Write (Res_Obj));
                  end;
               end if;
            end;
         end if;
         return AWS.Response.Acknowledge (AWS.Messages.S404);
      exception
         when E : others =>
            Ada.Text_IO.Put_Line (Ada.Text_IO.Standard_Error, "[FATAL] Dispatch Exception: " & Ada.Exceptions.Exception_Message (E));
            return AWS.Response.Build (Content_Type => "application/json", Message_Body => "{""error"":""Internal Error""}", Status_Code => AWS.Messages.S500);
      end;
   end Dispatch;

end Adelaide_Server_Pkg;
