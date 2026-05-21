with AWS.Status;
with AWS.Response;
with AWS.Messages;
with GNATCOLL.JSON; use GNATCOLL.JSON;
with Ada.Text_IO; use Ada.Text_IO;
with Model_Manager;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Streaming_Queue;

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

   function Dispatch (Request : AWS.Status.Data) return AWS.Response.Data is
      URI : constant String := AWS.Status.URI (Request);
      Method : constant String := AWS.Status.Method (Request);
   begin
      Put_Line ("[Server] " & Method & " " & URI);

      if URI = "/v1/models" or else URI = "/api/tags" then
         declare
            Resp   : JSON_Value := Create_Object;
            Models : JSON_Array := Empty_Array;
            M1     : JSON_Value := Create_Object;
         begin
            Set_Field (M1, "id", "adelaide-hybrid");
            Set_Field (M1, "name", "adelaide-hybrid");
            Append (Models, M1);
            Set_Field (Resp, "data", Models);
            Set_Field (Resp, "models", Models);
            return AWS.Response.Build
              (Content_Type => "application/json",
               Message_Body => Write (Resp));
         end;
      elsif URI = "/api/chat" or else URI = "/v1/chat/completions" then
         declare
            Result : Unbounded_String;
            Resp : JSON_Value := Create_Object;
            Choices : JSON_Array := Empty_Array;
            Choice : JSON_Value := Create_Object;
            Msg : JSON_Value := Create_Object;
         begin
            Model_Manager.Hybrid_Generate
              (Prompt     => "User request",
               Result     => Result,
               Session_ID => "web-api");

            Set_Field (Msg, "role", "assistant");
            Set_Field (Msg, "content", To_String (Result));
            Set_Field (Choice, "message", Msg);
            Append (Choices, Choice);
            Set_Field (Resp, "choices", Choices);
            Set_Field (Resp, "message", Msg);

            return AWS.Response.Build
              (Content_Type => "application/json",
               Message_Body => Write (Resp));
         end;
      elsif URI = "/api/embeddings" or else URI = "/v1/embeddings" then
         declare
            Resp : JSON_Value := Create_Object;
            Embeddings : JSON_Array := Empty_Array;
         begin
            Append (Embeddings, Create (0.1));
            Append (Embeddings, Create (0.2));
            Set_Field (Resp, "embedding", Embeddings);
            return AWS.Response.Build
              (Content_Type => "application/json",
               Message_Body => Write (Resp));
         end;
      else
         return AWS.Response.Build
           (Content_Type => "text/plain",
            Message_Body => "Adelaide API Endpoint",
            Status_Code  => AWS.Messages.S404);
      end if;
   exception
      when others =>
         return AWS.Response.Build
           (Content_Type => "application/json",
            Message_Body => "{}",
            Status_Code  => AWS.Messages.S500);
   end Dispatch;

end Adelaide_Server_Pkg;
