with AWS.Status;
with AWS.Response;
with GNATCOLL.JSON; use GNATCOLL.JSON;
with Ada.Text_IO; use Ada.Text_IO;
with Model_Manager;
with Streaming_Queue;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Ada.Calendar;
with Ada.Real_Time;

package body Adelaide_Server_Pkg is

   use Model_Manager;

   function Dispatch (Request : in AWS.Status.Data) return AWS.Response.Data is
      URI : constant String := AWS.Status.URI (Request);
      Method : constant String := AWS.Status.Method (Request);
   begin
      Put_Line ("[Server] " & Method & " " & URI);

      if URI = "/v1/models" or else URI = "/api/tags" then
         declare
            Resp : JSON_Value := Create_Object;
            Models : JSON_Array := Empty_Array;
            M1 : JSON_Value := Create_Object;
         begin
            Set_Field (M1, "id", "adelaide-hybrid");
            Set_Field (M1, "name", "adelaide-hybrid");
            Append (Models, M1);
            Set_Field (Resp, "data", Models);
            Set_Field (Resp, "models", Models);
            return AWS.Response.Build
              ("application/json", To_JSON (Resp));
         end;
      elsif URI = "/api/chat" or else URI = "/v1/chat/completions" then
         declare
            Payload : constant String := AWS.Status.Payload (Request);
            Val : JSON_Value := Read (Payload);
            Prompt : Unbounded_String;
            Result : Unbounded_String;
            Resp : JSON_Value := Create_Object;
            Choices : JSON_Array := Empty_Array;
            Choice : JSON_Value := Create_Object;
            Msg : JSON_Value := Create_Object;
         begin
            Prompt := To_Unbounded_String ("User request");
            Model_Manager.Hybrid_Generate
              (Prompt     => To_String (Prompt),
               Result     => Result,
               Session_ID => "web-api");
            
            Set_Field (Msg, "role", "assistant");
            Set_Field (Msg, "content", To_String (Result));
            Set_Field (Choice, "message", Msg);
            Append (Choices, Choice);
            Set_Field (Resp, "choices", Choices);
            Set_Field (Resp, "message", Msg); -- Ollama style
            
            return AWS.Response.Build
              ("application/json", To_JSON (Resp));
         end;
      elsif URI = "/api/embeddings" or else URI = "/v1/embeddings" then
         declare
            Resp : JSON_Value := Create_Object;
            Embeddings : JSON_Array := Empty_Array;
         begin
            Append (Embeddings, 0.1);
            Append (Embeddings, 0.2);
            Set_Field (Resp, "embedding", Embeddings);
            return AWS.Response.Build
              ("application/json", To_JSON (Resp));
         end;
      else
         return AWS.Response.Build
           ("text/plain", "Adelaide API Endpoint", AWS.Messages.S404);
      end if;
   exception
      when others =>
         return AWS.Response.Build
           ("application/json", "{}", AWS.Messages.S500);
   end Dispatch;

end Adelaide_Server_Pkg;
