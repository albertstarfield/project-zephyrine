with AWS.Status;
with AWS.Response;
with AWS.Messages;
with GNATCOLL.JSON; use GNATCOLL.JSON;
with Ada.Text_IO; use Ada.Text_IO;
with Model_Manager;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Streaming_Queue;
with Ada.Exceptions;

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
            Resp   : constant JSON_Value := Create_Object;
            Models : JSON_Array := Empty_Array;
            M1     : constant JSON_Value := Create_Object;
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
            --  Try Binary_Data if Payload is empty
            Payload : constant String := 
              (if AWS.Status.Payload (Request) /= "" 
               then AWS.Status.Payload (Request)
               else To_String (AWS.Status.Binary_Data (Request)));
            Val     : JSON_Value;
            Prompt  : Unbounded_String := To_Unbounded_String ("No payload");
            Images  : JSON_Array := Empty_Array;
            Result  : Unbounded_String;
            Resp    : constant JSON_Value := Create_Object;
            Choices : JSON_Array := Empty_Array;
            Choice  : constant JSON_Value := Create_Object;
            Msg_Out : constant JSON_Value := Create_Object;
         begin
            Put_Line ("[Server] Resolved Payload length: " & Payload'Length'Image);
            
            if Payload /= "" then
               Val := Read (Payload);
               if Val.Has_Field ("messages") then
                  declare
                     Msgs : constant JSON_Array := Get (Val, "messages");
                     Last : constant JSON_Value := Get (Msgs, Msgs.Length);
                  begin
                     Prompt := To_Unbounded_String
                       (String'(Get (Last, "content")));
                     if Last.Has_Field ("images") then
                        Images := Get (Last, "images");
                     end if;
                  end;
               elsif Val.Has_Field ("prompt") then
                  Prompt := To_Unbounded_String (String'(Get (Val, "prompt")));
               end if;
            end if;

            Model_Manager.Hybrid_Generate
              (Prompt     => To_String (Prompt),
               Result     => Result,
               Images     => Images,
               Session_ID => "web-api");

            Set_Field (Msg_Out, "role", "assistant");
            Set_Field (Msg_Out, "content", To_String (Result));
            Set_Field (Choice, "message", Msg_Out);
            Append (Choices, Choice);
            Set_Field (Resp, "choices", Choices);
            Set_Field (Resp, "message", Msg_Out);

            return AWS.Response.Build
              (Content_Type => "application/json",
               Message_Body => Write (Resp));
         end;
      elsif URI = "/api/embeddings" or else URI = "/v1/embeddings" then
         declare
            Resp : constant JSON_Value := Create_Object;
            Embeddings : JSON_Array := Empty_Array;
         begin
            Append (Embeddings, Create (Long_Float (0.1)));
            Append (Embeddings, Create (Long_Float (0.2)));
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
      when E : others =>
         Put_Line ("[Server] Error: " & Ada.Exceptions.Exception_Message (E));
         return AWS.Response.Build
           (Content_Type => "application/json",
            Message_Body => "{}",
            Status_Code  => AWS.Messages.S500);
   end Dispatch;

end Adelaide_Server_Pkg;
