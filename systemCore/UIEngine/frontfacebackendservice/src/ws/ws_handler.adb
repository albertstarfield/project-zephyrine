pragma Ada_2022;

with Ada.Text_IO;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with GNATCOLL.JSON;         use GNATCOLL.JSON;
with AWS.Net.WebSocket;
with Connection_Manager;
with Cognitive_Lang_Resp;

package body WS_Handler is

   pragma Suppress (Accessibility_Check);

   ------------
   -- Create --
   ------------
   function Create (Socket : AWS.Net.Socket_Access;
                    Request : AWS.Status.Data) return AWS.Net.WebSocket.Object'Class is
   begin
      Ada.Text_IO.Put_Line ("[WS] Factory: Creating new WebSocket Object...");
      return Handler'
        (AWS.Net.WebSocket.Object (AWS.Net.WebSocket.Create (Socket, Request)) 
         with null record);
   end Create;

   -------------
   -- On_Open --
   -------------
   overriding procedure On_Open (Socket : in out Handler; Message : String) is
      pragma Unreferenced (Message);
      User_ID : constant String := "debug-user-01";
   begin
      Ada.Text_IO.Put_Line ("[WS][Status] Client Connected: " & User_ID);
      Connection_Manager.Store.Add (User_ID, AWS.Net.WebSocket.Object (Socket));
   exception
      when E : others =>
         Ada.Text_IO.Put_Line ("[WS][Error] CRASH in On_Open: " & E'Image);
   end On_Open;

   ----------------
   -- On_Message --
   ----------------
   overriding procedure On_Message (Socket : in out Handler; Message : String) is
      JSON_Req     : JSON_Value;
      Prompt_Str   : Unbounded_String;
      System_Resp  : Unbounded_String;
      JSON_Res     : JSON_Value := Create_Object;
      Outgoing_Str : Unbounded_String;
   begin
      Ada.Text_IO.Put_Line ("[WS][RX] Received: " & Message);

      begin
         JSON_Req := Read (Message);
         
         if JSON_Req.Has_Field ("prompt") then
            -- FIX: Explicit typing to resolve GNATCOLL.JSON ambiguity
            declare
               -- Force the compiler to choose the String' Get interpretation
               Raw_Prompt : constant String := JSON_Req.Get ("prompt");
            begin
               Prompt_Str := To_Unbounded_String (Raw_Prompt);
               
               Ada.Text_IO.Put_Line ("[WS][Proc] Calling AI with: " & To_String (Prompt_Str));

               System_Resp := To_Unbounded_String
                 (Cognitive_Lang_Resp.Process_Input
                    (Input_Sequence => To_String (Prompt_Str),
                     Model_ID       => "Snowball-Enaga"));
               
               JSON_Res.Set_Field ("reply", To_String (System_Resp));
               JSON_Res.Set_Field ("type", "chat_update");
               
               Outgoing_Str := To_Unbounded_String (Write (JSON_Res));

               Ada.Text_IO.Put_Line ("[WS][TX] Sending: " & To_String (Outgoing_Str));

               AWS.Net.WebSocket.Send 
                 (AWS.Net.WebSocket.Object (Socket), 
                  To_String (Outgoing_Str));
            end;
         else
            Ada.Text_IO.Put_Line ("[WS][Warn] JSON missing 'prompt' field");
         end if;

      exception
         when E : others =>
            Ada.Text_IO.Put_Line ("[WS][Error] Exception in On_Message: " & E'Image);
            AWS.Net.WebSocket.Send 
              (AWS.Net.WebSocket.Object (Socket), 
               "{""error"": ""Processing Failed""}");
      end;
   end On_Message;

   --------------
   -- On_Close --
   --------------
   overriding procedure On_Close (Socket : in out Handler; Message : String) is
      pragma Unreferenced (Socket);
      User_ID : constant String := "debug-user-01";
   begin
      Ada.Text_IO.Put_Line ("[WS][Status] Disconnected: " & Message);
      Connection_Manager.Store.Remove (User_ID);
   end On_Close;

   --------------
   -- On_Error --
   --------------
   overriding procedure On_Error (Socket : in out Handler; Error : String) is
      pragma Unreferenced (Socket);
   begin
      Ada.Text_IO.Put_Line ("[WS][Error] Callback: " & Error);
   end On_Error;

end WS_Handler;