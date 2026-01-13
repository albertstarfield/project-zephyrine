pragma Ada_2022;

with Ada.Text_IO;
with AWS.Net.WebSocket;
with Connection_Manager;

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
      Ada.Text_IO.Put_Line ("[WS] Client Connected: " & User_ID);
      
      -- CRITICAL STABILITY FIX:
      -- We do NOT send a message here. We wait for the client to speak first.
      -- We do NOT store the socket yet (to avoid copy-destructor issues).
      
   exception
      when E : others =>
         Ada.Text_IO.Put_Line ("[WS] CRASH in On_Open: " & E'Image);
   end On_Open;

   ----------------
   -- On_Message --
   ----------------
   overriding procedure On_Message (Socket : in out Handler;
                                    Message : String) is
   begin
      Ada.Text_IO.Put_Line ("[WS] Received: " & Message);

      -- Echo back safely
      Socket.Send ("Echo: " & Message);

   exception
      when E : others =>
         Ada.Text_IO.Put_Line ("[WS] CRASH in On_Message: " & E'Image);
   end On_Message;

   --------------
   -- On_Close --
   --------------
   overriding procedure On_Close (Socket : in out Handler;
                                  Message : String) is
   begin
      Ada.Text_IO.Put_Line ("[WS] Client Disconnected: " & Message);
   end On_Close;

   --------------
   -- On_Error --
   --------------
   overriding procedure On_Error (Socket : in out Handler;
                                  Error  : String) is
   begin
      Ada.Text_IO.Put_Line ("[WS] Error Callback: " & Error);
   end On_Error;

end WS_Handler;