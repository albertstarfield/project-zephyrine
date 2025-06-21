-- Implements the logic for our WebSocket connections.
with Ada.Text_IO;

package body WebSocket_Handler is

   -- Our custom WebSocket object to hold connection-specific state.
   type Zephyrine_Socket is new AWS.Net.WebSocket.Object with null record;
      -- TODO: Add fields for User_ID, Chat_ID, etc.

   -- Overridden callbacks for our custom socket type.
   overriding procedure On_Open
     (Socket : in out Zephyrine_Socket; Message : String);
   overriding procedure On_Message
     (Socket : in out Zephyrine_Socket; Message : String);
   overriding procedure On_Close
     (Socket : in out Zephyrine_Socket; Message : String);
   overriding procedure On_Error
     (Socket : in out Zephyrine_Socket; Message : String);

   -- Implementation of the factory function
   function Create
     (Socket  : AWS.Net.WebSocket.Socket_Access;
      Request : AWS.Status.Data) return AWS.Net.WebSocket.Object'Class
   is
      New_Socket : constant Zephyrine_Socket'Class :=
        (AWS.Net.WebSocket.Object (AWS.Net.WebSocket.Create (Socket, Request)) with null record);
   begin
      Ada.Text_IO.Put_Line ("WebSocket: Factory creating new connection.");
      return New_Socket;
   end Create;

   -- Implementation of the callbacks
   procedure On_Open (Socket : in out Zephyrine_Socket; Message : String) is
   begin
      Ada.Text_IO.Put_Line ("WebSocket: Connection opened.");
      -- Example of sending a welcome message
      Socket.Send ("{""type"": ""hello"", ""message"": ""Welcome to Zephyrine backend (Ada)""}");
   end On_Open;

   procedure On_Message (Socket : in out Zephyrine_Socket; Message : String) is
   begin
      Ada.Text_IO.Put_Line ("WebSocket: Received message: " & Message);
      -- TODO: Implement the main message handling logic here.
      -- 1. Parse 'Message' string with GNATCOLL.JSON
      -- 2. Use a 'case' statement on the message 'type'
      -- 3. Call DB functions, OpenAI client, etc.
      -- 4. Use Socket.Send() to send back chunks or full responses.
      Socket.Send ("{""type"": ""ack"", ""received"": " & Message & "}");
   end On_Message;

   procedure On_Close (Socket : in out Zephyrine_Socket; Message : String) is
   begin
      Ada.Text_IO.Put_Line ("WebSocket: Connection closed.");
   end On_Close;

   procedure On_Error (Socket : in out Zephyrine_Socket; Message : String) is
   begin
      Ada.Text_IO.Put_Line ("WebSocket: Error: " & Message);
   end On_Error;

end WebSocket_Handler;