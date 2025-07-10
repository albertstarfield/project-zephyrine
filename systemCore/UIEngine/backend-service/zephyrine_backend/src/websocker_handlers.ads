-- Declares our WebSocket handler based on the AWS WebSocket documentation.
with AWS.Net.WebSocket;
with AWS.Status;

package WebSocket_Handler is

   -- This "factory" function is called by AWS when a client connects to /ws
   function Create
     (Socket  : AWS.Net.WebSocket.Socket_Access;
      Request : AWS.Status.Data) return AWS.Net.WebSocket.Object'Class;

end WebSocket_Handler;