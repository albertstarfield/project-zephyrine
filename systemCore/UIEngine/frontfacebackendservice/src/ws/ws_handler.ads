with AWS.Net.WebSocket;
with AWS.Status;
with AWS.Net;

package WS_Handler is
   pragma Elaborate_Body;

   type Handler is new AWS.Net.WebSocket.Object with null record;

   function Create (Socket : AWS.Net.Socket_Access;
                    Request : AWS.Status.Data) return AWS.Net.WebSocket.Object'Class;

   -- STRICT CONFORMANCE: All parameters named 'Message'
   overriding procedure On_Open (Socket : in out Handler; 
                                 Message : String);

   overriding procedure On_Message (Socket : in out Handler; 
                                    Message : String); -- FIX: Was 'Data'

   overriding procedure On_Close (Socket : in out Handler;
                                  Message : String);

   overriding procedure On_Error (Socket : in out Handler; 
                                  Error   : String);

end WS_Handler;