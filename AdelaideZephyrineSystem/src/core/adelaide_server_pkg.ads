pragma SPARK_Mode (Off);
-- thread: AWS HTTP client requires task protection
with AWS.Response;
with AWS.Status;
with Streaming_Queue;

package Adelaide_Server_Pkg is

   --  The main dispatch callback for Adelaide AWS Proxy Server
   function Dispatch (Request : AWS.Status.Data) return AWS.Response.Data with Pre => True, Post => True;

   --  Session Management for cross-component logging
   procedure Register (ID : String; Q : Streaming_Queue.Queue_Access) with Pre => True, Post => True;
   procedure Unregister (ID : String) with Pre => True, Post => True;
   procedure Push_Log (ID : String; Log : String) with Pre => True, Post => True;

   --  Last API endpoint tracker (thread-safe for heartbeat)
   procedure Set_Last_API (URI : String) with Pre => True, Post => True;
   function Get_Last_API return String with Pre => True, Post => True;

end Adelaide_Server_Pkg;
