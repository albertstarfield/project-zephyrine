pragma SPARK_Mode (Off);
with AWS.Response;
with AWS.Status;
with Streaming_Queue;

package Adelaide_Server_Pkg is

   --  The main dispatch callback for Adelaide AWS Proxy Server
   function Dispatch (Request : AWS.Status.Data) return AWS.Response.Data;

   --  Session Management for cross-component logging
   procedure Register (ID : String; Q : Streaming_Queue.Queue_Access);
   procedure Unregister (ID : String);
   procedure Push_Log (ID : String; Log : String);

end Adelaide_Server_Pkg;
