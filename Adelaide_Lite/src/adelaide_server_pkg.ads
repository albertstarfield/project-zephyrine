with AWS.Response;
with AWS.Status;

package Adelaide_Server_Pkg is
   pragma Spark_Mode (Off);

   --  The main dispatch callback for Adelaide AWS Proxy Server
   function Dispatch (Request : AWS.Status.Data) return AWS.Response.Data;

end Adelaide_Server_Pkg;
