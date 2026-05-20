with AWS.Response;
with AWS.Status;

package Adelaide_Server_Pkg is

   --  The main dispatch callback for Adelaide AWS Proxy Server
   function Dispatch (Request : AWS.Status.Data) return AWS.Response.Data;

end Adelaide_Server_Pkg;
