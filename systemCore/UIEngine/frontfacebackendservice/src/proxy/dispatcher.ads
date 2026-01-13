with AWS.Status;
with AWS.Response;

package Dispatcher is

   -- The main entry point for the AWS Server.
   -- It inspects Request.URI and routes to Handlers.
   function Callback (Request : AWS.Status.Data) return AWS.Response.Data;

end Dispatcher;