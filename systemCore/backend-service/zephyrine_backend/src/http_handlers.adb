-- Implements the logic for our HTTP handlers.
with AWS.MIME;
package body HTTP_Handlers is

   -- Implementation for the /health endpoint
   overriding function Dispatch
     (Self    : in Health_Check_Handler;
      Request : in AWS.Status.Data) return AWS.Response.Data
   is
      pragma Unreferenced (Self, Request);
   begin
      return AWS.Response.Build
        (Content_Type => AWS.MIME.Application_JSON,
         Body         => "{""status"": ""ok"", ""message"": ""HTTP and WebSocket server is healthy""}");
   end Dispatch;

end HTTP_Handlers;