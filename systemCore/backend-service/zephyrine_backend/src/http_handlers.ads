-- Declares our HTTP handlers based on the AWS dispatcher pattern.
with AWS.Response;
with AWS.Status;
with AWS.Services.Dispatchers.URI;

package HTTP_Handlers is

   -- A generic handler type for our application
   type Zephyrine_Handler is new AWS.Services.Dispatchers.URI.Handler with null record;

   -- Specific handler for the /health endpoint
   type Health_Check_Handler is new Zephyrine_Handler with null record;
   overriding function Dispatch
     (Self    : in Health_Check_Handler;
      Request : in AWS.Status.Data) return AWS.Response.Data;

   -- TODO: Declare other handlers for /primedready, /api/..., etc.

end HTTP_Handlers;