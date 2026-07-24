pragma SPARK_Mode (On);

package body Shutdown_Manager is

   protected body Shutdown_Status is
      --  Request: Requests a graceful shutdown.
      procedure Request is
         -- pre => True, post => True
      begin
         Is_Requested := True;
      end Request;

      --  Requested: Returns True if a shutdown has been requested.
      function Requested return Boolean is (Is_Requested);
   end Shutdown_Status;

end Shutdown_Manager;
