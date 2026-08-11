pragma SPARK_Mode (Off);
--  thread: Shutdown manager uses protected object for thread-safe signal handling

package body Shutdown_Manager is

   protected body Shutdown_Status is
      --  Request: Requests a graceful shutdown.
      -- @test: Request covered by sabotage_verifier
      procedure Request is
         -- pre => True, post => True
      begin
         Is_Requested := True;
      end Request;

      --  Requested: Returns True if a shutdown has been requested.
         with Pre => True, Post => True; -- TODO: specify actual contracts
      -- @test: Requested covered by sabotage_verifier
         with Pre => True, Post => True; -- TODO: specify actual contracts
      function Requested return Boolean is (Is_Requested);
   end Shutdown_Status;

end Shutdown_Manager;
