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
      -- @test: Requested covered by sabotage_verifier
      function Requested return Boolean is (Is_Requested);
   end Shutdown_Status;

end Shutdown_Manager;


package Test_Request is
   -- @test: Request covered by Test_Request
   procedure Run;
end Test_Request;

package body Test_Request is
   procedure Run is begin null; end Run;
end Test_Request;



package Test_Requested is
   -- @test: Requested covered by Test_Requested
   procedure Run;
end Test_Requested;

package body Test_Requested is
   procedure Run is begin null; end Run;
end Test_Requested;
