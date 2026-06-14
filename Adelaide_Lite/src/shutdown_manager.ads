pragma SPARK_Mode (On);

package Shutdown_Manager is

   --  Thread-safe shutdown signaling for the entire Adelaide platform.
   --  Used to gracefully stop background tasks (ELP monitor, watchdog, etc.)
   --  before the server finalized.
   protected Shutdown_Status is
      procedure Request;
      function Requested return Boolean;
   private
      Is_Requested : Boolean := False;
   end Shutdown_Status;

end Shutdown_Manager;
