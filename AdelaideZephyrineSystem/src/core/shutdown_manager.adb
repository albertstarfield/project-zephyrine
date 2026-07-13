pragma SPARK_Mode (On);

package body Shutdown_Manager is

   protected body Shutdown_Status is
      procedure Request is
      begin
         Is_Requested := True;
      end Request;

      function Requested return Boolean is (Is_Requested);
   end Shutdown_Status;

end Shutdown_Manager;
