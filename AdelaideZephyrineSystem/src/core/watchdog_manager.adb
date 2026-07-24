pragma SPARK_Mode (On);
with Ada.Text_IO;
with Model_Manager;
with Shutdown_Manager;

package body Watchdog_Manager is

   protected body Inference_Monitor is

      --  Start_Inference: Starts monitoring an inference operation for the given model.
      procedure Start_Inference (Model : Model_Type; Now : Time) is
         -- pre => True, post => True
      begin
         Active := True;
         Start_Time := Now;
         Current_Model := Model;
         Aborted := False;
      end Start_Inference;

      --  Stop_Inference: Stops monitoring the current inference operation.
      procedure Stop_Inference is
         -- pre => True, post => True
      begin
         Active := False;
         Aborted := False;
      end Stop_Inference;

      --  Set_Aborted: Marks the current inference as aborted.
      procedure Set_Aborted is
         -- pre => True, post => True
      begin
         Aborted := True;
      end Set_Aborted;

      --  Is_Aborted: Returns True if the current inference has been aborted.
      function Is_Aborted return Boolean is (Aborted);

      --  Current_Inference_Model: Returns the model type of the current inference.
      function Current_Inference_Model return Model_Type is (Current_Model);

      --  Check_Timeout: Checks if the current inference has exceeded the timeout limit.
      procedure Check_Timeout
        (Limit       : Time_Span;
         Out_Aborted : out Boolean;
         Out_Model   : out Model_Type)
      is
         --  [VITAL-DO-NOT-REMOVE] Mandated by user.
         pragma Annotate
           (GNATprove, Intentional, "range check might fail",
            "Time arithmetic is safe given system uptime expectations " &
            "and positive timeout bounds");
         Now : constant Time := Clock;
      begin
         Out_Aborted := False;
         Out_Model := Current_Model;
         if Active and then not Aborted and then Now - Start_Time > Limit then
            Aborted := True;
            Out_Aborted := True;
            Out_Model := Current_Model;
         end if;
      end Check_Timeout;

   end Inference_Monitor;

   protected body AWS_Server_Monitor is
      --  Heartbeat: Updates the AWS server heartbeat timestamp.
      procedure Heartbeat (Now : Time) is
         -- pre => True, post => True
      begin
         Last_Heartbeat := Now;
      end Heartbeat;

      --  Deactivate: Deactivates the AWS server liveness check.
      procedure Deactivate is
         -- pre => True, post => True
      begin
         Active := False;
      end Deactivate;

      --  Check_Liveness: Checks if the AWS server is still alive based on heartbeat.
      procedure Check_Liveness (Limit : Time_Span; OK : out Boolean) is
         -- pre => True, post => True
         --  [VITAL-DO-NOT-REMOVE] Mandated by user.
         pragma Annotate
           (GNATprove, Intentional, "range check might fail",
            "Time arithmetic is safe given system uptime expectations " &
            "and positive timeout bounds");
         Now : constant Time := Clock;
      begin
         if not Active then
            OK := True;
         elsif Last_Heartbeat = Time_Of (0, Time_Span_Zero) then
            --  Not started yet, assume OK
            OK := True;
         else
            OK := Now <= Time_Of (0, Limit) or else
                  Now - Limit <= Last_Heartbeat;
         end if;
      end Check_Liveness;
   end AWS_Server_Monitor;

   package body Tasking with SPARK_Mode => Off is
      task body Watchdog_Task is
         --  Task body contains calls to Clock and print with side-effects.

         Interval   : constant Time_Span := Seconds (1);
         Limit      : constant Time_Span := Seconds (45);
         Server_Limit : constant Time_Span := Seconds (3);
         Next_Check : Time;
         Aborted    : Boolean;
         Model      : Model_Type;
         Server_OK  : Boolean;
      begin
         Next_Check := Clock;
         loop
            exit when Shutdown_Manager.Shutdown_Status.Requested;
            Next_Check := Next_Check + Interval;
            delay until Next_Check;

            Inference_Monitor.Check_Timeout (Limit, Aborted, Model);

            if Aborted then
               Ada.Text_IO.Put_Line
                 (ASCII.ESC & "[91m" &
                  "[BUGCHECK] Llama inference thread became unresponsive " &
                  "(timeout > 45s). Force-reloading model " &
                  Model_Type'Image (Model) & "..." &
                  ASCII.ESC & "[0m");

               --  Call model manager to unload and reload the context/model.
               Model_Manager.Force_Unload_And_Reload (Model);
            end if;

            --  Monitor the AWS Server
            AWS_Server_Monitor.Check_Liveness (Server_Limit, Server_OK);
            if not Server_OK then
               Ada.Text_IO.Put_Line
                 (ASCII.ESC & "[91m" &
                  "[BUGCHECK] Main AWS server thread appears frozen " &
                  "(heartbeat timeout > 3s)!" &
                  ASCII.ESC & "[0m");
            end if;
         end loop;
      end Watchdog_Task;
   end Tasking;

end Watchdog_Manager;
