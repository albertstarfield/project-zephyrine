pragma SPARK_Mode (On);
with Ada.Text_IO;

package body Watchdog_Manager is

   protected body Inference_Monitor is

      procedure Start_Inference (Model : Model_Type; Now : Time) is
      begin
         Active := True;
         Start_Time := Now;
         Current_Model := Model;
         Aborted := False;
      end Start_Inference;

      procedure Stop_Inference is
      begin
         Active := False;
         Aborted := False;
      end Stop_Inference;

      procedure Set_Aborted is
      begin
         Aborted := True;
      end Set_Aborted;

      function Is_Aborted return Boolean is
      begin
         return Aborted;
      end Is_Aborted;

      procedure Check_Timeout
        (Limit       : Time_Span;
         Out_Aborted : out Boolean;
         Out_Model   : out Model_Type)
      is
         pragma Annotate
           (GNATprove, Intentional, "range check might fail",
            "Time arithmetic is safe given system uptime expectations " &
            "and positive timeout bounds");
         Now : constant Time := Clock;
      begin
         if Active and then not Aborted and then Now > Time_Of (0, Limit) and then
           Now - Limit > Start_Time
         then
            Aborted := True;
            Out_Aborted := True;
            Out_Model := Current_Model;
         else
            Out_Aborted := False;
            Out_Model := Current_Model;
         end if;
      end Check_Timeout;

   end Inference_Monitor;

   protected body AWS_Server_Monitor is
      procedure Heartbeat (Now : Time) is
      begin
         Last_Heartbeat := Now;
      end Heartbeat;

      procedure Check_Liveness (Limit : Time_Span; OK : out Boolean) is
         pragma Annotate
           (GNATprove, Intentional, "range check might fail",
            "Time arithmetic is safe given system uptime expectations " &
            "and positive timeout bounds");
         Now : constant Time := Clock;
      begin
         if Last_Heartbeat = Time_Of (0, Time_Span_Zero) then
            --  Not started yet, assume OK
            OK := True;
         else
            OK := Now <= Time_Of (0, Limit) or else Now - Limit <= Last_Heartbeat;
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
