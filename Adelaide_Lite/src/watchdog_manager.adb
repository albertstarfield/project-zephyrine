with Ada.Text_IO;

package body Watchdog_Manager is
   pragma Spark_Mode (On);

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
        (Now         : Time;
         Limit       : Time_Span;
         Out_Aborted : out Boolean;
         Out_Model   : out Model_Type)
      is
      begin
         if Active and then not Aborted and then
           (Now - Start_Time) > Limit
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

   task body Watchdog_Task is
      pragma SPARK_Mode (Off);
      --  Task body contains calls to Clock and print with side-effects.

      Interval   : constant Time_Span := Seconds (1);
      Limit      : constant Time_Span := Seconds (45);
      Now        : Time;
      Next_Check : Time;
      Aborted    : Boolean;
      Model      : Model_Type;
   begin
      Next_Check := Clock;
      loop
         Next_Check := Next_Check + Interval;
         delay until Next_Check;

         Now := Clock;
         Inference_Monitor.Check_Timeout (Now, Limit, Aborted, Model);

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
      end loop;
   end Watchdog_Task;

end Watchdog_Manager;
