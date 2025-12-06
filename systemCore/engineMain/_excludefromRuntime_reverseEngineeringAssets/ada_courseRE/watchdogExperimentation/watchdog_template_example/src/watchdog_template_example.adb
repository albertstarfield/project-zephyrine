with Ada.Text_IO;
with Ada.Exceptions;
with GNAT.OS_Lib; 

procedure Watchdog_Template_Example is
   use GNAT.OS_Lib; 
   use Ada.Text_IO;

   task type Watchdog_Task_Type is
      entry Signal_Shutdown;
   end Watchdog_Task_Type;

   Watchdog : Watchdog_Task_Type;

      task body Watchdog_Task_Type is
      -- Path is relative to the execution directory (bin/)
      Program_To_Run     : constant String := "./child_process";

      -- Use a concrete empty array for arguments, not an uninitialized pointer.
      Process_Args       : GNAT.OS_Lib.Argument_List := (1 .. 0 => new String'(""));

      Child_PID          : GNAT.OS_Lib.Process_Id := GNAT.OS_Lib.No_Process_Id;
      Shutdown_Requested : Boolean := False;

      procedure Log (Message : String) is
      begin
         Ada.Text_IO.Put_Line ("WATCHDOG: " & Message);
      end Log;

   begin
      Log ("Task started. Initial spawn incoming.");

      loop
         select
            accept Signal_Shutdown do
               Log ("Shutdown signal received. Terminating gracefully.");
               Shutdown_Requested := True;
            end Signal_Shutdown;
         or
            delay 2.0;

            if not GNAT.OS_Lib.Is_Alive (Child_PID) then
               Log ("Child process is not running. Spawning...");

               if GNAT.OS_Lib.Spawn
                 (Program_Name => Program_To_Run,
                  Args         => Process_Args,
                  Process_Id   => Child_PID)
               then
                  Log ("Successfully spawned child with PID: " &
                    GNAT.OS_Lib.To_String (Child_PID));
               else
                  Log ("*** FAILED to spawn child process. ***");
                  Log ("Make sure '" & Program_To_Run & "' can be executed.");
                  delay 3.0;
               end if;
            else
               Log ("Heartbeat: Child " &
                 GNAT.OS_Lib.To_String (Child_PID) & " is alive.");
            end if;
         end select;

         exit when Shutdown_Requested;
      end loop;

      Log ("Exited main loop. Performing final cleanup.");
      if GNAT.OS_Lib.Is_Alive (Child_PID) then
         Log ("Attempting to terminate child process " &
           GNAT.OS_Lib.To_String (Child_PID) & "...");
         if GNAT.OS_Lib.Terminate_Process (Child_PID) then
            Log ("Termination signal sent successfully.");
         else
            Log ("Failed to send termination signal.");
         end if;
      end if;

   exception
      when E : others =>
         Log ("!!! An unexpected error occurred in task: " &
           Ada.Exceptions.Exception_Message (E));
   end Watchdog_Task_Type;

begin
   Ada.Text_IO.Put_Line
     ("=====================================================");
   Ada.Text_IO.Put_Line
     (" Main thread started. Concurrent Watchdog is running.");
   Ada.Text_IO.Put_Line
     (" The watchdog will monitor and restart the child process.");
   Ada.Text_IO.Put_Line
     (" Press ENTER to signal a graceful shutdown.");
   Ada.Text_IO.Put_Line
     ("=====================================================");

   Ada.Text_IO.Skip_Line;

   Ada.Text_IO.Put_Line
     ("Main: ENTER detected. Signaling watchdog to shut down...");
   Watchdog.Signal_Shutdown;
   Ada.Text_IO.Put_Line
     ("Main: Shutdown signal sent. Main thread is now finished.");

exception
   when E : others =>
      Ada.Text_IO.Put_Line
        ("An unexpected error occurred in the main procedure: " &
         Ada.Exceptions.Exception_Message (E));
end Watchdog_Template_Example;