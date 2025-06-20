with Ada.Text_IO;
with Ada.Exceptions;
with Interfaces.C;
with Interfaces.C.Strings;

procedure Main is

   -- ===================================================================
   --  PART 1: DECLARATIONS
   -- ===================================================================
   
   -- FIX: Move all necessary 'use' clauses to the top level so they are
   -- visible to all nested units, including pragmas and task bodies.
   use Interfaces.C;
   use Interfaces.C.Strings;
   use type Interfaces.C.int;

   -- The "C_Bridge" package specification is declared here.
   package C_Bridge is
      type Process_Id is new int;
      No_Process_Id : constant Process_Id := -1;

      function Spawn (Program : String) return Process_Id;
      function Is_Alive (Pid : Process_Id) return Boolean;
      function Kill_Process (Pid : Process_Id) return Boolean;
   end C_Bridge;

   -- The "Watchdog_Task_Type" task specification is declared here.
   task type Watchdog_Task_Type is
      entry Signal_Shutdown;
   end Watchdog_Task_Type;

   -- The task instance is declared here.
   Watchdog : Watchdog_Task_Type;

   -- FIX: All C function bindings must be declared here in the main
   -- declarative part, where the necessary types from Interfaces.C are visible.
   function C_Fork return int;
   pragma Import (C, C_Fork, "fork");

   function C_Execvp (File : chars_ptr; Args : chars_ptr_array) return int;
   pragma Import (C, C_Execvp, "execvp");

   function C_Kill (Pid : int; Sig : int) return int;
   pragma Import (C, C_Kill, "kill");


   -- ===================================================================
   --  PART 2: IMPLEMENTATIONS (BODIES)
   -- ===================================================================

   -- The implementation for the C_Bridge package is nested inside Main.
   package body C_Bridge is
      function Spawn (Program : String) return Process_Id is
         -- The C array needs space for two things: the program name and a null terminator.
         C_Args      : aliased chars_ptr_array (0 .. 1);
         -- FIX: The intermediate _Ptr variable was incorrect and is removed.
         C_Program   : chars_ptr := New_String (Program);
         Child_PID   : int;
      begin
         C_Args(0) := New_String (Program);
         C_Args(1) := null_ptr;
         
         Child_PID := C_Fork;

         if Child_PID < 0 then
            return No_Process_Id;
         elsif Child_PID = 0 then -- This is the child process
            declare
               -- We call C_Execvp directly with the array.
               Return_Code : int := C_Execvp (C_Program, C_Args);
               procedure C_Exit (Status : int);
               pragma Import (C, C_Exit, "_exit");
            begin
               -- If execvp returns, it failed. Exit the child immediately.
               C_Exit (1);
            end;
            return No_Process_Id; -- Should not be reached
         else -- This is the parent process
            -- Clean up the memory we allocated for the C strings.
            Free(C_Program);
            Free(C_Args(0));
            return Process_Id (Child_PID);
         end if;
      end Spawn;

      function Is_Alive (Pid : Process_Id) return Boolean is
         -- On POSIX systems, we can check if a process is alive by sending it
         -- signal 0. This doesn't actually send a signal, but performs
         -- error checking to see if the PID is valid and we have permission.
         Signal_0    : constant int := 0;
         Return_Code : int;
      begin
         -- THE FIX: Add a guard clause to handle our special invalid PID value.
         if Pid = No_Process_Id then
            return False;
         end if;

         Return_Code := C_Kill (int (Pid), Signal_0);
         -- A return code of 0 means the process exists.
         return Return_Code = 0;
      end Is_Alive;

      function Kill_Process (Pid : Process_Id) return Boolean is
         Sig_Term    : constant int := 15;
         Return_Code : constant int := C_Kill (int (Pid), Sig_Term);
      begin
         return Return_Code = 0;
      end Kill_Process;
   end C_Bridge;

   -- The implementation for the Watchdog task is also nested inside Main.
   task body Watchdog_Task_Type is
      -- FIX: Make the operators for our custom Process_Id type visible here.
      use type C_Bridge.Process_Id;

      Program_To_Run     : constant String := "./bin/child_process";
      Child_PID          : C_Bridge.Process_Id := C_Bridge.No_Process_Id;
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
            if not C_Bridge.Is_Alive (Child_PID) then
               Log ("Child process is not running. Spawning...");
               Child_PID := C_Bridge.Spawn (Program => Program_To_Run);
               if Child_PID /= C_Bridge.No_Process_Id then
                  Log ("Successfully spawned child with PID: " & C_Bridge.Process_Id'Image (Child_PID));
               else
                  Log ("*** FAILED to spawn child process. ***");
                  Log ("Make sure '" & Program_To_Run & "' exists and is executable.");
                  delay 3.0;
               end if;
            else
               Log ("Heartbeat: Child " & C_Bridge.Process_Id'Image(Child_PID) & " is alive.");
            end if;
         end select;
         exit when Shutdown_Requested;
      end loop;

      Log ("Exited main loop. Performing final cleanup.");
      if C_Bridge.Is_Alive (Child_PID) then
         Log ("Attempting to terminate child process " & C_Bridge.Process_Id'Image(Child_PID) & "...");
         if C_Bridge.Kill_Process (Child_PID) then
            Log ("Termination signal sent successfully.");
         else
            Log ("Failed to send termination signal.");
         end if;
      end if;
   exception
      when E : others =>
         Log ("!!! An unexpected error occurred in task: " & Ada.Exceptions.Exception_Message (E));
   end Watchdog_Task_Type;


-- ===================================================================
--  PART 3: MAIN PROCEDURE EXECUTION
-- ===================================================================
begin
   Ada.Text_IO.Put_Line ("=====================================================");
   Ada.Text_IO.Put_Line (" Main thread started. Concurrent Watchdog is running.");
   Ada.Text_IO.Put_Line (" The watchdog will monitor and restart './child_process'.");
   Ada.Text_IO.Put_Line (" Press ENTER to signal a graceful shutdown.");
   Ada.Text_IO.Put_Line ("=====================================================");

   Ada.Text_IO.Skip_Line;

   Ada.Text_IO.Put_Line ("Main: ENTER detected. Signaling watchdog to shut down...");
   Watchdog.Signal_Shutdown;
   Ada.Text_IO.Put_Line ("Main: Shutdown signal sent. Main thread is now finished.");
   Ada.Text_IO.Put_Line ("The program will exit once the watchdog task completes its cleanup.");

exception
   when E : others =>
      Ada.Text_IO.Put_Line ("An unexpected error occurred in the main procedure: " &
                            Ada.Exceptions.Exception_Message (E));
end Main;