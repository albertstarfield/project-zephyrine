with Ada.Text_IO;
with Ada.Exceptions;
with Interfaces.C;
with Interfaces.C.Strings;

procedure Watchdog_Thread2 is

   -- ===================================================================
   --  PART 1: DECLARATIONS
   -- ===================================================================
   
   type String_Access_Array is array (Positive range <>) of access String;

   package C_Bridge is
      type Process_Id is new Interfaces.C.int;
      No_Process_Id : constant Process_Id := -1;

      function Spawn
        (Program : String;
         Args    : String_Access_Array) return Process_Id;

      function Is_Alive (Pid : Process_Id) return Boolean;
      function Kill_Process (Pid : Process_Id) return Boolean;
   end C_Bridge;

   -- ===================================================================
   --  PART 2: IMPLEMENTATIONS (BODIES)
   -- ===================================================================

   package body C_Bridge is
      -- FIX #1: Add use clauses INSIDE the package body to make C types visible here.
      use Interfaces;
      use Interfaces.C;
      use Interfaces.C.Strings;
      use type C.int;

      -- --- C Function Bindings ---
      function C_Fork return int;
      pragma Import (C, C_Fork, "fork");
      function C_Execvp (File : chars_ptr; Args : chars_ptr_array) return int;
      pragma Import (C, C_Execvp, "execvp");
      function C_Kill (Pid : int; Sig : int) return int;
      pragma Import (C, C_Kill, "kill");

      -- --- Ada Wrapper Implementations ---
      function Spawn
        (Program : String;
         Args    : String_Access_Array) return Process_Id
      is
         Arg_Count   : constant size_t := size_t(Args'Length + 1);
         -- FIX #2: Explicitly type the range bounds as size_t.
         C_Args      : aliased chars_ptr_array (size_t(0) .. Arg_Count);
         C_Program   : chars_ptr := New_String (Program);
         Child_PID   : int;
      begin
         C_Args(size_t(0)) := New_String (Program);
         for I in Args'Range loop
            -- FIX #2 (continued): Convert the loop index for array access.
            C_Args(size_t(I - Args'First + 1)) := New_String (Args(I).all);
         end loop;
         C_Args(Arg_Count) := null_ptr;
         
         Child_PID := C_Fork;

         if Child_PID < 0 then
            return No_Process_Id;
         elsif Child_PID = 0 then
            declare
               Return_Code : int := C_Execvp (C_Program, C_Args);
               procedure C_Exit (Status : int);
               pragma Import (C, C_Exit, "_exit");
            begin
               C_Exit (1);
            end;
            return No_Process_Id;
         else
            Free(C_Program);
            for Ptr of C_Args loop
               if Ptr /= null_ptr then
                  Free(Ptr);
               end if;
            end loop;
            return Process_Id (Child_PID);
         end if;
      end Spawn;

      function Is_Alive (Pid : Process_Id) return Boolean is
         Signal_0 : constant int := 0;
      begin
         if Pid = No_Process_Id then
            return False;
         end if;
         return C_Kill (int (Pid), Signal_0) = 0;
      end Is_Alive;

      function Kill_Process (Pid : Process_Id) return Boolean is
         Sig_Term : constant int := 15;
      begin
         return C_Kill (int (Pid), Sig_Term) = 0;
      end Kill_Process;
   end C_Bridge;
   
   -- ===================================================================
   --  PART 3: MAIN PROGRAM VARIABLES
   -- ===================================================================
   
   use type C_Bridge.Process_Id;

   Program_To_Run : constant String := "./Watchdog_Thread1";
   Process_Args   : constant String_Access_Array :=
     (new String'("--integrity-check-file=./launcher.py"),
      new String'("--"),
      new String'("python"),
      new String'("AdelaideAlbertCortex.py"));

   Child_PID      : C_Bridge.Process_Id := C_Bridge.No_Process_Id;

-- ===================================================================
--  PART 4: THE MAIN PROGRAM LOGIC
-- ===================================================================
begin
   Ada.Text_IO.Put_Line ("=====================================================");
   Ada.Text_IO.Put_Line (" ADA WATCHDOG (LEVEL 2) INITIALIZED.");
   Ada.Text_IO.Put_Line ("=====================================================");

   loop
      if not C_Bridge.Is_Alive (Child_PID) then
         Ada.Text_IO.Put_Line ("ADA_WATCHDOG: Monitored process (Go Watchdog) is not running. Spawning...");

         Child_PID := C_Bridge.Spawn (Program => Program_To_Run,
                                      Args    => Process_Args);
         
         if Child_PID /= C_Bridge.No_Process_Id then
            Ada.Text_IO.Put_Line ("ADA_WATCHDOG: Successfully spawned Go Watchdog with PID: " & C_Bridge.Process_Id'Image (Child_PID));
         else
            Ada.Text_IO.Put_Line ("ADA_WATCHDOG: *** FAILED to spawn Go Watchdog. ***");
            Ada.Text_IO.Put_Line ("ADA_WATCHDOG: Retrying in 5 seconds...");
            delay 5.0;
         end if;
      end if;

      delay 5.0;
   end loop;

exception
   when E : others =>
      Ada.Text_IO.Put_Line ("!!! A FATAL, UNEXPECTED ERROR occurred in the Ada Watchdog: " &
                            Ada.Exceptions.Exception_Message (E));
      delay 10.0;
end Watchdog_Thread2;