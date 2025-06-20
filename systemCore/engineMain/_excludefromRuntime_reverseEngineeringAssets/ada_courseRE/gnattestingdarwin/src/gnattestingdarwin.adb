-- The crucial line. It MUST have "GNAT."
with GNAT.OS_Lib;
with Ada.Text_IO;

-- Procedure name now matches the filename.
procedure gnatTestingDarwin is
   -- This line is also crucial.
   use GNAT.OS_Lib;
   use Ada.Text_IO;

   Program_Name : constant String := "zenity";
   Program_Args : constant Argument_List :=
     (1 => new String'("--entry"),
      2 => new String'("--text=Say Hi to the Sekai"));
      
   Child_PID    : Process_Id;

begin
   Put_Line ("Attempting to launch Zenity using GNAT.OS_Lib...");

   -- This Spawn call is now correct for GNAT.OS_Lib.
   if Spawn (Program_Name => Program_Name,
             Args         => Program_Args,
             Process_Id   => Child_PID)
   then
      -- This To_String call is also correct for GNAT.OS_Lib.
      Put_Line ("Successfully launched Zenity with PID: " & To_String (Child_PID));
      Put_Line ("Program will wait for 5 seconds, then exit.");
      Put_Line ("The Zenity window will be left running.");

      delay 5.0;

      Put_Line ("5 seconds have passed. Main program exiting.");

   else
      Put_Line ("*** FAILED to launch Zenity. ***");
      Put_Line ("Please ensure 'zenity' is installed and in your system's PATH.");
   end if;

exception
   when others =>
      Put_Line ("An unexpected error occurred.");
end gnatTestingDarwin;