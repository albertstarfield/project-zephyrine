with Ashell;
with Ada.Text_IO;

procedure LaunchProcess is
   use Ada.Text_IO;
   
   -- We will use 'dot notation' for Ashell to be explicit.
   
   Command : constant String := "zenity";
   Args    : constant Ashell.Argument_List :=
     ("--entry", "--text=Say Hi to the Sekai");
      
   Result  : Ashell.Process_Result;
   
begin
   Put_Line ("Attempting to launch Zenity using the 'ashell' library...");

   -- Run the command in the background (non-blocking).
   Result := Ashell.Execute (Command, Args, Mode => Ashell.Background);

   -- Check if the OS successfully launched the process.
   if Result.Exit_Code = 0 and then Result.Process.Is_Some then
      
      declare
         PID : constant Ashell.Process_Id_Type :=
           Ashell.Get_Process_Id (Result.Process);
      begin
         Put_Line ("Successfully launched Zenity with PID: " & PID'Image);
      end;
      
      Put_Line ("The program will now wait for 5 seconds...");
      delay 5.0;

      Put_Line ("5 seconds have passed. Main program is now exiting.");
      Put_Line ("The Zenity window will remain open as an orphan process.");
      
      -- We do nothing else. The procedure ends here.
      
   else
      Put_Line ("*** FAILED to launch Zenity. ***");
      Put_Line ("Standard Output from failed command: " & Result.Output);
      Put_Line ("Please ensure 'zenity' is installed and in your system's PATH.");
   end if;

exception
   when others =>
      Put_Line ("An unexpected error occurred.");
end LaunchProcess;