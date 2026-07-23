with GNAT.OS_Lib;
with Ada.Text_IO;

procedure my_final_test is
   use GNAT.OS_Lib;
   use Ada.Text_IO;

   Program_Name : constant String := "zenity";
   Program_Args : constant Argument_List :=
     (1 => new String'("--entry"),
      2 => new String'("--text=Say Hi to the Sekai"));
      
   Child_PID    : Process_Id;

begin
   Put_Line ("Attempting to launch Zenity using GNAT.OS_Lib...");

   if Spawn (Program_Name => Program_Name,
             Args         => Program_Args,
             Process_Id   => Child_PID)
   then
      Put_Line ("Successfully launched Zenity with PID: " & To_String (Child_PID));
      Put_Line ("Program will wait for 5 seconds, then exit.");
      delay 5.0;
      Put_Line ("5 seconds have passed. Main program exiting.");
   else
      Put_Line ("*** FAILED to launch Zenity. ***");
   end if;
end my_final_test;
