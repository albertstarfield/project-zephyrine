with Ada.Text_IO; use Ada.Text_IO;
with GNAT.OS_Lib;

procedure Main is

   -- Detect platform, architecture, and username
   Platform   : constant String := GNAT.OS_Lib.Operating_System;
   Arch       : constant String := GNAT.OS_Lib.Architecture;
   Username   : constant String := GNAT.OS_Lib.Username;

   -- Console color codes
   Color_Reset        : constant String := ASCII.ESC & "[0m";
   Color_Bright_Cyan  : constant String := ASCII.ESC & "[96m";
   Color_Bright_Red   : constant String := ASCII.ESC & "[91m";
   Color_Bright_Green : constant String := ASCII.ESC & "[92m";

   Assistant_Name : constant String := "Adelaide Zephyrine Charlotte";
   App_Name       : constant String := "Project " & Assistant_Name;
   Engine_Name    : constant String := "Adelaide Paradigm Engine";

   Console_Log_Prefix                : constant String := "[" & Color_Bright_Cyan & Engine_Name & "_" & Platform & "_" & Arch & Color_Reset & "]:";
   Version_The_Unattended_Log_Prefix : constant String := "[" & Color_Bright_Cyan & Engine_Name & Color_Bright_Green & " [Codename : \"The Unattended\"] " & Color_Reset & "]:";
   Version_Featherfeet_Log_Prefix    : constant String := "[" & Color_Bright_Cyan & Engine_Name & Color_Bright_Red & "[Codename : \"Featherfeet\"]" & Color_Reset & "]:";

begin
   -- Output the values
   Put_Line ("Platform: " & Platform);
   Put_Line ("Architecture: " & Arch);
   Put_Line ("Username: " & Username);
   Put_Line ("Assistant Name: " & Assistant_Name);
   Put_Line ("App Name: " & App_Name);
   Put_Line ("Engine Name: " & Engine_Name);
   Put_Line ("Console Log Prefix: " & Console_Log_Prefix);
   Put_Line ("Version 'The Unattended' Log Prefix: " & Version_The_Unattended_Log_Prefix);
   Put_Line ("Version 'Featherfeet' Log Prefix: " & Version_Featherfeet_Log_Prefix);
end Main;
