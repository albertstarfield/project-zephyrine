pragma SPARK_Mode (Off);
-- thread: Main orchestrator requires task protection

with AnsiAda;
with Ada.Text_IO; use Ada.Text_IO;
with Model_Manager;
with Knowledge_Manager;
with Ada.Exceptions;

--  AdelaideZephyrineSystem: Main entry point for the Adelaide Zephyrine System.
procedure AdelaideZephyrineSystem is
   -- pre => True, post => True
begin
   --  Initialize core systems (fatal on failure)
   begin
      Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Main]" &
                AnsiAda.Reset & " Initializing Adelaide Knowledge Core...");
      Model_Manager.Initialize;
      Knowledge_Manager.Initialize;
   exception
      when E : others =>
         Put_Line (AnsiAda.Foreground (AnsiAda.Red) & "[FATAL]" &
                   AnsiAda.Reset & " Init Error: " &
                   Ada.Exceptions.Exception_Message (E));
         return;
   end;

   --  Start background tasks (non-fatal on failure)
   begin
      Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Main]" &
                AnsiAda.Reset & " Starting background tasks...");
      Knowledge_Manager.Start_Tasks;
   exception
      when E : others =>
         Put_Line (AnsiAda.Foreground (AnsiAda.Red) & "[WARN]" &
                   AnsiAda.Reset & " Background task error: " &
                   Ada.Exceptions.Exception_Message (E));
   end;

   Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Main]" &
             AnsiAda.Reset & " Adelaide Knowledge Core is active.");
   Put_Line ("[+] AdelaideZephyrineSystem ready.");
   Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Main]" &
              AnsiAda.Reset);

   --  Main loop - continues listening even after errors
   loop
      begin
         declare
            Input : constant String := Get_Line;
         begin
            exit when Input = "q" or else Input = "Q";
         end;
      exception
         when others =>
            null;
      end;
   end loop;
end AdelaideZephyrineSystem;
