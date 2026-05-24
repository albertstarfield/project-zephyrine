with AnsiAda;
with Ada.Text_IO; use Ada.Text_IO;
with Model_Manager;
with Knowledge_Manager;
with Ada.Exceptions;

procedure Adelaide_Lite is
begin
   begin
      Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Main]" &
                AnsiAda.Reset & " Initializing Adelaide Knowledge Core...");
      Model_Manager.Initialize;
      Knowledge_Manager.Initialize;

      Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Main]" &
                AnsiAda.Reset & " Starting background tasks...");
      Knowledge_Manager.Start_Tasks;

      Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Main]" &
                AnsiAda.Reset & " Adelaide Knowledge Core is active.");
      Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Main]" &
                AnsiAda.Reset & " Press Q to shutdown.");

      loop
         declare
            Input : constant String := Get_Line;
         begin
            exit when Input = "q" or else Input = "Q";
         end;
      end loop;

   exception
      when E : others =>
         Put_Line (AnsiAda.Foreground (AnsiAda.Red) & "[FATAL]" &
                   AnsiAda.Reset & " Core Error: " &
                   Ada.Exceptions.Exception_Message (E));
   end;
end Adelaide_Lite;
