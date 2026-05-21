with Ada.Text_IO; use Ada.Text_IO;
with Adelaide_Server_Pkg;
with Model_Manager;
with Knowledge_Manager;
with AWS.Config.Set;
with AWS.Server;
with Ada.Exceptions;
with Ada.Command_Line;

procedure Adelaide_Server is
   WS : AWS.Server.HTTP;
   Conf : AWS.Config.Object := AWS.Config.Get_Current;
begin
   begin
      Put_Line ("[Main] Initializing Adelaide Intelligence Backend...");
      Model_Manager.Initialize;
      Knowledge_Manager.Initialize;
      
      AWS.Config.Set.Server_Port (Conf, 11420);
      AWS.Config.Set.Reuse_Address (Conf, True);
      
      Put_Line ("[Main] Adelaide-Lite Server starting on port 11420...");
      AWS.Server.Start (WS, "Adelaide",
                        Config   => Conf,
                        Callback => Adelaide_Server_Pkg.Dispatch'Access);
      
      Put_Line ("[Main] Initializing index crawl...");
      Knowledge_Manager.Start_Tasks;
      
      Put_Line ("[Main] Server is UP. Press Q to shutdown.");
      
      loop
         declare
            Input : String := Get_Line;
         begin
            exit when Input = "q" or else Input = "Q";
         end;
      end loop;
      
      Put_Line ("[Main] Shutting down...");
      AWS.Server.Stop (WS);
   exception
      when E : others =>
         Put_Line ("[FATAL] Server Error: " & Ada.Exceptions.Exception_Message (E));
   end;
end Adelaide_Server;
