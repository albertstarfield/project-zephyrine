with Ada.Text_IO;
with Ada.Exceptions;
with AWS.Server;
with Adelaide_Server_Pkg;

procedure Adelaide_Server is
   procedure Set_Darwin_Realtime;
   pragma Import (C, Set_Darwin_Realtime, "set_darwin_realtime");

   Server_Port : constant Positive := 11435;
   Web_Server  : AWS.Server.HTTP;
begin
   --  Apply Apple Silicon soft real-time thread constraint policy
   --  Set_Darwin_Realtime;

   Ada.Text_IO.Put_Line ("[+] Starting Adelaide AWS Proxy on port" &
                         Server_Port'Img & "...");

   --  Start AWS server with the custom Dispatcher callback
   AWS.Server.Start (
      Web_Server => Web_Server,
      Name       => "Adelaide-Proxy",
      Callback   => Adelaide_Server_Pkg.Dispatch'Access,
      Port       => Server_Port
   );

   Ada.Text_IO.Put_Line ("[+] Adelaide AWS Proxy is running on port 11435.");
   Ada.Text_IO.Put_Line ("[+] Proxying non-ML to Ollama (11434), " &
                         "ML/agentic flow to Python backend (11436).");

   --  Keep the main thread alive indefinitely to service incoming requests
   loop
      delay 3600.0;
   end loop;

exception
   when E : others =>
      Ada.Text_IO.Put_Line ("[-] Exception in Adelaide AWS Server: " &
                            Ada.Exceptions.Exception_Name (E));
      Ada.Text_IO.Put_Line ("[-] Message: " &
                            Ada.Exceptions.Exception_Message (E));
      Ada.Text_IO.Put_Line ("[-] Shutting down...");
      AWS.Server.Shutdown (Web_Server);
end Adelaide_Server;
