with Ada.Text_IO;
with Ada.Streams;
with Ada.Streams.Stream_IO;
with GNAT.OS_Lib;
with GNAT.Sockets;

procedure Branchpredictor_Dctd is
   use Ada.Text_IO;
   use GNAT.Sockets;

   Server_Socket_Path : constant String := "celestial_timestream_vector_helper.socket";
   
   procedure Start_Python_Server is
      Args : GNAT.OS_Lib.Argument_List := 
        (1 => new String'("src/vector_prediction_server.py"));
      PID  : GNAT.OS_Lib.Process_Id;
   begin
      -- Start Python ZMQ server as background process
      PID := GNAT.OS_Lib.Non_Blocking_Spawn 
        (Program_Name => "python3",
         Args => Args);
      
      Put_Line("Started Python ZMQ server (PID:" & PID'Img & ")");
      
      -- Give server time to start
      delay 2.0;
   end Start_Python_Server;

   procedure Clean_Socket_File is
      Success : Boolean;
   begin
      if GNAT.OS_Lib.Is_Regular_File (Server_Socket_Path) then
         GNAT.OS_Lib.Delete_File (Server_Socket_Path, Success);
         if Success then
            Put_Line("Cleaned up existing socket file");
         end if;
      end if;
   end Clean_Socket_File;

begin
   Put_Line("--- DCTd Branch Predictor Starting ---");
   
   -- Clean up any existing socket
   Clean_Socket_File;
   
   -- Start Python server
   Start_Python_Server;
   
   Put_Line("Daemon ready - Python server handling ZMQ requests");
   Put_Line("Press Ctrl+C to exit");
   
   -- Keep the main process alive
   loop
      delay 60.0;  -- Check every minute if we should exit
   end loop;
   
exception
   when others =>
      Put_Line("Daemon exiting...");
end Branchpredictor_Dctd;