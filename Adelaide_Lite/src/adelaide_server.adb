with Ada.Text_IO;
with Ada.Exceptions;
with AWS.Server;
with Adelaide_Server_Pkg;
with Model_Manager;
with Integrity_Utils;
with Interfaces; use Interfaces;

procedure Adelaide_Server is
   procedure Set_Darwin_Realtime;
   pragma Import (C, Set_Darwin_Realtime, "set_darwin_realtime");

   --  Perform a quick self-test of the formally verified integrity logic
   procedure Run_Integrity_Self_Test is
      use Integrity_Utils;
      Block_Size : constant Positive := 4;
      Data       : Byte_Array (1 .. 8) := (1, 2, 3, 4, 5, 6, 7, 8);
      CRCs       : CRC_Array (1 .. 2);
      Parity     : Byte_Array (1 .. 4) := (0, 0, 0, 0);
      Success    : Boolean;
   begin
      Ada.Text_IO.Put_Line ("[*] Running Formal Integrity Self-Test...");
      
      --  1. Generate CRCs and Parity
      CRCs (1) := Calculate_CRC32 (Data (1 .. 4));
      CRCs (2) := Calculate_CRC32 (Data (5 .. 8));
      Generate_Parity (Data, Block_Size, Parity);
      
      --  2. Simulate corruption
      Data (1) := 99; 
      
      --  3. Attempt Self-Patch
      Self_Patch (Data, Block_Size, CRCs, Parity, Success);
      
      if Success and then Data (1) = 1 then
         Ada.Text_IO.Put_Line ("[+] Integrity Core: VERIFIED (Self-Patch Operational)");
      else
         Ada.Text_IO.Put_Line ("[!] Integrity Core: FAILED. Check SPARK proofs.");
      end if;
   end Run_Integrity_Self_Test;

   Server_Port : constant Positive := 11420;
   Web_Server  : AWS.Server.HTTP;
begin
   --  Apply Apple Silicon soft real-time thread constraint policy
   Set_Darwin_Realtime;

   --  Verify data integrity subsystem
   Run_Integrity_Self_Test;

   --  Initialize Llama backend and models
   Model_Manager.Initialize;

   Ada.Text_IO.Put_Line ("[+] Starting Adelaide AWS Proxy on port" &
                         Server_Port'Img & "...");

   --  Start AWS server with the custom Dispatcher callback
   AWS.Server.Start (
      Web_Server => Web_Server,
      Name       => "Adelaide-Inference-Server",
      Callback   => Adelaide_Server_Pkg.Dispatch'Access,
      Port       => Server_Port
   );

   Ada.Text_IO.Put_Line ("[+] Adelaide-Lite is running on port" & Server_Port'Img);
   Ada.Text_IO.Put_Line ("[+] Internal llama.cpp engine initialized with GPU acceleration.");

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
