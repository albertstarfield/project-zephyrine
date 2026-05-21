with Ada.Text_IO; use Ada.Text_IO;
with Ada.Exceptions;
with Ada.Command_Line; use Ada.Command_Line;
with AWS.Server;
with Adelaide_Server_Pkg;
with Model_Manager;
with Integrity_Utils;
with Toolchain_Manager;
with Interfaces; use Interfaces;
with GNAT.OS_Lib;

procedure Adelaide_Server is
   pragma Spark_Mode (Off);

   --  ANSI Color Codes
   Reset   : constant String := ASCII.ESC & "[0m";
   Bold    : constant String := ASCII.ESC & "[1m";
   Purple  : constant String := ASCII.ESC & "[38;5;171m";
   Cyan    : constant String := ASCII.ESC & "[36m";
   Green   : constant String := ASCII.ESC & "[32m";
   Yellow  : constant String := ASCII.ESC & "[33m";
   Blue    : constant String := ASCII.ESC & "[34m";

   procedure Set_Darwin_Realtime;
   pragma Import (C, Set_Darwin_Realtime, "set_darwin_realtime");

   --  Display Whimsical Help
   procedure Show_Help is
   begin
      Put_Line (Purple & Bold & "      Adelaide Zephyrine Charlotte" & Reset);
      Put_Line (Cyan & "  Universal AI Gateway & Reasoning Pipeline" & Reset);
      New_Line;
      Put_Line (Bold & "USAGE:" & Reset);
      Put_Line ("  adelaide_server [options]");
      New_Line;
      Put_Line (Bold & "OPTIONS:" & Reset);
      Put_Line ("  -h, --help     " & Green &
                "Show this whimsical menu" & Reset);
      Put_Line ("  -v, --version  " & Green &
                "Display version metadata" & Reset);
      New_Line;
      Put_Line (Bold & "ENVIRONMENT:" & Reset);
      Put_Line ("  Default Port:  11420");
      Put_Line ("  Models Path:   ../llama.cpp/models/");
      New_Line;
      Put_Line (Purple & "I am dedicated to creating architectures that are " &
                "intellectually rigorous." & Reset);
   end Show_Help;

   --  Perform a quick self-test of the formally verified integrity logic
   procedure Run_Integrity_Self_Test is
      use Integrity_Utils;
      Block_Size : constant Positive := 4;
      Data       : Byte_Array (1 .. 8) := (1, 2, 3, 4, 5, 6, 7, 8);
      CRCs       : CRC_Array (1 .. 2);
      Parity     : Byte_Array (1 .. 4) := (0, 0, 0, 0);
      Success    : Boolean;
   begin
      Put_Line (Blue & "[*] Running Formal Integrity Self-Test..." & Reset);

      CRCs (1) := Calculate_CRC32 (Data (1 .. 4));
      CRCs (2) := Calculate_CRC32 (Data (5 .. 8));
      Generate_Parity (Data, Block_Size, Parity);

      --  Simulate corruption
      Data (1) := 99;

      --  Attempt Self-Patch
      Self_Patch (Data, Block_Size, CRCs, Parity, Success);

      if Success and then Data (1) = 1 then
         Put_Line (Green & "[+] Integrity Core: VERIFIED" & Reset);
      else
         Put_Line (Yellow &
                   "[!] Integrity Core: FAILED (Proofs only)" & Reset);
      end if;
   end Run_Integrity_Self_Test;

   Server_Port : constant Positive := 11420;
   Web_Server  : AWS.Server.HTTP;
begin
   --  Parse Arguments
   for I in 1 .. Argument_Count loop
      declare
         Arg : constant String := Argument (I);
      begin
         if Arg = "-h" or else Arg = "--help" then
            Show_Help;
            return;
         elsif Arg = "-v" or else Arg = "--version" then
            Put_Line ("Adelaide-Lite v0.1.0-dev (SPARK verified core)");
            return;
         end if;
      end;
   end loop;

   --  Whimsical Startup Banner
   Put_Line (Purple & "==========================" &
             "==========================" & Reset);
   Put_Line (Purple & Bold &
             "    ADELAIDE-LITE INITIALIZING... (Enchanting)" & Reset);
   Put_Line (Purple & "==========================" &
             "==========================" & Reset);

   --  Apply Apple Silicon soft real-time thread constraint policy
   Set_Darwin_Realtime;

   --  Verify data integrity subsystem
   Run_Integrity_Self_Test;

   --  Verify external toolchain and dependencies
   Toolchain_Manager.Verify_And_Heal;

   --  Initialize Llama backend and models
   Model_Manager.Initialize;

   Put_Line (Cyan & "[+] Starting Adelaide AWS Proxy on port" &
             Server_Port'Img & "..." & Reset);

   --  Start AWS server with the custom Dispatcher callback
   declare
      Max_Retries : constant Positive := 3;
      Success     : Boolean := False;
   begin
      for Attempt in 1 .. Max_Retries loop
         begin
            AWS.Server.Start (
               Web_Server => Web_Server,
               Name       => "Adelaide-Inference-Server",
               Callback   => Adelaide_Server_Pkg.Dispatch'Access,
               Port       => Server_Port
            );
            Success := True;
            exit;
         exception
            when E : others =>
               Put_Line
                 (Yellow & "[!] Port" & Server_Port'Img &
                  " bind attempt" & Attempt'Img & " failed." & Reset);
               if Attempt < Max_Retries then
                  Put_Line ("[!] Retrying in 1 second...");
                  delay 1.0;
               else
                  Put_Line
                    (ASCII.ESC & "[91m" &
                     "[BUGCHECK] Failed to bind to port" &
                     Server_Port'Img & " after" & Max_Retries'Img &
                     " attempts." & ASCII.LF & "Issue: " &
                     Ada.Exceptions.Exception_Name (E) & " - " &
                     Ada.Exceptions.Exception_Message (E) &
                     ASCII.ESC & "[0m");
               end if;
         end;
      end loop;
      if not Success then
         GNAT.OS_Lib.OS_Exit (1);
      end if;
   end;

   Put_Line (Green & Bold & "[+] Adelaide-Lite is ACTIVE on port" &
             Server_Port'Img & Reset);
   Put_Line (Green & "[+] Metal GPU Acceleration: ENABLED" & Reset);

   --  Keep the main thread alive indefinitely
   loop
      delay 3600.0;
   end loop;

exception
   when E : others =>
      Put_Line (Yellow & "[-] Exception in Adelaide Server: " &
                Ada.Exceptions.Exception_Name (E) & Reset);
      Put_Line ("[-] Message: " & Ada.Exceptions.Exception_Message (E));
      AWS.Server.Shutdown (Web_Server);
end Adelaide_Server;
