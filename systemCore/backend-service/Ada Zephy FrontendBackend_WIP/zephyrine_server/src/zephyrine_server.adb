-- File: src/zephyrine_server.adb (DEFINITIVE AND SYNTACTICALLY CORRECT)

with Ada.Text_IO;
with Zephyrine.HTTP_Client;
with Ada.Exceptions; -- Added for the top-level exception handler

-- <<< THE FIX: The entire file content is now correctly wrapped in the main procedure.
procedure Zephyrine_Server is
begin
   Ada.Text_IO.Put_Line ("--- Zephyrine Server (Manual Transmission Edition) ---");
   Ada.Text_IO.Put_Line ("--- Step 2: Testing Pure Ada HTTP Client Layer ---");

   -- This inner block is good practice, as it isolates the HTTP test.
   -- If it fails, the program can continue (or in this case, end gracefully).
   begin
      declare
         -- NOTE: We are now using a non-secure http endpoint for the test.
         Response : constant String := Zephyrine.HTTP_Client.Post
           (URL     => "http://httpbin.org/post",
            Payload => "{""message"": ""Hello from pure Ada sockets!""}");
      begin
         Ada.Text_IO.Put_Line ("HTTP POST Request Successful!");
         Ada.Text_IO.Put_Line ("Server Response:");
         Ada.Text_IO.Put_Line (Response);
      end;
   exception
      when Zephyrine.HTTP_Client.HTTP_Error =>
         Ada.Text_IO.Put_Line ("HTTP POST Request Failed.");
   end;

exception
   -- A top-level handler for any unexpected errors.
   when E : others =>
      Ada.Text_IO.Put_Line ("A totally unexpected error occurred: " & Ada.Exceptions.Exception_Name(E));
      raise; -- Re-raise the exception to get a non-zero exit code.

end Zephyrine_Server;