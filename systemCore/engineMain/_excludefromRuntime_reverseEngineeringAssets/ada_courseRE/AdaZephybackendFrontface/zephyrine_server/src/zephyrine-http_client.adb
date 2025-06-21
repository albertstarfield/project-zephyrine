-- File: src/zephyrine-http_client.adb (DEFINITIVE, WITH DETAILED CONNECT DEBUGGING)

with Ada.Strings.Fixed;
with Interfaces.C;
with Interfaces.C.Strings;
with System;
with Ada.Unchecked_Conversion;
with Ada.Exceptions;
with Ada.Text_IO;

with Zephyrine.C_Sockets;

package body Zephyrine.HTTP_Client is

   use type Zephyrine.C_Sockets.Socket_FD;
   use type Interfaces.C.long;
   use type Interfaces.C.int;
   use type System.Address;

   function Create_Request (URL : String; Payload : String) return String is
      Host_Start          : constant Natural := Ada.Strings.Fixed.Index(URL, "://") + 3;
      Search_Slice        : constant String := URL(Host_Start .. URL'Last);
      Path_Index_In_Slice : constant Natural := Ada.Strings.Fixed.Index(Search_Slice, "/");
   begin
      if Path_Index_In_Slice = 0 then
         declare Host : constant String := Search_Slice; Path : constant String := "/";
         begin return "POST " & Path & " HTTP/1.1" & ASCII.CR & ASCII.LF & "Host: " & Host & ASCII.CR & ASCII.LF & "Content-Type: application/json" & ASCII.CR & ASCII.LF & "Content-Length:" & Payload'Length'Image & ASCII.CR & ASCII.LF & "Connection: close" & ASCII.CR & ASCII.LF & ASCII.CR & ASCII.LF & Payload; end;
      else
         declare Host : constant String := Search_Slice(Search_Slice'First .. Path_Index_In_Slice - 1); Path : constant String := Search_Slice(Path_Index_In_Slice .. Search_Slice'Last);
         begin return "POST " & Path & " HTTP/1.1" & ASCII.CR & ASCII.LF & "Host: " & Host & ASCII.CR & ASCII.LF & "Content-Type: application/json" & ASCII.CR & ASCII.LF & "Content-Length:" & Payload'Length'Image & ASCII.CR & ASCII.LF & "Connection: close" & ASCII.CR & ASCII.LF & ASCII.CR & ASCII.LF & Payload; end;
      end if;
   end Create_Request;

   function Post (URL : String; Payload : String) return String is
      Host_Start : constant Natural := Ada.Strings.Fixed.Index(URL, "://") + 3;
      Search_Slice : constant String := URL(Host_Start .. URL'Last);
      Path_Index_In_Slice : constant Natural := Ada.Strings.Fixed.Index(Search_Slice, "/");
      Host : constant String := (if Path_Index_In_Slice = 0 then Search_Slice else Search_Slice(Search_Slice'First .. Path_Index_In_Slice - 1));
      Port : constant String := "80";

      Request  : constant String := Create_Request (URL, Payload);
      Sock_FD  : C_Sockets.Socket_FD := -1;
      Response : String := "";
      Addr_Info_Results : System.Address := System.Null_Address;
      type Addr_Info_Access is access all C_Sockets.Addr_Info;
      function To_Addr_Info_Ptr is new Ada.Unchecked_Conversion(System.Address, Addr_Info_Access);
   begin
      begin
         -- (Optional Debugging: Uncomment to see the request being sent)
         -- Ada.Text_IO.Put_Line("--- HTTP Request ---");
         -- Ada.Text_IO.Put_Line(Request);
         -- Ada.Text_IO.Put_Line("--- End Request ---");

         -- Step 1: DNS Lookup
         declare
            C_Host : Interfaces.C.Strings.chars_ptr := Interfaces.C.Strings.New_String(Host);
            C_Port : Interfaces.C.Strings.chars_ptr := Interfaces.C.Strings.New_String(Port);
            Result : Interfaces.C.int;
         begin
            Result := C_Sockets.getaddrinfo(C_Host, C_Port, System.Null_Address, Addr_Info_Results);
            Interfaces.C.Strings.Free(C_Host);
            Interfaces.C.Strings.Free(C_Port);
            if Result /= 0 then raise HTTP_Error with "DNS lookup failed for " & Host; end if;
         end;

         -- Step 2 & 3: Iterate addresses and connect
         declare
            Current_Addr_Ptr : System.Address := Addr_Info_Results;
         begin
            while Current_Addr_Ptr /= System.Null_Address and Sock_FD < 0 loop
               declare
                  Current_Addr_Info : constant Addr_Info_Access := To_Addr_Info_Ptr(Current_Addr_Ptr);
                  Temp_FD : C_Sockets.Socket_FD;
               begin
                  -- Ada.Text_IO.Put_Line ("[DEBUG] Trying address (family " & Current_Addr_Info.Ai_Family'Image & ")...");
                  Temp_FD := C_Sockets.socket(Current_Addr_Info.Ai_Family, Current_Addr_Info.Ai_Socktype, Current_Addr_Info.Ai_Protocol);
                  if Temp_FD >= 0 then
                     if C_Sockets.connect(Temp_FD, Current_Addr_Info.Ai_Addr, Current_Addr_Info.Ai_Addrlen) = 0 then
                        -- SUCCESS! We have a working socket.
                        Sock_FD := Temp_FD;
                     else
                        -- <<< THE DEFINITIVE PORTABLE ERROR HANDLING >>>
                        declare
                           Err_Num : Interfaces.C.int;
                        begin
                           -- Use the correct function based on the OS
                           if OS = "windows" then
                              Err_Num := C_Sockets.Get_Errno;
                           else -- Linux and macOS use the pointer-based version
                              Err_Num := C_Sockets.Get_Errno_Addr.all;
                           end if;

                           declare
                              Err_Msg_Ptr : constant Interfaces.C.Strings.chars_ptr := C_Sockets.strerror(Err_Num);
                              Err_Msg     : constant String := Interfaces.C.Strings.Value(Err_Msg_Ptr);
                           begin
                              -- Ada.Text_IO.Put_Line ("[DEBUG] connect() failed: " & Err_Msg);
                              null; -- Keep this block for debugging
                           end;
                        end;
                        -- End of error handling block
                        declare Dummy : Interfaces.C.int := C_Sockets.close(Temp_FD);
                        begin pragma Unreferenced(Dummy); end;
                        Sock_FD := -1; -- Reset to invalid state
                     end if;
                  end if;
               end;
               Current_Addr_Ptr := To_Addr_Info_Ptr(Current_Addr_Ptr).Ai_Next;
            end loop;
         end;

         if Sock_FD < 0 then
            raise HTTP_Error with "Failed to connect to " & Host & " after trying all addresses.";
         end if;

         -- Step 4: Send the request
         declare
            C_Request    : aliased Interfaces.C.char_array := Interfaces.C.To_C(Request);
            Request_Len  : constant Interfaces.C.size_t := Interfaces.C.size_t(Request'Length);
            Sent_Bytes   : Interfaces.C.long := C_Sockets.send(Sock_FD, C_Request'Address, Request_Len, 0);
         begin
            if Sent_Bytes < Interfaces.C.long(Request_Len) then raise HTTP_Error with "Failed to send full request"; end if;
         end;

         -- Step 5: Receive the response
         loop
            declare
               Buffer           : aliased Interfaces.C.char_array(0..1023);
               Bytes_Read       : constant Interfaces.C.long := C_Sockets.recv(Sock_FD, Buffer'Address, Buffer'Length, 0);
               Num_Bytes_Read   : constant Natural := Natural(Bytes_Read);
            begin
               if Bytes_Read <= 0 then exit; end if;
               declare
                  Slice_Upper_Bound : constant Interfaces.C.size_t := Interfaces.C.size_t(Num_Bytes_Read - 1);
                  Chunk             : constant String := Interfaces.C.To_Ada(Buffer(0 .. Slice_Upper_Bound));
               begin
                  Response := Response & Chunk;
               end;
            end;
         end loop;
      exception
         when E : others =>
            if Addr_Info_Results /= System.Null_Address then C_Sockets.freeaddrinfo(Addr_Info_Results); end if;
            if Sock_FD >= 0 then declare Dummy : Interfaces.C.int := C_Sockets.close(Sock_FD); begin pragma Unreferenced(Dummy); end; end if;
            raise HTTP_Error with "Connection or communication failed: " & Ada.Exceptions.Exception_Message(E);
      end;
      if Addr_Info_Results /= System.Null_Address then C_Sockets.freeaddrinfo(Addr_Info_Results); end if;
      if Sock_FD >= 0 then declare Dummy : Interfaces.C.int := C_Sockets.close(Sock_FD); begin pragma Unreferenced(Dummy); end; end if;
      return Response;
   end Post;
end Zephyrine.HTTP_Client;