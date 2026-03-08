-- File: src/zephyrine-c_sockets.ads (DEFINITIVE AND CORRECT)

with Interfaces.C;
with Interfaces.C.Strings;
with System;

package Zephyrine.C_Sockets is
   -- (All the other declarations for socket, connect, etc. are unchanged and correct)
   AF_INET     : constant := 2;
   SOCK_STREAM : constant := 1;
   type Socket_FD is new Interfaces.C.int;
   type Addr_Info is record
      Ai_Flags     : Interfaces.C.int; Ai_Family    : Interfaces.C.int;
      Ai_Socktype  : Interfaces.C.int; Ai_Protocol  : Interfaces.C.int;
      Ai_Addrlen   : Interfaces.C.unsigned; Ai_Addr      : System.Address;
      Ai_Canonname : Interfaces.C.Strings.chars_ptr; Ai_Next      : System.Address;
   end record;
   pragma Convention (C, Addr_Info);
   function getaddrinfo (Nodename, Servname : Interfaces.C.Strings.chars_ptr; Hints : System.Address; Res : out System.Address) return Interfaces.C.int;
   pragma Import (C, getaddrinfo);
   procedure freeaddrinfo (Res : System.Address);
   pragma Import (C, freeaddrinfo);
   function socket (Domain, Typ, Protocol : Interfaces.C.int) return Socket_FD;
   pragma Import (C, socket);
   function connect (Sock_FD : Socket_FD; Addr : System.Address; Addrlen : Interfaces.C.unsigned) return Interfaces.C.int;
   pragma Import (C, connect);
   function send (Sock_FD : Socket_FD; Msg : System.Address; Len : Interfaces.C.size_t; Flags : Interfaces.C.int) return Interfaces.C.long;
   pragma Import (C, send);
   function recv (Sock_FD : Socket_FD; Buf : System.Address; Len : Interfaces.C.size_t; Flags : Interfaces.C.int) return Interfaces.C.long;
   pragma Import (C, recv);
   function close (Sock_FD : Socket_FD) return Interfaces.C.int;
   pragma Import (C, close);
   function strerror (Errnum : Interfaces.C.int) return Interfaces.C.Strings.chars_ptr;
   pragma Import (C, strerror);


   -- <<< THE PORTABILITY FIX IS IMPLEMENTED HERE >>>
   -- We declare the functions that might have different names or conventions.
   -- The actual 'pragma Import' is now handled by the config.pra file.

   -- This function will be imported as either __error or __errno_location
   function Get_Errno_Addr return access Interfaces.C.int;

   -- This function will be imported as WSAGetLastError on Windows
   function Get_Errno return Interfaces.C.int;

end Zephyrine.C_Sockets;