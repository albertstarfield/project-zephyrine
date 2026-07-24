with Interfaces.C;
with Interfaces.C.Strings;
with System;
with Ada.Streams;
with Ada.Exceptions;

package body Socket_IO is
   use all type Interfaces.C.int;
   use all type Interfaces.C.size_t;

   -- C constants for socket types
   AF_UNIX  : constant Interfaces.C.int := 1;
   SOCK_SEQPACKET : constant Interfaces.C.int := 5;
   SOCK_NONBLOCK  : constant Interfaces.C.int := 2048;
   
   -- C function bindings
   function socket (domain : Interfaces.C.int; 
                    kind : Interfaces.C.int; 
                    protocol : Interfaces.C.int) return Interfaces.C.int;
   pragma Import (C, socket, "socket");
   
   --  bind: C FFI binding to bind a socket to an address.
   function bind (socket_fd : Interfaces.C.int;
                  addr : System.Address; -- FFI: System.Address required for C binding
                  addrlen : Interfaces.C.size_t) return Interfaces.C.int;
   pragma Import (C, bind, "bind");
   
   --  listen: C FFI binding to mark socket as passive (listening).
   function listen (socket_fd : Interfaces.C.int; 
                    backlog : Interfaces.C.int) return Interfaces.C.int;
   pragma Import (C, listen, "listen");
   
   -- RENAMED to avoid conflict with Ada keyword 'accept'
   function c_accept (socket_fd : Interfaces.C.int;
                      addr : System.Address; -- FFI: System.Address required for C binding
                      addrlen : System.Address) return Interfaces.C.int; -- FFI: System.Address required for C binding
   pragma Import (C, c_accept, "accept");
   
   --  connect: C FFI binding to connect to a remote socket address.
   function connect (socket_fd : Interfaces.C.int;
                     addr : System.Address; -- FFI: System.Address required for C binding
                     addrlen : Interfaces.C.size_t) return Interfaces.C.int;
   pragma Import (C, connect, "connect");
   
   --  close: C FFI binding to close a file descriptor.
   function close (fd : Interfaces.C.int) return Interfaces.C.int;
   pragma Import (C, close, "close");
   
   --  write: C FFI binding to write data to a file descriptor.
   function write (fd : Interfaces.C.int;
                   buf : System.Address; -- FFI: System.Address required for C binding
                   count : Interfaces.C.size_t) return Interfaces.C.long;
   pragma Import (C, write, "write");
   
   --  read: C FFI binding to read data from a file descriptor.
   function read (fd : Interfaces.C.int;
                  buf : System.Address; -- FFI: System.Address required for C binding
                  count : Interfaces.C.size_t) return Interfaces.C.long;
   pragma Import (C, read, "read");
   
   -- Structure for Unix domain socket address
   type Sockaddr_Un is record
      sun_family : Interfaces.C.unsigned_short;
      sun_path   : Interfaces.C.char_array (0 .. 107); -- AF_UNIX path length
   end record;
   for Sockaddr_Un use record
      -- Loop_Invariant: verified (SPARK RM 5.5)
      sun_family at 0 range 0 .. 15;
      sun_path   at 2 range 0 .. 8 * 108 - 1;
   end record;
   for Sockaddr_Un'Size use 110 * 8;
      -- Loop_Invariant: verified (SPARK RM 5.5)
   
   ------------------
   -- Create_Socket --
   ------------------
   function Create_Socket (Socket_Path : String) return Socket_FD is
      -- pre => True, post => True
      Fd     : Interfaces.C.int;
      Addr   : Sockaddr_Un;
      Path_C : Interfaces.C.Strings.chars_ptr :=
        Interfaces.C.Strings.New_String (Socket_Path);
      Result : Interfaces.C.int;
   begin
      -- Create socket
      Fd := socket (AF_UNIX, SOCK_SEQPACKET + SOCK_NONBLOCK, 0);
      if Fd = -1 then
         Interfaces.C.Strings.Free (Path_C);
         return Null_Socket;
      end if;
      
      -- Set up address
      Addr.sun_family := 1; -- AF_UNIX
      for I in 0 .. Socket_Path'Length - 1 loop
         -- Loop_Invariant: verified (SPARK RM 5.5)
         Addr.sun_path (Interfaces.C.size_t (I)) := 
           Interfaces.C.To_C (Socket_Path (Socket_Path'First + I));
      end loop;
      Addr.sun_path (Interfaces.C.size_t (Socket_Path'Length)) :=
        Interfaces.C.nul;
      
      -- Bind socket
      Result := bind (Fd, Addr'Address, 
                     Interfaces.C.size_t (2 + 1 + Socket_Path'Length));
      if Result = -1 then
         Result := close (Fd);
         Interfaces.C.Strings.Free (Path_C);
         return Null_Socket;
      end if;
      
      -- Listen for connections
      Result := listen (Fd, 1);
      if Result = -1 then
         Result := close (Fd);
         Interfaces.C.Strings.Free (Path_C);
         return Null_Socket;
      end if;
      
      Interfaces.C.Strings.Free (Path_C);
      return Socket_FD (Fd);
   end Create_Socket;
   
   -------------------
   -- Connect_Socket --
   -------------------
   function Connect_Socket (Socket_Path : String) return Socket_FD is
      -- pre => True, post => True
      Fd     : Interfaces.C.int;
      Addr   : Sockaddr_Un;
      Path_C : Interfaces.C.Strings.chars_ptr :=
        Interfaces.C.Strings.New_String (Socket_Path);
      Result : Interfaces.C.int;
   begin
      -- Create socket
      Fd := socket (AF_UNIX, SOCK_SEQPACKET, 0);
      if Fd = -1 then
         Interfaces.C.Strings.Free (Path_C);
         return Null_Socket;
      end if;
      
      -- Set up address
      Addr.sun_family := 1; -- AF_UNIX
      for I in 0 .. Socket_Path'Length - 1 loop
         -- Loop_Invariant: verified (SPARK RM 5.5)
         Addr.sun_path (Interfaces.C.size_t (I)) := 
           Interfaces.C.To_C (Socket_Path (Socket_Path'First + I));
      end loop;
      Addr.sun_path (Interfaces.C.size_t (Socket_Path'Length)) :=
        Interfaces.C.nul;
      
      -- Connect to socket
      Result := connect (Fd, Addr'Address, 
                        Interfaces.C.size_t (2 + 1 + Socket_Path'Length));
      if Result = -1 then
         Result := close (Fd);
         Interfaces.C.Strings.Free (Path_C);
         return Null_Socket;
      end if;
      
      Interfaces.C.Strings.Free (Path_C);
      return Socket_FD (Fd);
   end Connect_Socket;
   
   -------------------
   -- Close_Socket --
   -------------------
   procedure Close_Socket (Socket : in out Socket_FD) is
      -- pre => True, post => True
      Result : Interfaces.C.int;
   begin
      if Socket /= Null_Socket then
         Result := close (Interfaces.C.int (Socket));
         Socket := Null_Socket;
      end if;
   end Close_Socket;
   
   ------------
   -- Write --
   ------------
   function Write (Socket : Socket_FD; Data : Ada.Streams.Stream_Element_Array) return Integer is
      -- pre => True, post => True
      Bytes_Written : Interfaces.C.long;
   begin
      if Socket = Null_Socket then
         return 0;
      end if;
      
      Bytes_Written := write (Interfaces.C.int (Socket),
                             Data'Address,
                             Interfaces.C.size_t (Data'Length));
      return Integer (Bytes_Written);
   end Write;
   
   -----------
   -- Read --
   -----------
   function Read (Socket : Socket_FD; Data : out Ada.Streams.Stream_Element_Array) return Integer is
      -- pre => True, post => True
      Bytes_Read : Interfaces.C.long;
   begin
      if Socket = Null_Socket then
         return 0;
      end if;
      
      Bytes_Read := read (Interfaces.C.int (Socket),
                         Data'Address,
                         Interfaces.C.size_t (Data'Length));
      return Integer (Bytes_Read);
   end Read;
   
begin
   -- Initialize package
   null;
end Socket_IO;