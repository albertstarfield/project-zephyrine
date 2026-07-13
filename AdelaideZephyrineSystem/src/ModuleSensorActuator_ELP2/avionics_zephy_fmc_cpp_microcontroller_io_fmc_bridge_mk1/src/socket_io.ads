with Interfaces.C;
with Interfaces.C.Strings;
with Ada.Streams;

-- This package provides a safe, high-level Ada wrapper for Unix domain socket
-- communication with error detection.
package Socket_IO
  with SPARK_Mode => Off -- This package contains C bindings and is not provable.
is
   -- A new type to represent a socket file descriptor
   type Socket_FD is private;
   -- A constant representing an invalid or unopened socket
   Null_Socket : constant Socket_FD;
   
   -- Creates a Unix domain socket at the specified path
   -- Returns a valid Socket_FD on success or Null_Socket on failure
   function Create_Socket (Socket_Path : String) return Socket_FD;
   
   -- Connects to an existing Unix domain socket
   -- Returns a valid Socket_FD on success or Null_Socket on failure
   function Connect_Socket (Socket_Path : String) return Socket_FD;
   
   -- Closes the socket and releases the file descriptor
   procedure Close_Socket (Socket : in out Socket_FD);
   
   -- Writes a data buffer to the socket
   -- Returns the number of bytes successfully written
   function Write (Socket : Socket_FD; Data : Ada.Streams.Stream_Element_Array) return Integer;
   
   -- Reads data from the socket into a buffer
   -- Returns the number of bytes successfully read
   function Read (Socket : Socket_FD; Data : out Ada.Streams.Stream_Element_Array) return Integer;
   
   -- Maximum size of a single message
   Max_Message_Size : constant := 16;
   
private
   type Socket_FD is new Interfaces.C.int;
   Null_Socket : constant Socket_FD := -1;
end Socket_IO;
