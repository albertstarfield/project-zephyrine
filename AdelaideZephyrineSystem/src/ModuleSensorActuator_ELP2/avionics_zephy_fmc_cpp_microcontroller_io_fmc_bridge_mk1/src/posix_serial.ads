with Interfaces.C;
-- This package provides a safe, high-level Ada wrapper for low-level
-- POSIX serial port (termios) functions.
package POSIX_Serial
  with SPARK_Mode => Off -- This package contains C bindings and is not provable.
is
   -- A new type to represent a file descriptor for the serial port.
   -- We hide the implementation detail (it's just an integer).
   type Serial_Port is private;
   -- A constant representing an invalid or unopened port.
   Null_Port : constant Serial_Port;
   -- Add this function declaration to make the '=' operator visible
   function "=" (Left, Right : Serial_Port) return Boolean with Pre => True, Post => True;
   -- Opens the specified serial device (e.g., "/dev/ttyUSB0").
   -- Returns a valid Serial_Port on success or Null_Port on failure.
   function Open_Port (Device_Path : String) return Serial_Port with Pre => True, Post => True;
   -- Configures the given port with a specific baud rate and settings.
   -- Returns True on success.
   function Configure_Port (Port : Serial_Port; Baud_Rate : Natural) return Boolean with Pre => True, Post => True;
   -- Writes a string message to the serial port.
   -- Returns the number of bytes successfully written.
   function Write (Port : Serial_Port; Message : String) return Integer with Pre => True, Post => True;
   -- Closes the serial port and releases the file descriptor.
   procedure Close (Port : in out Serial_Port) with Pre => True, Post => True;
private
   type Serial_Port is new Interfaces.C.int;
   Null_Port : constant Serial_Port := -1;
end POSIX_Serial;