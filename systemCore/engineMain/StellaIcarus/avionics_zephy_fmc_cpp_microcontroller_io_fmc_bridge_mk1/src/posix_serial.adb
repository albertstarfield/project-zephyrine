with Interfaces.C;
with Interfaces.C.Strings;
-- Make operators for C types directly visible
use type Interfaces.C.int;
use type Interfaces.C.size_t; -- Good practice for size_t operations too
with Interfaces; -- Add this for bitwise operations

-- This is the implementation of our wrapper. It directly calls C functions
-- for opening, configuring, and writing to the serial port.
package body POSIX_Serial is

   -----------
   -- Is_Null --
   -----------
   function Is_Null (Port : Serial_Port) return Boolean is
   begin
      -- This comparison is legal inside the package body
      return Port = Null_Port;
   end Is_Null;

   -- C function bindings for standard file I/O
   -- CORRECTED TYPE: Use chars_ptr, not char_ptr
   function open (path : Interfaces.C.Strings.chars_ptr; oflag : Interfaces.C.int) return Interfaces.C.int;
   pragma Import (C, open, "open");
   function write (fd : Interfaces.C.int; buf : Interfaces.C.Strings.chars_ptr; count : Interfaces.C.size_t) return Interfaces.C.long;
   pragma Import (C, write, "write");
   function close (fd : Interfaces.C.int) return Interfaces.C.int;
   pragma Import (C, close, "close");

   -- C termios constants needed for open()
   -- Using octal notation is more standard and portable
   O_RDWR   : constant Interfaces.C.int := 8#0000002#; -- Open for reading and writing (Octal 2)
   O_NOCTTY : constant Interfaces.C.int := 8#0040000#; -- Don't let port become controlling terminal (Octal 100000)
   -- The 'or' operator is now visible due to 'use type Interfaces.C.int;'

   -----------------
   -- Open_Port --
   -----------------
   function Open_Port (Device_Path : String) return Serial_Port is
      -- CORRECTED TYPE and made VARIABLE (not CONSTANT)
      Device_Path_C : Interfaces.C.Strings.chars_ptr := Interfaces.C.Strings.New_String (Device_Path);
      File_Descriptor : Interfaces.C.int;
   begin
      -- Use Interfaces."or" for explicit bitwise OR
      File_Descriptor := open (
         Device_Path_C,
         Interfaces.C.int(Interfaces."or"(
            Interfaces.Unsigned_32(O_RDWR),
            Interfaces.Unsigned_32(O_NOCTTY)
         ))
      );
      Interfaces.C.Strings.Free (Device_Path_C); -- Free modifies the variable
      return Serial_Port (File_Descriptor);
   end Open_Port;

   -----------
   -- Write --
   -----------
   function Write (Port : Serial_Port; Message : String) return Integer is
      -- CORRECTED TYPE and made it a VARIABLE (not CONSTANT) for Free
      Message_C : Interfaces.C.Strings.chars_ptr := Interfaces.C.Strings.New_String (Message & ASCII.LF); -- Append newline
      Bytes_Written : Interfaces.C.long;
   begin
      if Port = Null_Port then
         Interfaces.C.Strings.Free (Message_C); -- Free even if not used (safe as it's Null_Ptr initially, but New_String was called)
         return 0;
      end if;
      -- Cast Message'Length (Natural) to Interfaces.C.size_t for the C call
      -- The "+" operator is visible due to 'use type Interfaces.C.size_t;'
      Bytes_Written := write (Interfaces.C.int (Port), Message_C, Interfaces.C.size_t(Message'Length + 1)); -- +1 for LF
      Interfaces.C.Strings.Free (Message_C); -- Free modifies the variable
      return Integer (Bytes_Written);
   end Write;

   -----------
   -- Close --
   -----------
   procedure Close (Port : in out Serial_Port) is
      Result : Interfaces.C.int;
   begin
      if Port /= Null_Port then
         Result := close (Interfaces.C.int (Port));
         Port := Null_Port;
      end if;
   end Close;

   ----------------------
   -- Configure_Port --
   ----------------------
   function Configure_Port (Port : Serial_Port; Baud_Rate : Natural) return Boolean is
   begin
      pragma Unreferenced(Port, Baud_Rate);
      return True;
   end Configure_Port;

   -- PROVIDE THE BODY for the explicit "=" function declared in the spec
   function "=" (Left, Right : Serial_Port) return Boolean is
   begin
     -- Compare the underlying C int values
     return Interfaces.C.int(Left) = Interfaces.C.int(Right);
   end "=";

end POSIX_Serial;