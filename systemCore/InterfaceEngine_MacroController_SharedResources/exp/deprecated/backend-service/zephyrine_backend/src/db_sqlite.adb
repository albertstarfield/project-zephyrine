-- The implementation of our SQLite C wrapper.
with Ada.Text_IO;
with System;

package body DB_SQLite is

   subtype C_Int is Interfaces.C.int;
   OK : constant C_Int := 0;

   -- Import the C functions we need from libsqlite3.
   -- This tells the Ada compiler to link against these C functions.
   function C_sqlite3_open (filename : Interfaces.C.Strings.chars_ptr; ppDb : out System.Address) return C_Int;
   pragma Import (C, C_sqlite3_open, "sqlite3_open");

   function C_sqlite3_close (pDb : System.Address) return C_Int;
   pragma Import (C, C_sqlite3_close, "sqlite3_close");

   function C_sqlite3_exec
     (pDb       : System.Address;
      sql       : Interfaces.C.Strings.chars_ptr;
      callback  : System.Address;
      arg       : System.Address;
      errmsg    : out System.Address) return C_Int;
   pragma Import (C, C_sqlite3_exec, "sqlite3_exec");

   function C_sqlite3_errmsg (pDb : System.Address) return Interfaces.C.Strings.chars_ptr;
   pragma Import (C, C_sqlite3_errmsg, "sqlite3_errmsg");

   -- Implementation of the safe Ada wrappers
   function Open (Path : String) return Database_Connection is
      DB_Ptr   : System.Address;
      C_Path   : constant Interfaces.C.Strings.chars_ptr := Interfaces.C.Strings.New_String (Path);
      Result   : C_Int;
      Ret_Conn : Database_Connection;
   begin
      Result := C_sqlite3_open (C_Path, DB_Ptr);
      Interfaces.C.Strings.Free (C_Path);
      if Result /= OK then
         Ada.Text_IO.Put_Line ("ERROR: Could not open database: " & Path);
         return Null_Connection;
      else
         Ret_Conn.Handle := DB_Ptr;
         return Ret_Conn;
      end if;
   end Open;

   procedure Close (Conn : in out Database_Connection) is
   begin
      if Conn /= Null_Connection then
         if C_sqlite3_close (Conn.Handle) /= OK then
            Ada.Text_IO.Put_Line ("WARNING: Error closing database.");
         end if;
         Conn := Null_Connection;
      end if;
   end Close;

   procedure Exec (Conn : Database_Connection; SQL : String) is
      C_SQL  : constant Interfaces.C.Strings.chars_ptr := Interfaces.C.Strings.New_String (SQL);
      Result : C_Int;
      ErrMsg_Ptr : System.Address;
   begin
      if Conn = Null_Connection then
         raise Program_Error with "Attempted to execute SQL on a null database connection";
      end if;
      Result := C_sqlite3_exec (Conn.Handle, C_SQL, System.Null_Address, System.Null_Address, ErrMsg_Ptr);
      Interfaces.C.Strings.Free (C_SQL);
      if Result /= OK then
         declare
            Error_Message : constant String := Interfaces.C.Strings.Value (C_sqlite3_errmsg (Conn.Handle));
         begin
            Ada.Text_IO.Put_Line ("SQL Error: " & Error_Message);
            raise Program_Error with "SQL Execution Failed: " & Error_Message;
         end;
      end if;
   end Exec;

end DB_SQLite;