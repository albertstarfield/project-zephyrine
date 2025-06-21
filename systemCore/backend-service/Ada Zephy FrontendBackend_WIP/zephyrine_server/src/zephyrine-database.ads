private with System;
private with Zephyrine.C_Types;
with Ada.Strings.Unbounded; -- <<< FIX: Added for the Unbounded_String type

package Zephyrine.Database is
   type Connection is limited private;
   Database_Error : exception;

   procedure Open (DB : out Connection; Path : in String);
   procedure Close (DB : in out Connection);
   procedure Execute (DB : in Connection; SQL : in String);

   -- --- NEW: API for handling queries that return data ---

   -- <<< FIX: Changed the element type from 'String' to 'Unbounded_String'
   type Row_Data is array (Positive range <>) of Ada.Strings.Unbounded.Unbounded_String;

   type Row_Callback is access procedure (Row : in Row_Data);

   procedure Query (DB : in Connection; SQL : in String; Callback : in Row_Callback);

private
   type Connection is limited record
      Handle : C_Types.Opaque_Handle := C_Types.Opaque_Handle(System.Null_Address);
   end record;
end Zephyrine.Database;