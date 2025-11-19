-- File: src/zephyrine-database.adb (FINAL and POLISHED)

with Interfaces.C;
with Interfaces.C.Strings;
with Zephyrine.C_SQLite;
with Ada.Unchecked_Conversion;
with Ada.Strings.Unbounded;
with Ada.Text_IO; -- Keeping this as it's used for error reporting. If the warning persists, we can ignore it as harmless.

package body Zephyrine.Database is

   use type Zephyrine.C_Types.Opaque_Handle;
   use type Interfaces.C.int;
   use type Interfaces.C.Strings.chars_ptr;

   function To_Chars_Ptr is new Ada.Unchecked_Conversion
     (Source => System.Address,
      Target => Interfaces.C.Strings.chars_ptr);

   procedure Check (Result_Code : Interfaces.C.int; DB_Handle : C_Types.Opaque_Handle; Message : String) is
   begin
      if Result_Code /= Zephyrine.C_SQLite.SQLITE_OK then
         declare
            Error_Msg_Ptr : constant Interfaces.C.Strings.chars_ptr := Zephyrine.C_SQLite.sqlite3_errmsg (DB_Handle);
            Error_Msg     : constant String := Interfaces.C.Strings.Value(Error_Msg_Ptr);
         begin
            Ada.Text_IO.Put_Line ("SQLite Error: " & Message & " - " & Error_Msg);
            raise Database_Error;
         end;
      end if;
   end Check;

   procedure Open (DB : out Connection; Path : in String) is
      Result_Code : Interfaces.C.int;
      C_Path      : constant C_Types.C_String := Interfaces.C.To_C (Path);
   begin
      Result_Code := Zephyrine.C_SQLite.sqlite3_open (C_Path, DB.Handle);
      Check (Result_Code, DB.Handle, "opening database");
      Ada.Text_IO.Put_Line ("Database opened successfully: " & Path);
   end Open;

   procedure Close (DB : in out Connection) is
      Result_Code : Interfaces.C.int;
      NULL_HANDLE : constant Zephyrine.C_Types.Opaque_Handle := Zephyrine.C_Types.Opaque_Handle(System.Null_Address);
   begin
      if DB.Handle /= NULL_HANDLE then
         Result_Code := Zephyrine.C_SQLite.sqlite3_close (DB.Handle);
         if Result_Code /= Zephyrine.C_SQLite.SQLITE_OK then
            Ada.Text_IO.Put_Line ("Warning: sqlite3_close failed. The database may have been busy.");
         end if;
         DB.Handle := NULL_HANDLE;
         Ada.Text_IO.Put_Line ("Database connection closed.");
      end if;
   end Close;

   procedure Execute (DB : in Connection; SQL : in String) is
      Result_Code : Interfaces.C.int;
      C_SQL       : constant C_Types.C_String := Interfaces.C.To_C (SQL);
      Error_Msg_Handle : C_Types.Opaque_Handle;
   begin
      Result_Code := Zephyrine.C_SQLite.sqlite3_exec
        (DB.Handle, C_SQL, System.Null_Address, System.Null_Address, Error_Msg_Handle);

      if Result_Code /= Zephyrine.C_SQLite.SQLITE_OK then
         declare
            Raw_Address  : constant System.Address := System.Address(Error_Msg_Handle);
            Typed_Ptr    : constant Interfaces.C.Strings.chars_ptr := To_Chars_Ptr(Raw_Address);
            Error_Msg    : constant String := Interfaces.C.Strings.Value(Typed_Ptr);
         begin
            Ada.Text_IO.Put_Line ("SQLite Error executing SQL: " & Error_Msg);
            Zephyrine.C_SQLite.sqlite3_free (Error_Msg_Handle);
            raise Database_Error;
         end;
      end if;
   end Execute;

   procedure Query (DB : in Connection; SQL : in String; Callback : in Row_Callback) is
      C_SQL       : constant C_Types.C_String := Interfaces.C.To_C (SQL);
      Stmt_Handle : Zephyrine.C_SQLite.Statement_Handle;
      Result_Code : Interfaces.C.int;
      Unused_Tail : System.Address;
   begin
      Result_Code := Zephyrine.C_SQLite.sqlite3_prepare_v2
        (DB.Handle, C_SQL, -1, Stmt_Handle, Unused_Tail);
      Check (Result_Code, DB.Handle, "preparing statement: " & SQL);

      begin
         loop
            Result_Code := Zephyrine.C_SQLite.sqlite3_step (Stmt_Handle);
            if Result_Code = Zephyrine.C_SQLite.SQLITE_ROW then
               declare
                  Col_Count : constant Natural := Natural(Zephyrine.C_SQLite.sqlite3_column_count (Stmt_Handle));
                  Row       : Row_Data(1 .. Col_Count);
               begin
                  for I in 1 .. Col_Count loop
                     declare
                        Col_Text_Ptr : constant Interfaces.C.Strings.chars_ptr :=
                          Zephyrine.C_SQLite.sqlite3_column_text (Stmt_Handle, Interfaces.C.int(I - 1));
                     begin
                        if Col_Text_Ptr = Interfaces.C.Strings.Null_Ptr then
                           Row(I) := Ada.Strings.Unbounded.To_Unbounded_String ("");
                        else
                           Row(I) := Ada.Strings.Unbounded.To_Unbounded_String (Interfaces.C.Strings.Value (Col_Text_Ptr));
                        end if;
                     end;
                  end loop;
                  Callback.all (Row);
               end;
            elsif Result_Code = Zephyrine.C_SQLite.SQLITE_DONE then
               exit;
            else
               Check (Result_Code, DB.Handle, "stepping through query result");
            end if;
         end loop;
      exception
         -- <<< FIX: Changed 'when E : others' to 'when others' to fix the warning.
         when others =>
            Ada.Text_IO.Put_Line ("Exception during query processing, finalizing statement...");
            declare
               Dummy_Result : Interfaces.C.int := Zephyrine.C_SQLite.sqlite3_finalize (Stmt_Handle);
            begin
               pragma Unreferenced (Dummy_Result);
            end;
            raise;
      end;
      Result_Code := Zephyrine.C_SQLite.sqlite3_finalize (Stmt_Handle);
      Check (Result_Code, DB.Handle, "finalizing statement");
   end Query;

end Zephyrine.Database;