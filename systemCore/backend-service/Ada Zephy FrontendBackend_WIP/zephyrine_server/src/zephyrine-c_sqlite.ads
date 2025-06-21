-- File: src/zephyrine-c_sqlite.ads (Complete and Corrected)

-- This package contains the raw, unsafe bindings to libsqlite3.
-- No logic should be in here, only C imports.

with System;
with Interfaces.C;
with Interfaces.C.Strings; -- Provides the correct 'chars_ptr' type
with Zephyrine.C_Types;

package Zephyrine.C_SQLite is

   -- We now need a distinct handle for a prepared statement, which is also
   -- an opaque pointer from C.
   type Statement_Handle is new C_Types.Opaque_Handle;

   -- ===================================================================
   -- Core Database Connection Functions
   -- ===================================================================
   function sqlite3_open (Filename : C_Types.C_String; DB_Handle : out C_Types.Opaque_Handle) return Interfaces.C.int;
   pragma Import (C, sqlite3_open);

   function sqlite3_close (DB_Handle : C_Types.Opaque_Handle) return Interfaces.C.int;
   pragma Import (C, sqlite3_close);

   procedure sqlite3_free (Ptr : C_Types.Opaque_Handle);
   pragma Import (C, sqlite3_free);

   function sqlite3_errmsg (DB_Handle : C_Types.Opaque_Handle) return Interfaces.C.Strings.chars_ptr;
   pragma Import (C, sqlite3_errmsg);

   -- Simple "fire-and-forget" execution function
   function sqlite3_exec
     (DB_Handle    : C_Types.Opaque_Handle;
      SQL          : C_Types.C_String;
      Callback     : System.Address;
      Callback_Arg : System.Address;
      Errmsg       : out C_Types.Opaque_Handle)
      return Interfaces.C.int;
   pragma Import (C, sqlite3_exec);

   -- ===================================================================
   -- Functions for the "Prepare -> Step -> Finalize" Query Cycle
   -- ===================================================================

   -- Compiles an SQL statement into a ready-to-run bytecode program
   function sqlite3_prepare_v2
     (DB_Handle   : C_Types.Opaque_Handle;
      SQL         : C_Types.C_String;
      Num_Bytes   : Interfaces.C.int;   -- Use -1 to read the whole C string
      Stmt_Handle : out Statement_Handle;
      Tail        : out System.Address) -- Pointer to unused portion of SQL
      return Interfaces.C.int;
   pragma Import (C, sqlite3_prepare_v2);

   -- Executes a prepared statement one step (one row) at a time
   function sqlite3_step (Stmt_Handle : Statement_Handle) return Interfaces.C.int;
   pragma Import (C, sqlite3_step);

   -- Destroys a prepared statement to free resources
   function sqlite3_finalize (Stmt_Handle : Statement_Handle) return Interfaces.C.int;
   pragma Import (C, sqlite3_finalize);

   -- Gets the number of columns in the result set of a prepared statement
   function sqlite3_column_count (Stmt_Handle : Statement_Handle) return Interfaces.C.int;
   pragma Import (C, sqlite3_column_count);

   -- Gets the data from a specific column of the current result row as text
   function sqlite3_column_text (Stmt_Handle : Statement_Handle; Column_Index : Interfaces.C.int) return Interfaces.C.Strings.chars_ptr;
   pragma Import (C, sqlite3_column_text);


   -- ===================================================================
   -- Common SQLite C API Result Codes
   -- ===================================================================
   SQLITE_OK   : constant Interfaces.C.int := 0;
   SQLITE_ROW  : constant Interfaces.C.int := 100; -- A row of data is ready
   SQLITE_DONE : constant Interfaces.C.int := 101; -- The statement has finished executing

end Zephyrine.C_SQLite;