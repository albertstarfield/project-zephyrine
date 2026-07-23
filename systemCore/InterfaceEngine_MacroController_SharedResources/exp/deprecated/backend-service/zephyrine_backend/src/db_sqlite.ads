-- The "manual transmission": A safe Ada interface to the libsqlite3 C library.
with Interfaces.C;
with Interfaces.C.Strings;

package DB_SQLite is

   -- Opaque type hiding the C pointer to the database connection.
   type Database_Connection is private;
   Null_Connection : constant Database_Connection;

   -- Public API for database operations
   function Open (Path : String) return Database_Connection;
   procedure Close (Conn : in out Database_Connection);
   procedure Exec (Conn : Database_Connection; SQL : String);

   -- TODO: Add functions for querying data (e.g., Get_Messages)

private
   type DB_Handle is new System.Address;
   type Database_Connection is record
      Handle : DB_Handle := System.Null_Address;
   end record;
   Null_Connection : constant Database_Connection := (Handle => System.Null_Address);
end DB_SQLite;