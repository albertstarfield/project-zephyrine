with Ada.Text_IO; use Ada.Text_IO;
with Ada_Sqlite3; use Ada_Sqlite3;
with Ada.Exceptions;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;

package body Database_Manager is

   DB_File : constant String := "adelaide_memory.db";

   type DB_Access is access all Ada_Sqlite3.Database;
   Main_DB_Ptr : DB_Access := null;

   ----------------
   -- Initialize --
   ----------------
   procedure Initialize is
   begin
      if Main_DB_Ptr /= null then
         return;
      end if;

      Main_DB_Ptr := new Ada_Sqlite3.Database'(Open (DB_File));

      --  Create table if not exists
      begin
         Execute (Main_DB_Ptr.all,
                  "CREATE TABLE IF NOT EXISTS memories (" &
                  "id INTEGER PRIMARY KEY AUTOINCREMENT," &
                  "input TEXT," &
                  "response TEXT," &
                  "timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)");
         Put_Line ("[DB] SQLite memory database (ada_sqlite3) initialized.");
      exception
         when E : others =>
            Put_Line ("[DB] Initialization error: " &
                      Ada.Exceptions.Exception_Name (E) & ": " &
                      Ada.Exceptions.Exception_Message (E));
      end;
   end Initialize;

   --------------
   -- Remember --
   --------------
   procedure Remember (User_Input : String; Assistant_Response : String) is
   begin
      if Main_DB_Ptr = null then
         return;
      end if;

      declare
         Stmt : Statement := Prepare
           (Main_DB_Ptr.all,
            "INSERT INTO memories (input, response) VALUES (?, ?)");
      begin
         Bind_Text (Stmt, 1, User_Input);
         Bind_Text (Stmt, 2, Assistant_Response);
         Step (Stmt);
      end;
   exception
      when others =>
         Put_Line ("[DB] Error saving memory.");
   end Remember;

   ------------
   -- Recall --
   ------------
   function Recall (Query : String) return String is
      Result : Unbounded_String := Null_Unbounded_String;
   begin
      if Main_DB_Ptr = null then
         return "";
      end if;

      declare
         Stmt : Statement := Prepare
           (Main_DB_Ptr.all,
            "SELECT response FROM memories WHERE input LIKE ? LIMIT 1");
      begin
         Bind_Text (Stmt, 1, "%" & Query & "%");

         if Step (Stmt) = ROW then
            Result := To_Unbounded_String (Column_Text (Stmt, 0));
         end if;
      end;

      return To_String (Result);
   exception
      when others =>
         return "";
   end Recall;

   -----------
   -- Close --
   -----------
   procedure Close is
   begin
      null;
   end Close;

end Database_Manager;
