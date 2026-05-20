with Ada.Text_IO; use Ada.Text_IO;
with GNATCOLL.SQL.SQLite;
with GNATCOLL.SQL.Exec;   use GNATCOLL.SQL.Exec;
with Ada.Exceptions;

package body Database_Manager is

   DB_File : constant String := "adelaide_memory.db";
   DB      : Database_Connection;

   ----------------
   -- Initialize --
   ----------------
   procedure Initialize is
      --  Correctly use Database_Description and Build_Connection
      Description : constant Database_Description :=
        GNATCOLL.SQL.SQLite.Setup (DB_File);
   begin
      DB := Description.Build_Connection;

      --  Create table if not exists
      begin
         Execute (DB, "CREATE TABLE IF NOT EXISTS memories (" &
                     "id INTEGER PRIMARY KEY AUTOINCREMENT," &
                     "input TEXT," &
                     "response TEXT," &
                     "timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)");
         Put_Line ("[DB] SQLite memory database initialized.");
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
      --  Note: In a full implementation, I'd use parameterized queries.
      --  For now, direct execution for the prototype.
      Execute (DB, "INSERT INTO memories (input, response) VALUES ('" &
              User_Input & "', '" & Assistant_Response & "')");
   exception
      when others =>
         Put_Line ("[DB] Error saving memory.");
   end Remember;

   ------------
   -- Recall --
   ------------
   function Recall (Query : String) return String is
      Result : Unbounded_String := Null_Unbounded_String;
      Cursor : Forward_Cursor;
   begin
      --  Search for similar inputs (crude keyword match for prototype)
      Cursor.Fetch (DB, "SELECT response FROM memories WHERE input LIKE '%" &
                   Query & "%' LIMIT 1");

      if Has_Row (Cursor) then
         Result := To_Unbounded_String (Value (Cursor, 0));
      end if;

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
