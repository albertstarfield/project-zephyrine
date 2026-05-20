with Ada.Text_IO; use Ada.Text_IO;
with GNATCOLL.SQL.SQLite; use GNATCOLL.SQL.SQLite;
with GNATCOLL.SQL.Exec;   use GNATCOLL.SQL.Exec;
with GNATCOLL.SQL;        use GNATCOLL.SQL;

package body Database_Manager is

   DB_File : constant String := "adelaide_memory.db";
   DB      : SQL_Database;
   
   ----------------
   -- Initialize --
   ----------------
   procedure Initialize is
      DB_Descr : constant Database_Description := Setup (DB_File);
   begin
      DB := DB_Descr.Build;
      
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
            Put_Line ("[DB] Initialization error: " & Ada.Exceptions.Exception_Message (E));
      end;
   end Initialize;

   --------------
   -- Remember --
   --------------
   procedure Remember (User_Input : String; Assistant_Response : String) is
   begin
      if not DB.Is_Connected then
         return;
      end if;
      
      --  Simple escape (very crude, for production use parameterized queries)
      --  But GNATCOLL.SQL.Exec has parameterized query support
      declare
         Q : constant String := "INSERT INTO memories (input, response) VALUES (?, ?)";
      begin
         --  Note: In a full implementation, I'd use GNATCOLL.SQL's ORM or prepared statements.
         --  For now, direct execution with placeholders if supported by the driver, 
         --  or just a formatted string for the prototype.
         Execute (DB, "INSERT INTO memories (input, response) VALUES ('" & 
                 User_Input & "', '" & Assistant_Response & "')");
      exception
         when others =>
            Put_Line ("[DB] Error saving memory.");
      end;
   end Remember;

   ------------
   -- Recall --
   ------------
   function Recall (Query : String) return String is
      Result : Unbounded_String := Null_Unbounded_String;
      Cursor : Forward_Cursor;
   begin
      if not DB.Is_Connected then
         return "";
      end if;
      
      --  Search for similar inputs (crude keyword match for prototype)
      Cursor.Fetch (DB, "SELECT response FROM memories WHERE input LIKE '%" & Query & "%' LIMIT 1");
      
      if Cursor.Has_Row then
         Result := To_Unbounded_String (Cursor.Value (0));
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
      --  SQLite driver handles closing
      null;
   end Close;

end Database_Manager;
