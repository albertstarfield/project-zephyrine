pragma Ada_2022;
with Ada.Text_IO;
with Ada.Strings.Fixed;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with GNATCOLL.SQL.Exec;     use GNATCOLL.SQL.Exec;
with GNATCOLL.SQL.Sqlite;
with UUIDs;
with UUIDs.V4;

package body Database is

   -----------------------
   -- Helper: Get_Conn --
   -----------------------
   function Get_Conn return Database_Connection is
      Descr : constant Database_Description :=
        GNATCOLL.SQL.Sqlite.Setup ("frontface.db");
   begin
      return Descr.Build_Connection;
   end Get_Conn;

   -----------------
   -- Create_Chat --
   -----------------
   function Create_Chat (Title : String) return String is
      Conn : Database_Connection := Get_Conn;
      ID   : constant String := UUIDs.V4.UUID4'Image;
   begin
      Execute
        (Conn,
         "INSERT INTO chats (id, title) VALUES (?, ?)",
         Params => [1 => +ID, 2 => +Title]);
      Commit (Conn);
      Free (Conn);
      return ID;
   exception
      when others =>
         Free (Conn);
         raise;
   end Create_Chat;

   ------------------
   -- Save_Message --
   ------------------
   procedure Save_Message (Chat_ID : String; Role : String; Content : String) is
      Conn : Database_Connection := Get_Conn;
      ID   : constant String := UUIDs.V4.UUID4'Image;
   begin
      Execute
        (Conn,
         "INSERT INTO messages (id, chat_id, role, content) VALUES (?, ?, ?, ?)",
         Params =>
           [1 => +ID,
            2 => +Chat_ID,
            3 => +Role,
            4 => +Content]);

      Commit (Conn);
      Free (Conn);
   exception
      when others =>
         Free (Conn);
         raise;
   end Save_Message;

   ----------------------
   -- Get_Chat_History --
   ----------------------
   function Get_Chat_History (Chat_ID : String)
      return Models.Message_Vectors.Vector
   is
      Conn   : Database_Connection := Get_Conn;
      Result : Forward_Cursor;
      List   : Models.Message_Vectors.Vector;
      Item   : Models.Message;
   begin
      Fetch
        (Connection => Conn,
         Result     => Result,
         Query      => "SELECT id, chat_id, role, content, created_at " &
                       "FROM messages WHERE chat_id = ? ORDER BY created_at ASC",
         Params     => [1 => +Chat_ID]);

      while Has_Row (Result) loop
         Item.ID         := To_Unbounded_String (Result.Value (0));
         Item.Chat_ID    := To_Unbounded_String (Result.Value (1));
         Item.Role       := To_Unbounded_String (Result.Value (2));
         Item.Content    := To_Unbounded_String (Result.Value (3));
         Item.Created_At := To_Unbounded_String (Result.Value (4));
         List.Append (Item);
         Next (Result);
      end loop;
      Free (Conn);
      return List;
   exception
      when others =>
         Free (Conn);
         raise;
   end Get_Chat_History;

   -------------------
   -- Get_All_Chats --
   -------------------
   function Get_All_Chats return Models.Chat_Vectors.Vector is
      Conn   : Database_Connection := Get_Conn;
      Result : Forward_Cursor;
      List   : Models.Chat_Vectors.Vector;
      Item   : Models.Chat;
   begin
      Fetch
        (Connection => Conn,
         Result     => Result,
         Query      => "SELECT id, title, created_at FROM chats " &
                       "ORDER BY created_at DESC");
      while Has_Row (Result) loop
         Item.ID         := To_Unbounded_String (Result.Value (0));
         Item.Title      := To_Unbounded_String (Result.Value (1));
         Item.Created_At := To_Unbounded_String (Result.Value (2));
         List.Append (Item);
         Next (Result);
      end loop;

      Free (Conn);
      return List;
   exception
      when others =>
         Free (Conn);
         raise;
   end Get_All_Chats;

   -----------------------
   -- Store_File_Record --
   -----------------------
   procedure Store_File_Record 
     (Filename, Filetype, User_ID, LLM_ID, Status : String) is
      Conn : Database_Connection := Get_Conn;
      -- FIX: Trim the leading space from 'Image
      Raw_ID : constant String := UUIDs.V4.UUID4'Image;
      ID : constant String := Ada.Strings.Fixed.Trim (Raw_ID, Ada.Strings.Both);
   begin
      Execute
        (Conn,
         "INSERT INTO files (id, user_id, filename, filetype, llm_id, status) " &
         "VALUES (?, ?, ?, ?, ?, ?)",
         Params =>
           [1 => +ID,
            2 => +User_ID,
            3 => +Filename,
            4 => +Filetype,
            5 => +LLM_ID,
            6 => +Status]);
      Commit (Conn);
      Free (Conn);
   exception
      when others =>
         Free (Conn);
         raise;
   end Store_File_Record;

   ---------------
   -- Get_Files --
   ---------------
   function Get_Files (User_ID : String) return Models.File_Vectors.Vector is
      Conn   : Database_Connection := Get_Conn;
      Result : Forward_Cursor;
      List   : Models.File_Vectors.Vector;
      Item   : Models.File_Record;
   begin
      Fetch
        (Connection => Conn,
         Result     => Result,
         Query      => "SELECT id, user_id, filename, filetype, llm_id, status, uploaded_at " &
                       "FROM files WHERE user_id = ? ORDER BY uploaded_at DESC",
         Params     => [1 => +User_ID]);

      while Has_Row (Result) loop
         Item.ID          := To_Unbounded_String (Result.Value (0));
         Item.User_ID     := To_Unbounded_String (Result.Value (1));
         Item.Filename    := To_Unbounded_String (Result.Value (2));
         Item.Filetype    := To_Unbounded_String (Result.Value (3));
         Item.LLM_ID      := To_Unbounded_String (Result.Value (4));
         Item.Status      := To_Unbounded_String (Result.Value (5));
         Item.Uploaded_At := To_Unbounded_String (Result.Value (6));
         List.Append (Item);
         Next (Result);
      end loop;

      Free (Conn);
      return List;
   exception
      when others =>
         Free (Conn);
         raise;
   end Get_Files;

   ----------------
   -- Initialize --
   ----------------
   procedure Initialize (DB_Path : String := "frontface.db") is
      Descr : GNATCOLL.SQL.Exec.Database_Description;
      Conn  : GNATCOLL.SQL.Exec.Database_Connection;
   begin
      Ada.Text_IO.Put_Line ("[Database] Configuring SQLite: " & DB_Path);
      Descr := GNATCOLL.SQL.Sqlite.Setup (DB_Path);
      Conn  := Descr.Build_Connection;

      -- 1. Create Chats Table
      Execute
        (Conn,
         "CREATE TABLE IF NOT EXISTS chats (" &
         "   id TEXT PRIMARY KEY, " &
         "   title TEXT NOT NULL, " &
         "   created_at DATETIME DEFAULT CURRENT_TIMESTAMP" &
         ")");

      if not Success (Conn) then
         Ada.Text_IO.Put_Line ("[Error] Creating chats table: " & Error (Conn));
      end if;

      -- 2. Create Messages Table
      Execute
        (Conn,
         "CREATE TABLE IF NOT EXISTS messages (" &
         "   id TEXT PRIMARY KEY, " &
         "   chat_id TEXT NOT NULL, " &
         "   role TEXT NOT NULL, " &
         "   content TEXT NOT NULL, " &
         "   created_at DATETIME DEFAULT CURRENT_TIMESTAMP, " &
         "   FOREIGN KEY(chat_id) REFERENCES chats(id)" &
         ")");
         
      if not Success (Conn) then
         Ada.Text_IO.Put_Line ("[Error] Creating messages table: " & Error (Conn));
      end if;

      -- 3. NEW: Create Files Table
      Execute
        (Conn,
         "CREATE TABLE IF NOT EXISTS files (" &
         "   id TEXT PRIMARY KEY, " &
         "   user_id TEXT NOT NULL, " &
         "   filename TEXT NOT NULL, " &
         "   filetype TEXT NOT NULL, " &
         "   llm_id TEXT, " &
         "   status TEXT NOT NULL, " &
         "   uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP" &
         ")");
         
      if not Success (Conn) then
         Ada.Text_IO.Put_Line ("[Error] Creating files table: " & Error (Conn));
      end if;

      Commit (Conn);
      Free (Conn);
      Ada.Text_IO.Put_Line ("[Database] Schema initialized successfully.");
   exception
      when others =>
         Ada.Text_IO.Put_Line ("[Fatal] Database init failed.");
         Free (Conn);
         raise;
   end Initialize;

end Database;