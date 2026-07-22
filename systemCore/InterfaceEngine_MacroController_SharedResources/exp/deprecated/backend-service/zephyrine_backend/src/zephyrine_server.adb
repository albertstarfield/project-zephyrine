-- The main procedure that ties everything together.
with Ada.Text_IO;
with Ada.Exceptions;
with Ada.Command_Line;
with AWS.Server;
with AWS.Services.Dispatchers.URI;
with AWS.Net.WebSocket.Registry;
with AWS.Net.WebSocket.Registry.Control;

-- Our application packages
with Config;
with DB_SQLite;
with HTTP_Handlers;
with WebSocket_Handler;

procedure Zephyrine_Server is
   App_Conf   : Config.App_Config;
   DB_Conn    : DB_SQLite.Database_Connection;
   Srv        : AWS.Server.HTTP;
   Root       : AWS.Services.Dispatchers.URI.Handler;

   -- Handler instances
   Health_H   : HTTP_Handlers.Health_Check_Handler;

   -- This is the corrected way to define a multi-line string constant in Ada.
   -- Each line is a separate string, concatenated with the '&' operator.
   Create_Tables_SQL : constant String :=
      "CREATE TABLE IF NOT EXISTS chats (" &
      "  id TEXT PRIMARY KEY, user_id TEXT, title TEXT," &
      "  created_at DATETIME, updated_at DATETIME);" &
      "CREATE TABLE IF NOT EXISTS messages (" &
      "  id TEXT PRIMARY KEY, chat_id TEXT NOT NULL, user_id TEXT," &
      "  sender TEXT NOT NULL, content TEXT NOT NULL, created_at DATETIME," &
      "  FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE);" &
      "CREATE TABLE IF NOT EXISTS generated_images (" &
      "  id TEXT PRIMARY KEY, user_id TEXT NOT NULL, prompt TEXT NOT NULL," &
      "  image_url TEXT NOT NULL, created_at DATETIME);" &
      "CREATE TABLE IF NOT EXISTS fine_tuning_files (" &
      "  id TEXT PRIMARY KEY, user_id TEXT NOT NULL, filename TEXT NOT NULL," &
      "  filetype TEXT NOT NULL, status TEXT NOT NULL, uploaded_at DATETIME," &
      "  llm_file_id TEXT);";

begin
   Ada.Text_IO.Put_Line ("--- Starting Project Zephyrine Backend (Ada Version) ---");

   -- 1. Load Configuration
   Config.Load (App_Conf);
   Ada.Text_IO.Put_Line ("INFO: Server configured for port: " & Config.To_String(App_Conf.Port));

   -- 2. Initialize Database
   DB_Conn := DB_SQLite.Open (Config.To_String(App_Conf.DB_Path));
   if DB_Conn = DB_SQLite.Null_Connection then
      Ada.Text_IO.Put_Line ("FATAL: Failed to initialize database.");
      Ada.Command_Line.Set_Exit_Status (Ada.Command_Line.Failure);
      return;
   end if;

   DB_SQLite.Exec (DB_Conn, Create_Tables_SQL);
   Ada.Text_IO.Put_Line ("INFO: Database tables initialized.");

   -- 3. Register HTTP and WebSocket Handlers
   AWS.Services.Dispatchers.URI.Register (Root, "/health", Health_H'Access);
   -- TODO: Register other HTTP handlers here...

   AWS.Net.WebSocket.Registry.Register ("/ws", WebSocket_Handler.Create'Access);
   Ada.Text_IO.Put_Line ("INFO: Registered /health and /ws endpoints.");

   -- 4. Start Servers
   AWS.Server.Start (Srv, Root, Port => Integer'Value(Config.To_String(App_Conf.Port)));
   AWS.Net.WebSocket.Registry.Control.Start;
   Ada.Text_IO.Put_Line ("INFO: HTTP and WebSocket servers are running.");

   -- 5. Keep the main thread alive
   delay Duration'Last;

exception
   when E : others =>
      Ada.Text_IO.Put_Line ("FATAL: An unhandled exception occurred: " & Ada.Exceptions.Exception_Message (E));
      DB_SQLite.Close(DB_Conn);
      AWS.Server.Shutdown(Srv);
      Ada.Command_Line.Set_Exit_Status (Ada.Command_Line.Failure);
end Zephyrine_Server;