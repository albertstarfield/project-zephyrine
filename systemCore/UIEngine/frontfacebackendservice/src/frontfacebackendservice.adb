pragma Ada_2022;

-- 1. Standard Library
with Ada.Text_IO;           use Ada.Text_IO;
with Ada.Command_Line;
with Ada.Strings.Fixed;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;

-- 2. AWS (Web Server)
with AWS.Server;
with AWS.Config;            use AWS.Config;
with AWS.Config.Set;        use AWS.Config.Set;
with AWS.Net.WebSocket.Registry;
with AWS.Net.WebSocket.Registry.Control;

-- 3. GNATCOLL JSON
with GNATCOLL.JSON;         use GNATCOLL.JSON;

-- 4. GNATCOLL SQL
with GNATCOLL.SQL.Sessions;
with GNATCOLL.SQL.Sqlite;
-- Registers the SQLite driver automatically

-- 5. UUIDs
with UUIDs;                 use UUIDs;
with UUIDs.V4;

-- Helper Modular Components
with Database;      
with Handlers;              use Handlers;
with Dispatcher;            use Dispatcher;
with Cognitive_Lang_Resp;

with Connection_Manager;
use Connection_Manager;
with WS_Handler;            use WS_Handler;


procedure Frontfacebackendservice is
   
   -- AWS Server Objects
   WS      : AWS.Server.HTTP;
   Conf    : AWS.Config.Object;

   -- Test Objects for Linking Check
   J_Obj   : JSON_Value := Create_Object;
   ID      : UUID := V4.UUID4;

   -- Configuration Variables (Matching Go Config)
   Target_Host : Unbounded_String := To_Unbounded_String ("127.0.0.1");
   Target_Port : Integer          := 8080;
   
   Arg_Count : constant Natural := Ada.Command_Line.Argument_Count;

   ----------------
   -- Parse_Args --
   ----------------
   procedure Parse_Args is
      I : Integer := 1;
      
      procedure Parse_Address (Host_Port : String) is
         Idx : constant Natural := Ada.Strings.Fixed.Index (Host_Port, ":");
      begin
         if Idx > 0 then
            Target_Host := To_Unbounded_String (Host_Port (Host_Port'First .. Idx - 1));
            Target_Port := Integer'Value (Host_Port (Idx + 1 .. Host_Port'Last));
         else
            Target_Host := To_Unbounded_String (Host_Port);
            -- Keep default port if not specified
         end if;
      exception
         when others =>
            Put_Line ("[Warning] Invalid format for --frontfaceProxyTarget. Using default.");
      end Parse_Address;

   begin
      while I <= Arg_Count loop
         declare
            Arg : constant String := Ada.Command_Line.Argument (I);
         begin
            if Arg = "--frontfaceProxyTarget" and then I < Arg_Count then
               Parse_Address (Ada.Command_Line.Argument (I + 1));
               I := I + 1;
            elsif Arg = "--backendCognitiveTarget" and then I < Arg_Count then
               Cognitive_Lang_Resp.Configure_Backend (Ada.Command_Line.Argument (I + 1));
               I := I + 1;
            end if;
         end;
         I := I + 1;
      end loop;
   end Parse_Args;

begin
   Put_Line ("--- Phase 1: Dependency Check ---");
   Put_Line ("[OK] UUID Library Linked: " & ID'Image);
   J_Obj.Set_Field ("system_status", "nominal");
   Put_Line ("[OK] JSON Library Linked: " & J_Obj.Write);
   Put_Line ("[OK] SQLite Driver Registered (Implicitly)");
   Put_Line ("[OK] AWS Web Server Linked");

   -- Phase 1.5: Configuration Parsing
   Parse_Args;

   Put_Line ("[Debug database Init], state0");
   Database.Initialize;
   Put_Line ("[Debug database Init], state1");

   Put_Line("websocket register debug");
   AWS.Net.WebSocket.Registry.Register ("/ws/chat", WS_Handler.Create'Access);

   Put_Line("register / for clients that connect directly to the base URL");
   -- NEW: Registers root / for clients that connect directly to the base URL
   AWS.Net.WebSocket.Registry.Register ("/", WS_Handler.Create'Access);

   Put_Line ("[Debug Dispatcher Unit Init], Dispatcher prestate");
   
   -- 1. Apply Configuration: Host & Port
   AWS.Config.Set.Server_Port (Conf, Target_Port);
   AWS.Config.Set.Server_Host (Conf, To_String (Target_Host));
   AWS.Config.Set.Reuse_Address (Conf, True); -- Go typically allows reuse
   
   -- 2. Apply Configuration: Timeouts (Matching Go's 600s for LLM support)
   -- Go: ReadTimeout -> AWS: Receive_Timeout
   AWS.Config.Set.Receive_Timeout (Conf, 600.0); 
   -- Go: WriteTimeout -> AWS: Send_Timeout
   AWS.Config.Set.Send_Timeout (Conf, 600.0);

   -- 3. Apply Configuration: Concurrency
   -- Go handles high concurrency via goroutines. AWS defaults to ~30 tasks.
   -- We increase this to support simultaneous LLM streams + static file requests.
   AWS.Config.Set.Max_Connection (Conf, 512);

   Put_Line ("[AWS] Starting HTTP Server on " & To_String(Target_Host) & ":" & Target_Port'Image & "...");
   
   AWS.Server.Start 
     (Web_Server => WS,
      Callback   => Dispatcher.Callback'Access,
      Config     => Conf);

   Put_Line ("System Running. Press Ctrl+C to stop.");
   
   -- 4. Wait for Signal (Matching Go's signal.Notify)
   -- Forever waits for SIGINT/SIGTERM properly in service mode.
   AWS.Server.Wait (AWS.Server.Forever);
   
   Put_Line ("[Debug Dispatcher proxy Unit Init], Dispatcher Pass");
   AWS.Net.WebSocket.Registry.Control.Shutdown;
   AWS.Server.Shutdown (WS);
   Put_Line ("[Debug Dispatcher Unit Quit], Dispatcher Pass");

   Put_Line ("---------------------------------");
   Put_Line ("System ready for Next Phase porting.");

end Frontfacebackendservice;