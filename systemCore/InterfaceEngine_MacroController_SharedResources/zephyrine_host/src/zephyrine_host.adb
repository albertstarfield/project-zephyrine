pragma Ada_2022;

with Ada.Text_IO;
with Ada.Command_Line;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Ada.Strings.Fixed;
with Ada.Exceptions;

with GNAT.OS_Lib;

with AWS.Config.Set;
with AWS.Server;
with AWS.Status;
with AWS.Response;
with AWS.MIME;
with AWS.Client;
with AWS.Messages;

procedure Zephyrine_Host is
   WS       : AWS.Server.HTTP;
   Config   : AWS.Config.Object;
   Root_Dir : constant String := "dist";

   -- Default Configuration
   Server_Port : Positive := 8080;
   Server_Host : Unbounded_String := To_Unbounded_String ("127.0.0.1");
   
   -- Backend Configuration (The Middleware Target)
   Backend_Target : Unbounded_String := Null_Unbounded_String;

   -----------------------
   -- Callback Function --
   -----------------------
   function Callback (Request : AWS.Status.Data) return AWS.Response.Data is
      URI  : constant String := AWS.Status.URI (Request);
      Path : constant String := Root_Dir & URI;
      
      -- Helper to determine if we should proxy
      function Is_Backend_Route return Boolean is
      begin
         return (URI'Length >= 4 and then URI (1 .. 4) = "/api")
           or else (URI'Length >= 3 and then URI (1 .. 3) = "/ws")
           
           -- 1. OpenAI Standard (Fixes /v1/files, /v1/chat, /v1/models)
           or else (URI'Length >= 3 and then URI (1 .. 3) = "/v1")

           -- 2. Your Custom System Routes
           or else (URI = "/ZephyCortexConfig")
           or else (URI = "/zepzepadaui")
           
           -- 3. The Stream Preview (NOW ENABLED)
           or else (URI = "/instrumentviewportdatastreamlowpriopreview")

           -- 4. Legacy Health Check
           or else (URI'Length >= 12 and then URI (1 .. 12) = "/primedready");
      end Is_Backend_Route;

   begin
      -- 1. SERVE DYNAMIC CONFIGURATION (The "Injection" Step)
      -- This allows React to know where the backend is without rebuilding.
      if URI = "/env-config.js" then
         declare
            JS_Config : constant String := 
              "window.FrontendBackendRecieve = """ & To_String (Backend_Target) & """;";
         begin
            return AWS.Response.Build
              (Content_Type => AWS.MIME.Text_Javascript,
               Message_Body => JS_Config);
         end;

      -- 2. Check for Static Files (Frontend Assets)
      elsif GNAT.OS_Lib.Is_Regular_File (Path) then
         return AWS.Response.File
           (Content_Type => AWS.MIME.Content_Type (Path),
            Filename     => Path);

      -- 3. Proxy to Backend (Middleware Logic)
      elsif Length (Backend_Target) > 0 and then Is_Backend_Route then
         declare
            Target_URI : Unbounded_String := To_Unbounded_String (URI);
            Result     : AWS.Response.Data;
            Method     : constant String := AWS.Status.Method (Request);
         begin
            if URI = "/instrumentviewportdatastreamlowpriopreview" then
               Target_URI := To_Unbounded_String ("/instrumentviewportdatastreamlowpriopreview");
            end if;

            Ada.Text_IO.Put_Line ("[Proxy] Forwarding " & Method & " " & URI & " to " & To_String (Backend_Target) & To_String (Target_URI));

            if Method = "POST" then
               Result := AWS.Client.Post
                 (URL          => To_String (Backend_Target) & To_String (Target_URI),
                  Data         => AWS.Status.Payload (Request),
                  Content_Type => AWS.Status.Content_Type (Request));
            else
               Result := AWS.Client.Get
                 (URL => To_String (Backend_Target) & To_String (Target_URI));
            end if;
               
            return Result;

         exception
            when E : others =>
               Ada.Text_IO.Put_Line ("[Error] Proxy Failed: " & Ada.Exceptions.Exception_Message (E));
               return AWS.Response.Build
                 (Status_Code  => AWS.Messages.S502,
                  Content_Type => AWS.MIME.Text_Plain,
                  Message_Body => "Bad Gateway: Could not reach Backend Service");
         end;

      -- 4. Fallback to Index.html (SPA Routing)
      else
         return AWS.Response.File
           (Content_Type => AWS.MIME.Text_HTML,
            Filename     => Root_Dir & "/index.html");
      end if;
   end Callback;

   ----------------
   -- Parse_Args --
   ----------------
   procedure Parse_Args is
      use Ada.Command_Line;
      Idx : Positive := 1;
      
      procedure Parse_Host_Port (Arg : String; Host : out Unbounded_String; Port : out Positive) is
         Colon_Index : constant Natural := Ada.Strings.Fixed.Index (Arg, ":");
      begin
         if Colon_Index > 0 then
            Host := To_Unbounded_String (Arg (Arg'First .. Colon_Index - 1));
            Port := Positive'Value (Arg (Colon_Index + 1 .. Arg'Last));
         else
            Host := To_Unbounded_String (Arg);
         end if;
      exception
         when others =>
            Ada.Text_IO.Put_Line ("Invalid host:port format: " & Arg);
      end Parse_Host_Port;

   begin
      while Idx <= Argument_Count loop
         declare
            Arg : constant String := Argument (Idx);
         begin
            if Arg = "--frontface" and then Idx < Argument_Count then
               Parse_Host_Port (Argument (Idx + 1), Server_Host, Server_Port);
               Idx := Idx + 1;
               
            elsif Arg = "--backendRecieve" and then Idx < Argument_Count then
               declare
                  Raw_Val : constant String := Argument (Idx + 1);
               begin
                  -- Ensure protocol is present
                  if Raw_Val'Length > 4 and then Raw_Val (1..4) = "http" then
                     Backend_Target := To_Unbounded_String (Raw_Val);
                  else
                     Backend_Target := To_Unbounded_String ("http://" & Raw_Val);
                  end if;
               end;
               Idx := Idx + 1;
            end if;
         end;
         Idx := Idx + 1;
      end loop;
   exception
      when others =>
         Ada.Text_IO.Put_Line ("Invalid arguments. Using defaults.");
   end Parse_Args;

begin
   Parse_Args;

   AWS.Config.Set.Server_Port (Config, Server_Port);
   AWS.Config.Set.Server_Host (Config, To_String (Server_Host));
   AWS.Config.Set.Reuse_Address (Config, True);
   AWS.Config.Set.Max_Connection (Config, 512);

   Ada.Text_IO.Put_Line ("--------------------------------------");
   Ada.Text_IO.Put_Line ("Zephyrine Host Active (Ada 2022)");
   Ada.Text_IO.Put_Line ("Frontface (UI): " & To_String (Server_Host) & ":" & Positive'Image (Server_Port));
   if Length (Backend_Target) > 0 then
      Ada.Text_IO.Put_Line ("Target        : " & To_String (Backend_Target));
   else
      Ada.Text_IO.Put_Line ("Target        : [Wait] No target configured");
   end if;
   Ada.Text_IO.Put_Line ("--------------------------------------");

   AWS.Server.Start (WS, Callback'Unrestricted_Access, Config);
   AWS.Server.Wait (AWS.Server.Q_Key_Pressed);
   AWS.Server.Shutdown (WS);
end Zephyrine_Host;