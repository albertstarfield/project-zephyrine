pragma Ada_2022;

with Ada.Text_IO;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Ada.Strings.Fixed;
with Ada.Exceptions;

with AWS.Messages;          use AWS.Messages;
with AWS.MIME;
with AWS.Status;
with AWS.Response;
with AWS.Client;            -- Required for Proxying
with AWS.Net;               use AWS.Net;
with AWS.Net.WebSocket;     use AWS.Net.WebSocket;

with GNATCOLL.JSON;
with Cognitive_Lang_Resp;
with Database;              -- Now we use the Database
with Models;

package body Handlers is

   ------------------
   -- Health_Check --
   ------------------
   function Health_Check (Request : AWS.Status.Data) return AWS.Response.Data is
      pragma Unreferenced (Request);
      use GNATCOLL.JSON;
      JSON_Res : JSON_Value := Create_Object;
   begin
      JSON_Res.Set_Field ("system_status", "nominal");
      return AWS.Response.Build
        (Content_Type => AWS.MIME.Application_JSON,
         Message_Body => Write (JSON_Res));
   end Health_Check;

   -----------------
   -- File_Upload --
   -----------------
   function File_Upload (Request : AWS.Status.Data) return AWS.Response.Data is
      CT : constant String := AWS.Status.Content_Type (Request);
      use GNATCOLL.JSON;
   begin
      -- CASE 1: Metadata Save (JSON)
      -- This handles the "Save metadata to your backend" step from React
      if Ada.Strings.Fixed.Index (CT, "application/json") > 0 then
         Ada.Text_IO.Put_Line ("[Handler] Received File Metadata JSON");
         declare
            Payload : constant String := AWS.Status.Payload (Request);
            Meta    : JSON_Value;
            Ack_Obj : JSON_Value := Create_Object;
         begin
            Meta := Read (Payload);
            
            -- Call Database to store the record
            -- We assume the JSON keys match the React frontend
            Database.Store_File_Record
              (Filename => Meta.Get ("filename"),
               Filetype => Meta.Get ("filetype"),
               User_ID  => Meta.Get ("userId"),
               LLM_ID   => Meta.Get ("llmFileId"), -- Might be null, check frontend
               Status   => Meta.Get ("status"));

            Ack_Obj.Set_Field ("status", "success");
            Ack_Obj.Set_Field ("message", "Metadata saved to DB");
            Ack_Obj.Set_Field ("file", Meta);

            return AWS.Response.Build
              (Status_Code  => AWS.Messages.S200,
               Content_Type => AWS.MIME.Application_JSON,
               Message_Body => Write (Ack_Obj));
         exception
            when E : others =>
               Ada.Text_IO.Put_Line ("DB Error: " & Ada.Exceptions.Exception_Message (E));
               return AWS.Response.Build 
                 (Status_Code  => AWS.Messages.S500,
                  Content_Type => AWS.MIME.Text_Plain,
                  Message_Body => "Database Error");
         end;

      -- CASE 2: Actual File Binary (Multipart)
      -- This handles the "uploadSingleFileToLLM" step from React
      elsif Ada.Strings.Fixed.Index (CT, "multipart/form-data") > 0 then
         declare
            -- Dynamic Forwarding URL
            Target_URL : constant String := 
               Cognitive_Lang_Resp.Get_Backend_URL & "/v1/files";
            Result     : AWS.Response.Data;
         begin
            Ada.Text_IO.Put_Line ("[Handler] Proxying File to: " & Target_URL);

            -- Forward the exact payload and headers (boundary included)
            Result := AWS.Client.Post
              (URL          => Target_URL,
               Data         => AWS.Status.Payload (Request),
               Content_Type => AWS.Status.Content_Type (Request));

            return Result;
         exception
            when E : others =>
               return AWS.Response.Build 
                 (Status_Code  => AWS.Messages.S502, 
                  Content_Type => AWS.MIME.Text_Plain,
                  Message_Body => "Proxy Failed: " & Ada.Exceptions.Exception_Message (E));
         end;

      else
         return AWS.Response.Build 
           (Status_Code  => AWS.Messages.S400, 
            Content_Type => AWS.MIME.Text_Plain,
            Message_Body => "Unsupported Content-Type");
      end if;
   end File_Upload;

   ------------------
   -- File_History --
   ------------------
   function File_History (Request : AWS.Status.Data) return AWS.Response.Data is
      use GNATCOLL.JSON;
      User_ID : constant String := AWS.Status.Parameter (Request, "userId");
      Files   : Models.File_Vectors.Vector;
      Arr     : JSON_Array := Empty_Array;
      Res     : JSON_Value := Create_Object;
   begin
      -- Fetch from Database
      Files := Database.Get_Files (User_ID);

      for F of Files loop
         declare
            Item : JSON_Value := Create_Object;
         begin
            Item.Set_Field ("id", To_String (F.ID));
            Item.Set_Field ("filename", To_String (F.Filename));
            Item.Set_Field ("filetype", To_String (F.Filetype));
            Item.Set_Field ("status", To_String (F.Status));
            Item.Set_Field ("uploaded_at", To_String (F.Uploaded_At));
            Item.Set_Field ("llm_file_id", To_String (F.LLM_ID));
            Append (Arr, Item);
         end;
      end loop;

      Res.Set_Field ("data", Arr);
      
      return AWS.Response.Build
         (Status_Code  => S200, 
          Content_Type => AWS.MIME.Application_JSON, 
          Message_Body => Write (Res));
   end File_History;

   ----------------------
   -- Chat_Completions --
   ----------------------
   function Chat_Completions (Request : AWS.Status.Data) return AWS.Response.Data is
      use GNATCOLL.JSON;
      Body_Str       : constant String := AWS.Status.Payload (Request);
      JSON_Req       : JSON_Value;
      Input_Sequence : Unbounded_String;
      System_Output  : Unbounded_String;
      JSON_Res       : JSON_Value := Create_Object;
   begin
      begin
         JSON_Req := Read (Body_Str);
         -- FIXED: Explicit typing to resolve ambiguity
         declare
            Prompt : constant String := JSON_Req.Get ("prompt");
         begin
            Input_Sequence := To_Unbounded_String (Prompt);
         end;
      exception
         when others =>
            return AWS.Response.Build 
              (Status_Code  => AWS.Messages.S400, 
               Content_Type => AWS.MIME.Text_Plain,
               Message_Body => "Invalid Data Sequence");
      end;

      System_Output := To_Unbounded_String
        (Cognitive_Lang_Resp.Process_Input
           (Input_Sequence => To_String (Input_Sequence),
            Model_ID       => "Snowball-Enaga"));

      JSON_Res.Set_Field ("reply", To_String (System_Output));
      JSON_Res.Set_Field ("model", "Snowball-Enaga");
      JSON_Res.Set_Field ("status", "nominal");
      return AWS.Response.Build 
        (Content_Type => AWS.MIME.Application_JSON, 
         Message_Body => Write (JSON_Res));
   end Chat_Completions;

   -----------------------
   -- Image_Generations --
   -----------------------
   function Image_Generations (Request : AWS.Status.Data) return AWS.Response.Data is
      Target_URL : constant String := 
         Cognitive_Lang_Resp.Get_Backend_URL & "/api/v1/images/generations";
      Result     : AWS.Response.Data;
   begin
      Ada.Text_IO.Put_Line ("[Handler] Proxying Image Gen to: " & Target_URL);
      Result := AWS.Client.Post
        (URL          => Target_URL,
         Data         => AWS.Status.Payload (Request),
         Content_Type => AWS.Status.Content_Type (Request));
      return Result;
   exception
      when E : others =>
         return AWS.Response.Build 
           (Status_Code  => S502, 
            Content_Type => AWS.MIME.Text_Plain,
            Message_Body => "Image Gen Failed: " & Ada.Exceptions.Exception_Message (E));
   end Image_Generations;

   ---------------------
   -- WebSocket_Entry --
   ---------------------
   function WebSocket_Entry (Request : AWS.Status.Data) return AWS.Response.Data is
      pragma Unreferenced (Request);
   begin
      return AWS.Response.Build 
        (Status_Code  => S200, 
         Content_Type => AWS.MIME.Text_Plain,
         Message_Body => "WebSocket Endpoint");
   end WebSocket_Entry;

   ---------------------------------
   -- Instrument_Viewport_Preview --
   -- (PFD Data Feed)             --
   ---------------------------------
   function Instrument_Viewport_Preview (Request : AWS.Status.Data) return AWS.Response.Data is
      pragma Unreferenced (Request);
      
      -- 1. Construct the target URL (e.g., http://localhost:11434/instrument...)
      Target_URL : constant String := 
         Cognitive_Lang_Resp.Get_Backend_URL & "/instrumentviewportdatastreamlowpriopreview";
      
      Result : AWS.Response.Data;
   begin
      Ada.Text_IO.Put_Line ("[Handler] Proxying Telemetry to: " & Target_URL);

      -- 2. Fetch the REAL data from your Main Server (Python/Ollama/Flask)
      Result := AWS.Client.Get(URL => Target_URL);

      -- 3. Return the exact JSON received from the main server
      return Result;
      
   exception
      when E : others =>
         -- If the main server is offline, return a 502 Bad Gateway so the Frontend knows to show "NO CARRIER"
         Ada.Text_IO.Put_Line ("[Error] Telemetry Stream Failed: " & Ada.Exceptions.Exception_Message (E));
         return AWS.Response.Build 
           (Status_Code  => AWS.Messages.S502, 
            Content_Type => AWS.MIME.Text_Plain, 
            Message_Body => "Upstream Flight Computer Offline");
   end Instrument_Viewport_Preview;


   -------------------------
   -- Audio_Transcriptions --
   -- (STT Proxy)        --
   -------------------------
   function Audio_Transcriptions (Request : AWS.Status.Data) 
      return AWS.Response.Data is
      
      Target_URL : constant String := 
         Cognitive_Lang_Resp.Get_Backend_URL & "/v1/audio/transcriptions";
      Result     : AWS.Response.Data;
   begin
      Ada.Text_IO.Put_Line 
        ("[Handler] Proxying Audio Transcription to: " & Target_URL);

      -- Forward multipart/form-data (audio file)
      Result := AWS.Client.Post
        (URL          => Target_URL,
         Data         => AWS.Status.Payload (Request),
         Content_Type => AWS.Status.Content_Type (Request));
      return Result;
   exception
      when E : others =>
         -- FIX: Added Content_Type and split long line
         return AWS.Response.Build 
           (Status_Code  => AWS.Messages.S502, 
            Content_Type => AWS.MIME.Text_Plain,
            Message_Body => "STT Failed: " & 
                            Ada.Exceptions.Exception_Message (E));
   end Audio_Transcriptions;

   ------------------
   -- Audio_Speech --
   -- (TTS Proxy)  --
   ------------------
   function Audio_Speech (Request : AWS.Status.Data) return AWS.Response.Data is
      Target_URL : constant String := 
         Cognitive_Lang_Resp.Get_Backend_URL & "/v1/audio/speech";
      Result     : AWS.Response.Data;
   begin
      Ada.Text_IO.Put_Line ("[Handler] Proxying TTS Request to: " & Target_URL);

      -- Forward JSON, receive Binary (Audio Blob)
      Result := AWS.Client.Post
        (URL          => Target_URL,
         Data         => AWS.Status.Payload (Request),
         Content_Type => AWS.Status.Content_Type (Request));
      return Result;
   exception
      when E : others =>
         -- FIX: Added Content_Type and split long line
         return AWS.Response.Build 
           (Status_Code  => AWS.Messages.S502, 
            Content_Type => AWS.MIME.Text_Plain,
            Message_Body => "TTS Failed: " & 
                            Ada.Exceptions.Exception_Message (E));
   end Audio_Speech;

   function Primed_Ready (Request : AWS.Status.Data) return AWS.Response.Data is
      pragma Unreferenced (Request);
      
      Target_URL : constant String := 
         Cognitive_Lang_Resp.Get_Backend_URL & "/primedready";
      Result     : AWS.Response.Data;
   begin
      Ada.Text_IO.Put_Line ("[Handler] Checking System Readiness: " & Target_URL);
      
      -- Proxy the GET request directly to the AI Engine
      Result := AWS.Client.Get(URL => Target_URL);
      
      return Result;
   exception
      when E : others =>
         -- If AI Engine is down, return 503 so frontend keeps waiting
         Ada.Text_IO.Put_Line ("[Warn] PrimedReady Check Failed: " & 
                               Ada.Exceptions.Exception_Message (E));
         return AWS.Response.Build 
           (Status_Code  => AWS.Messages.S503,
            Content_Type => AWS.MIME.Application_JSON,
            Message_Body => "{""status"": ""initializing"", ""ready"": false}");
   end Primed_Ready;

end Handlers;