pragma Ada_2022;
with AWS.MIME;
with AWS.Messages;
with AWS.Response;
with AWS.Status; use AWS.Status;
with AWS.Response.Set; use AWS.Response.Set;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with AWS.Net.WebSocket.Registry.Control;
with Handlers;

package body Dispatcher is

   --------------
   -- Add_CORS --
   --------------
   function Add_CORS (Resp : AWS.Response.Data) return AWS.Response.Data is
      R : AWS.Response.Data := Resp;
   begin
      AWS.Response.Set.Add_Header (R, "Access-Control-Allow-Origin", "*");
      AWS.Response.Set.Add_Header (R, "Access-Control-Allow-Methods", "GET, POST, OPTIONS");  
      AWS.Response.Set.Add_Header (R, "Access-Control-Allow-Headers", "Content-Type");
      return R;
   end Add_CORS;

   --------------
   -- Callback --
   --------------
   function Callback (Request : AWS.Status.Data) return AWS.Response.Data is
      URI : constant String := AWS.Status.URI (Request);
   begin
      -- 1. Global CORS Preflight
      if AWS.Status.Method (Request) = AWS.Status.OPTIONS then
         return Add_CORS 
           (AWS.Response.Build (Content_Type => AWS.MIME.Text_Plain, Message_Body => ""));
      end if;

      -- 2. Router Switch
      if URI = "/health" then
         return Add_CORS (Handlers.Health_Check (Request));

      elsif URI = "/api/v1/files" and then AWS.Status.Method (Request) = AWS.Status.POST then
         return Add_CORS (Handlers.File_Upload (Request));

      elsif URI = "/primedready" then
         return Add_CORS (Handlers.Primed_Ready (Request));

      elsif URI = "/api/v1/chat/completions" and then AWS.Status.Method (Request) = AWS.Status.POST then
         return Add_CORS (Handlers.Chat_Completions (Request));

      -- FIXED: Changed 'Method' to 'AWS.Status.Method (Request)'
      elsif URI = "/api/v1/audio/transcriptions" and then AWS.Status.Method (Request) = AWS.Status.POST then
         return Add_CORS (Handlers.Audio_Transcriptions (Request));

      -- FIXED: Changed 'Method' to 'AWS.Status.Method (Request)'
      elsif URI = "/api/v1/audio/speech" and then AWS.Status.Method (Request) = AWS.Status.POST then
         return Add_CORS (Handlers.Audio_Speech (Request));

      -- 3. WebSocket Hand-off
      elsif URI = "/ws/chat" or else URI = "/" then
         return AWS.Response.WebSocket;
         
      else
         return Add_CORS 
           (AWS.Response.Build 
              (Status_Code  => AWS.Messages.S404, 
               Content_Type => AWS.MIME.Text_Plain,
               Message_Body => "Not Found: " & URI));
      end if;
      
   exception
      when others =>
         return AWS.Response.Build 
           (Status_Code  => AWS.Messages.S500, 
            Content_Type => AWS.MIME.Text_Plain,
            Message_Body => "Internal Server Error");
   end Callback;

end Dispatcher;