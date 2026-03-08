pragma Ada_2022;
with AWS.MIME;
with AWS.Messages;
with AWS.Response;
with AWS.Response.Set;
with AWS.Status; use AWS.Status;
with AWS.Response.Set; use AWS.Response.Set;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with AWS.Net.WebSocket.Registry.Control;
with Ada.Text_IO;
with Ada.Exceptions;
with AWS.Headers; use AWS.Headers;
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

   procedure Debug_Headers (Request : AWS.Status.Data) is
      Header_List : constant AWS.Headers.List := AWS.Status.Header (Request);
   begin
      Ada.Text_IO.Put_Line ("--- [Incoming Headers] ---");
      -- AWS.Headers.Count returns the number of headers present
      for I in 1 .. AWS.Headers.Count (Header_List) loop
         -- FIX: Use Get_Name and Get_Value as per AWS API
         Ada.Text_IO.Put_Line (AWS.Headers.Get_Name (Header_List, I) & ": " & 
                               AWS.Headers.Get_Value (Header_List, I));
      end loop;
      Ada.Text_IO.Put_Line ("--------------------------");
   end Debug_Headers;

   procedure Debug_Response (Resp : AWS.Response.Data) is
      use AWS.Messages;
      use AWS.Headers;
      -- Extract the header list from the response object
      H_List : constant AWS.Headers.List := AWS.Response.Header (Resp);
   begin
      Ada.Text_IO.Put_Line ("--- [Outgoing WebSocket Response] ---");
      
      -- Print the Status Code (e.g., 101 Switching Protocols)
      Ada.Text_IO.Put_Line ("Status: " & Image (AWS.Response.Status_Code (Resp)));

      -- Iterate through all headers currently attached to the response
      for I in 1 .. Count (H_List) loop
         Ada.Text_IO.Put_Line (Get_Name (H_List, I) & ": " & Get_Value (H_List, I));
      end loop;
      
      Ada.Text_IO.Put_Line ("--------------------------------------");
   end Debug_Response;

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

      elsif URI = "/instrumentviewportdatastreamlowpriopreview" then
         return Add_CORS (Handlers.Instrument_Viewport_Preview (Request));

      elsif URI = "/primedready" then
         --return Add_CORS (Handlers.Primed_Ready (Request));
         return Handlers.Primed_Ready (Request);

      elsif URI = "/api/v1/chat/completions" and then AWS.Status.Method (Request) = AWS.Status.POST then
         return Add_CORS (Handlers.Chat_Completions (Request));

      -- FIXED: Changed 'Method' to 'AWS.Status.Method (Request)'
      elsif URI = "/api/v1/audio/transcriptions" and then AWS.Status.Method (Request) = AWS.Status.POST then
         return Add_CORS (Handlers.Audio_Transcriptions (Request));

      -- FIXED: Changed 'Method' to 'AWS.Status.Method (Request)'
      elsif URI = "/api/v1/audio/speech" and then AWS.Status.Method (Request) = AWS.Status.POST then
         --return Add_CORS (Handlers.Audio_Speech (Request));
         return Handlers.Audio_Speech (Request);

      -- 3. WebSocket Hand-off
      elsif URI = "/ws/chat" or else URI = "/" then
         declare
            -- Create the base WebSocket response (Status 101)
            WS_Resp : AWS.Response.Data := AWS.Response.WebSocket;
         begin
            Debug_Headers (Request);
            --return Add_CORS (WS_Resp);
            --Ada.Text_IO.Put_Line("Inside here of the WS RESP" & (WS_Resp)); -- This doesn't work like usual, use other procedure to do style mixing or 'image if it's not object
            Debug_Response(WS_Resp);
            return WS_Resp;

         end;
         
      else
         return Add_CORS 
           (AWS.Response.Build 
              (Status_Code  => AWS.Messages.S404, 
               Content_Type => AWS.MIME.Text_Plain,
               Message_Body => "Not Found: " & URI));
      end if;
      
   exception
      when E : others =>
         -- If this prints, the browser receives a 500 error -> "Invalid Frame Header"
         Ada.Text_IO.Put_Line ("[CRITICAL ERROR] Dispatcher Exception: " & 
                               Ada.Exceptions.Exception_Message (E));
         return AWS.Response.Build 
           (Status_Code  => AWS.Messages.S500,
            Content_Type => AWS.MIME.Text_Plain,
            Message_Body => "Internal Server Error");
   end Callback;

end Dispatcher;