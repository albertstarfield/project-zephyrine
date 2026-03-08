with AWS.Status;
with AWS.Response;

package Handlers is

   -- GET /health
   -- Returns 200 OK "System Ready"
   function Health_Check (Request : AWS.Status.Data) return AWS.Response.Data;

   -- POST /api/v1/files
   -- (Stub) Handles file uploads
   function File_Upload (Request : AWS.Status.Data) return AWS.Response.Data;

   -- POST /api/v1/chat/completions
   -- (Stub) Proxies to LLM
   function Chat_Completions (Request : AWS.Status.Data)
      return AWS.Response.Data;

   -- WS /ws/chat
   -- (Stub) Upgrade to WebSocket
   function WebSocket_Entry (Request : AWS.Status.Data)
      return AWS.Response.Data;

   -- Provides a low-priority snapshot of system viewport telemetry.
   function Instrument_Viewport_Preview (Request : AWS.Status.Data)
      return AWS.Response.Data;

   -- POST /api/v1/audio/transcriptions
   -- Proxies audio files to STT backend
   function Audio_Transcriptions (Request : AWS.Status.Data) 
      return AWS.Response.Data;

   -- POST /api/v1/audio/speech
   -- Proxies text to TTS backend
   function Audio_Speech (Request : AWS.Status.Data) 
      return AWS.Response.Data;

   function Primed_Ready (Request : AWS.Status.Data)
      return AWS.Response.Data;

end Handlers;