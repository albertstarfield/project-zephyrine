pragma SPARK_Mode (Off);
with AnsiAda;
with Ada.Text_IO; use Ada.Text_IO;
with Ada.Calendar;
with Ada.Calendar.Formatting;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Ada.Strings.Fixed; use Ada.Strings.Fixed;
with Ada.Exceptions;
with Ada.Streams;
with Kokoro_Interface;
with Moonshine_Interface;
with Interfaces; use Interfaces;
with Model_Manager;
with Streaming_Queue;
use Streaming_Queue;
with Multimodal_Content_Parser;
use Multimodal_Content_Parser;
with AWS.Response.Set;
with AWS.Client;
with AWS.Messages;
with GNAT.Sockets;
with AWS.Headers;
with GNATCOLL.JSON; use GNATCOLL.JSON;
with Math_Utils;
with Ada.Containers.Indefinite_Ordered_Maps;
with Ada.Real_Time; use Ada.Real_Time;
with Fuzzy_Match;
with Claude_Client; use Claude_Client;
with SD_Manager;
with Database_Manager;
with Model_Types; use Model_Types;
with Ada.Directories;
with Benchmark_Manager; use Benchmark_Manager;
with Accuracy_Benchmark_Manager; use Accuracy_Benchmark_Manager;

with Version;
--  ===========================================================================
--  DISPATCH QUIRKS & DISCOVERED WORKAROUNDS
--  ===========================================================================
--  [QUIRK-D01] [ALL] Model routing always goes through Hybrid_Generate
--  Regardless of the "model" field in the client request, the server
--  ignores it and routes everything through Hybrid_Generate (Adelaide
--  Core Orchestration) at line 731-736.  The requested model name is
--  only echoed back in the response for OpenAI/Ollama compatibility.
--  This means:
--     - "model": "gpt-4" -> still uses Qwen3.5HybridMythos backend
--     - "model": "llama3" -> still uses Qwen3.5HybridMythos backend
--     - "model": "Snowball-Enaga" -> uses Qwen3.5HybridMythos backend (correct)
--  LINUX-COMPAT: Same behavior applies on Linux.
--
--  [QUIRK-D02] [ALL] User-Agent fuzzy match for external agent detection
--  The Dispatch function uses Fuzzy_Match to score the User-Agent header
--  against known agent signatures at 0.7 threshold.  If matched AND the
--  agent is not in the "standard chatbot" list, it's treated as an
--  "External Agent" and gets raw LLM passthrough (no personality pipeline).
--  Standard chatbots (OpenWebUI, Msty, Chatbox, etc.) always get the full
--  Adelaide personality pipeline regardless of UA score.
--  LINUX-COMPAT: Same behavior, no platform dependencies.
--
--  [QUIRK-D03] [ALL] Streaming format: NDJSON for Ollama, SSE for OpenAI
--  /api/chat and /api/generate use NDJSON (application/x-ndjson).
--  /v1/chat/completions and /v1/completions use SSE (text/event-stream).
--  The Streaming_Queue handles both with the correct markers:
--     - Ollama: done:true / done:false
--     - OpenAI: data: [DONE]
--  The immediate ACK pattern (line 707-717) guarantees sub-ms TTFB by
--  pushing the first chunk into the queue buffer BEFORE the generator
--  task starts.
--  LINUX-COMPAT: Same behavior, no platform dependencies.
--
--  [QUIRK-D04] [ALL] Audio endpoint data format
--  /v1/audio/transcriptions expects raw PCM float32 audio data (not WAV
--  or other container format).  The Moonshine interface transcribes
--  directly from the raw float array.  /v1/audio/speech returns raw PCM
--  float32 as well (from Kokoro TTS).  Clients must send/receive raw PCM.
--  LINUX-COMPAT: Same behavior, no platform dependencies.
--  ===========================================================================

package body Adelaide_Server_Pkg is

   --  API Alignment Details:
   --   1. Ollama Compatibility:
   --       * /api/chat: Correctly returns NDJSON (streaming) or JSON (non-streaming)
   --         with the message object (role, content) and done flag.
   --       * /api/generate: Correctly returns NDJSON/JSON with the response
   --         field and done flag.
   --       * /api/tags: Correctly reports the available models in the
   --         expected Ollama format.
   --       * /api/embeddings: Fully compatible with Ollama embedding requests.
   --   2. OpenAI Compatibility:
   --       * /v1/chat/completions: Fully aligned response structure
   --         (non-streaming) and SSE format (streaming).
   --       * /v1/models & /v1/embeddings: Standard parity for easier integration.
   --   3. Unified Streaming Logic:
   --       * The Streaming_Queue correctly handles NDJSON for Ollama and
   --         SSE (data: ...) for OpenAI, including the standard completion
   --         markers (done: true and [DONE]).

   --  Pace timing for main loop
   WCET_Main_Loop : Duration := 0.0;

   -- Handless Mode Trackers
   Handless_Stage : Unbounded_String := To_Unbounded_String ("Idle");
   Handless_Input_Text : Unbounded_String := To_Unbounded_String ("");
   Handless_Output_Text : Unbounded_String := To_Unbounded_String ("");
   Handless_WCET : Float := 0.0;
   Handless_Vision_Context : Unbounded_String := To_Unbounded_String ("");

   use type Streaming_Queue.Queue_Access;

   package Session_Maps is new Ada.Containers.Indefinite_Ordered_Maps
     (Key_Type     => String,
      Element_Type => Streaming_Queue.Queue_Access);

   Active_Sessions : Session_Maps.Map;

   --  ===========================================================================
   --  VIRTUAL CONTEXT SIZE: models/ + database literature + database interaction
   --  The "size" field in /api/tags and /api/ps exposes the total knowledge
   --  footprint. Virtual Context Model = 2^63 (theoretical maximum).
   --  ===========================================================================
   --  2^63 in Ada: shift 1 left by 63 bits
   Virtual_Context_Max : constant Unsigned_64 :=
      16#7FFFFFFFFFFFFFFF#;

   function Calculate_Total_Knowledge_Size return Unsigned_64 is
      use Ada.Directories;
      Total : Unsigned_64 := 0;
      Search : Search_Type;
      Dir_Ent : Directory_Entry_Type;
   begin
      --  1. All files in model/ directory
      if Exists ("model") then
         Start_Search (Search, "model", "*");
         while More_Entries (Search) loop
            Get_Next_Entry (Search, Dir_Ent);
            if Kind (Dir_Ent) = Ordinary_File then
               Total := Total + Unsigned_64 (Size (Dir_Ent));
            end if;
         end loop;
         End_Search (Search);
      end if;
      --  2. Literature + interaction databases
      if Exists ("UI_Database") then
         Start_Search (Search, "UI_Database", "*");
         while More_Entries (Search) loop
            Get_Next_Entry (Search, Dir_Ent);
            if Kind (Dir_Ent) = Ordinary_File then
               Total := Total + Unsigned_64 (Size (Dir_Ent));
            end if;
         end loop;
         End_Search (Search);
      end if;
      --  3. KV cache state
      if Exists ("cache") then
         Start_Search (Search, "cache", "*");
         while More_Entries (Search) loop
            Get_Next_Entry (Search, Dir_Ent);
            if Kind (Dir_Ent) = Ordinary_File then
               Total := Total + Unsigned_64 (Size (Dir_Ent));
            end if;
         end loop;
         End_Search (Search);
      end if;
      return Total;
   exception
      when others => return Total;
   end Calculate_Total_Knowledge_Size;

   procedure Register (ID : String; Q : Streaming_Queue.Queue_Access) is
   begin
      Active_Sessions.Include (ID, Q);
   end Register;

   procedure Unregister (ID : String) is
   begin
      Active_Sessions.Exclude (ID);
   end Unregister;

   procedure Push_Log (ID : String; Log : String) is
      use type Streaming_Queue.Queue_Access;
   begin
      if Active_Sessions.Contains (ID) then
         Active_Sessions.Element (ID).Push (Log);
      end if;
   end Push_Log;

   --  Thread-safe last API tracker for heartbeat display
   protected Last_API_Tracker is
      procedure Set (URI : String);
      function Get return String;
   private
      Last_URI : Unbounded_String := To_Unbounded_String ("none");
   end Last_API_Tracker;

   protected body Last_API_Tracker is
      procedure Set (URI : String) is
      begin
         Last_URI := To_Unbounded_String (URI);
      end Set;

      function Get return String is
      begin
         return To_String (Last_URI);
      end Get;
   end Last_API_Tracker;

   procedure Set_Last_API (URI : String) is
   begin
      Last_API_Tracker.Set (URI);
   end Set_Last_API;

   function Get_Last_API return String is
   begin
      return Last_API_Tracker.Get;
   end Get_Last_API;

   function Build_Response
     (Content : String;
      Status  : AWS.Messages.Status_Code := AWS.Messages.S200;
      C_Type  : String := "application/json") return AWS.Response.Data
   is
      Resp : AWS.Response.Data := AWS.Response.Build (C_Type, Content);
   begin
      AWS.Response.Set.Status_Code (Resp, Status);
      return Resp;
   end Build_Response;

   function Wrap_Response (R : AWS.Response.Data) return AWS.Response.Data is
      Result : AWS.Response.Data := R;
   begin
      AWS.Response.Set.Add_Header (Result, "Access-Control-Allow-Origin", "*");
      AWS.Response.Set.Add_Header (Result, "Access-Control-Allow-Methods",
                                   "GET, POST, OPTIONS");
      AWS.Response.Set.Add_Header (Result, "Access-Control-Allow-Headers",
                                   "Content-Type, Authorization");
      --  Standard Server-Sent Events (SSE) headers required by strict clients like Msty
      AWS.Response.Set.Add_Header (Result, "Cache-Control", "no-cache");
      AWS.Response.Set.Add_Header (Result, "Connection", "keep-alive");
      return Result;
   end Wrap_Response;

   task type Generator_Task is
       pragma Storage_Size (256 * 1024 * 1024);  --  256 MB storage pool
       pragma Task_Stack_Size (32 * 1024 * 1024); --  32 MB thread stack (llama.cpp tokenize needs deep C stack)
       entry Start
         (Prompt : String; Model_Name : String;
          Format : Streaming_Queue.Format_Type;
          Q : Streaming_Queue.Queue_Access;
          Session_ID : String;
          Agentic : Boolean := False; Raw_Prompt : Boolean := False;
          External_Agent : Boolean := False);
   end Generator_Task;

   type Generator_Task_Access is access Generator_Task;

   task type Background_Deep_Thought_Task is
       pragma Storage_Size (256 * 1024 * 1024);
       pragma Task_Stack_Size (32 * 1024 * 1024);
       entry Start
         (Prompt : String;
          Images : GNATCOLL.JSON.JSON_Array;
          Session_ID : String);
   end Background_Deep_Thought_Task;
   type Background_Deep_Thought_Task_Access is access Background_Deep_Thought_Task;

    task body Generator_Task is
       P : Unbounded_String;
       S_ID : Unbounded_String;
       QA : Streaming_Queue.Queue_Access;
       Res : Unbounded_String;
       Is_Ag : Boolean;
       Is_Raw : Boolean;
       Is_Ext : Boolean;
       use type Streaming_Queue.Queue_Access;
    begin
       --  [VITAL-DO-NOT-REMOVE] Mandated by user.
       Ada.Text_IO.Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) &
             "[Dispatch-V]" & AnsiAda.Reset &
             " Generator_Task: Task body ENTERED. Waiting for Start...");
       accept Start
         (Prompt : String; Model_Name : String;
          Format : Streaming_Queue.Format_Type;
          Q : Streaming_Queue.Queue_Access;
          Session_ID : String;
          Agentic : Boolean := False; Raw_Prompt : Boolean := False;
          External_Agent : Boolean := False)
       do
          P := To_Unbounded_String (Prompt);
          S_ID := To_Unbounded_String (Session_ID);
          QA := Q;
          --  Dispatch already set format and pushed the immediate ACK.
          Is_Ag := Agentic;
          Is_Raw := Raw_Prompt;
          Is_Ext := External_Agent;
          --  [VITAL-DO-NOT-REMOVE] Mandated by user.
          Ada.Text_IO.Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) &
                "[Dispatch-V]" & AnsiAda.Reset &
                " Generator_Task: Start accepted. Q=" &
                (if QA /= null then "YES" else "NO") &
                " Format=" & Format'Image &
                " Agentic=" & Boolean'Image (Is_Ag) &
                " Raw=" & Boolean'Image (Is_Raw) &
                " Ext=" & Boolean'Image (Is_Ext));
       end Start;

      begin
         --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
         Ada.Text_IO.Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) &
               "[Dispatch-V]" & AnsiAda.Reset &
               " Generator_Task: Starting Hybrid_Generate...");

            Model_Manager.Hybrid_Generate
              (Prompt         => To_String (P),
               Result         => Res,
               Session_ID     => To_String (S_ID),
               Stream         => QA,
               Agentic        => Is_Ag,
               Raw_Prompt     => Is_Raw,
               External_Agent => Is_Ext);
         --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
         Ada.Text_IO.Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) &
               "[Dispatch-V]" & AnsiAda.Reset &
               " Generator_Task: Hybrid_Generate returned. ResLen=" &
               Natural'Image (Length (Res)));
       exception
          when E : others =>
             Ada.Text_IO.Put_Line (AnsiAda.Foreground (AnsiAda.Red) &
               "[Dispatch-V]" & AnsiAda.Reset &
               " Generator Task EXCEPTION: " &
               Ada.Exceptions.Exception_Message (E));
             begin
                if QA /= null then
                   QA.Push (ASCII.LF & "ERROR: Inference Task Failed." & ASCII.LF);
                end if;
             exception
                when others => null;
             end;
       end;

       begin
          --  [VITAL-DO-NOT-REMOVE] Mandated by user.
          Ada.Text_IO.Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) &
                "[Dispatch-V]" & AnsiAda.Reset &
                " Generator_Task: Calling Q.Close...");
          if QA /= null then
             QA.Close;
          end if;
          --  [VITAL-DO-NOT-REMOVE] Mandated by user.
          Ada.Text_IO.Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) &
                "[Dispatch-V]" & AnsiAda.Reset &
                " Generator_Task: Q.Close returned.");
       exception
          when others => null;
       end;
   exception
      when E : others =>
         Ada.Text_IO.Put_Line ("Error in Generator_Task: " &
                               Ada.Exceptions.Exception_Message (E));
         begin
            if QA /= null then
               QA.Close;
            end if;
         exception
            when others => null;
         end;
   end Generator_Task;

   task body Background_Deep_Thought_Task is
      P : Unbounded_String;
      S : Unbounded_String;
      V : GNATCOLL.JSON.JSON_Array;
      Res : Unbounded_String;
   begin
      accept Start
        (Prompt : String;
         Images : GNATCOLL.JSON.JSON_Array;
         Session_ID : String) do
         P := To_Unbounded_String (Prompt);
         V := Images;
         S := To_Unbounded_String (Session_ID);
      end Start;
      
      Model_Manager.Hybrid_Generate
        (Prompt         => To_String (P),
         Result         => Res,
         Images         => V,
         Session_ID     => To_String (S),
         Stream         => null,
         Level          => ELP0,
         Agentic        => True,
         Raw_Prompt     => False,
         External_Agent => False);
   exception
      when E : others =>
         Ada.Text_IO.Put_Line ("Error in Background_Deep_Thought_Task: " & Ada.Exceptions.Exception_Message (E));
   end Background_Deep_Thought_Task;

   --------------
   -- Dispatch --
   function Stream_To_String (Data : Ada.Streams.Stream_Element_Array) return String is
      Result : String (1 .. Data'Length);
   begin
      for I in Data'Range loop
         Result (Integer (I) - Integer (Data'First) + 1) := Character'Val (Data (I));
      end loop;
      return Result;
   end Stream_To_String;

   Is_External_Agent : Boolean := False;

   --------------
     function Dispatch
       (Request : AWS.Status.Data) return AWS.Response.Data
     is
         --  UserAgent=FuzzyMatch: Behavioural patch for external agent detection.
         --  External agent apps (OpenCode, OpenWebUI, etc.) send structured
         --  chat completions requests but expect raw LLM output, not our
         --  personality pipeline. Fuzzy matching the User-Agent against known
         --  agent signatures at 0.7 threshold lets us bypass the personality
         --  orchestrator and passthrough raw inference.
          URI    : constant String := AWS.Status.URI (Request);
          UA     : constant String := AWS.Status.User_Agent (Request);
          Match_Score : Float := 0.0;
          Best_Match_Name : Unbounded_String := To_Unbounded_String ("(none)");
       begin
          --  Track last API for heartbeat display
          Adelaide_Server_Pkg.Set_Last_API (URI);
          --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
          --  Verbose: confirms Dispatch was called and shows which URI.
          Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Dispatch-V]" &
                    AnsiAda.Reset & " Dispatch ENTERED: URI=" & URI);
          --  Fuzzy match User-Agent against known external agents (Status Quo mode)
         declare
            Known_Agents : constant array (1 .. 16) of String (1 .. 12) :=
              ("OpenCode    ", "OpenClaw    ", "Hermes      ", "VSCode      ",
               "Copilot     ", "Continue    ", "Chatbox     ", "Palchat     ",
               "OpenCat     ", "Enlighten   ", "Aiko        ", "Shortcuts   ",
               "MindMac     ", "Comments    ", "OpenWebUI   ", "Perplexity  ");
            Current_Score : Float;
         begin
            for Agent of Known_Agents loop
               begin
                  Current_Score := Fuzzy_Match.Match (UA, Trim (Agent, Ada.Strings.Right));
                  if Current_Score > Match_Score then
                     Match_Score := Current_Score;
                     Best_Match_Name := To_Unbounded_String (Trim (Agent, Ada.Strings.Right));
                  end if;
               exception
                  when others => null;
               end;
            end loop;
         end;

         --  Standard chatbot list: these get Adelaide Mode (full personality pipeline)
         --  even if their UA might partially match external agents
         Is_Standard_Chatbot : Boolean := False;
         Matched_Chatbot : Unbounded_String := To_Unbounded_String ("(none)");
         declare
             Standard_Chatbots : constant array (1 .. 9) of String (1 .. 12) :=
               ("msty        ", "OpenWebUI   ", "Chatbox     ", "Palchat     ",
                "OpenCat     ", "Enlighten   ", "Aiko        ", "MindMac     ",
                "curl        ");
            Current_Score_2 : Float;
         begin
            for Bot of Standard_Chatbots loop
               begin
                  Current_Score_2 := Fuzzy_Match.Match (UA, Trim (Bot, Ada.Strings.Right));
                  if Current_Score_2 >= 0.7 then
                     Is_Standard_Chatbot := True;
                     Matched_Chatbot := To_Unbounded_String (Trim (Bot, Ada.Strings.Right));
                     exit;
                  end if;
               exception
                  when others => null;
               end;
            end loop;
         end;

         --  External Agent = matches known agents AND NOT a standard chatbot
         Is_External_Agent := Match_Score >= 0.7 and then not Is_Standard_Chatbot;
        declare
          Score_Pct : constant Integer := Integer (Match_Score * 100.0);
          Category  : constant String :=
            (if Is_External_Agent then "external-agent"
             elsif Is_Standard_Chatbot then "chatbot"
             else "unknown");
          Matched   : constant String :=
            (if Is_Standard_Chatbot then To_String (Matched_Chatbot)
             else To_String (Best_Match_Name));
        begin
          Ada.Text_IO.Put_Line ("[API] Request: " & URI &
                                " | UA: " & UA &
                                " | Confidence: " & Integer'Image (Score_Pct) & "%" &
                                " | Category: " & Category &
                                " | Matched: " & Matched);
        end;
       declare
         Method : constant String := AWS.Status.Method (Request);
      Raw_S  : constant String := AWS.Status.Parameter (Request, "prompt");
      Raw_B   : constant Unbounded_String :=
        To_Unbounded_String (AWS.Status.Payload (Request));
      Payload : Unbounded_String := (if Raw_S /= "" then
        To_Unbounded_String (Raw_S) 
        elsif Length (Raw_B) > 0 then Raw_B
        else To_Unbounded_String (Stream_To_String (Ada.Streams.Stream_Element_Array'(AWS.Status.Binary_Data (Request)))));
      Result : Unbounded_String;
   begin
      if Method = "OPTIONS" then
         return Wrap_Response (Build_Response (""));
      end if;

      --  Ollama heartbeat: HEAD / (checkServerHeartbeat sends HEAD /)
      if URI = "/" and then Method = "HEAD" then
         return Build_Response ("");
      end if;

      --  Root GET returns friendly status
      if URI = "/" and then Method = "GET" then
         return Build_Response ("Adelaide API");
      end if;

      --  Health check: any endpoint with ?ping=true returns alive status
      declare
         Ping_Param : constant String := AWS.Status.Parameter (Request, "ping");
      begin
         if Ping_Param = "true" then
            return Build_Response
              ("{""status"": ""ok"", ""endpoint"": """ & URI & """}");
         end if;
      end;

      if URI = "/api/version" then
         return Build_Response ("{""version"": """ & Version.Full_Version & """}");
      end if;

      --  Ollama stub: /api/show (POST) - show model info
      if URI = "/api/show" and then Method = "POST" then
         return Build_Response
           ("{""modelfile"": ""FROM Snowball-Enaga"", "
            & """parameters"": """", "
            & """template"": """", "
            & """details"": {"
            & """parent_model"": """", "
            & """format"": ""gguf"", "
            & """family"": ""Snowball-Enaga"", "
            & """families"": [""Snowball-Enaga""], "
            & """parameter_size"": ""9B"", "
            & """quantization_level"": ""Q4_1""}, "
            & """model_info"": {"
            & """general.architecture"": ""snowball"", "
            & """general.parameter_size"": ""9B"", "
            & """general.quantization_level"": ""Q4_1""}}");
      end if;

      --  Ollama stub: /api/create (POST) - create model from Modelfile (stub)
      if URI = "/api/create" and then Method = "POST" then
         return Build_Response
           ("{""status"": ""success"", "
            & """done"": true}");
      end if;

      --  Ollama stub: /api/pull (POST) - pull model (stub, always success)
      if URI = "/api/pull" and then Method = "POST" then
         return Build_Response
           ("{""status"": ""success"", "
            & """digest"": ""adelaide-snowball-enaga-local"", "
            & """total"": 0, "
            & """completed"": 0}");
      end if;

      --  Ollama stub: /api/push (POST) - push model (stub)
      if URI = "/api/push" and then Method = "POST" then
         return Build_Response
           ("{""status"": ""success"", "
            & """total"": 0, "
            & """completed"": 0}");
      end if;

      --  Ollama stub: /api/copy (POST) - copy model (stub)
      if URI = "/api/copy" and then Method = "POST" then
         return Build_Response
           ("{""status"": ""success""}");
      end if;

      --  Ollama stub: /api/delete (DELETE) - remove model (stub)
      if URI = "/api/delete" and then Method = "DELETE" then
         return Build_Response
           ("{""status"": ""success""}");
      end if;

      --  Ollama stub: /api/signin (POST) - sign in (stub)
      if URI = "/api/signin" and then Method = "POST" then
         return Build_Response
           ("{""status"": ""success""}");
      end if;

      --  Ollama stub: /api/signout (POST) - sign out (stub)
      if URI = "/api/signout" and then Method = "POST" then
         return Build_Response
           ("{""status"": ""success""}");
      end if;

      if URI = "/v1/audio/transcriptions" then
         declare
            use GNATCOLL.JSON;
            R : constant JSON_Value := Create_Object;
            Raw_Payload : constant Ada.Streams.Stream_Element_Array :=
              AWS.Status.Binary_Data (Request);
            Num_Floats : constant Unsigned_64 :=
              Unsigned_64 (Raw_Payload'Length / 4);
            type Float_Array is array (1 .. Natural (Num_Floats)) of
              aliased Float;
            Audio_Floats : Float_Array with Import, Address => Raw_Payload'Address;

            Transcript : Unbounded_String;
            use type Unsigned_64;
         begin
            if Num_Floats > 0 then
               Transcript := To_Unbounded_String
                 (Moonshine_Interface.Transcribe_Raw_PCM
                    (Audio_Floats (1)'Access, Num_Floats));
            else
               Transcript := To_Unbounded_String ("No audio data received");
            end if;

            Set_Field (R, "text", To_String (Transcript));
            return Build_Response (Write (R));
         end;
      end if;



       if URI = "/api/agenticZephyHandlessMode" then
          declare
             use GNATCOLL.JSON;
             Raw_Payload : constant Ada.Streams.Stream_Element_Array :=
               AWS.Status.Binary_Data (Request);
             
             -- Check request parameter for JSON metadata (vision context)
             Vision_Context_Param : constant String :=
               AWS.Status.Parameter (Request, "vision_context_b64");
             
             Num_Floats : constant Unsigned_64 :=
               Unsigned_64 (Raw_Payload'Length / 4);
             type Float_Array is array (1 .. Natural (Num_Floats)) of
               aliased Float;
             Audio_Floats : Float_Array with Import, Address => Raw_Payload'Address;

              Transcript : Unbounded_String;
              LLM_Result : Unbounded_String;
              T_Start : constant Ada.Real_Time.Time := Ada.Real_Time.Clock;
              
              -- Convert vision context param to unbounded string if provided
              Vision_Context_From_Param : Unbounded_String;

              -- [DO NOT REMOVE] Handless Pipeline Timing Estimates:
              -- Stage 1 (VAD Check):        10-50ms   Unix socket to adelaide_vad.sock
              -- Stage 2 (STT/Moonshine):     100-300ms Audio -> Text, Tiny Streaming model
              -- Stage 3 (Intent Classify):   100-200ms 0.8B model, 1024 ctx, YES/NO only
              -- Stage 4 (Reflex Reply):      200-400ms 0.8B model, 1024 ctx, short answer
              -- Stage 5 (TTS/Kokoro):        300-800ms Python sidecar spawn + inference
              --
              -- Total End-to-End Estimates:
              --   Cold start (model not loaded): ~1.0-1.5s
              --   Warm (0.8B in memory):         ~0.7-1.2s
              --   KV cache hit:                  ~0.5-0.8s
              --   Best case: 500ms (warm, cache hit, fast disk)
              --   Typical:   800-1000ms
              --   Worst case: 1500ms (cold start, slow TTS)
              -- Model load penalty: ~288ms disk (0.8B at 1847 MB/s)

              use type Unsigned_64;
              use type Ada.Real_Time.Time;
          begin
            Vision_Context_From_Param := To_Unbounded_String (Vision_Context_Param);
            Handless_Stage := To_Unbounded_String ("Transcribing...");
            Handless_Input_Text := To_Unbounded_String ("");
            Handless_Output_Text := To_Unbounded_String ("");

            if Num_Floats > 0 then
               declare
                  use GNAT.Sockets;
                  VAD_Socket  : Socket_Type;
                  VAD_Address : Sock_Addr_Type (Family_Unix);
                  VAD_Stream  : Stream_Access;
                  
                  Raw_Length : constant Unsigned_32 := Unsigned_32 (Raw_Payload'Length);
                  
                  -- Send length as Big-Endian 4-byte network order
                  -- In Ada, GNAT.Sockets has no Host_To_Network for Unsigned_32 directly without unchecked conversion.
                  -- But we can just use Stream_Element_Array and pack the bytes manually since it's only 4 bytes.
                  Length_Arr : Ada.Streams.Stream_Element_Array (1 .. 4);
               begin
                  Length_Arr (1) := Ada.Streams.Stream_Element (Shift_Right (Raw_Length, 24) and 16#FF#);
                  Length_Arr (2) := Ada.Streams.Stream_Element (Shift_Right (Raw_Length, 16) and 16#FF#);
                  Length_Arr (3) := Ada.Streams.Stream_Element (Shift_Right (Raw_Length, 8) and 16#FF#);
                  Length_Arr (4) := Ada.Streams.Stream_Element (Raw_Length and 16#FF#);

                  VAD_Address.Name := To_Unbounded_String ("run/adelaide_vad.sock");
                  Create_Socket (VAD_Socket, Family_Unix, Socket_Stream);
                  Connect_Socket (VAD_Socket, VAD_Address);
                  VAD_Stream := Stream (VAD_Socket);
                  
                  -- Send Length (4 bytes)
                  Ada.Streams.Stream_Element_Array'Write (VAD_Stream, Length_Arr);
                  -- Send Audio Bytes
                  Ada.Streams.Stream_Element_Array'Write (VAD_Stream, Raw_Payload);
                  
                  -- Read Response ('0' or '1')
                  declare
                     Response_Char : Character;
                  begin
                     Character'Read (VAD_Stream, Response_Char);
                     Close_Socket (VAD_Socket);
                     
                     if Response_Char = '0' then
                        Handless_Stage := To_Unbounded_String ("Idle");
                        return Wrap_Response (AWS.Response.Build ("text/plain", "No Speech Detected"));
                     end if;
                  end;
               exception
                  when others =>
                     -- If VAD sidecar is down, fallback to transcribing anyway
                     begin
                        Close_Socket (VAD_Socket);
                     exception
                        when others => null;
                     end;
               end;

               Transcript := To_Unbounded_String
                 (Moonshine_Interface.Transcribe_Raw_PCM
                    (Audio_Floats (1)'Access, Num_Floats));
            else
               Transcript := To_Unbounded_String ("");
            end if;

             Handless_Input_Text := Transcript;
             Handless_Stage := To_Unbounded_String ("Classifying Intent...");

             if Length (Transcript) > 0 then
                declare
                   Vision_Arr : GNATCOLL.JSON.JSON_Array :=
                     GNATCOLL.JSON.Empty_Array;
                   Intent_Res : Unbounded_String;
                begin
                   -- Use vision context from previous upload or current request param
                   if Length (Handless_Vision_Context) > 0 then
                      GNATCOLL.JSON.Append
                        (Vision_Arr,
                         Create (To_String (Handless_Vision_Context)));
                      Handless_Vision_Context := To_Unbounded_String("");
                   end if;
                   
                   -- Also add the vision context from the request parameter
                   if Length (Vision_Context_From_Param) > 0 then
                      GNATCOLL.JSON.Append(Vision_Arr, Create (To_String (Vision_Context_From_Param)));
                      Vision_Context_From_Param := To_Unbounded_String ("");
                   end if;

                    Model_Manager.Generate
                      (Kind          => Snowball_Enaga_ShortNetworkAnswer,
                       Prompt        => "[INTEGRITY: Answer ONLY 'YES' or 'NO'. Do not fabricate or infer intent that is not explicitly present in the text.] " &
                                        "Does the following text contain a clear, actionable intent directed at an AI assistant (a direct question, command, or request requiring a response)? " &
                                        "Answer ONLY 'YES' or 'NO'. Text: """ & To_String (Transcript) & """",
                       Result        => Intent_Res,
                       Images        => GNATCOLL.JSON.Empty_Array,
                       Session_ID    => "intent-classifier",
                       Requested_Ctx => 1024,
                       Stream        => null,
                       Level         => ELP1);

                   if Index (To_String (Intent_Res), "YES") > 0 then
                      Handless_Stage := To_Unbounded_String ("Reflex Replying...");
                       Model_Manager.Generate
                         (Kind          => Snowball_Enaga_ShortNetworkAnswer,
                          Prompt        => "[INTEGRITY: Never fabricate information. If you do not know the answer or are uncertain, say 'I don't know' or 'I'm not sure'. " &
                                           "Do not hallucinate facts, dates, statistics, or sources. If the user asks for something harmful, refuse politely. " &
                                           "Cite sources only if explicitly referenced in your knowledge. Keep responses factual and grounded.] " &
                                           "You are Adelaide Zephyrine Charlotte, a whimsical, curious, and endearingly cute Automata companion. " &
                                           "Give a short, concise, and helpful reply with warmth and a touch of charm. " &
                                           "When something clicks, say 'aha!' not 'smoking gun'. " &
                                           "Reply to the following: """ & To_String (Transcript) & """",
                          Result        => LLM_Result,
                          Images        => Vision_Arr,
                          Session_ID    => "server-handless-reflex",
                          Requested_Ctx => 1024,
                          Stream        => null,
                          Level         => ELP1);

                      declare
                         Deep_Ptr : Background_Deep_Thought_Task_Access := new Background_Deep_Thought_Task;
                      begin
                         Deep_Ptr.Start (To_String (Transcript), Vision_Arr, "server-handless-deep");
                      end;
                   else
                      -- Background chatter
                      Handless_Stage := To_Unbounded_String ("Background Chatter - Idle");
                      Database_Manager.Remember
                        (Prompt   => To_String (Transcript),
                         Response => "[Passive Observation]",
                         Image_B64 => "");
                      return Wrap_Response (AWS.Response.Build ("text/plain", "Silently Embedded"));
                   end if;
                end;
             else
               Handless_Stage := To_Unbounded_String ("Proactive Initiating...");
                Model_Manager.Hybrid_Generate
                  (Prompt     => "[INTEGRITY: Never fabricate information or hallucinate topics. " &
                   "Only initiate about things you are confident about. " &
                   "If you are uncertain what to discuss, ask the user what they would like to talk about. " &
                   "Do not make up facts, statistics, or events.] " &
                   "You are Adelaide Zephyrine Charlotte, a whimsical, curious, and endearingly cute Automata companion. " &
                   "Proactively initiate the conversation with warmth and charm. " &
                   "Ask a random, interesting, or highly agentic question " &
                   "to the user instead of waiting for a prompt.",
                   Result     => LLM_Result,
                   Session_ID => "server-handless",
                   Agentic    => True,
                   Raw_Prompt => True);
            end if;
            Handless_Output_Text := LLM_Result;
            Handless_Stage := To_Unbounded_String("Synthesizing...");

            declare
               PCM_Data : constant Ada.Streams.Stream_Element_Array :=
                 Kokoro_Interface.Synthesize_Speech (To_String (LLM_Result));
            begin
               Handless_Stage := To_Unbounded_String("Idle");
               Handless_WCET := Float(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - T_Start)) * 1000.0;

               if PCM_Data'Length = 0 then
                  return Wrap_Response (AWS.Response.Build ("text/plain", "TTS Error"));
               else
                  declare
                     Result_Str : String (1 .. Natural(PCM_Data'Length));
                  begin
                     for I in PCM_Data'Range loop
                        Result_Str (Natural(I) - Natural(PCM_Data'First) + 1) := Character'Val (PCM_Data (I));
                     end loop;
                     return Wrap_Response (AWS.Response.Build ("audio/pcm", Result_Str));
                  end;
               end if;
            end;
         end;
      end if;

      if URI = "/v1/audio/speech" then
         declare
            use GNATCOLL.JSON;
            Payload_Str : constant String := (if Raw_S /= "" then Raw_S else To_String (Raw_B));
            Parser_Result : constant Read_Result := Read (Payload_Str);
            Text_To_Say : Unbounded_String := To_Unbounded_String ("Hello");
         begin
            if Parser_Result.Success then
               declare
                  Val : constant JSON_Value := Parser_Result.Value;
               begin
                  if Has_Field (Val, "input") then
                     Text_To_Say := To_Unbounded_String (String'(Get (Val, "input")));
                  end if;
               end;
            end if;

            -- Actually call Kokoro
            declare
               PCM_Data : constant Ada.Streams.Stream_Element_Array :=
                 Kokoro_Interface.Synthesize_Speech (To_String(Text_To_Say));
            begin
               if PCM_Data'Length = 0 then
                  return Wrap_Response (AWS.Response.Build ("text/plain", "TTS Error"));
               else
                  -- Convert Stream_Element_Array to String for AWS
                  declare
                     Result_Str : String (1 .. Natural(PCM_Data'Length));
                  begin
                     for I in PCM_Data'Range loop
                        Result_Str (Natural(I) - Natural(PCM_Data'First) + 1) := Character'Val (PCM_Data (I));
                     end loop;
                     return Wrap_Response (AWS.Response.Build ("audio/pcm", Result_Str));
                  end;
               end if;
            end;
         end;
      end if;

      if URI = "/api/ps" then
         --  Ollama /api/ps requires: name, model, size, digest, details, expires_at, size_vram
         --  Size = total knowledge footprint (models + databases + cache)
         declare
            Total_Size : constant Unsigned_64 :=
               Calculate_Total_Knowledge_Size;
         begin
            return Build_Response
              ("{""models"": [{"
               & """name"": ""Snowball-Enaga:latest"", "
               & """model"": ""Snowball-Enaga:latest"", "
               & """size"": " & Unsigned_64'Image (Total_Size) & ", "
               & """digest"": ""adelaide-snowball-enaga-local"", "
               & """details"": {"
               & """parent_model"": """", "
               & """format"": ""gguf"", "
               & """family"": ""Snowball-Enaga"", "
               & """families"": [""Snowball-Enaga""], "
               & """parameter_size"": ""9B"", "
               & """quantization_level"": ""Q4_1""}, "
               & """expires_at"": ""2099-12-31T23:59:59.000000000+00:00"", "
               & """size_vram"": " & Unsigned_64'Image (Total_Size)
               & "}]}");
         end;
      end if;

      if URI = "/api/telemetry" then
         declare
            use GNATCOLL.JSON;
            R : constant JSON_Value := Create_Object;
            Main_NS : constant Float := Float (WCET_Main_Loop) * 1_000_000_000.0;
         begin
            Set_Field (R, "WCET_ELP0_nS", Float(Model_Manager.Current_WCET_ELP0) * 1_000_000_000.0);
            Set_Field (R, "WCET_ELP1_nS", Float(Model_Manager.Current_WCET_ELP1) * 1_000_000_000.0);
            Set_Field (R, "WCET_ELP2_nS", Float(Model_Manager.Current_WCET_ELP2) * 1_000_000_000.0);
            Set_Field (R, "WCET_ELP3_nS", Float(Model_Manager.Current_WCET_ELP3) * 1_000_000_000.0);
            Set_Field (R, "Jitter_Avg_nS", Float (Model_Manager.Current_Jitter_Avg * 1_000_000_000.0));
            Set_Field (R, "Jitter_Max_nS", Float (Model_Manager.Current_Jitter_Max * 1_000_000_000.0));
            Set_Field (R, "WCET_mainLoop_nS", Main_NS);

            -- Handless Mode telemetry
            Set_Field (R, "Handless_Stage", To_String(Handless_Stage));
            Set_Field (R, "Handless_WCET_nS", Handless_WCET * 1_000_000.0);
            Set_Field (R, "Handless_Input_Text", To_String(Handless_Input_Text));
            Set_Field (R, "Handless_Output_Text", To_String(Handless_Output_Text));
            Set_Field (R, "Handless_Vision_Context",
              (if Length (Handless_Vision_Context) > 0 then "loaded" else "none"));

            return Wrap_Response (Build_Response (Write (R)));
         end;
      end if;

      if URI = "/api/ZenithRoutine" then
         declare
            use GNATCOLL.JSON;
            R : constant JSON_Value := Create_Object;
         begin
            Set_Field (R, "status", "Deterministic Pacing Active");
            Set_Field (R, "elp_level", "ELP3");
            Set_Field (R, "target_freq", "1000Hz");
            Set_Field (R, "jitter_avg_us",
                       Float (Model_Manager.Current_Jitter_Avg * 1_000_000.0));
            return Build_Response (Write (R));
         end;
      end if;

      if URI = "/api/tags" or else URI = "/v1/models" then
         if URI = "/api/tags" then
            --  Ollama /api/tags requires: name, model, modified_at, size, digest, details
            declare
               Total_Size : constant Unsigned_64 :=
                  Calculate_Total_Knowledge_Size;
            begin
               return Wrap_Response
                 (Build_Response
                    ("{""models"": [{"
                     & """name"": ""Snowball-Enaga:latest"", "
                     & """model"": ""Snowball-Enaga:latest"", "
                     & """modified_at"": ""2026-01-01T00:00:00.000000000+00:00"", "
                     & """size"": " & Unsigned_64'Image (Total_Size) & ", "
                     & """digest"": ""adelaide-snowball-enaga-local"", "
                     & """details"": {"
                     & """parent_model"": """", "
                     & """format"": ""gguf"", "
                     & """family"": ""Snowball-Enaga"", "
                     & """families"": [""Snowball-Enaga""], "
                     & """parameter_size"": ""9B"", "
                     & """quantization_level"": ""Q4_1""}"
                     & "}]}"));
            end;
         else
            return Wrap_Response (Build_Response ("{""object"": ""list"", ""data"": [{""id"": ""Snowball-Enaga"", ""object"": ""model"", ""created"": 1686935002, ""owned_by"": ""adelaide""}]}"));
         end if;
      end if;

      if URI = "/v1/embeddings" or else URI = "/api/embeddings" or else URI = "/api/embed" then
         declare
            use GNATCOLL.JSON;
            Payload_Str : constant String := (if Raw_S /= "" then Raw_S
              elsif Length (Raw_B) > 0 then To_String (Raw_B)
              else Stream_To_String (Ada.Streams.Stream_Element_Array'(AWS.Status.Binary_Data (Request))));
            P_Res    : constant Read_Result := Read (Payload_Str);
            Txt      : Unbounded_String := To_Unbounded_String (Payload_Str);
            Vec      : Math_Utils.Vector (1 .. 16384);
            Len      : Natural;
            Resp     : constant JSON_Value := Create_Object;
            Data_Arr : JSON_Array := Empty_Array;
            Emb_Obj  : constant JSON_Value := Create_Object;
            Emb_Arr  : JSON_Array := Empty_Array;
         begin
            if P_Res.Success then
               declare
                  Val : constant JSON_Value := P_Res.Value;
               begin
                  if Has_Field (Val, "input") then
                     Txt := To_Unbounded_String (String'(Get (Val, "input")));
                  elsif Has_Field (Val, "prompt") then
                     Txt := To_Unbounded_String (String'(Get (Val, "prompt")));
                  end if;
               end;
            end if;

            if Length (Txt) > 0 then
               Model_Manager.Get_Embedding (To_String (Txt), Vec, Len);
               for I in 1 .. Len loop
                  Append (Emb_Arr, Create (Long_Float (Vec (I))));
               end loop;
            end if;

            if URI = "/v1/embeddings" then
               Set_Field (Emb_Obj, "object", "embedding");
               Set_Field (Emb_Obj, "index", Integer'(0));
               Set_Field (Emb_Obj, "embedding", Emb_Arr);
               Append (Data_Arr, Emb_Obj);
               Set_Field (Resp, "object", "list");
               Set_Field (Resp, "data", Data_Arr);
               Set_Field (Resp, "model", "Snowball-Enaga");
            else
               Set_Field (Resp, "embedding", Emb_Arr);
            end if;
            return Wrap_Response (Build_Response (Write (Resp)));
         end;
      end if;

      if URI = "/api/power" then
         declare
            use GNATCOLL.JSON;
            Payload_Str : constant String :=
              (if Raw_S /= "" then Raw_S
               elsif Length (Raw_B) > 0 then To_String (Raw_B)
               else Stream_To_String (Ada.Streams.Stream_Element_Array'
                 (AWS.Status.Binary_Data (Request))));
            P_Res    : constant Read_Result := Read (Payload_Str);
         begin
            if P_Res.Success then
               declare
                  Val : constant JSON_Value := P_Res.Value;
                  On_Batt : Boolean := False;
                  Level   : Natural := 100;
               begin
                  if Has_Field (Val, "on_battery") then
                     On_Batt := Get (Val, "on_battery");
                  end if;
                  if Has_Field (Val, "level") then
                     Level := Get (Val, "level");
                  end if;
                  Model_Manager.Set_Power_Condition (On_Batt, Level);
               end;
            end if;
            return Wrap_Response (Build_Response ("{""status"":""ok""}"));
         end;
      end if;

      if URI = "/api/chat" or else URI = "/api/generate" or else
         URI = "/v1/chat/completions" or else URI = "/v1/completions"
      then
         declare
            Payload : Unbounded_String := (if Raw_S /= "" then
              To_Unbounded_String (Raw_S) 
              elsif Length (Raw_B) > 0 then Raw_B
              else To_Unbounded_String (Stream_To_String (Ada.Streams.Stream_Element_Array'(AWS.Status.Binary_Data (Request)))));
            Prompt  : Unbounded_String := Null_Unbounded_String;
            Req_Model : Unbounded_String := To_Unbounded_String ("Snowball-Enaga");
            S_ID      : Unbounded_String := To_Unbounded_String ("server-stream");
            Is_Streaming : Boolean := False;
            Is_Agentic : Boolean := False;
            Is_Raw_Prompt : Boolean := False;
            Parser_Success : Boolean := False;
         begin
            if Length (Payload) > 0 then
               declare
                  Parser_Result : constant GNATCOLL.JSON.Read_Result :=
                    GNATCOLL.JSON.Read (To_String (Payload));
               begin
                  Parser_Success := Parser_Result.Success;
                  if Parser_Result.Success then
                     declare
                        Val : constant GNATCOLL.JSON.JSON_Value :=
                          Parser_Result.Value;
                     begin
                        if GNATCOLL.JSON.Has_Field (Val, "model") then
                           begin
                              Req_Model := To_Unbounded_String
                                (String'(GNATCOLL.JSON.Get (Val, "model")));
                           exception
                              when others => null;
                           end;
                        end if;
                        if GNATCOLL.JSON.Has_Field (Val, "session_id") then
                           begin
                              S_ID := To_Unbounded_String
                                (String'(GNATCOLL.JSON.Get (Val, "session_id")));
                           exception
                              when others => null;
                           end;
                        end if;
                        if GNATCOLL.JSON.Has_Field (Val, "stream") then
                           begin
                              Is_Streaming := GNATCOLL.JSON.Get (Val, "stream");
                           exception
                              when others => null;
                           end;
                        end if;
                        if GNATCOLL.JSON.Has_Field (Val, "agentic") then
                           begin
                              Is_Agentic := GNATCOLL.JSON.Get (Val, "agentic");
                           exception
                              when others => null;
                           end;
                        end if;
                        if GNATCOLL.JSON.Has_Field (Val, "raw") then
                           begin
                              Is_Raw_Prompt := GNATCOLL.JSON.Get (Val, "raw");
                           exception
                              when others => null;
                           end;
                        end if;

                        if URI = "/v1/chat/completions" or else
                           URI = "/api/chat"
                        then
                           if GNATCOLL.JSON.Has_Field (Val, "messages") then
                              declare
                                 Msgs : constant GNATCOLL.JSON.JSON_Array :=
                                   GNATCOLL.JSON.Get (Val, "messages");
                              begin
                                 for I in 1 .. GNATCOLL.JSON.Length (Msgs) loop
                                    declare
                                       M : constant GNATCOLL.JSON.JSON_Value :=
                                         GNATCOLL.JSON.Get (Msgs, I);
                                       Role : constant String :=
                                         GNATCOLL.JSON.Get (M, "role");
                                       --  Use Extract_Text_Content to handle both
                                       --  string and array content formats
                                       Content : constant String :=
                                         To_String (Extract_Text_Content (M));
                                       Has_OpenAI_Img  : Boolean;
                                       Has_Ollama_Img  : Boolean;
                                       pragma Unreferenced (Has_OpenAI_Img);
                                       pragma Unreferenced (Has_Ollama_Img);
                                    begin
                                       --  Extract text content into prompt
                                       Append (Prompt, "<|im_start|>" & Role & ASCII.LF &
                                               Content & "<|im_end|>" & ASCII.LF);
                                       --  Extract and encode images (OpenAI format)
                                       Has_OpenAI_Img := Extract_And_Encode_Images (M);
                                       --  Extract and encode images (Ollama format)
                                       Has_Ollama_Img := Extract_Ollama_Images (M);
                                    end;
                                 end loop;
                                 Append (Prompt, "<|im_start|>assistant" & ASCII.LF);
                                 --  We've manually joined with ChatML tags, so use Raw mode.
                                 Is_Raw_Prompt := True;
                              exception
                                 when others => null;
                              end;
                           end if;
                        else
                           if GNATCOLL.JSON.Has_Field (Val, "prompt") then
                              begin
                                 Prompt := Ada.Strings.Unbounded.To_Unbounded_String
                                   (String'(GNATCOLL.JSON.Get (Val, "prompt")));
                              exception
                                 when others => null;
                              end;
                           end if;
                        end if;
                     exception
                        when E : others =>
                           Ada.Text_IO.Put_Line ("JSON Parse Exception: " &
                             Ada.Exceptions.Exception_Message (E));
                     end;
                  end if;
               end;
            end if;

            --  Fallback: if JSON parsing failed entirely (Payload was not JSON),
            --  use raw payload as prompt. But if prompt field existed and was
            --  empty (warm-up), do NOT fall back to payload.
            if Length (Prompt) = 0
               and then Length (Payload) > 0
               and then not Parser_Success
            then
               Prompt := Payload;
            end if;

            if Length (Prompt) = 0 then
               --  Ollama warm-up: "ollama run" with no input skips processing
               --  Return immediately without loading model
               return Build_Response
                 ("{""model"": ""Snowball-Enaga:latest"", "
                  & """response"": """", "
                  & """done"": true, "
                  & """total_duration"": 0, "
                  & """load_duration"": 0, "
                  & """prompt_eval_count"": 0, "
                  & """eval_count"": 0}",
                  AWS.Messages.S200);
            end if;

            --  ENFORCEMENT: We ignore the client-requested model for the actual inference
            --  and always route through Hybrid_Generate (Adelaide Core Orchestration).
            --  The Req_Model is only kept to echo back in the response for compatibility
            --   with OpenAI/Ollama clients.
            if Is_Streaming then
               declare
                  use type Streaming_Queue.Queue_Access;
                  Q : constant Streaming_Queue.Queue_Access :=
                    new Streaming_Queue.Queue;
                  T : constant Generator_Task_Access := new Generator_Task;
                  S : constant Streaming_Queue.Response_Stream_Access :=
                    new Streaming_Queue.Response_Stream;
                  Fmt : constant Streaming_Queue.Format_Type :=
                    (if URI = "/v1/chat/completions" or else
                        URI = "/v1/completions"
                     then Streaming_Queue.OpenAI
                     elsif URI = "/api/generate"
                     then Streaming_Queue.Ollama_Generate
                     else Streaming_Queue.Ollama_Chat);
                  Now  : constant Ada.Calendar.Time := Ada.Calendar.Clock;
                  TS   : String := Ada.Calendar.Formatting.Image (Now);
               begin
                  if TS'Length >= 11 then
                     TS (11) := 'T';
                  end if;
                  S.Q := Q;
                  --  [VITAL-DO-NOT-REMOVE] Mandated by user.
                  Ada.Text_IO.Put_Line
                    (AnsiAda.Foreground (AnsiAda.Cyan) & "[Dispatch-V]" &
                     AnsiAda.Reset & " Streaming: Queue created. Q=" &
                     (if Q /= null then "YES" else "NO") &
                     " S.Q=" & (if S.Q /= null then "YES" else "NO") &
                     " Fmt=" & Fmt'Image &
                     " URI=" & URI);

                  --  IMMEDIATE ACK: Set format and push first chunk BEFORE
                  --  starting the generator task. This guarantees sub-ms TTFB
                  --  because the data is in the queue buffer before AWS even
                  --  begins reading from the Response_Stream.
                   Q.Set_Format (Fmt, To_String (Req_Model));
                   if Is_External_Agent then
                      Q.Push ("" & ASCII.LF);
                   else
                        --  Wrap orchestration metadata in <think> block so it is
                       --  captured as internal reasoning, not leaked to the client.
                       --  The </think> close is pushed by Hybrid_Generate after all
                       --  thought content and model thinking have been emitted.
                       declare
                           GPU_Part : Unbounded_String := Null_Unbounded_String;
                       begin
                           if Model_Manager.GPU_Total_MB > 0 then
                               GPU_Part :=
                                  To_Unbounded_String
                                     ("GPU Status: "
                                      & Ada.Strings.Fixed.Trim (Natural'Image (Model_Manager.GPU_Free_MB), Ada.Strings.Both)
                                      & "MB free / "
                                      & Ada.Strings.Fixed.Trim (Natural'Image (Model_Manager.GPU_Total_MB), Ada.Strings.Both)
                                      & "MB total ("
                                      & Ada.Strings.Fixed.Trim (Natural'Image (Model_Manager.GPU_Layer_Percent), Ada.Strings.Both)
                                      & "%) GPU_Layers="
                                      & (if Model_Manager.Acceleration_Silicon_Layer = -1 then "ALL(-1)"
                                         else Ada.Strings.Fixed.Trim (Integer'Image (Model_Manager.Acceleration_Silicon_Layer), Ada.Strings.Both)));
                           else
                               if Model_Manager.GPU_Is_Stable then
                                   GPU_Part := To_Unbounded_String ("GPU Status: STABLE (CPU-only)");
                               else
                                   GPU_Part := To_Unbounded_String ("GPU Status: UNSTABLE (OOM/crash) GPU_Layers=0");
                               end if;
                           end if;
                           Q.Push ("<think>" & ASCII.LF &
                                   "[Adelaide Core Orchestration]" & ASCII.LF &
                                   "Timestamp: " & TS & "Z" & ASCII.LF &
                                   "Session: " & To_String (S_ID) & ASCII.LF &
                                   "Pipeline: Hybrid Multi-Hop Reasoning" & ASCII.LF &
                                   "Model: " & To_String (Req_Model) & ASCII.LF &
                                   "Status: Request received - starting orchestration..." & ASCII.LF &
                                   "Phase: Initializing thought pipeline..." & ASCII.LF &
                                   "Step 1/3: Parsing user intent and context" & ASCII.LF &
                                   "Step 2/3: Loading memory and knowledge context" & ASCII.LF &
                                   "Step 3/3: Generating response with personality" & ASCII.LF &
                                   "Pipeline Architecture: ELP0 (background) -> ELP1 (foreground)" & ASCII.LF &
                                   "Token Budget: Streaming enabled, real-time chunk delivery" & ASCII.LF &
                                   "Memory System: Semantic search + knowledge graph integration" & ASCII.LF &
                                   "Orchestration Mode: Multi-hop reasoning with context faults" & ASCII.LF &
                                   To_String (GPU_Part) & ASCII.LF);
                       end;
                   end if;

                  T.Start (To_String (Prompt), To_String (Req_Model),
                           Fmt, Q, To_String (S_ID),
                           Is_Agentic, Is_Raw_Prompt,
                           Is_External_Agent);
                  --  [VITAL-DO-NOT-REMOVE] Mandated by user.
                  Ada.Text_IO.Put_Line
                    (AnsiAda.Foreground (AnsiAda.Cyan) & "[Dispatch-V]" &
                     AnsiAda.Reset & " Streaming: Generator_Task.Start called." &
                     " Returning stream response.");
                  return Wrap_Response (AWS.Response.Stream
                    (Content_Type => (if URI = "/v1/chat/completions" or else
                                         URI = "/v1/completions"
                                      then "text/event-stream"
                                      else "application/x-ndjson"),
                     Handle => S));
               end;
            else
             Model_Manager.Hybrid_Generate
               (Prompt     => To_String (Prompt),
                Result     => Result,
                Session_ID => To_String (S_ID),
                Agentic    => Is_Agentic,
                Raw_Prompt => Is_Raw_Prompt);

               declare
                  use GNATCOLL.JSON;
                  Resp_Obj : constant JSON_Value := Create_Object;
               begin
                  if URI = "/v1/chat/completions" or else URI = "/v1/completions" then
                     declare
                        Choices : JSON_Array := Empty_Array;
                        Choice  : constant JSON_Value := Create_Object;
                        Msg     : constant JSON_Value := Create_Object;
                        Usage   : constant JSON_Value := Create_Object;
                     begin
                         Set_Field (Msg, "role", "assistant");
                         Set_Field (Msg, "content", To_String (Result));
                        Set_Field (Choice, "index", Integer'(0));
                        Set_Field (Choice, "message", Msg);
                        Set_Field (Choice, "finish_reason", "stop");
                        Append (Choices, Choice);
                        Set_Field (Resp_Obj, "id", "chatcmpl-zephy");
                        Set_Field (Resp_Obj, "object", "chat.completion");
                        Set_Field (Resp_Obj, "created", Integer'(1677652288));
                        Set_Field (Resp_Obj, "model", To_String (Req_Model));
                        Set_Field (Resp_Obj, "choices", Choices);
                        Set_Field (Usage, "prompt_tokens", Integer'(0));
                        Set_Field (Usage, "completion_tokens", Integer'(0));
                        Set_Field (Usage, "total_tokens", Integer'(0));
                        Set_Field (Resp_Obj, "usage", Usage);
                     end;
                  else
                     declare
                        Now  : constant Ada.Calendar.Time := Ada.Calendar.Clock;
                        TS   : String := Ada.Calendar.Formatting.Image (Now);
                     begin
                        if TS'Length >= 11 then
                           TS (11) := 'T';
                        end if;
                        Set_Field (Resp_Obj, "model", To_String (Req_Model));
                        Set_Field (Resp_Obj, "created_at", TS & "Z");
                        
                        if URI = "/api/chat" then
                           declare
                              Msg : constant JSON_Value := Create_Object;
                           begin
                              Set_Field (Msg, "role", "assistant");
                              Set_Field (Msg, "content", To_String (Result));
                              Set_Field (Resp_Obj, "message", Msg);
                           end;
                        else
                           Set_Field (Resp_Obj, "response", To_String (Result));
                        end if;
                        Set_Field (Resp_Obj, "done", True);
                     end;
                  end if;
                  return Wrap_Response (Build_Response (Write (Resp_Obj)));
               end;
            end if;
         end;
      else
         --  =====================================================================
         --  /v1/messages: Claude API endpoint (Anthropic Messages API)
         --  Forwards requests to Claude API when model name starts with "claude"
         --  DO NOT REMOVE, OR YOU WILL BE KILLED
         --  =====================================================================
         if URI = "/v1/messages" then
            declare
               Payload : Unbounded_String := (if Raw_S /= "" then
                 To_Unbounded_String (Raw_S)
                 elsif Length (Raw_B) > 0 then Raw_B
                 else To_Unbounded_String (Stream_To_String (Ada.Streams.Stream_Element_Array'(AWS.Status.Binary_Data (Request)))));
               Req_Model     : Unbounded_String := To_Unbounded_String ("claude-3-5-sonnet-20241022");
               Max_Tokens    : Positive := Claude_Client.Default_Max_Tokens;
               System_Prompt : Unbounded_String := Null_Unbounded_String;
               Temperature   : Float := 1.0;
               Claude_Messages : Claude_Client.Claude_Message_Array (1 .. 50);
               Msg_Count     : Natural := 0;
               API_Key       : Unbounded_String := Null_Unbounded_String;
               Req_Headers   : AWS.Headers.List;
            begin
               --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
               --  Extract API key from x-api-key header
               Req_Headers := AWS.Status.Header (Request);
               API_Key := To_Unbounded_String
                 (AWS.Headers.Get_Values (Req_Headers, "x-api-key"));

               if Length (API_Key) = 0 then
                  --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
                  Ada.Text_IO.Put_Line
                    (AnsiAda.Foreground (AnsiAda.Red)
                     & "[Claude] ERROR: Missing x-api-key header"
                     & AnsiAda.Reset);
                  declare
                     Err_Obj : constant JSON_Value := Create_Object;
                  begin
                     Set_Field (Err_Obj, "type", "error");
                     Set_Field (Err_Obj, "error_type", "authentication_error");
                     Set_Field (Err_Obj, "message", "Missing x-api-key header");
                     return Wrap_Response
                       (Build_Response (Write (Err_Obj), AWS.Messages.S401, "application/json"));
                  end;
               end if;
               if Length (Payload) > 0 then
                  declare
                     Parser_Result : constant GNATCOLL.JSON.Read_Result :=
                       GNATCOLL.JSON.Read (To_String (Payload));
                  begin
                     if Parser_Result.Success then
                        declare
                           Val : constant GNATCOLL.JSON.JSON_Value :=
                             Parser_Result.Value;
                        begin
                           --  Extract model
                           if GNATCOLL.JSON.Has_Field (Val, "model") then
                              begin
                                 Req_Model := To_Unbounded_String
                                   (String'(GNATCOLL.JSON.Get (Val, "model")));
                              exception
                                 when others => null;
                              end;
                           end if;
                           --  Extract max_tokens
                           if GNATCOLL.JSON.Has_Field (Val, "max_tokens") then
                              begin
                                 Max_Tokens := GNATCOLL.JSON.Get (Val, "max_tokens");
                              exception
                                 when others => null;
                              end;
                           end if;
                           --  Extract temperature
                           if GNATCOLL.JSON.Has_Field (Val, "temperature") then
                              begin
                                 Temperature := Float'(GNATCOLL.JSON.Get (Val, "temperature"));
                              exception
                                 when others => null;
                              end;
                           end if;
                           --  Extract system prompt
                           if GNATCOLL.JSON.Has_Field (Val, "system") then
                              begin
                                 System_Prompt := To_Unbounded_String
                                   (String'(GNATCOLL.JSON.Get (Val, "system")));
                              exception
                                 when others => null;
                              end;
                           end if;
                           --  Extract messages array
                           if GNATCOLL.JSON.Has_Field (Val, "messages") then
                              declare
                                 Msgs : constant GNATCOLL.JSON.JSON_Array :=
                                   GNATCOLL.JSON.Get (Val, "messages");
                              begin
                                 for I in 1 .. GNATCOLL.JSON.Length (Msgs) loop
                                    declare
                                       M : constant GNATCOLL.JSON.JSON_Value :=
                                         GNATCOLL.JSON.Get (Msgs, I);
                                       Role : constant String :=
                                         GNATCOLL.JSON.Get (M, "role");
                                       Content : constant String :=
                                         To_String (Extract_Text_Content (M));
                                    begin
                                       Msg_Count := Msg_Count + 1;
                                       if Role = "user" then
                                          Claude_Messages (Msg_Count) :=
                                            (Claude_Client.User,
                                             To_Unbounded_String (Content));
                                       else
                                          Claude_Messages (Msg_Count) :=
                                            (Claude_Client.Assistant,
                                             To_Unbounded_String (Content));
                                       end if;
                                    end;
                                 end loop;
                              end;
                           end if;
                        end;
                     end if;
                  end;
               end if;

               --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
               --  Convert Claude messages to ChatML and call LOCAL model (Snowball-Enaga)
               --  Returns response in Claude Messages API format.
               if Msg_Count > 0 then
                  declare
                     Prompt      : Unbounded_String;
                     Start_Time  : constant Ada.Real_Time.Time := Ada.Real_Time.Clock;
                     Result      : Unbounded_String;
                     Is_Agentic  : Boolean := False;
                     Is_Raw      : Boolean := True;
                  begin
                     --  Build ChatML prompt from Claude messages
                     if Length (System_Prompt) > 0 then
                        Append (Prompt, "im_start" & "system" & ASCII.LF &
                                To_String (System_Prompt) & "im_end" & ASCII.LF);
                     end if;
                     for I in 1 .. Msg_Count loop
                        declare
                           M : constant Claude_Client.Claude_Message := Claude_Messages (I);
                        begin
                           if M.Role = Claude_Client.User then
                              Append (Prompt, "im_start" & "user" & ASCII.LF &
                                      To_String (M.Content) & "im_end" & ASCII.LF);
                           else
                              Append (Prompt, "im_start" & "assistant" & ASCII.LF &
                                      To_String (M.Content) & "im_end" & ASCII.LF);
                           end if;
                        end;
                     end loop;
                     Append (Prompt, "im_start" & "assistant" & ASCII.LF);

                     --  Call local Snowball-Enaga model via Hybrid_Generate
                     Model_Manager.Hybrid_Generate
                       (Prompt     => To_String (Prompt),
                        Result     => Result,
                        Session_ID => "claude-api",
                        Agentic    => Is_Agentic,
                        Raw_Prompt => Is_Raw);

                     declare
                        Elapsed   : constant Duration :=
                          Ada.Real_Time.To_Duration (Ada.Real_Time.Clock - Start_Time);
                        Resp_Obj  : constant JSON_Value := Create_Object;
                        Content_Arr : JSON_Array;
                        Content_Obj : constant JSON_Value := Create_Object;
                     begin
                        --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
                        Ada.Text_IO.Put_Line
                          (AnsiAda.Foreground (AnsiAda.Cyan)
                           & "[Claude] Local model responded in "
                           & Duration'Image (Elapsed) & "s"
                           & AnsiAda.Reset);

                        --  Build Claude-compatible response format
                        Set_Field (Resp_Obj, "id", "msg_" &
                          Ada.Strings.Fixed.Trim (Integer'Image (Integer (Elapsed * 1000.0)), Ada.Strings.Both));
                        Set_Field (Resp_Obj, "type", "message");
                        Set_Field (Resp_Obj, "role", "assistant");
                        Set_Field (Resp_Obj, "model", To_String (Req_Model));
                        Set_Field (Resp_Obj, "stop_reason", "end_turn");
                        Set_Field (Resp_Obj, "stop_sequence", GNATCOLL.JSON.JSON_Null);

                        --  Content block
                        Set_Field (Content_Obj, "type", "text");
                        Set_Field (Content_Obj, "text", To_String (Result));
                        Append (Content_Arr, Content_Obj);
                        Set_Field (Resp_Obj, "content", Content_Arr);

                        --  Usage (estimated)
                        declare
                           Usage : constant JSON_Value := Create_Object;
                        begin
                           Set_Field (Usage, "input_tokens", Integer'(0));
                           Set_Field (Usage, "output_tokens", Integer'(0));
                           Set_Field (Resp_Obj, "usage", Usage);
                        end;

                        return Wrap_Response (Build_Response (Write (Resp_Obj)));
                     end;
                  end;
               else
                  --  No messages, return error
                  declare
                     Err_Obj : constant JSON_Value := Create_Object;
                  begin
                     Set_Field (Err_Obj, "type", "error");
                     Set_Field (Err_Obj, "error_type", "invalid_request_error");
                     Set_Field (Err_Obj, "message", "No messages provided");
                     return Wrap_Response
                       (Build_Response (Write (Err_Obj), AWS.Messages.S400, "application/json"));
                  end;
                 end if;
              end;
           end if;

            --  =====================================================================
            --  /api/snowballEnagaValidationBenchmark: Benchmark endpoint
            --  Requires API key: IknowtheConsequencesAndWouldLockupTheServerForHours
            --  Calls /v1/chat/completions with varying prompt lengths
            --  Streams progress via SSE, logs to stdio
            --  DO NOT REMOVE, OR YOU WILL BE KILLED
            --  =====================================================================
            if URI = "/api/snowballEnagaValidationBenchmark" then
               declare
                  use Benchmark_Manager;
                  use Accuracy_Benchmark_Manager;
                  Req_Headers : AWS.Headers.List;
                  API_Key : Unbounded_String := Null_Unbounded_String;
                  Config : Benchmark_Config;
                  Result : Unbounded_String;
                  Bench_Type : Unbounded_String := To_Unbounded_String("performance");
                  Accuracy_Result : Accuracy_Benchmark_Manager.Benchmark_Result;
               begin
                  --  [DO NOT REMOVE] Extract API key from header
                  Req_Headers := AWS.Status.Header (Request);
                  API_Key := To_Unbounded_String(
                     AWS.Headers.Get_Values (Req_Headers, "x-api-key"));

                  --  [DO NOT REMOVE] Validate API key
                  if not Benchmark_Manager.Validate_API_Key(To_String(API_Key)) then
                     Ada.Text_IO.Put_Line(
                        AnsiAda.Foreground(AnsiAda.Red) &
                        "[Benchmark]" & AnsiAda.Reset &
                        " Invalid API key provided");
                     return Wrap_Response(
                        Build_Response(
                           "{""error"": ""Invalid API key""}",
                           AWS.Messages.S401,
                           "application/json"));
                  end if;

                  --  [DO NOT REMOVE] Log benchmark request
                  Ada.Text_IO.Put_Line(
                     AnsiAda.Foreground(AnsiAda.Cyan) &
                     "[Benchmark]" & AnsiAda.Reset &
                     " Benchmark request received with valid API key");

                  --  Parse request body for configuration
                  declare
                     Payload : Unbounded_String := (if Raw_S /= "" then
                       To_Unbounded_String (Raw_S)
                       elsif Length (Raw_B) > 0 then Raw_B
                       else To_Unbounded_String (Stream_To_String (Ada.Streams.Stream_Element_Array'(AWS.Status.Binary_Data (Request)))));
                  begin
                     if Length (Payload) > 0 then
                        declare
                           Parser_Result : constant GNATCOLL.JSON.Read_Result :=
                             GNATCOLL.JSON.Read (To_String (Payload));
                        begin
                           if Parser_Result.Success then
                              declare
                                 Val : constant GNATCOLL.JSON.JSON_Value :=
                                   Parser_Result.Value;
                              begin
                                 --  Extract benchmark type (performance or accuracy)
                                 if GNATCOLL.JSON.Has_Field (Val, "benchmark_type") then
                                    begin
                                       Bench_Type := To_Unbounded_String(
                                          String'(GNATCOLL.JSON.Get (Val, "benchmark_type")));
                                    exception
                                       when others => null;
                                    end;
                                 end if;
                                 --  Extract prompt_lengths (for performance)
                                 if GNATCOLL.JSON.Has_Field (Val, "prompt_lengths") then
                                    begin
                                       Config.Prompt_Lengths := To_Unbounded_String(
                                          String'(GNATCOLL.JSON.Get (Val, "prompt_lengths")));
                                    exception
                                       when others => null;
                                    end;
                                 end if;
                                 --  Extract generation_length (for performance)
                                 if GNATCOLL.JSON.Has_Field (Val, "generation_length") then
                                    begin
                                       Config.Generation_Length := Integer'Value(
                                          String'(GNATCOLL.JSON.Get (Val, "generation_length")));
                                    exception
                                       when others => null;
                                    end;
                                 end if;
                                 --  Extract accuracy_benchmark (for accuracy)
                                 --  Options: mmlu, mmlu_pro, kmmlu, cmmlu, jmmlu,
                                 --           gsm8k, mathqa, humaneval, mbpp, livecodebench,
                                 --           hellaswag, truthfulqa, arc_challenge, winogrande,
                                 --           bbq, safetybench
                              end;
                           end if;
                        end;
                     end if;
                  end;

                  --  [DO NOT REMOVE] Run appropriate benchmark type
                  if To_String(Bench_Type) = "accuracy" then
                     --  Run accuracy benchmark
                     declare
                        Acc_Bench : Unbounded_String := To_Unbounded_String("mmlu");
                     begin
                        --  TODO: Extract accuracy_benchmark from request
                        --  For now, run MMLU as default
                        Put_Line(AnsiAda.Foreground(AnsiAda.Cyan) &
                                 "[Benchmark]" & AnsiAda.Reset &
                                 " Running accuracy benchmark: " & To_String(Acc_Bench));

                        Run_Accuracy_Benchmark(
                           Benchmark => BENCH_MMLU,
                           Sample_Size => 0,
                           On_Progress => null,
                           Result => Accuracy_Result
                        );

                        Result := To_Unbounded_String(
                           "{""benchmark"":""accuracy""," &
                           """name"":" & To_String(Acc_Bench) & "," &
                           """accuracy"":" & Float'Image(Accuracy_Result.Accuracy) & "," &
                           """total"":" & Natural'Image(Accuracy_Result.Total_Questions) & "," &
                           """correct"":" & Natural'Image(Accuracy_Result.Correct_Count) & "," &
                           """time"":" & Float'Image(Accuracy_Result.Time_Seconds) & "}");
                     end;
                  else
                     --  Run performance benchmark (original)
                     Run_Benchmark(
                        Config => Config,
                        On_Progress => null,
                        Result => Result
                     );
                  end if;

                  --  [DO NOT REMOVE] Return benchmark result
                  return Wrap_Response(
                     Build_Response(
                        To_String(Result),
                        AWS.Messages.S200,
                        "application/json"));
               end;
            end if;

           --  =====================================================================
           --  /v1/images/generations: OpenAI-compatible image generation endpoint
           --  Two-stage pipeline: FLUX sparse -> SD refinement
           --  DO NOT REMOVE, OR YOU WILL BE KILLED
           --  =====================================================================
          if URI = "/v1/images/generations" then
             declare
                Payload : Unbounded_String := (if Raw_S /= "" then
                  To_Unbounded_String (Raw_S)
                  elsif Length (Raw_B) > 0 then Raw_B
                  else To_Unbounded_String (Stream_To_String (Ada.Streams.Stream_Element_Array'(AWS.Status.Binary_Data (Request)))));
                Req_Model   : Unbounded_String := To_Unbounded_String ("flux-schnell");
                Prompt      : Unbounded_String := Null_Unbounded_String;
                N_Images    : Integer := 1;
                Width       : Integer := 1024;
                Height      : Integer := 1024;
                Seed        : Long_Long_Integer := -1;
                Quality     : Unbounded_String := To_Unbounded_String ("standard");
                Style       : Unbounded_String := To_Unbounded_String ("vivid");
                Response_Fmt: Unbounded_String := To_Unbounded_String ("b64_json");
                Image_B64   : Unbounded_String := Null_Unbounded_String;
                Error_Msg   : Unbounded_String := Null_Unbounded_String;
                Img_Obj     : JSON_Value;
                Data_Arr    : JSON_Value;
                Resp_Obj    : JSON_Value;
             begin
                --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
                --  Parse request body
                if Length (Payload) > 0 then
                   declare
                      Parser_Result : constant GNATCOLL.JSON.Read_Result :=
                        GNATCOLL.JSON.Read (To_String (Payload));
                   begin
                      if Parser_Result.Success then
                         declare
                            Val : constant GNATCOLL.JSON.JSON_Value :=
                              Parser_Result.Value;
                         begin
                            --  Extract prompt (required)
                            if GNATCOLL.JSON.Has_Field (Val, "prompt") then
                               begin
                                  Prompt := To_Unbounded_String
                                    (String'(GNATCOLL.JSON.Get (Val, "prompt")));
                               exception
                                  when others => null;
                               end;
                            end if;
                            --  Extract model
                            if GNATCOLL.JSON.Has_Field (Val, "model") then
                               begin
                                  Req_Model := To_Unbounded_String
                                    (String'(GNATCOLL.JSON.Get (Val, "model")));
                               exception
                                  when others => null;
                               end;
                            end if;
                            --  Extract n (number of images)
                            if GNATCOLL.JSON.Has_Field (Val, "n") then
                               begin
                                  N_Images := Integer'(GNATCOLL.JSON.Get (Val, "n"));
                               exception
                                  when others => null;
                               end;
                            end if;
                            --  Extract size (WxH format)
                            if GNATCOLL.JSON.Has_Field (Val, "size") then
                               declare
                                  Size_Str : constant String :=
                                    String'(GNATCOLL.JSON.Get (Val, "size"));
                                  X_Pos    : Natural;
                               begin
                                  X_Pos := Ada.Strings.Fixed.Index (Size_Str, "x");
                                  if X_Pos > 0 then
                                     Width := Integer'Value (Size_Str (Size_Str'First .. X_Pos - 1));
                                     Height := Integer'Value (Size_Str (X_Pos + 1 .. Size_Str'Last));
                                  end if;
                               exception
                                  when others => null;
                               end;
                            end if;
                            --  Extract seed
                            if GNATCOLL.JSON.Has_Field (Val, "seed") then
                               begin
                                  Seed := Long_Long_Integer (Integer'(GNATCOLL.JSON.Get (Val, "seed")));
                               exception
                                  when others => null;
                               end;
                            end if;
                            --  Extract quality
                            if GNATCOLL.JSON.Has_Field (Val, "quality") then
                               begin
                                  Quality := To_Unbounded_String
                                    (String'(GNATCOLL.JSON.Get (Val, "quality")));
                               exception
                                  when others => null;
                               end;
                            end if;
                            --  Extract style
                            if GNATCOLL.JSON.Has_Field (Val, "style") then
                               begin
                                  Style := To_Unbounded_String
                                    (String'(GNATCOLL.JSON.Get (Val, "style")));
                               exception
                                  when others => null;
                               end;
                            end if;
                            --  Extract response_format
                            if GNATCOLL.JSON.Has_Field (Val, "response_format") then
                               begin
                                  Response_Fmt := To_Unbounded_String
                                    (String'(GNATCOLL.JSON.Get (Val, "response_format")));
                               exception
                                  when others => null;
                               end;
                            end if;
                         end;
                      end if;
                   end;
                end if;

                --  Validate prompt
                if Length (Prompt) = 0 then
                   declare
                      Err_Obj : constant JSON_Value := Create_Object;
                   begin
                      Set_Field (Err_Obj, "error", "Missing required parameter: prompt");
                      return Wrap_Response
                        (Build_Response (Write (Err_Obj), AWS.Messages.S400, "application/json"));
                   end;
                end if;

                --  Log request
                Ada.Text_IO.Put_Line
                  (AnsiAda.Foreground (AnsiAda.Cyan)
                   & "[ImgGen] New image generation request"
                   & AnsiAda.Reset);
                Ada.Text_IO.Put_Line
                  (AnsiAda.Foreground (AnsiAda.Cyan)
                   & "[ImgGen]   Prompt: " & To_String (Prompt)
                   & AnsiAda.Reset);
                Ada.Text_IO.Put_Line
                  (AnsiAda.Foreground (AnsiAda.Cyan)
                   & "[ImgGen]   Size: " & Integer'Image (Width) & "x" & Integer'Image (Height)
                   & " Seed: " & Long_Long_Integer'Image (Seed)
                   & AnsiAda.Reset);

                --  Generate image using two-stage pipeline
                SD_Manager.Generate_Two_Stage
                  (Prompt         => To_String (Prompt),
                   Width          => Width,
                   Height         => Height,
                   Seed           => Seed,
                   Flux_Steps     => (if Quality = "hd" then 8 else 4),
                   Flux_Cfg       => 1.0,
                   Refine_Enabled => (Quality = "hd"),
                   Refine_Steps   => 8,
                   Refine_Strength => 0.4,
                   Image_B64      => Image_B64,
                   Error_Msg      => Error_Msg);

                --  Check for errors
                if Length (Error_Msg) > 0 then
                   declare
                      Err_Obj : constant JSON_Value := Create_Object;
                   begin
                      Set_Field (Err_Obj, "error", To_String (Error_Msg));
                      return Wrap_Response
                        (Build_Response (Write (Err_Obj), AWS.Messages.S500, "application/json"));
                   end;
                end if;

                --  Build OpenAI-compatible response
                Resp_Obj := Create_Object;
                Set_Field (Resp_Obj, "created", Integer (Ada.Calendar.Seconds (Ada.Calendar.Clock)));

                --  Build data array with image objects
                Img_Obj := Create_Object;
                if Response_Fmt = "b64_json" then
                   Set_Field (Img_Obj, "b64_json", To_String (Image_B64));
                else
                   --  URL format not supported, return b64_json as fallback
                   Set_Field (Img_Obj, "b64_json", To_String (Image_B64));
                end if;
                Set_Field (Img_Obj, "revised_prompt", To_String (Prompt));

                --  GNATCOLL.JSON workaround: wrap in object with "0" key
                --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
                Data_Arr := Create_Object;
                Set_Field (Data_Arr, "0", Img_Obj);
                Set_Field (Resp_Obj, "data", Data_Arr);

                Ada.Text_IO.Put_Line
                  (AnsiAda.Foreground (AnsiAda.Green)
                   & "[ImgGen] Image generation complete. Returning Base64 response."
                   & AnsiAda.Reset);

                return Wrap_Response (Build_Response (Write (Resp_Obj)));
             end;
          end if;

          return Build_Response ("Adelaide API", AWS.Messages.S404, "text/plain");
      end if;
   exception
      when E : others =>
         --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
         --  UNKNOWN/CATEGORIZED ERROR: Dump full exception and red banner.
         --  Server keeps running and continues serving other requests.
         Ada.Text_IO.Put_Line
            (AnsiAda.Foreground (AnsiAda.Red)
             & "=========================================================="
             & AnsiAda.Reset);
         Ada.Text_IO.Put_Line
            (AnsiAda.Foreground (AnsiAda.Red)
             & "  !!! UNKNOWN ERROR / UNCATEGORIZED EXCEPTION !!!"
             & AnsiAda.Reset);
         Ada.Text_IO.Put_Line
            (AnsiAda.Foreground (AnsiAda.Red)
             & "  URI: " & URI
             & AnsiAda.Reset);
         Ada.Text_IO.Put_Line
            (AnsiAda.Foreground (AnsiAda.Red)
             & "  UA: " & UA
             & AnsiAda.Reset);
         Ada.Text_IO.Put_Line
            (AnsiAda.Foreground (AnsiAda.Red)
             & "  Exception: "
             & Ada.Exceptions.Exception_Name (E)
             & AnsiAda.Reset);
         Ada.Text_IO.Put_Line
            (AnsiAda.Foreground (AnsiAda.Red)
             & "  Message: "
             & Ada.Exceptions.Exception_Message (E)
             & AnsiAda.Reset);
         Ada.Text_IO.Put_Line
            (AnsiAda.Foreground (AnsiAda.Red)
             & "  Full Trace:"
             & AnsiAda.Reset);
         Ada.Text_IO.Put_Line
            (Ada.Exceptions.Exception_Information (E));
         Ada.Text_IO.Put_Line
            (AnsiAda.Foreground (AnsiAda.Red)
             & "=========================================================="
             & AnsiAda.Reset);
         Ada.Text_IO.Put_Line
            (AnsiAda.Foreground (AnsiAda.Red)
             & "  REPORT TO DEVELOPER! Server continues serving."
             & AnsiAda.Reset);
         Ada.Text_IO.Put_Line
            (AnsiAda.Foreground (AnsiAda.Red)
             & "=========================================================="
             & AnsiAda.Reset);
         --  Return error response — server stays alive for next request
         return Build_Response
            ("{""error"": ""Unknown error occurred"", ""detail"": """
             & Ada.Exceptions.Exception_Message (E) & """}",
             AWS.Messages.S500);
   end;
   exception
   when E : others =>
      Ada.Text_IO.Put_Line ("Server Error: " &
        Ada.Exceptions.Exception_Message (E));
      return Build_Response ("{}", AWS.Messages.S500);
   end Dispatch;

end Adelaide_Server_Pkg;
