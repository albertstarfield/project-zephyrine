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
with Interfaces;
with Model_Manager;
with Streaming_Queue;
use Streaming_Queue;
with Multimodal_Content_Parser;
use Multimodal_Content_Parser;
with AWS.Response.Set;
with AWS.Messages;
with GNATCOLL.JSON;
with Math_Utils;
with Ada.Containers.Indefinite_Ordered_Maps;
with Ada.Real_Time;
with Fuzzy_Match;

--  ===========================================================================
--  DISPATCH QUIRKS & DISCOVERED WORKAROUNDS
--  ===========================================================================
--  [QUIRK-D01] [ALL] Model routing always goes through Hybrid_Generate
--  Regardless of the "model" field in the client request, the server
--  ignores it and routes everything through Hybrid_Generate (Adelaide
--  Core Orchestration) at line 731-736.  The requested model name is
--  only echoed back in the response for OpenAI/Ollama compatibility.
--  This means:
--     - "model": "gpt-4" -> still uses Qwen3.5-9B backend
--     - "model": "llama3" -> still uses Qwen3.5-9B backend
--     - "model": "Snowball-Enaga" -> uses Qwen3.5-9B backend (correct)
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
         return Build_Response ("{""version"": ""Project-Zephyrine-0.27""}");
      end if;

      if URI = "/v1/audio/transcriptions" then
         declare
            use GNATCOLL.JSON;
            R : constant JSON_Value := Create_Object;
            Raw_Payload : constant Ada.Streams.Stream_Element_Array :=
              AWS.Status.Binary_Data (Request);
            Num_Floats : constant Interfaces.Unsigned_64 :=
              Interfaces.Unsigned_64 (Raw_Payload'Length / 4);
            type Float_Array is array (1 .. Natural (Num_Floats)) of
              aliased Float;
            Audio_Floats : Float_Array with Import, Address => Raw_Payload'Address;

            Transcript : Unbounded_String;
            use type Interfaces.Unsigned_64;
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
             
             Num_Floats : constant Interfaces.Unsigned_64 :=
               Interfaces.Unsigned_64 (Raw_Payload'Length / 4);
             type Float_Array is array (1 .. Natural (Num_Floats)) of
               aliased Float;
             Audio_Floats : Float_Array with Import, Address => Raw_Payload'Address;

             Transcript : Unbounded_String;
             LLM_Result : Unbounded_String;
             T_Start : constant Ada.Real_Time.Time := Ada.Real_Time.Clock;
             
             -- Convert vision context param to unbounded string if provided
             Vision_Context_From_Param : Unbounded_String;

             use type Interfaces.Unsigned_64;
             use type Ada.Real_Time.Time;
          begin
            Vision_Context_From_Param := To_Unbounded_String (Vision_Context_Param);
            Handless_Stage := To_Unbounded_String ("Transcribing...");
            Handless_Input_Text := To_Unbounded_String ("");
            Handless_Output_Text := To_Unbounded_String ("");

            if Num_Floats > 0 then
               Transcript := To_Unbounded_String
                 (Moonshine_Interface.Transcribe_Raw_PCM
                    (Audio_Floats (1)'Access, Num_Floats));
            else
               Transcript := To_Unbounded_String ("");
            end if;

             Handless_Input_Text := Transcript;
             Handless_Stage := To_Unbounded_String ("Generating...");

             if Length (Transcript) > 0 then
                declare
                   Vision_Arr : GNATCOLL.JSON.JSON_Array :=
                     GNATCOLL.JSON.Empty_Array;
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

                   Model_Manager.Hybrid_Generate
                     (Prompt     => To_String (Transcript),
                      Result     => LLM_Result,
                      Images     => Vision_Arr,
                      Session_ID => "server-handless",
                      Agentic    => True,
                      Raw_Prompt => False);
                end;
             else
               Model_Manager.Hybrid_Generate
                 (Prompt     => "Proactively initiate the conversation. " &
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
         return Build_Response ("{""models"": [{""name"": ""metamodel-ELP0"", " &
           """size"": 0, ""size_vram"": 0}, {""name"": ""metamodel-ELP1"", " &
           """size"": 0, ""size_vram"": 0}]}");
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
            return Wrap_Response (Build_Response ("{""models"": [{""name"": ""Snowball-Enaga""}]}"));
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

            if Length (Prompt) = 0 and then Length (Payload) > 0 then
               Prompt := Payload; -- Fallback if parsing fails or fields missing
            end if;

            if Length (Prompt) = 0 then
               return Build_Response ("{""response"": """"}",
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
                       Q.Push ("<think>" & ASCII.LF &
                               "[Adelaide Core Orchestration]" & ASCII.LF &
                               "Timestamp: " & TS & "Z" & ASCII.LF &
                               "Session: " & To_String (S_ID) & ASCII.LF &
                               "Pipeline: Hybrid Multi-Hop Reasoning" & ASCII.LF &
                               "Model: " & To_String (Req_Model) & ASCII.LF &
                               "Status: Request received - starting orchestration..." & ASCII.LF);
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
         return Build_Response ("Adelaide API", AWS.Messages.S404, "text/plain");
      end if;
   exception
      when E : others =>
         Ada.Text_IO.Put_Line ("Dispatch Error: " &
           Ada.Exceptions.Exception_Message (E));
         return Build_Response ("{}", AWS.Messages.S500);
   end;
   exception
   when E : others =>
      Ada.Text_IO.Put_Line ("Server Error: " &
        Ada.Exceptions.Exception_Message (E));
      return Build_Response ("{}", AWS.Messages.S500);
   end Dispatch;

end Adelaide_Server_Pkg;
