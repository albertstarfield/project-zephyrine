with AnsiAda;
with Ada.Text_IO;
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
with AWS.Response.Set;
with AWS.Messages;
with GNATCOLL.JSON;
with Math_Utils;
with Ada.Containers.Indefinite_Ordered_Maps;
with Ada.Real_Time;

package body Adelaide_Server_Pkg is

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
      entry Start
        (Prompt : String; Model_Name : String;
         Format : Streaming_Queue.Format_Type;
         Q : Streaming_Queue.Queue_Access;
         Agentic : Boolean := False; Raw_Prompt : Boolean := False);
   end Generator_Task;

   type Generator_Task_Access is access Generator_Task;

   task body Generator_Task is
      P : Unbounded_String;
      QA : Streaming_Queue.Queue_Access;
      Res : Unbounded_String;
      Is_Ag : Boolean;
      Is_Raw : Boolean;
      use type Streaming_Queue.Queue_Access;
   begin
      accept Start
        (Prompt : String; Model_Name : String;
         Format : Streaming_Queue.Format_Type;
         Q : Streaming_Queue.Queue_Access;
         Agentic : Boolean := False; Raw_Prompt : Boolean := False)
      do
         P := To_Unbounded_String (Prompt);
         QA := Q;
         Is_Ag := Agentic;
         Is_Raw := Raw_Prompt;
      end Start;

      Model_Manager.Hybrid_Generate
        (Prompt     => To_String (P),
         Result     => Res,
         Session_ID => "server-stream",
         Stream     => QA,
         Agentic    => Is_Ag,
         Raw_Prompt => Is_Raw);

      QA.Close;
   exception
      when E : others =>
         Ada.Text_IO.Put_Line ("Error in Generator_Task: " &
                               Ada.Exceptions.Exception_Message (E));
         if QA /= null then
            QA.Close;
         end if;
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

   --------------
   function Dispatch
     (Request : AWS.Status.Data) return AWS.Response.Data
   is
      URI    : constant String := AWS.Status.URI (Request);
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
             Vision_Context_From_Param : Unbounded_String := To_Unbounded_String("");
             if Length (Vision_Context_Param) > 0 then
                Vision_Context_From_Param := To_Unbounded_String (Vision_Context_Param);
             end if;

             use type Interfaces.Unsigned_64;
             use type Ada.Real_Time.Time;
          begin
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

             -- Parse optional vision context from JSON payload if provided
             declare
               Payload_Str : constant String := (if Num_Floats > 0 then
                 -- For PCM audio, we need to extract any JSON metadata that was prepended
                 "" 
                 else To_String (Raw_B)
               end if);
             
               Parser_Result : constant Read_Result := Read (Payload_Str);
             begin
               if Parser_Result.Success then
                 declare
                   Val : constant JSON_Value := Parser_Result.Value;
                 begin
                   if Has_Field (Val, "vision_context_b64") then
                     Vision_Context_B64 := String'(Get (Val, "vision_context_b64"));
                   end if;
                 end;
               end if;
             exception
               when others => null;  -- Silently ignore JSON parse errors
             end;

             if Length (Transcript) > 0 then
                declare
                   Vision_Arr : GNATCOLL.JSON.JSON_Array :=
                     GNATCOLL.JSON.Empty_Array;
                begin
                   if Length (Handless_Vision_Context) > 0 then
                      GNATCOLL.JSON.Append
                        (Vision_Arr,
                         Create (To_String (Handless_Vision_Context)));
                      Handless_Vision_Context := To_Unbounded_String("");
                   end if;
                   
                   -- Also add the new vision context from this request
                   if Length (Vision_Context_B64) > 0 then
                      GNATCOLL.JSON.Append(Vision_Arr, Create (Vision_Context_B64));
                      Vision_Context_B64 := "";
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
            Main_US : constant Float := Float (WCET_Main_Loop) * 1_000_000.0;
         begin
            Set_Field (R, "WCET_ELP0", Float(Model_Manager.Current_WCET_ELP0));
            Set_Field (R, "WCET_ELP1", Float(Model_Manager.Current_WCET_ELP1));
            Set_Field (R, "WCET_ELP2", Float(Model_Manager.Current_WCET_ELP2));
            Set_Field (R, "WCET_ELP3", Float(Model_Manager.Current_WCET_ELP3));
            Set_Field (R, "Jitter_Avg_uS", Float (Model_Manager.Current_Jitter_Avg * 1_000_000.0));
            Set_Field (R, "Jitter_Max_uS", Float (Model_Manager.Current_Jitter_Max * 1_000_000.0));
            Set_Field (R, "WCET_mainLoop_uS", Main_US);

            -- Handless Mode telemetry
            Set_Field (R, "Handless_Stage", To_String(Handless_Stage));
            Set_Field (R, "Handless_WCET", Handless_WCET);
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

      if URI = "/api/tags" then
         return Build_Response ("{""models"": [{""name"": ""qwen:0.8b""}]}");
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
            Req_Model : Unbounded_String := To_Unbounded_String ("qwen:0.8b");
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
                                       Content : constant String :=
                                         GNATCOLL.JSON.Get (M, "content");
                                    begin
                                       Append (Prompt, "<|im_start|>" & Role & ASCII.LF &
                                               Content & "<|im_end|>" & ASCII.LF);
                                    end;
                                 end loop;
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
            Ada.Text_IO.Put_Line ("[DEBUG] Payload: " & To_String (Payload));
            Ada.Text_IO.Put_Line ("[DEBUG] Parsed Prompt: " & To_String (Prompt));

            if Length (Prompt) = 0 then
               return Build_Response ("{""response"": """"}",
                                      AWS.Messages.S200);
            end if;

            if Is_Streaming then
               declare
                  use type Streaming_Queue.Queue_Access;
                  Q : constant Streaming_Queue.Queue_Access :=
                    new Streaming_Queue.Queue;
                  T : constant Generator_Task_Access := new Generator_Task;
                  S : constant Streaming_Queue.Response_Stream_Access :=
                    new Streaming_Queue.Response_Stream;
               begin
                  S.Q := Q;
                  T.Start (To_String (Prompt), To_String (Req_Model),
                           (if URI = "/v1/chat/completions" or else
                               URI = "/v1/completions"
                            then Streaming_Queue.OpenAI
                            else Streaming_Queue.Ollama), Q,
                           Is_Agentic, Is_Raw_Prompt);
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
                  Session_ID => "server-sync",
                  Agentic    => Is_Agentic,
                  Raw_Prompt => Is_Raw_Prompt);

               declare
                  R : constant GNATCOLL.JSON.JSON_Value :=
                    GNATCOLL.JSON.Create_Object;
               begin
                  GNATCOLL.JSON.Set_Field (R, "model", To_String (Req_Model));
                  GNATCOLL.JSON.Set_Field (R, "response", To_String (Result));
                  GNATCOLL.JSON.Set_Field (R, "done", True);
                  return Build_Response (GNATCOLL.JSON.Write (R));
               end;
            end if;
         end;

      elsif URI = "/api/embeddings" or else URI = "/api/embed" then
         declare
            Payload : constant String := (if Raw_S /= "" then Raw_S
              else To_String (Raw_B));
            Vec     : Math_Utils.Vector (1 .. 4096) := [others => 0.0];
            Len     : Natural := 0;
            Emb_Arr : GNATCOLL.JSON.JSON_Array := GNATCOLL.JSON.Empty_Array;
            R       : constant GNATCOLL.JSON.JSON_Value :=
              GNATCOLL.JSON.Create_Object;
         begin
            Model_Manager.Get_Embedding (Payload, Vec, Len);
            for I in 1 .. (if Len > 0 then Len else 128) loop
               GNATCOLL.JSON.Append (Emb_Arr, GNATCOLL.JSON.Create
                 (Long_Float (if Len > 0 then Vec (I) else 0.1)));
            end loop;
            GNATCOLL.JSON.Set_Field (R, "embedding", Emb_Arr);
            return Build_Response (GNATCOLL.JSON.Write (R));
         end;
      else
         return Build_Response ("Adelaide API", AWS.Messages.S404, "text/plain");
      end if;
   exception
      when E : others =>
         Ada.Text_IO.Put_Line ("Server Error: " &
           Ada.Exceptions.Exception_Message (E));
         return Build_Response ("{}", AWS.Messages.S500);
   end Dispatch;

end Adelaide_Server_Pkg;
