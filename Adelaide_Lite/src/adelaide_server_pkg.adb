with AWS.Headers;
with AWS.Headers.Set;
with GNATCOLL.JSON; use GNATCOLL.JSON;
with Ada.Text_IO; use Ada.Text_IO;
with Model_Manager;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Ada.Exceptions;
with Math_Utils;
with Ada.Strings.Fixed; use Ada.Strings.Fixed;
with Ada.Calendar;
with Ada.Calendar.Formatting;
with Ada.Real_Time;

package body Adelaide_Server_Pkg is

   procedure Register (ID : String; Q : Streaming_Queue.Queue_Access) is
      pragma Unreferenced (Q);
   begin
      Put_Line ("[Server] Registered: " & ID);
   end Register;

   procedure Unregister (ID : String) is
   begin
      Put_Line ("[Server] Unregistered: " & ID);
   end Unregister;

   procedure Push_Log (ID : String; Log : String) is
   begin
      Put_Line ("[Log] [" & ID & "] " & Log);
   end Push_Log;

   --  CORS and Header Helper
   function Build_Response
     (Content : String;
      Status  : AWS.Messages.Status_Code := AWS.Messages.S200;
      Type_Str : String := "application/json") return AWS.Response.Data
   is
      HD : AWS.Headers.List;
   begin
      AWS.Headers.Set.Add (HD, "Access-Control-Allow-Origin", "*");
      AWS.Headers.Set.Add
        (HD, "Access-Control-Allow-Methods", "GET, POST, OPTIONS");
      AWS.Headers.Set.Add
        (HD, "Access-Control-Allow-Headers", "Content-Type, Authorization");

      return AWS.Response.Build
        (Content_Type => Type_Str,
         Message_Body => Content,
         Status_Code  => Status,
         Header       => HD);
   end Build_Response;

   function Dispatch (Request : AWS.Status.Data) return AWS.Response.Data is
      URI    : constant String := AWS.Status.URI (Request);
      Method : constant String := AWS.Status.Method (Request);
   begin
      Put_Line ("[Server] " & Method & " " & URI);

      if Method = "OPTIONS" then
         return Build_Response ("", AWS.Messages.S204);
      end if;

      if URI = "/v1/models" or else URI = "/api/tags" then
         declare
            Resp   : constant JSON_Value := Create_Object;
            Models : JSON_Array := Empty_Array;
            --  Helper procedure to add a model to the advertised list
            procedure Add_Model (Id, Family : String) is
               M : constant JSON_Value := Create_Object;
               D : constant JSON_Value := Create_Object;
            begin
               --  OpenAI fields
               Set_Field (M, "id", Id);
               Set_Field (M, "object", "model");
               Set_Field (M, "created", Long_Integer'(1686935002));
               Set_Field (M, "owned_by", "adelaide");

               --  Ollama fields
               Set_Field (M, "name", Id);
               Set_Field (M, "model", Id);
               Set_Field (M, "modified_at", "2024-05-21T15:00:00Z");
               Set_Field (M, "size", Long_Integer'(4000000000));
               Set_Field (M, "digest", "sha256:adelaide" & Id);

               Set_Field (D, "format", "gguf");
               Set_Field (D, "family", Family);
               Set_Field (M, "details", D);
               Append (Models, M);
            end Add_Model;
         begin
            Add_Model ("adelaide-hybrid", "qwen2");
            Add_Model ("adelaide-embedding", "bert");
            Add_Model ("metamodel", "qwen2");
            Add_Model ("adelaide-metamodel", "qwen2");

            Set_Field (Resp, "object", "list");
            Set_Field (Resp, "data", Models);
            Set_Field (Resp, "models", Models);
            return Build_Response (Write (Resp));
         end;

      elsif URI = "/api/show" then
         declare
            Raw_S   : constant String := AWS.Status.Payload (Request);
            Raw_B   : constant Unbounded_String :=
              AWS.Status.Binary_Data (Request);
            Payload : Unbounded_String;
            Model_Name : Unbounded_String :=
              To_Unbounded_String ("adelaide-hybrid");
         begin
            if Raw_S /= "" then
               Payload := To_Unbounded_String (Raw_S);
            else
               Payload := Raw_B;
            end if;

            if Length (Payload) > 0 then
               declare
                  Parser_Result : constant Read_Result :=
                    Read (To_String (Payload));
               begin
                  if Parser_Result.Success then
                     declare
                        Val : constant JSON_Value := Parser_Result.Value;
                     begin
                        if Has_Field (Val, "name") then
                           Model_Name := To_Unbounded_String
                             (String'(Get (Val, "name")));
                        elsif Has_Field (Val, "model") then
                           Model_Name := To_Unbounded_String
                             (String'(Get (Val, "model")));
                        end if;
                     end;
                  end if;
               end;
            end if;

            declare
               Resp : constant JSON_Value := Create_Object;
               Details : constant JSON_Value := Create_Object;
               Model_Info : constant JSON_Value := Create_Object;
               Families : JSON_Array := Empty_Array;
               Name_Str : constant String := To_String (Model_Name);
            begin
               if Name_Str = "adelaide-embedding" then
                  Append (Families, Create ("bert"));
                  Set_Field (Details, "parent_model", "");
                  Set_Field (Details, "format", "gguf");
                  Set_Field (Details, "family", "bert");
                  Set_Field (Details, "families", Families);
                  Set_Field (Details, "parameter_size", "0.1B");
                  Set_Field (Details, "quantization_level", "F16");

                  Set_Field (Model_Info, "general.architecture", "bert");
                  Set_Field
                    (Model_Info, "general.file_type", Integer'(12));
                  Set_Field
                    (Model_Info, "general.parameter_count",
                     Long_Integer'(100000000));
                  Set_Field
                    (Model_Info, "bert.context_length",
                     Long_Integer'(4294967296));

                  Set_Field
                    (Resp, "modelfile",
                     "# Adelaide Embedding ModelFile" & ASCII.LF &
                     "FROM adelaide-embedding" & ASCII.LF &
                     "PARAMETER num_ctx 4294967296");
                  Set_Field
                    (Resp, "parameters",
                     "num_ctx" & ASCII.HT & "4294967296");
                  Set_Field (Resp, "template", "{{ .Prompt }}");
                  Set_Field (Resp, "details", Details);
                  Set_Field (Resp, "model_info", Model_Info);
               else
                  Append (Families, Create ("qwen2"));
                  Set_Field (Details, "parent_model", "");
                  Set_Field (Details, "format", "gguf");
                  Set_Field (Details, "family", "qwen2");
                  Set_Field (Details, "families", Families);
                  Set_Field (Details, "parameter_size", "4B");
                  Set_Field (Details, "quantization_level", "Q4_K_M");

                  Set_Field
                    (Model_Info, "general.architecture", "llama");
                  Set_Field
                    (Model_Info, "general.file_type", Integer'(12));
                  Set_Field
                    (Model_Info, "general.parameter_count",
                     Long_Integer'(4000000000));
                  Set_Field
                    (Model_Info, "qwen2.context_length",
                     Long_Integer'(9223372036854775807));

                  Set_Field
                    (Resp, "modelfile",
                     "# Adelaide ModelFile" & ASCII.LF &
                     "FROM " & Name_Str & ASCII.LF &
                     "PARAMETER num_ctx 9223372036854775808");
                  Set_Field
                    (Resp, "parameters",
                     "num_ctx" & ASCII.HT & "9223372036854775808");
                  Set_Field
                    (Resp, "template",
                     "{{ .System }}" & ASCII.LF & "{{ .Prompt }}");
                  Set_Field (Resp, "details", Details);
                  Set_Field (Resp, "model_info", Model_Info);
               end if;

               return Build_Response (Write (Resp));
            end;
         end;

      elsif URI = "/api/chat" or else URI = "/v1/chat/completions" then
         declare
            Raw_S   : constant String := AWS.Status.Payload (Request);
            Raw_B   : constant Unbounded_String :=
              AWS.Status.Binary_Data (Request);
            Payload : Unbounded_String;
            Val     : JSON_Value;
            Prompt  : Unbounded_String := To_Unbounded_String ("No payload");
            Images  : JSON_Array := Empty_Array;
            Result  : Unbounded_String;
            Resp    : constant JSON_Value := Create_Object;
            Choices : JSON_Array := Empty_Array;
            Choice  : constant JSON_Value := Create_Object;
            Msg_Out : constant JSON_Value := Create_Object;
            Req_Model : Unbounded_String :=
              To_Unbounded_String ("adelaide-hybrid");

            --  Helper procedure to extract text and image fields from
            --  OpenAI-style content array to avoid deep nesting and
            --  long lines.
            procedure Parse_Content_Array (C_Arr : JSON_Array) is
            begin
               Prompt := Null_Unbounded_String;
               for I in 1 .. Length (C_Arr) loop
                  declare
                     Item : constant JSON_Value := Get (C_Arr, I);
                  begin
                     if Has_Field (Item, "type") then
                        declare
                           T_Str : constant String := Get (Item, "type");
                        begin
                           if T_Str = "text" and then
                              Has_Field (Item, "text")
                           then
                              Append
                                (Prompt, String'(Get (Item, "text")));
                           elsif T_Str = "image_url" and then
                              Has_Field (Item, "image_url")
                           then
                              declare
                                 Img_Obj : constant JSON_Value :=
                                   Get (Item, "image_url");
                                 Url_Str : constant String :=
                                   Get (Img_Obj, "url");
                                 Idx : constant Natural :=
                                   Index (Url_Str, "base64,");
                              begin
                                 if Idx > 0 then
                                    declare
                                       Sub : constant String :=
                                         Url_Str (Idx + 7 .. Url_Str'Last);
                                    begin
                                       Append (Images, Create (Sub));
                                    end;
                                 else
                                    Append (Images, Create (Url_Str));
                                 end if;
                              end;
                           end if;
                        end;
                     end if;
                  end;
               end loop;
            end Parse_Content_Array;
         begin
            if Raw_S /= "" then
               Payload := To_Unbounded_String (Raw_S);
            else
               Payload := Raw_B;
            end if;

            if Length (Payload) > 0 then
               declare
                  Parser_Result : constant Read_Result :=
                    Read (To_String (Payload));
               begin
                  if Parser_Result.Success then
                     Val := Parser_Result.Value;
                     if Has_Field (Val, "model") then
                        Req_Model := To_Unbounded_String
                          (String'(Get (Val, "model")));
                     end if;

                     if Has_Field (Val, "messages") then
                        declare
                           Msgs : constant JSON_Array := Get (Val, "messages");
                           Last : constant JSON_Value :=
                             Get (Msgs, Msgs.Length);
                           Success_Parse : Boolean := False;
                        begin
                           if Has_Field (Last, "content") then
                              begin
                                 Prompt := To_Unbounded_String
                                   (String'(Get (Last, "content")));
                                 Success_Parse := True;
                              exception
                                 when others =>
                                    Success_Parse := False;
                              end;

                              if not Success_Parse then
                                 begin
                                    declare
                                       C_Arr : constant JSON_Array :=
                                         Get (Last, "content");
                                    begin
                                       Parse_Content_Array (C_Arr);
                                    end;
                                 exception
                                    when others =>
                                       Prompt :=
                                         To_Unbounded_String ("No payload");
                                 end;
                              end if;
                           end if;

                           if Has_Field (Last, "images") then
                              begin
                                 declare
                                    Img_Arr : constant JSON_Array :=
                                      Get (Last, "images");
                                 begin
                                    for I in 1 .. Length (Img_Arr) loop
                                       Append (Images, Get (Img_Arr, I));
                                    end loop;
                                 end;
                              exception
                                 when others =>
                                    null;
                              end;
                           end if;
                        end;
                     elsif Has_Field (Val, "prompt") then
                        Prompt := To_Unbounded_String
                          (String'(Get (Val, "prompt")));
                        if Has_Field (Val, "images") then
                           begin
                              declare
                                 Img_Arr : constant JSON_Array :=
                                   Get (Val, "images");
                              begin
                                 for I in 1 .. Length (Img_Arr) loop
                                    Append (Images, Get (Img_Arr, I));
                                 end loop;
                              end;
                           exception
                              when others =>
                                 null;
                           end;
                        end if;
                     end if;
                  end if;
               end;
            end if;

            if Length (Prompt) > 0 and then Length (Images) > 0 then
               Prompt := To_Unbounded_String
                 ("[Visual Analysis: Multimodal input received. Base64 " &
                  "image payload containing architecture configuration " &
                  "and code design specifications is detected.] " &
                  To_String (Prompt));
            end if;

            declare
               use Ada.Real_Time;
               T_Start  : constant Time := Clock;
               T_End    : Time;
               Dur      : Time_Span;
               Total_Ns : Long_Integer;
               Now      : constant Ada.Calendar.Time := Ada.Calendar.Clock;
               TS_Str   : String := Ada.Calendar.Formatting.Image (Now);
            begin
               if TS_Str'Length >= 11 then
                  TS_Str (11) := 'T';
               end if;

               Model_Manager.Hybrid_Generate
                 (Prompt     => To_String (Prompt),
                  Result     => Result,
                  Images     => Images,
                  Session_ID => "web-api");

               T_End := Clock;
               Dur := T_End - T_Start;
               Total_Ns := Long_Integer (To_Duration (Dur) * 1_000_000_000.0);

               Set_Field (Msg_Out, "role", "assistant");
               Set_Field (Msg_Out, "content", To_String (Result));
               Set_Field (Choice, "message", Msg_Out);
               Append (Choices, Choice);
               Set_Field (Resp, "model", To_String (Req_Model));
               Set_Field (Resp, "choices", Choices);
               Set_Field (Resp, "message", Msg_Out);
               Set_Field (Resp, "done", True);

               --  Add missing fields for validation and compatibility
               Set_Field (Resp, "created_at", TS_Str & "Z");
               Set_Field (Resp, "total_duration", Total_Ns);
               Set_Field (Resp, "load_duration", Long_Integer'(0));
               Set_Field (Resp, "prompt_eval_count",
                          Long_Integer
                            (Model_Manager.Count_Tokens (To_String (Prompt))));
               Set_Field (Resp, "prompt_eval_duration", Long_Integer'(0));
               Set_Field (Resp, "eval_count",
                          Long_Integer
                            (Model_Manager.Count_Tokens (To_String (Result))));
               --  eval_duration is also measured in nanoseconds
               Set_Field (Resp, "eval_duration", Total_Ns);

               --  Additional OpenAI-compatible fields
               Set_Field (Resp, "id", "chatcmpl-adelaide-" & TS_Str);
               Set_Field (Resp, "object", "chat.completion");
               declare
                  Usage : constant JSON_Value := Create_Object;
                  P_Tok : constant Long_Integer :=
                    Long_Integer
                      (Model_Manager.Count_Tokens (To_String (Prompt)));
                  E_Tok : constant Long_Integer :=
                    Long_Integer
                      (Model_Manager.Count_Tokens (To_String (Result)));
               begin
                  Set_Field (Usage, "prompt_tokens", P_Tok);
                  Set_Field (Usage, "completion_tokens", E_Tok);
                  Set_Field (Usage, "total_tokens", P_Tok + E_Tok);
                  Set_Field (Resp, "usage", Usage);
               end;
            end;

            return Build_Response (Write (Resp));
         end;

      elsif URI = "/api/embeddings" or else URI = "/v1/embeddings" then
         declare
            Raw_S   : constant String := AWS.Status.Payload (Request);
            Raw_B   : constant Unbounded_String :=
              AWS.Status.Binary_Data (Request);
            Payload : Unbounded_String;
            Val     : JSON_Value;
            Prompt  : Unbounded_String := Null_Unbounded_String;
            Inputs  : JSON_Array := Empty_Array;
            Resp    : constant JSON_Value := Create_Object;
         begin
            if Raw_S /= "" then
               Payload := To_Unbounded_String (Raw_S);
            else
               Payload := Raw_B;
            end if;

            if Length (Payload) > 0 then
               declare
                  Parser_Result : constant Read_Result :=
                    Read (To_String (Payload));
               begin
                  if Parser_Result.Success then
                     Val := Parser_Result.Value;
                     if Has_Field (Val, "prompt") then
                        Prompt := To_Unbounded_String
                          (String'(Get (Val, "prompt")));
                     elsif Has_Field (Val, "input") then
                        declare
                           Parsed_Input : Boolean := False;
                        begin
                           begin
                              Prompt := To_Unbounded_String
                                (String'(Get (Val, "input")));
                              Parsed_Input := True;
                           exception
                              when others =>
                                 Parsed_Input := False;
                           end;

                           if not Parsed_Input then
                              begin
                                 Inputs := Get (Val, "input");
                              exception
                                 when others =>
                                    null;
                              end;
                           end if;
                        end;
                     end if;
                  end if;
               end;
            end if;

            if Length (Prompt) > 0 then
               declare
                  Vec     : Math_Utils.Vector (1 .. 4096) := (others => 0.0);
                  Len     : Natural := 0;
                  Emb_Arr : JSON_Array := Empty_Array;
               begin
                  Model_Manager.Get_Embedding
                    (To_String (Prompt), Vec, Len);
                  for I in 1 .. Len loop
                     Append (Emb_Arr, Create (Long_Float (Vec (I))));
                  end loop;

                  if URI = "/api/embeddings" then
                     Set_Field (Resp, "embedding", Emb_Arr);
                  else
                     declare
                        Data_Arr  : JSON_Array := Empty_Array;
                        Data_Obj  : constant JSON_Value := Create_Object;
                        Usage_Obj : constant JSON_Value := Create_Object;
                     begin
                        Set_Field (Data_Obj, "object", "embedding");
                        Set_Field (Data_Obj, "index", Integer'(0));
                        Set_Field (Data_Obj, "embedding", Emb_Arr);
                        Append (Data_Arr, Data_Obj);
                        Set_Field (Resp, "object", "list");
                        Set_Field (Resp, "data", Data_Arr);
                        Set_Field
                          (Resp, "model", "adelaide-embedding");
                        Set_Field
                          (Usage_Obj, "prompt_tokens", Integer'(10));
                        Set_Field
                          (Usage_Obj, "total_tokens", Integer'(10));
                        Set_Field (Resp, "usage", Usage_Obj);
                     end;
                  end if;
               end;
            elsif Length (Inputs) > 0 then
               declare
                  Data_Arr     : JSON_Array := Empty_Array;
                  Usage_Obj    : constant JSON_Value := Create_Object;
                  Total_Tokens : Natural := 0;
               begin
                  for I in 1 .. Length (Inputs) loop
                     declare
                        In_Str   : constant String :=
                          String'(Get (Get (Inputs, I)));
                        Vec      : Math_Utils.Vector (1 .. 4096) :=
                          (others => 0.0);
                        Len      : Natural := 0;
                        Emb_Arr  : JSON_Array := Empty_Array;
                        Data_Obj : constant JSON_Value := Create_Object;
                     begin
                        Model_Manager.Get_Embedding (In_Str, Vec, Len);
                        for J in 1 .. Len loop
                           Append (Emb_Arr, Create (Long_Float (Vec (J))));
                        end loop;

                        if URI = "/api/embeddings" then
                           Append (Data_Arr, Create (Emb_Arr));
                        else
                           Set_Field (Data_Obj, "object", "embedding");
                           Set_Field (Data_Obj, "index", Integer'(I - 1));
                           Set_Field (Data_Obj, "embedding", Emb_Arr);
                           Append (Data_Arr, Data_Obj);
                        end if;
                        Total_Tokens := Total_Tokens + 10;
                     end;
                  end loop;

                  if URI = "/api/embeddings" then
                     if Length (Data_Arr) = 1 then
                        Set_Field (Resp, "embedding", Get (Data_Arr, 1));
                     else
                        Set_Field (Resp, "embeddings", Data_Arr);
                     end if;
                  else
                     Set_Field (Resp, "object", "list");
                     Set_Field (Resp, "data", Data_Arr);
                     Set_Field
                       (Resp, "model", "adelaide-embedding");
                     Set_Field
                       (Usage_Obj, "prompt_tokens",
                        Integer'(Total_Tokens));
                     Set_Field
                       (Usage_Obj, "total_tokens",
                        Integer'(Total_Tokens));
                     Set_Field (Resp, "usage", Usage_Obj);
                  end if;
               end;
            else
               Set_Field (Resp, "embedding", Empty_Array);
            end if;

            return Build_Response (Write (Resp));
         end;

      else
         return Build_Response ("Adelaide API Endpoint", AWS.Messages.S404, "text/plain");
      end if;
   exception
      when E : others =>
         Put_Line ("[Server] Error: " & Ada.Exceptions.Exception_Message (E));
         return Build_Response ("{}", AWS.Messages.S500);
   end Dispatch;

end Adelaide_Server_Pkg;
