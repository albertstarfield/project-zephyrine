with AWS.Status;
with AWS.Response;
with AWS.Messages;
with GNATCOLL.JSON; use GNATCOLL.JSON;
with Ada.Text_IO; use Ada.Text_IO;
with Model_Manager;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Streaming_Queue;
with Ada.Exceptions;
with Math_Utils;
with Ada.Strings.Fixed; use Ada.Strings.Fixed;

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

   function Dispatch (Request : AWS.Status.Data) return AWS.Response.Data is
      URI    : constant String := AWS.Status.URI (Request);
      Method : constant String := AWS.Status.Method (Request);
   begin
      Put_Line ("[Server] " & Method & " " & URI);

      if URI = "/v1/models" or else URI = "/api/tags" then
         declare
            Resp   : constant JSON_Value := Create_Object;
            Models : JSON_Array := Empty_Array;
            M1     : constant JSON_Value := Create_Object;
            M2     : constant JSON_Value := Create_Object;
         begin
            Set_Field (M1, "id", "adelaide-hybrid");
            Set_Field (M1, "name", "adelaide-hybrid");
            declare
               D1 : constant JSON_Value := Create_Object;
            begin
               Set_Field (D1, "format", "gguf");
               Set_Field (D1, "family", "qwen2");
               Set_Field (M1, "details", D1);
            end;
            Append (Models, M1);

            Set_Field (M2, "id", "adelaide-embedding");
            Set_Field (M2, "name", "adelaide-embedding");
            declare
               D2 : constant JSON_Value := Create_Object;
            begin
               Set_Field (D2, "format", "gguf");
               Set_Field (D2, "family", "bert");
               Set_Field (M2, "details", D2);
            end;
            Append (Models, M2);

            Set_Field (Resp, "data", Models);
            Set_Field (Resp, "models", Models);
            return AWS.Response.Build
              (Content_Type => "application/json",
               Message_Body => Write (Resp));
         end;

      elsif URI = "/api/show" then
         declare
            Raw_S : constant String := AWS.Status.Payload (Request);
            Model_Name : Unbounded_String :=
              To_Unbounded_String ("adelaide-hybrid");
         begin
            if Raw_S /= "" then
               declare
                  Parser_Result : constant Read_Result := Read (Raw_S);
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
                  Set_Field (Model_Info, "general.file_type", 12);
                  Set_Field (Model_Info, "general.parameter_count", 100000000);
                  Set_Field
                    (Model_Info, "bert.context_length",
                     Create (Long_Float (4294967296.0)));

                  Set_Field (Resp, "modelfile",
                             "# Adelaide Embedding ModelFile" & ASCII.LF &
                             "FROM adelaide-embedding" & ASCII.LF &
                             "PARAMETER num_ctx 4294967296");
                  Set_Field (Resp, "parameters",
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

                  Set_Field (Model_Info, "general.architecture", "llama");
                  Set_Field (Model_Info, "general.file_type", 12);
                  Set_Field (Model_Info, "general.parameter_count", 4000000000);
                  Set_Field
                    (Model_Info, "qwen2.context_length",
                     Create (Long_Float (9223372036854775808.0)));

                  Set_Field (Resp, "modelfile",
                             "# Adelaide ModelFile" & ASCII.LF &
                             "FROM adelaide-hybrid" & ASCII.LF &
                             "PARAMETER num_ctx 9223372036854775808");
                  Set_Field (Resp, "parameters",
                             "num_ctx" & ASCII.HT & "9223372036854775808");
                  Set_Field (Resp, "template",
                             "{{ .System }}" & ASCII.LF & "{{ .Prompt }}");
                  Set_Field (Resp, "details", Details);
                  Set_Field (Resp, "model_info", Model_Info);
               end if;

               return AWS.Response.Build
                 (Content_Type => "application/json",
                  Message_Body => Write (Resp));
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
                                       Prompt := Null_Unbounded_String;
                                       for I in 1 .. Length (C_Arr) loop
                                          declare
                                             Item : constant JSON_Value :=
                                               Get (C_Arr, I);
                                          begin
                                             if Has_Field (Item, "type") then
                                                declare
                                                   T_Str : constant String :=
                                                     Get (Item, "type");
                                                begin
                                                   if T_Str = "text" and then
                                                      Has_Field (Item, "text")
                                                   then
                                                      Append
                                                        (Prompt,
                                                         String'(Get
                                                           (Item, "text")));
                                                   elsif T_Str = "image_url"
                                                     and then Has_Field
                                                       (Item, "image_url")
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
                                                                 Url_Str
                                                                   (Idx + 7 ..
                                                                    Url_Str'Last);
                                                            begin
                                                               Append
                                                                 (Images,
                                                                  Create
                                                                    (Sub));
                                                            end;
                                                         else
                                                            Append
                                                              (Images,
                                                               Create
                                                                 (Url_Str));
                                                         end if;
                                                      end;
                                                   end if;
                                                end;
                                             end if;
                                          end;
                                       end loop;
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

            Model_Manager.Hybrid_Generate
              (Prompt     => To_String (Prompt),
               Result     => Result,
               Images     => Images,
               Session_ID => "web-api");

            Set_Field (Msg_Out, "role", "assistant");
            Set_Field (Msg_Out, "content", To_String (Result));
            Set_Field (Choice, "message", Msg_Out);
            Append (Choices, Choice);
            Set_Field (Resp, "model", "adelaide-hybrid");
            Set_Field (Resp, "choices", Choices);
            Set_Field (Resp, "message", Msg_Out);
            Set_Field (Resp, "done", True);

            return AWS.Response.Build
              (Content_Type => "application/json",
               Message_Body => Write (Resp));
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
                  Vec : Math_Utils.Vector (1 .. 4096) := (others => 0.0);
                  Len : Natural := 0;
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
                        Set_Field (Data_Obj, "index", 0);
                        Set_Field (Data_Obj, "embedding", Emb_Arr);
                        Append (Data_Arr, Data_Obj);
                        Set_Field (Resp, "object", "list");
                        Set_Field (Resp, "data", Data_Arr);
                        Set_Field (Resp, "model", "adelaide-embedding");
                        Set_Field (Usage_Obj, "prompt_tokens", 10);
                        Set_Field (Usage_Obj, "total_tokens", 10);
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
                        In_Str  : constant String := Get (Inputs, I);
                        Vec     : Math_Utils.Vector (1 .. 4096) :=
                          (others => 0.0);
                        Len     : Natural := 0;
                        Emb_Arr : JSON_Array := Empty_Array;
                        Data_Obj : constant JSON_Value := Create_Object;
                     begin
                        Model_Manager.Get_Embedding (In_Str, Vec, Len);
                        for J in 1 .. Len loop
                           Append (Emb_Arr, Create (Long_Float (Vec (J))));
                        end loop;

                        if URI = "/api/embeddings" then
                           Append (Data_Arr, Emb_Arr);
                        else
                           Set_Field (Data_Obj, "object", "embedding");
                           Set_Field (Data_Obj, "index", I - 1);
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
                     Set_Field (Resp, "model", "adelaide-embedding");
                     Set_Field (Usage_Obj, "prompt_tokens", Total_Tokens);
                     Set_Field (Usage_Obj, "total_tokens", Total_Tokens);
                     Set_Field (Resp, "usage", Usage_Obj);
                  end if;
               end;
            else
               Set_Field (Resp, "embedding", Empty_Array);
            end if;

            return AWS.Response.Build
              (Content_Type => "application/json",
               Message_Body => Write (Resp));
         end;

      else
         return AWS.Response.Build
           (Content_Type => "text/plain",
            Message_Body => "Adelaide API Endpoint",
            Status_Code  => AWS.Messages.S404);
      end if;
   exception
      when E : others =>
         Put_Line ("[Server] Error: " & Ada.Exceptions.Exception_Message (E));
         return AWS.Response.Build
           (Content_Type => "application/json",
            Message_Body => "{}",
            Status_Code  => AWS.Messages.S500);
   end Dispatch;

end Adelaide_Server_Pkg;
