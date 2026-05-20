with Ada.Text_IO;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Ada.Exceptions;
with AWS.Client;
with AWS.Headers;
with AWS.Messages;
with AWS.Response.Set;
with GNATCOLL.JSON;
with Math_Utils;
with Model_Manager;

package body Adelaide_Server_Pkg is

   OLLAMA_PORT : constant String := "11434";
   OLLAMA_URL  : constant String := "http://localhost:" & OLLAMA_PORT;

   --  Helper to convert JSON Array to Vector
   function JSON_Array_To_Vector
     (Arr : GNATCOLL.JSON.JSON_Array) return Math_Utils.Vector
   is
      use GNATCOLL.JSON;
      Len : constant Natural := Length (Arr);
      Vec : Math_Utils.Vector (1 .. Len);
   begin
      for I in 1 .. Len loop
         Vec (I) := Get (Get (Arr, I));
      end loop;
      return Vec;
   end JSON_Array_To_Vector;

   --  Filter and clone request headers to pass to downstream services
   function Filter_Request_Headers
     (Req : AWS.Status.Data) return AWS.Headers.List
   is
      Orig_Headers : constant AWS.Headers.List := AWS.Status.Header (Req);
      New_Headers  : AWS.Headers.List;
   begin
      for I in 1 .. AWS.Headers.Count (Orig_Headers) loop
         declare
            El : constant AWS.Headers.Element :=
              AWS.Headers.Get (Orig_Headers, I);
            Name : constant String := To_String (El.Name);
            Val  : constant String := To_String (El.Value);
         begin
            --  Skip headers handled/overridden by AWS Client or proxy layer
            if Name /= "Host" and then
               Name /= "host" and then
               Name /= "Content-Length" and then
               Name /= "content-length" and then
               Name /= "Connection" and then
               Name /= "connection"
            then
               AWS.Headers.Add (New_Headers, Name, Val);
            end if;
         end;
      end loop;
      return New_Headers;
   end Filter_Request_Headers;

   --  Helper to copy and clean up response headers from downstream response
   function Clone_And_Filter_Response
     (Resp : AWS.Response.Data) return AWS.Response.Data
   is
      use AWS.Response;
      New_Resp : Data := Build
        (Content_Type => Content_Type (Resp),
         Message_Body => String'(Message_Body (Resp)),
         Status_Code  => Status_Code (Resp));
      Orig_Headers : constant AWS.Headers.List := Header (Resp);
   begin
      --  Copy original headers except transfer/content length/connection
      for I in 1 .. AWS.Headers.Count (Orig_Headers) loop
         declare
            El : constant AWS.Headers.Element :=
              AWS.Headers.Get (Orig_Headers, I);
            Name : constant String := To_String (El.Name);
            Val  : constant String := To_String (El.Value);
         begin
            if Name /= "Content-Length" and then
               Name /= "content-length" and then
               Name /= "Transfer-Encoding" and then
               Name /= "transfer-encoding" and then
               Name /= "Content-Encoding" and then
               Name /= "content-encoding" and then
               Name /= "Connection" and then
               Name /= "connection"
            then
               AWS.Response.Set.Add_Header (New_Resp, Name, Val);
            end if;
         end;
      end loop;

      --  Inject CORS headers for browser/client compatibility
      AWS.Response.Set.Add_Header
        (New_Resp, "Access-Control-Allow-Origin", "*");
      AWS.Response.Set.Add_Header
        (New_Resp, "Access-Control-Allow-Methods",
         "GET, POST, PUT, DELETE, OPTIONS, HEAD");
      AWS.Response.Set.Add_Header
        (New_Resp, "Access-Control-Allow-Headers",
         "Content-Type, Authorization, X-Requested-With");

      return New_Resp;
   end Clone_And_Filter_Response;

   --  Extract prompt string from request body
   function Extract_Prompt (Body_Str : String) return String is
      use GNATCOLL.JSON;
      Res : constant Read_Result := Read (Body_Str);
   begin
      if not Res.Success then
         return "";
      end if;
      declare
         Val : constant JSON_Value := Res.Value;
      begin
         if Val.Kind /= JSON_Object_Type then
            return "";
         end if;

         if Has_Field (Val, "prompt") then
            declare
               Prompt_Val : constant JSON_Value := Get (Val, "prompt");
            begin
               if Prompt_Val.Kind = JSON_String_Type then
                  return Get (Prompt_Val);
               end if;
            end;
         elsif Has_Field (Val, "messages") then
            declare
               Msgs_Val : constant JSON_Value := Get (Val, "messages");
            begin
               if Msgs_Val.Kind = JSON_Array_Type then
                  declare
                     Arr : constant JSON_Array := Get (Msgs_Val);
                     Len : constant Natural := Length (Arr);
                  begin
                     if Len > 0 then
                        declare
                           Last_Msg : constant JSON_Value := Get (Arr, Len);
                        begin
                           if Last_Msg.Kind = JSON_Object_Type and then
                              Has_Field (Last_Msg, "content")
                           then
                              declare
                                 Content_Val : constant JSON_Value :=
                                   Get (Last_Msg, "content");
                              begin
                                 if Content_Val.Kind = JSON_String_Type then
                                    return Get (Content_Val);
                                 end if;
                              end;
                           end if;
                        end;
                     end if;
                  end;
               end if;
            end;
         end if;
      end;
      return "";
   exception
      when others =>
         return "";
   end Extract_Prompt;

   --  Get text embedding vector from local Model_Manager
   function Get_Query_Embedding (Prompt : String) return Math_Utils.Vector is
   begin
      return Model_Manager.Get_Embedding (Prompt);
   exception
      when E : others =>
         Ada.Text_IO.Put_Line
           ("[Embedding Error] " & Ada.Exceptions.Exception_Message (E));
         return (1 .. 0 => 0.0);
   end Get_Query_Embedding;

   --  Load JSON Cache Array from file path options
   function Load_Cache return GNATCOLL.JSON.JSON_Array is
      use GNATCOLL.JSON;
      Res : Read_Result;
   begin
      Res := Read_File ("Adelaide_Lite/python/response_cache.json");
      if Res.Success and then Res.Value.Kind = JSON_Array_Type then
         return Get (Res.Value);
      end if;

      return Empty_Array;
   end Load_Cache;

   --  Format cached response to match API endpoint specification
   function Format_Cache_Response
     (URI_Str : String;
      Cached_Resp : String;
      Similarity : Float) return String
   is
      use GNATCOLL.JSON;
      Res_Obj : constant JSON_Value := Create_Object;
      Sim_Str : constant String := Float'Image (Similarity);
      Think_Header : constant String :=
         "<think>" & ASCII.LF &
         "[Adelaide-Lite Memory Thoughts Match - Similarity:" &
         Sim_Str & "]" & ASCII.LF &
         "</think>" & ASCII.LF;
      Is_Match : constant Boolean := Similarity > 0.0;
      Full_Content : constant String :=
        (if Is_Match then Think_Header & Cached_Resp else Cached_Resp);
   begin
      if URI_Str = "/api/chat" or else URI_Str = "/api/chat/" then
         Set_Field (Res_Obj, String'("model"), String'("adelaide-hybrid"));
         declare
            Msg_Obj : constant JSON_Value := Create_Object;
         begin
            Set_Field (Msg_Obj, String'("role"), String'("assistant"));
            Set_Field (Msg_Obj, String'("content"), Full_Content);
            Set_Field (Res_Obj, String'("message"), Msg_Obj);
         end;
         Set_Field (Res_Obj, String'("done"), True);
      elsif URI_Str = "/v1/chat/completions" then
         Set_Field (Res_Obj, String'("id"), String'("chatcmpl-adelaide"));
         Set_Field (Res_Obj, String'("object"), String'("chat.completion"));
         Set_Field (Res_Obj, String'("model"), String'("adelaide-hybrid"));
         declare
            Choices_Arr : JSON_Array;
            Choice_Obj  : constant JSON_Value := Create_Object;
            Msg_Obj     : constant JSON_Value := Create_Object;
         begin
            Set_Field (Msg_Obj, String'("role"), String'("assistant"));
            Set_Field (Msg_Obj, String'("content"), Full_Content);
            Set_Field (Choice_Obj, String'("index"), Integer'(0));
            Set_Field (Choice_Obj, String'("message"), Msg_Obj);
            Set_Field (Choice_Obj, String'("finish_reason"), String'("stop"));
            Append (Choices_Arr, Choice_Obj);
            Set_Field (Res_Obj, String'("choices"), Choices_Arr);
         end;
      else
         --  default /api/generate
         Set_Field (Res_Obj, String'("model"), String'("adelaide-hybrid"));
         Set_Field (Res_Obj, String'("response"), Full_Content);
         Set_Field (Res_Obj, String'("done"), True);
      end if;
      return Write (Res_Obj);
   end Format_Cache_Response;

   --  Forward GET requests directly to local Ollama
   function Forward_Get (Request : AWS.Status.Data) return AWS.Response.Data is
      use AWS.Status;
      URI_Str : constant String := URI (Request);
      URL     : constant String := OLLAMA_URL & URI_Str;
      Headers : constant AWS.Headers.List := Filter_Request_Headers (Request);
   begin
      return Clone_And_Filter_Response
        (AWS.Client.Get
           (URL     => URL,
            Headers => Headers));
   exception
      when E : others =>
         return AWS.Response.Build
           (Content_Type => "application/json",
            Message_Body => "{""error"":""Proxy GET error: " &
              Ada.Exceptions.Exception_Message (E) & """}",
            Status_Code  => AWS.Messages.S502);
   end Forward_Get;

   --  Forward POST requests to specified downstream endpoint
   function Forward_Post
     (Request : AWS.Status.Data;
      Body_Str : String) return AWS.Response.Data
   is
      use AWS.Status;
      URL : constant String := OLLAMA_URL & URI (Request);
      Headers : constant AWS.Headers.List := Filter_Request_Headers (Request);
   begin
      return Clone_And_Filter_Response
        (AWS.Client.Post
           (URL          => URL,
            Data         => Body_Str,
            Content_Type => Content_Type (Request),
            Headers      => Headers));
   exception
      when E : others =>
         return AWS.Response.Build
           (Content_Type => "application/json",
            Message_Body => "{""error"":""Proxy POST error: " &
              Ada.Exceptions.Exception_Message (E) & """}",
            Status_Code  => AWS.Messages.S502);
   end Forward_Post;

   --------------
   -- Dispatch --
   --------------

   function Dispatch (Request : AWS.Status.Data) return AWS.Response.Data is
      use AWS.Status;
      URI_Str : constant String := URI (Request);
      Method_Val : constant Request_Method := Method (Request);

      --  Inject CORS headers helper
      procedure Set_CORS (Resp : in out AWS.Response.Data) is
      begin
         AWS.Response.Set.Add_Header (Resp, "Access-Control-Allow-Origin", "*");
         AWS.Response.Set.Add_Header
           (Resp, "Access-Control-Allow-Methods",
            "GET, POST, PUT, DELETE, OPTIONS, HEAD");
         AWS.Response.Set.Add_Header
           (Resp, "Access-Control-Allow-Headers",
            "Content-Type, Authorization, X-Requested-With");
      end Set_CORS;
   begin
      Ada.Text_IO.Put_Line ("[Request] " & Method_Val'Img & " " & URI_Str);

      --  1. Preflight/CORS OPTIONS handling
      if Method_Val = OPTIONS then
         declare
            Resp : AWS.Response.Data :=
              AWS.Response.Acknowledge (AWS.Messages.S200);
         begin
            Set_CORS (Resp);
            return Resp;
         end;
      end if;

      --  2. Handle HEAD / (Health check for Ollama CLI)
      if Method_Val = HEAD and then URI_Str = "/" then
         declare
            Resp : AWS.Response.Data :=
              AWS.Response.Acknowledge (AWS.Messages.S200);
         begin
            Set_CORS (Resp);
            return Resp;
         end;
      end if;

      --  3. Handle GET routes
      if Method_Val = GET then
         if URI_Str = "/" then
            declare
               Resp : AWS.Response.Data :=
                 AWS.Response.Build (Content_Type => "text/plain",
                                     Message_Body => "Ollama is running");
            begin
               Set_CORS (Resp);
               return Resp;
            end;
         elsif URI_Str = "/api/version" then
            declare
               Resp : AWS.Response.Data :=
                 AWS.Response.Build (Content_Type => "application/json",
                                     Message_Body => "{""version"":""0.1.48""}");
            begin
               Set_CORS (Resp);
               return Resp;
            end;
         elsif URI_Str = "/api/tags" or else URI_Str = "/tags" then
            declare
               use GNATCOLL.JSON;
               Res_Obj    : constant JSON_Value := Create_Object;
               Models_Arr : JSON_Array;
               function Create_Model_Info
                 (Name : String; Size : Long_Long_Integer) return JSON_Value
               is
                  M : constant JSON_Value := Create_Object;
                  D : constant JSON_Value := Create_Object;
               begin
                  Set_Field (M, "name", Name);
                  Set_Field (M, "model", Name);
                  Set_Field (M, "modified_at",
                             String'("2024-05-20T11:42:00Z"));
                  Set_Field (M, "size", Create (Size));
                  Set_Field (M, "digest", String'("adelaide-lite-v1"));
                  Set_Field (D, "format", String'("gguf"));
                  Set_Field (D, "family", String'("qwen"));
                  Set_Field (M, "details", D);
                  return M;
               end Create_Model_Info;
               Resp : AWS.Response.Data;
            begin
               Append (Models_Arr,
                       Create_Model_Info ("adelaide-hybrid", 3_100_000_000));
               Set_Field (Res_Obj, "models", Models_Arr);
               Resp := AWS.Response.Build
                 (Content_Type => "application/json",
                  Message_Body => Write (Res_Obj));
               Set_CORS (Resp);
               return Resp;
            end;
         elsif URI_Str = "/api/ps" then
            declare
               use GNATCOLL.JSON;
               Res_Obj    : constant JSON_Value := Create_Object;
               Models_Arr : JSON_Array;
               function Create_Running_Info (Name : String) return JSON_Value is
                  M : constant JSON_Value := Create_Object;
                  D : constant JSON_Value := Create_Object;
               begin
                  Set_Field (M, "name", Name);
                  Set_Field (M, "model", Name);
                  Set_Field (M, "size", Create (Long_Long_Integer'(0)));
                  Set_Field (M, "digest", String'("adelaide-lite-run"));
                  Set_Field (D, "format", String'("gguf"));
                  Set_Field (M, "details", D);
                  Set_Field (M, "expires_at", String'("2026-05-20T23:59:59Z"));
                  Set_Field (M, "size_vram", Create (Long_Long_Integer'(0)));
                  return M;
               end Create_Running_Info;
               Resp : AWS.Response.Data;
            begin
               if Model_Manager.Is_Loaded (Model_Manager.Qwen_0_8B) then
                  Append (Models_Arr, Create_Running_Info ("qwen3.5:0.8b"));
               end if;
               if Model_Manager.Is_Loaded (Model_Manager.Qwen_4B) then
                  Append (Models_Arr, Create_Running_Info ("qwen3.5:4b"));
               end if;
               Set_Field (Res_Obj, "models", Models_Arr);
               Resp := AWS.Response.Build
                 (Content_Type => "application/json",
                  Message_Body => Write (Res_Obj));
               Set_CORS (Resp);
               return Resp;
            end;
         else
            return Forward_Get (Request);
         end if;
      end if;

      --  4. Handle POST request
      if Method_Val = POST then
         declare
            Body_Str : constant String := To_String (Binary_Data (Request));
         begin
            if URI_Str = "/api/show" then
               declare
                  use GNATCOLL.JSON;
                  Res_Obj : constant JSON_Value := Create_Object;
                  Det_Obj : constant JSON_Value := Create_Object;
               begin
                  Set_Field (Res_Obj, "modelfile",
                             String'("FROM adelaide-hybrid"));
                  Set_Field (Res_Obj, "parameters", String'("stop <think>"));
                  Set_Field (Res_Obj, "template", String'("{{ .Prompt }}"));
                  Set_Field (Res_Obj, "system", String'("You are Adelaide."));
                  Set_Field (Det_Obj, "format", String'("gguf"));
                  Set_Field (Det_Obj, "family", String'("qwen"));
                  Set_Field (Res_Obj, "details", Det_Obj);
                  declare
                     Resp : AWS.Response.Data :=
                       AWS.Response.Build (Content_Type => "application/json",
                                           Message_Body => Write (Res_Obj));
                  begin
                     Set_CORS (Resp);
                     return Resp;
                  end;
               end;
            elsif URI_Str = "/api/chat" or else
               URI_Str = "/api/generate" or else
               URI_Str = "/v1/chat/completions" or else
               URI_Str = "/think"
            then
               declare
                  Prompt : constant String := Extract_Prompt (Body_Str);
               begin
                  if Prompt /= "" then
                     declare
                        Query_Vec : constant Math_Utils.Vector :=
                          Get_Query_Embedding (Prompt);
                     begin
                        if Query_Vec'Length > 0 then
                           declare
                              use GNATCOLL.JSON;
                              Cache_Arr : constant JSON_Array := Load_Cache;
                              Len : constant Natural := Length (Cache_Arr);
                              Max_Sim   : Float := -2.0;
                              Best_Resp : Unbounded_String;
                           begin
                              for I in 1 .. Len loop
                                 declare
                                    Entry_Val : constant JSON_Value :=
                                      Get (Cache_Arr, I);
                                 begin
                                    if Entry_Val.Kind = JSON_Object_Type and then
                                       Has_Field (Entry_Val, "embedding") and then
                                       Has_Field (Entry_Val, "response")
                                    then
                                       declare
                                          Arr : constant JSON_Array :=
                                            Get (Get (Entry_Val, "embedding"));
                                          Resp_Val : constant String :=
                                            Get (Get (Entry_Val, "response"));
                                       begin
                                          if Length (Arr) = Query_Vec'Length then
                                             declare
                                                Entry_Vec : constant
                                                  Math_Utils.Vector :=
                                                    JSON_Array_To_Vector (Arr);
                                                Sim : constant Float :=
                                                  Math_Utils.Cosine_Similarity
                                                    (Query_Vec, Entry_Vec);
                                             begin
                                                if Sim > Max_Sim then
                                                   Max_Sim := Sim;
                                                   Best_Resp :=
                                                     To_Unbounded_String
                                                       (Resp_Val);
                                                end if;
                                             end;
                                          end if;
                                       end;
                                    end if;
                                 end;
                              end loop;

                              if Max_Sim >= 0.85 and then Max_Sim < 0.98 then
                                 declare
                                    Formatted_Resp : constant String :=
                                      Format_Cache_Response
                                        (URI_Str, To_String (Best_Resp),
                                         Max_Sim);
                                    Resp : AWS.Response.Data :=
                                      AWS.Response.Build
                                        (Content_Type => "application/json",
                                         Message_Body => Formatted_Resp);
                                 begin
                                    Set_CORS (Resp);
                                    return Resp;
                                 end;
                              end if;
                           end;
                        end if;
                     end;
                  end if;

                  --  Cache miss: internal Model_Manager
                  declare
                     use GNATCOLL.JSON;
                     Res : constant Read_Result := Read (Body_Str);
                     Model_Name : Unbounded_String :=
                       To_Unbounded_String ("qwen3.5:0.8b");
                  begin
                     if Res.Success and then Has_Field (Res.Value, "model") then
                        Model_Name := To_Unbounded_String
                          (String'(Get (Get (Res.Value, "model"))));
                     end if;

                     declare
                        Model_Str : constant String := To_String (Model_Name);
                        Kind : constant Model_Manager.Model_Type :=
                          Model_Manager.Get_Kind_For_Model_Name (Model_Str);
                        Gen_Text : constant String :=
                          (if Model_Str = "adelaide-hybrid"
                           then Model_Manager.Hybrid_Generate (Prompt)
                           else Model_Manager.Generate (Kind, Prompt));
                        Formatted_Resp : constant String :=
                          Format_Cache_Response (URI_Str, Gen_Text, 0.0);
                        Resp : AWS.Response.Data :=
                          AWS.Response.Build
                            (Content_Type => "application/json",
                             Message_Body => Formatted_Resp);
                     begin
                        Set_CORS (Resp);
                        return Resp;
                     end;
                  end;
               end;
            elsif URI_Str = "/api/embed" or else URI_Str = "/api/embeddings" then
               declare
                  Prompt : constant String := Extract_Prompt (Body_Str);
                  Vec    : constant Math_Utils.Vector :=
                    Get_Query_Embedding (Prompt);
                  use GNATCOLL.JSON;
                  Res_Obj : constant JSON_Value := Create_Object;
                  Arr     : JSON_Array;
               begin
                  for I in Vec'Range loop
                     Append (Arr, Create (Vec (I)));
                  end loop;
                  Set_Field (Res_Obj, "embedding", Arr);
                  declare
                     Resp : AWS.Response.Data :=
                       AWS.Response.Build
                         (Content_Type => "application/json",
                          Message_Body => Write (Res_Obj));
                  begin
                     Set_CORS (Resp);
                     return Resp;
                  end;
               end;
            else
               return Forward_Post (Request, Body_Str);
            end if;
         end;
      end if;

      return AWS.Response.Acknowledge (AWS.Messages.S404);
   end Dispatch;

end Adelaide_Server_Pkg;
