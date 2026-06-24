--  claude_client.adb
--  Claude API client implementation for Anthropic's Messages API.
--  Uses AWS.HTTP_Client for HTTP POST requests.
--  DO NOT REMOVE, OR YOU WILL BE KILLED

with Ada.Text_IO;           use Ada.Text_IO;
with Ada.Strings;           use Ada.Strings;
with Ada.Strings.Fixed;     use Ada.Strings.Fixed;
with Ada.Strings.Maps;      use Ada.Strings.Maps;
with Ada.Real_Time;         use Ada.Real_Time;
with AWS.Client;
with AWS.Response;
with AWS.Messages;          use AWS.Messages;
with AWS.Headers;
with AWS.Headers.Set;
with GNATCOLL.JSON;         use GNATCOLL.JSON;

package body Claude_Client is

   --  Model name prefixes that identify Claude models
   Claude_Prefixes : constant array (1 .. 4) of String (1 .. 6) :=
     ("claude", "Claude", "CLAUDE", "anthro");

   function Is_Claude_Model (Model_Name : String) return Boolean is
   begin
      for Prefix of Claude_Prefixes loop
         if Model_Name'Length >= Prefix'Length and then
            Model_Name (Model_Name'First .. Model_Name'First + Prefix'Length - 1) = Prefix
         then
            return True;
         end if;
      end loop;
      return False;
   end Is_Claude_Model;

   --  Helper: Escape a string for JSON
   function Escape_JSON (S : String) return String is
      Result : Unbounded_String;
   begin
      for C of S loop
         case C is
            when '\' =>
               Append (Result, "\\");
            when '"' =>
               Append (Result, "\""");
            when ASCII.LF =>
               Append (Result, "\n");
            when ASCII.CR =>
               Append (Result, "\r");
            when ASCII.HT =>
               Append (Result, "\t");
            when others =>
               Append (Result, C);
         end case;
      end loop;
      return To_String (Result);
   end Escape_JSON;

   --  Build the JSON request body for Claude Messages API
   function Build_Request_Body
     (Model         : String;
      Messages      : Claude_Message_Array;
      Max_Tokens    : Positive;
      System_Prompt : String;
      Temperature   : Float)
      return String
   is
      Body_Str : Unbounded_String;
   begin
      Append (Body_Str, "{");
      Append (Body_Str, """model"": """ & Escape_JSON (Model) & """,");
      Append (Body_Str, """max_tokens"": " & Trim (Integer'Image (Max_Tokens), Both) & ",");
      Append (Body_Str, """temperature"": " & Trim (Float'Image (Temperature), Both) & ",");

      --  System prompt (optional)
      if System_Prompt'Length > 0 then
         Append (Body_Str, """system"": """ & Escape_JSON (System_Prompt) & """,");
      end if;

      --  Messages array
      Append (Body_Str, """messages"": [");
      for I in Messages'Range loop
         Append (Body_Str, "{");
         Append (Body_Str, """role"": """ &
           (if Messages (I).Role = User then "user" else "assistant") & """,");
         Append (Body_Str, """content"": """ &
           Escape_JSON (To_String (Messages (I).Content)) & """");
         Append (Body_Str, "}");
         if I < Messages'Last then
            Append (Body_Str, ",");
         end if;
      end loop;
      Append (Body_Str, "]");

      Append (Body_Str, "}");
      return To_String (Body_Str);
   end Build_Request_Body;

   function Send_Message
     (API_Key       : String;
      Model         : String;
      Messages      : Claude_Message_Array;
      Max_Tokens    : Positive := Default_Max_Tokens;
      System_Prompt : String   := "";
      Temperature   : Float    := 1.0)
      return String
   is
      URL        : constant String := Claude_Base_URL & "/v1/messages";
      Request_Body : constant String :=
        Build_Request_Body (Model, Messages, Max_Tokens, System_Prompt, Temperature);
      Headers    : AWS.Headers.List;
      Response   : AWS.Response.Data;
      Content    : Unbounded_String;
   begin
      Put_Line ("[Claude] Sending request to " & URL);
      Put_Line ("[Claude] Model: " & Model);
      Put_Line ("[Claude] Max_Tokens: " & Trim (Integer'Image (Max_Tokens), Both));

      --  Build headers
      Headers := AWS.Headers.Empty_List;
      AWS.Headers.Set.Add (Headers, "x-api-key", API_Key);
      AWS.Headers.Set.Add (Headers, "anthropic-version", API_Version);
      AWS.Headers.Set.Add (Headers, "content-type", "application/json");

      --  Send POST request
      Response := AWS.Client.Post
        (URL          => URL,
         Data         => Request_Body,
         Content_Type => "application/json",
         Headers      => Headers);

      if AWS.Response.Status_Code (Response) not in Success then
         declare
            Err_Body : constant String := AWS.Response.Message_Body (Response);
         begin
            Put_Line ("[Claude] ERROR: HTTP " &
              AWS.Messages.Status_Code'Image (AWS.Response.Status_Code (Response)) & ": " & Err_Body);
            raise Program_Error
              with "Claude API HTTP error: " & Err_Body;
         end;
      end if;

      Content := To_Unbounded_String (AWS.Response.Message_Body (Response));
      Put_Line ("[Claude] Response received, length: " &
        Trim (Integer'Image (Length (Content)), Both) & " bytes");
      return To_String (Content);
   end Send_Message;

   function Get_Response_Text
     (API_Key       : String;
      Model         : String;
      Messages      : Claude_Message_Array;
      Max_Tokens    : Positive := Default_Max_Tokens;
      System_Prompt : String   := "";
      Temperature   : Float    := 1.0)
      return String
   is
      JSON_Response : constant String :=
        Send_Message (API_Key, Model, Messages, Max_Tokens, System_Prompt, Temperature);
   begin
      return Parse_Response_Content (JSON_Response);
   end Get_Response_Text;

   function Parse_Response_Content (JSON_Response : String) return String is
      Parsed : constant GNATCOLL.JSON.JSON_Value :=
        GNATCOLL.JSON.Read (JSON_Response);
      Content : GNATCOLL.JSON.JSON_Array;
      Result  : Unbounded_String;
   begin
      if GNATCOLL.JSON.Has_Field (Parsed, "content") then
         Content := GNATCOLL.JSON.Get (Parsed, "content");
         for I in 1 .. GNATCOLL.JSON.Length (Content) loop
            declare
               Block : constant GNATCOLL.JSON.JSON_Value :=
                 GNATCOLL.JSON.Get (Content, I);
            begin
               if GNATCOLL.JSON.Has_Field (Block, "type") and then
                  String'(GNATCOLL.JSON.Get (Block, "type")) = "text"
               then
                  if GNATCOLL.JSON.Has_Field (Block, "text") then
                     Append (Result, String'(GNATCOLL.JSON.Get (Block, "text")));
                  end if;
               end if;
            end;
         end loop;
      end if;
      return To_String (Result);
   end Parse_Response_Content;

end Claude_Client;
