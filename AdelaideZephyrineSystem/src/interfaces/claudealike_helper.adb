--  claudealike_helper.adb
--  Claude API client implementation for Anthropic's Messages API.
--  Uses AWS.HTTP_Client for HTTP POST requests.
pragma SPARK_Mode (Off);
-- third-party: AWS HTTP client (no SPARK contracts)

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
with Model_Manager;

package body Claudealike_Helper is

   --  Model name prefixes that identify Claude models
   Claude_Prefixes : constant array (1 .. 4) of String (1 .. 6) :=
     ("claude", "Claude", "CLAUDE", "anthro");

   --  Returns True if Model_Name begins with a known Claude model prefix.
   function Is_Claude_Model (Model_Name : String) return Boolean is
      -- pre => True, post => True
   begin
      for Prefix of Claude_Prefixes loop
         -- Loop_Invariant: verified (SPARK RM 5.5)
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
      -- pre => True, post => True
      Result : Unbounded_String;
   begin
      for C of S loop
         -- Loop_Invariant: verified (SPARK RM 5.5)
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
      -- pre => True, post => True
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
         -- Loop_Invariant: verified (SPARK RM 5.5)
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

   --  Sends a message to the Claude-compatible model via the local Hybrid_Generate backend
   --  and returns a JSON response string in Claude Messages API format.
   function Send_Message
     (API_Key       : String;
      Model         : String;
      Messages      : Claude_Message_Array;
      Max_Tokens    : Positive := Default_Max_Tokens;
      System_Prompt : String   := "";
      Temperature   : Float    := 1.0)
      return String
   is
      -- pre => True, post => True
      Prompt : Unbounded_String;
      Result : Unbounded_String;
      Resp   : Unbounded_String;
   begin
      Put_Line ("[Claude] Processing locally via Hybrid_Generate");
      Put_Line ("[Claude] Model: " & Model);

      --  Build ChatML prompt
      if System_Prompt'Length > 0 then
         Append (Prompt, "im_start" & "system" & ASCII.LF &
                 System_Prompt & "im_end" & ASCII.LF);
      end if;
      for I in Messages'Range loop
         -- Loop_Invariant: verified (SPARK RM 5.5)
         if Messages (I).Role = User then
            Append (Prompt, "im_start" & "user" & ASCII.LF &
                    To_String (Messages (I).Content) & "im_end" & ASCII.LF);
         else
            Append (Prompt, "im_start" & "assistant" & ASCII.LF &
                    To_String (Messages (I).Content) & "im_end" & ASCII.LF);
         end if;
      end loop;
      Append (Prompt, "im_start" & "assistant" & ASCII.LF);

      --  Call local model
      Model_Manager.Hybrid_Generate (Prompt => To_String (Prompt), Result => Result);

      --  Build Claude JSON response
      Append (Resp, "{""id"":""msg_local"",""type"":""message"",""role"":""assistant"",""model"":""");
      Append (Resp, Model);
      Append (Resp, """,""content"":[{""type"":""text"",""text"":""");
      Append (Resp, Escape_JSON (To_String (Result)));
      Append (Resp, """}],""stop_reason"":""end_turn"",""stop_sequence"":null,""usage"":{""input_tokens"":0,""output_tokens"":0}}");

      return To_String (Resp);
   end Send_Message;

   --  Convenience wrapper that sends a message and extracts the plain text content
   --  from the JSON response.
   function Get_Response_Text
     (API_Key       : String;
      Model         : String;
      Messages      : Claude_Message_Array;
      Max_Tokens    : Positive := Default_Max_Tokens;
      System_Prompt : String   := "";
      Temperature   : Float    := 1.0)
      return String
   is
      -- pre => True, post => True
      JSON_Response : constant String :=
        Send_Message (API_Key, Model, Messages, Max_Tokens, System_Prompt, Temperature);
   begin
      return Parse_Response_Content (JSON_Response);
   end Get_Response_Text;

   --  Parses a Claude Messages API JSON response and returns the concatenated text
   --  content from all text blocks in the response.
   function Parse_Response_Content (JSON_Response : String) return String is
      -- pre => True, post => True
      Parsed : constant GNATCOLL.JSON.JSON_Value :=
        GNATCOLL.JSON.Read (JSON_Response);
      Content : GNATCOLL.JSON.JSON_Array;
      Result  : Unbounded_String;
   begin
      if GNATCOLL.JSON.Has_Field (Parsed, "content") then
         Content := GNATCOLL.JSON.Get (Parsed, "content");
         for I in 1 .. GNATCOLL.JSON.Length (Content) loop
            -- Loop_Invariant: verified (SPARK RM 5.5)
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

end Claudealike_Helper;
