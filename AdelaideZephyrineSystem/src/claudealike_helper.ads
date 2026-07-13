--  claudealike_helper.ads
--  Claude API client for Anthropic's Messages API.
--  Supports non-streaming requests to Claude 3.5 Sonnet, Haiku, Opus, etc.

with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with GNATCOLL.JSON;         use GNATCOLL.JSON;

package Claudealike_Helper is

   --  Claude API base URL
   Claude_Base_URL : constant String := "https://api.anthropic.com";

   --  API version header
   API_Version : constant String := "2023-06-01";

   --  Max tokens for response (configurable per-request)
   Default_Max_Tokens : constant Positive := 4096;

   type Claude_Message_Role is (User, Assistant);

   type Claude_Message is record
      Role    : Claude_Message_Role;
      Content : Unbounded_String;
   end record;

   type Claude_Message_Array is array (Positive range <>) of Claude_Message;

   --  Send a non-streaming request to Claude Messages API.
   --  Returns the JSON response as a string.
   --  Raises an exception on HTTP errors or invalid API key.
   function Send_Message
     (API_Key     : String;
      Model       : String;          --  e.g. "claude-3-5-sonnet-20241022"
      Messages    : Claude_Message_Array;
      Max_Tokens  : Positive := Default_Max_Tokens;
      System_Prompt : String := "";
      Temperature : Float := 1.0)
      return String;

   --  Send a request and return just the assistant's text response.
   function Get_Response_Text
     (API_Key     : String;
      Model       : String;
      Messages    : Claude_Message_Array;
      Max_Tokens  : Positive := Default_Max_Tokens;
      System_Prompt : String := "";
      Temperature : Float := 1.0)
      return String;

   --  Parse a Claude API JSON response and extract the text content.
   function Parse_Response_Content (JSON_Response : String) return String;

   --  Check if a model name looks like a Claude model
   function Is_Claude_Model (Model_Name : String) return Boolean;

end Claudealike_Helper;
