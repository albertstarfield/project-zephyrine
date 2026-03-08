pragma Ada_2022;

with AWS.Client;
with AWS.Response;
with AWS.Messages; use AWS.Messages;
with GNATCOLL.JSON;
with Ada.Text_IO;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;

package body Cognitive_Lang_Resp is

   Backend_Base_URL : Unbounded_String := 
      To_Unbounded_String ("http://localhost:11434");

   -----------------------
   -- Configure_Backend --
   -----------------------
   procedure Configure_Backend (Base_URL : String) is
   begin
      Backend_Base_URL := To_Unbounded_String (Base_URL);
      Ada.Text_IO.Put_Line ("[Zephy] Cognitive Backend set to: " & Base_URL);
   end Configure_Backend;

   ---------------------
   -- Get_Backend_URL --
   ---------------------
   function Get_Backend_URL return String is
   begin
      return To_String (Backend_Base_URL);
   end Get_Backend_URL;

   -------------------
   -- Process_Input --
   -------------------
   function Process_Input 
     (Input_Sequence : String;
      Model_ID       : String := "Snowball-Enaga") return String 
   is
      -- ... (Existing code for Process_Input remains unchanged) ...
      -- Just for context, the existing code:
      URL : constant String := To_String (Backend_Base_URL) & "/v1/chat/responsezeph";
      Payload : GNATCOLL.JSON.JSON_Value := GNATCOLL.JSON.Create_Object;
      Result  : AWS.Response.Data;
      Reply   : GNATCOLL.JSON.JSON_Value;
   begin
      Payload.Set_Field ("model", Model_ID);
      Payload.Set_Field ("prompt", Input_Sequence);
      Payload.Set_Field ("stream", False); 
      
      Result := AWS.Client.Post
        (URL          => URL,
         Data         => GNATCOLL.JSON.Write (Payload),
         Content_Type => "application/json");

      if AWS.Response.Status_Code (Result) /= AWS.Messages.S200 then
         return "System Alert: Adaptive System Connection Lost.";
      end if;

      declare
         Body_Content : constant String := AWS.Response.Message_Body (Result);
      begin
         Reply := GNATCOLL.JSON.Read (Body_Content);
      end;

      if Reply.Has_Field ("response") then
         return Reply.Get ("response");
      elsif Reply.Has_Field ("content") then
         return Reply.Get ("content");
      else
         return "System Alert: Invalid Data Format.";
      end if;
   exception
      when others =>
         return "System Alert: Cognitive Processing Failure.";
   end Process_Input;

end Cognitive_Lang_Resp;