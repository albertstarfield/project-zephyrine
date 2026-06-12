pragma SPARK_Mode (Off);
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Ada.Text_IO;
with GNATCOLL.JSON; use GNATCOLL.JSON;
with Image_Encoder;

--  Implementation of OpenAI content parsing utilities.
--  Why: This module handles the complexity of parsing OpenAI-compatible
--       message content formats, including vision/multipart content.
package body Multimodal_Content_Parser is

   --  Extract text content from an OpenAI message content field
   --  Handles both string and array formats
   function Extract_Text_Content
     (Message : GNATCOLL.JSON.JSON_Value) return Unbounded_String
   is
      Result : Unbounded_String := Null_Unbounded_String;
   begin
      if not GNATCOLL.JSON.Has_Field (Message, "content") then
         return Result;
      end if;

      declare
         Content : constant GNATCOLL.JSON.JSON_Value :=
           GNATCOLL.JSON.Get (Message, "content");
      begin
         if Content.Kind = GNATCOLL.JSON.JSON_String_Type then
            --  Simple string content
            Result := To_Unbounded_String
              (String'(GNATCOLL.JSON.Get (Content)));
         elsif Content.Kind = GNATCOLL.JSON.JSON_Array_Type then
            --  Multipart array content
            declare
               Parts : constant GNATCOLL.JSON.JSON_Array :=
                 GNATCOLL.JSON.Get (Content);
            begin
               for J in 1 .. GNATCOLL.JSON.Length (Parts) loop
                  declare
                     Part : constant GNATCOLL.JSON.JSON_Value :=
                       GNATCOLL.JSON.Get (Parts, J);
                     Part_Type : constant String :=
                       GNATCOLL.JSON.Get (Part, "type");
                  begin
                     if Part_Type = "text" then
                        if GNATCOLL.JSON.Has_Field (Part, "text") then
                           if Length (Result) > 0 then
                              Append (Result, ASCII.LF);
                           end if;
                           Append (Result,
                             To_Unbounded_String
                               (String'(GNATCOLL.JSON.Get (Part, "text"))));
                        end if;
                     end if;
                  end;
               end loop;
            end;
         end if;
      end;

      return Result;
   end Extract_Text_Content;

   --  Extract and encode images from an OpenAI message content field
   --  Returns True if any images were found and encoded
   function Extract_And_Encode_Images
     (Message : GNATCOLL.JSON.JSON_Value) return Boolean
   is
      Found_Images : Boolean := False;
   begin
      if not GNATCOLL.JSON.Has_Field (Message, "content") then
         return False;
      end if;

      declare
         Content : constant GNATCOLL.JSON.JSON_Value :=
           GNATCOLL.JSON.Get (Message, "content");
      begin
         if Content.Kind /= GNATCOLL.JSON.JSON_Array_Type then
            --  Not an array, no images possible
            return False;
         end if;

         declare
            Parts : constant GNATCOLL.JSON.JSON_Array :=
              GNATCOLL.JSON.Get (Content);
         begin
            for J in 1 .. GNATCOLL.JSON.Length (Parts) loop
               declare
                  Part : constant GNATCOLL.JSON.JSON_Value :=
                    GNATCOLL.JSON.Get (Parts, J);
                  Part_Type : constant String :=
                    GNATCOLL.JSON.Get (Part, "type");
               begin
                  if Part_Type = "image_url" then
                     --  Found an image URL
                     if GNATCOLL.JSON.Has_Field (Part, "image_url") then
                        declare
                           Image_URL_Obj : constant GNATCOLL.JSON.JSON_Value :=
                             GNATCOLL.JSON.Get (Part, "image_url");
                           URL : constant String :=
                             GNATCOLL.JSON.Get (Image_URL_Obj, "url");
                        begin
                           --  Check if it's a base64 data URL
                           if URL'Length > 22 and then
                             URL (URL'First .. URL'First + 21) =
                             "data:image/jpeg;base64,"
                           then
                              --  TODO: Decode base64 and encode image
                              --  For now, log that we found an image
                              Ada.Text_IO.Put_Line
                                ("[OpenAI_Content_Parser] Found base64 image");
                              Found_Images := True;
                           elsif URL'Length > 22 and then
                             URL (URL'First .. URL'First + 21) =
                             "data:image/png;base64,"
                           then
                              Ada.Text_IO.Put_Line
                                ("[OpenAI_Content_Parser] Found base64 PNG image");
                              Found_Images := True;
                           elsif URL'Length > 7 and then
                             URL (URL'First .. URL'First + 6) = "http://"
                           then
                              --  HTTP URL - would need to fetch
                              Ada.Text_IO.Put_Line
                                ("[OpenAI_Content_Parser] Found HTTP image URL");
                              Found_Images := True;
                           elsif URL'Length > 8 and then
                             URL (URL'First .. URL'First + 7) = "https://"
                           then
                              --  HTTPS URL - would need to fetch
                              Ada.Text_IO.Put_Line
                                ("[OpenAI_Content_Parser] Found HTTPS image URL");
                              Found_Images := True;
                           end if;
                        end;
                     end if;
                  end if;
               end;
            end loop;
         end;
      end;

      return Found_Images;
   end Extract_And_Encode_Images;

   --  Check if a message contains image content
   function Has_Images
     (Message : GNATCOLL.JSON.JSON_Value) return Boolean
   is
      Dummy : Unbounded_String;
      pragma Unreferenced (Dummy);
   begin
      --  Use Extract_And_Encode_Images but don't actually encode
      --  This is a bit inefficient but keeps the code simple
      if not GNATCOLL.JSON.Has_Field (Message, "content") then
         return False;
      end if;

      declare
         Content : constant GNATCOLL.JSON.JSON_Value :=
           GNATCOLL.JSON.Get (Message, "content");
      begin
         if Content.Kind /= GNATCOLL.JSON.JSON_Array_Type then
            return False;
         end if;

         declare
            Parts : constant GNATCOLL.JSON.JSON_Array :=
              GNATCOLL.JSON.Get (Content);
         begin
            for J in 1 .. GNATCOLL.JSON.Length (Parts) loop
               declare
                  Part : constant GNATCOLL.JSON.JSON_Value :=
                    GNATCOLL.JSON.Get (Parts, J);
                  Part_Type : constant String :=
                    GNATCOLL.JSON.Get (Part, "type");
               begin
                  if Part_Type = "image_url" then
                     return True;
                  end if;
               end;
            end loop;
         end;
      end;

      return False;
   end Has_Images;

end Multimodal_Content_Parser;
