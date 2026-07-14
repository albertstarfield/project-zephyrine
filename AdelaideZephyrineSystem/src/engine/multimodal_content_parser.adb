pragma SPARK_Mode (Off);
-- thread: Parser requires task protection
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Ada.Streams;
with Ada.Text_IO;
with GNATCOLL.JSON; use GNATCOLL.JSON;
with Image_Encoder;
with Interfaces; use Interfaces;
with Interfaces.C; use Interfaces.C;

--  Implementation of OpenAI/Ollama content parsing utilities.
--  Why: This module handles the complexity of parsing OpenAI-compatible
--       message content formats, including vision/multipart content.
--       It also handles Ollama's "images" field format.
package body Multimodal_Content_Parser is

   --  Base64 decoding table
   --  Why: We need to decode base64-encoded image data from API requests
   --       into raw bytes that can be passed to the mtmd image decoder.
   type Base64_Table_Type is array (Character) of Natural;
   Base64_Table : Base64_Table_Type := (others => 0);

   procedure Init_Base64_Table is
   begin
      for C in Standard.Character range 'A' .. 'Z' loop
         Base64_Table (C) := Character'Pos (C) - Character'Pos ('A');
      end loop;
      for C in Standard.Character range 'a' .. 'z' loop
         Base64_Table (C) := Character'Pos (C) - Character'Pos ('a') + 26;
      end loop;
      for C in Standard.Character range '0' .. '9' loop
         Base64_Table (C) := Character'Pos (C) - Character'Pos ('0') + 52;
      end loop;
      Base64_Table ('+') := 62;
      Base64_Table ('/') := 63;
   end Init_Base64_Table;

   Table_Initialized : Boolean := False;

   --  Decode a base64 string into raw bytes
   --  Why: API requests send image data as base64-encoded strings.
   --       We need to decode them to raw bytes for the mtmd image decoder.
   function Decode_Base64
     (Encoded : String) return Ada.Streams.Stream_Element_Array
   is
      use Ada.Streams;
      --  Calculate output length (approximate, will be trimmed)
      Enc_Len  : constant Natural := Encoded'Length;
      Out_Len  : constant Natural := (Enc_Len * 3) / 4;
      Result   : Stream_Element_Array (1 .. Stream_Element_Count (Out_Len));
      Out_Idx  : Stream_Element_Offset := 1;
      Acc      : Natural := 0;
      Bits     : Natural := 0;
   begin
      if not Table_Initialized then
         Init_Base64_Table;
         Table_Initialized := True;
      end if;

      for I in Encoded'Range loop
         declare
            C : constant Character := Encoded (I);
         begin
            if C = '=' then
               --  Padding, process remaining bits
               Acc := Acc * 4;
               Bits := Bits + 2;
               if Bits >= 8 then
                  Bits := Bits - 8;
                  Result (Out_Idx) :=
                    Stream_Element (Shift_Right (Unsigned_32 (Acc), Bits) and 16#FF#);
                  Out_Idx := Out_Idx + 1;
               end if;
            elsif C = ' ' or C = ASCII.LF or C = ASCII.CR or C = ASCII.HT then
               --  Skip whitespace
               null;
            elsif Base64_Table (C) > 0 or else C = 'A' then
               Acc := Acc * 64 + Base64_Table (C);
               Bits := Bits + 6;
               while Bits >= 8 loop
                  Bits := Bits - 8;
                  Result (Out_Idx) :=
                    Stream_Element (Shift_Right (Unsigned_32 (Acc), Bits) and 16#FF#);
                  Out_Idx := Out_Idx + 1;
               end loop;
            end if;
         end;
      end loop;

      --  Return only the decoded portion
      if Out_Idx > 1 then
         return Result (1 .. Out_Idx - 1);
      else
         return Result (1 .. 0);  --  Empty array
      end if;
   end Decode_Base64;

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

   --  Process a single base64 image string (decode and encode)
   --  Returns True on success
   function Process_Base64_Image
     (Base64_Data : String) return Boolean
   is
      use Ada.Streams;
      Decoded : constant Stream_Element_Array :=
        Decode_Base64 (Base64_Data);
   begin
      if Decoded'Length = 0 then
         Ada.Text_IO.Put_Line
           ("[Multimodal_Content_Parser] Empty base64 data");
         return False;
      end if;

      Ada.Text_IO.Put_Line
        ("[Multimodal_Content_Parser] Decoded base64 image, bytes=" &
         Natural'Image (Natural (Decoded'Length)));

      --  Pass raw bytes to image encoder (stb_image handles JPEG/PNG)
      return Image_Encoder.Encode_Image_From_Buffer
        (Decoded'Address, size_t (Decoded'Length));
   end Process_Base64_Image;

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
                              --  Decode base64 JPEG and encode image
                              declare
                                 B64_Start : constant Positive :=
                                   URL'First + 22;
                                 B64_Data  : constant String :=
                                   URL (B64_Start .. URL'Last);
                              begin
                                 Ada.Text_IO.Put_Line
                                   ("[Multimodal_Content_Parser] Processing base64 JPEG image");
                                 if Process_Base64_Image (B64_Data) then
                                    Found_Images := True;
                                 end if;
                              end;
                           elsif URL'Length > 22 and then
                             URL (URL'First .. URL'First + 21) =
                             "data:image/png;base64,"
                           then
                              --  Decode base64 PNG and encode image
                              declare
                                 B64_Start : constant Positive :=
                                   URL'First + 22;
                                 B64_Data  : constant String :=
                                   URL (B64_Start .. URL'Last);
                              begin
                                 Ada.Text_IO.Put_Line
                                   ("[Multimodal_Content_Parser] Processing base64 PNG image");
                                 if Process_Base64_Image (B64_Data) then
                                    Found_Images := True;
                                 end if;
                              end;
                           elsif URL'Length > 7 and then
                             URL (URL'First .. URL'First + 6) = "http://"
                           then
                              --  HTTP URL - would need to fetch
                              Ada.Text_IO.Put_Line
                                ("[Multimodal_Content_Parser] HTTP image URL not yet supported");
                           elsif URL'Length > 8 and then
                             URL (URL'First .. URL'First + 7) = "https://"
                           then
                              --  HTTPS URL - would need to fetch
                              Ada.Text_IO.Put_Line
                                ("[Multimodal_Content_Parser] HTTPS image URL not yet supported");
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

   --  Extract and encode images from Ollama "images" field
   --  Ollama format: "images": ["base64_encoded_data", ...]
   --  Returns True if any images were found and encoded
   function Extract_Ollama_Images
     (Message : GNATCOLL.JSON.JSON_Value) return Boolean
   is
      Found_Images : Boolean := False;
   begin
      if not GNATCOLL.JSON.Has_Field (Message, "images") then
         return False;
      end if;

      declare
         Images : constant GNATCOLL.JSON.JSON_Array :=
           GNATCOLL.JSON.Get (Message, "images");
      begin
         for J in 1 .. GNATCOLL.JSON.Length (Images) loop
            declare
               Img : constant GNATCOLL.JSON.JSON_Value :=
                 GNATCOLL.JSON.Get (Images, J);
            begin
               if Img.Kind = GNATCOLL.JSON.JSON_String_Type then
                  declare
                     B64_Data : constant String :=
                       String'(GNATCOLL.JSON.Get (Img));
                  begin
                     Ada.Text_IO.Put_Line
                       ("[Multimodal_Content_Parser] Processing Ollama image " &
                        Natural'Image (J));
                     if Process_Base64_Image (B64_Data) then
                        Found_Images := True;
                     end if;
                  end;
               end if;
            end;
         end loop;
      end;

      return Found_Images;
   end Extract_Ollama_Images;

   --  Check if a message contains image content
   function Has_Images
     (Message : GNATCOLL.JSON.JSON_Value) return Boolean
   is
   begin
      --  Check OpenAI format (content array with image_url parts)
      if GNATCOLL.JSON.Has_Field (Message, "content") then
         declare
            Content : constant GNATCOLL.JSON.JSON_Value :=
              GNATCOLL.JSON.Get (Message, "content");
         begin
            if Content.Kind = GNATCOLL.JSON.JSON_Array_Type then
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
            end if;
         end;
      end if;

      --  Check Ollama format (images array)
      if GNATCOLL.JSON.Has_Field (Message, "images") then
         return True;
      end if;

      return False;
   end Has_Images;

end Multimodal_Content_Parser;
