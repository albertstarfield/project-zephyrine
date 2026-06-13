pragma SPARK_Mode (Off);
with Interfaces.C; use Interfaces.C;
with Interfaces.C.Strings; use Interfaces.C.Strings;
with System;
with Ada.Text_IO;
with Ada.Unchecked_Deallocation;
with Ada.Streams.Stream_IO;
with Model_Manager;
with Model_Types; use Model_Types;
with Mtmd_Interface; use Mtmd_Interface;

--  Implementation of the image encoding pipeline.
--  Why: This module wraps the mtmd API calls for image encoding.
--       The mtmd API handles the CLIP vision encoder and projection
--       into the text model's embedding space.
package body Image_Encoder is

   --  State for the last encoded image
   type Image_Encoding_State is record
      Bitmap      : Mtmd_Bitmap := Null_Mtmd_Bitmap;
      Chunks      : Mtmd_Input_Chunks := Mtmd_Input_Chunks (System.Null_Address);
      N_Tokens    : Natural := 0;
      Embeddings  : System.Address := System.Null_Address;
      Is_Valid    : Boolean := False;
   end record;

   Last_Image : Image_Encoding_State;

   --  Helper: Get the default media marker as an Ada string
   function Get_Marker return String is
      Marker_Ptr : chars_ptr := Mtmd_Default_Marker_Safe;
   begin
      if Marker_Ptr = Null_Ptr then
         return "<__media__>";
      end if;
      return Interfaces.C.Strings.Value (Marker_Ptr);
   end Get_Marker;

   --  Encode an image from raw RGB pixels into embeddings
   --  Input: Raw RGB pixel data (nx * ny * 3 bytes in RGBRGBRGB... format)
   --  Output: Embedding data written to the mtmd context
   --  Returns: True on success, False on failure
   function Encode_Image
     (Nx         : unsigned;
      Ny         : unsigned;
      Pixel_Data : System.Address) return Boolean
   is
      Mtmd_Ctx : Mtmd_Context;
      Bitmap   : Mtmd_Bitmap;
      Chunks   : Mtmd_Input_Chunks;
      Marker   : constant String := Get_Marker;
      Prompt   : constant String := "Describe this image in detail." & Marker;
      Text_Ptr : chars_ptr;
      Result   : int;
   begin
      --  Clean up any previous encoding
      Free_Last_Image;

      --  Get the mtmd context
      Mtmd_Ctx := Model_Manager.Get_Mtmd_Context (Model_Types.MMProj);
      if Mtmd_Ctx = Null_Mtmd_Context then
         Ada.Text_IO.Put_Line ("[Image_Encoder] MMProj not loaded");
         return False;
      end if;

      --  Create bitmap from raw pixels
      Bitmap := Mtmd_Bitmap_Init_Safe (Nx, Ny, Pixel_Data);
      if Bitmap = Null_Mtmd_Bitmap then
         Ada.Text_IO.Put_Line ("[Image_Encoder] Failed to create bitmap");
         return False;
      end if;

      --  Create input chunks list
      Chunks := Mtmd_Input_Chunks_Init_Safe;
      if Chunks = Mtmd_Input_Chunks (System.Null_Address) then
         Ada.Text_IO.Put_Line ("[Image_Encoder] Failed to create chunks");
         Mtmd_Bitmap_Free_Safe (Bitmap);
         return False;
      end if;

      --  Tokenize the prompt with the image
      --  The marker in the prompt will be replaced with the image chunk
      Text_Ptr := New_String (Prompt);
      begin
         Result := Mtmd_Tokenize_Safe
           (Ctx           => Mtmd_Ctx,
            Output        => Chunks,
            Text          => Text_Ptr,
            Add_Special   => True,
            Parse_Special => True,
            Bitmaps       => Bitmap'Address,
            N_Bitmaps     => 1);
      end;
      Free (Text_Ptr);

      if Result /= 0 then
         Ada.Text_IO.Put_Line
           ("[Image_Encoder] mtmd_tokenize failed: " & int'Image (Result));
         Mtmd_Input_Chunks_Free_Safe (Chunks);
         Mtmd_Bitmap_Free_Safe (Bitmap);
         return False;
      end if;

      --  Iterate chunks and encode image chunks
      declare
         N_Chunks : constant size_t :=
           Mtmd_Input_Chunks_Size_Safe (Chunks);
      begin
         for I in 0 .. N_Chunks - 1 loop
            declare
               Chunk : constant Mtmd_Input_Chunk :=
                 Mtmd_Input_Chunks_Get_Safe (Chunks, I);
               Chunk_Type : constant int :=
                 Mtmd_Input_Chunk_Get_Type_Safe (Chunk);
            begin
               --  MTMD_INPUT_CHUNK_TYPE_IMAGE = 1
               if Chunk_Type = 1 then
                  --  Found an image chunk - encode it
                  declare
                     Enc_Result : constant int :=
                       Mtmd_Encode_Chunk_Safe (Mtmd_Ctx, Chunk);
                  begin
                     if Enc_Result /= 0 then
                        Ada.Text_IO.Put_Line
                          ("[Image_Encoder] mtmd_encode_chunk failed: " &
                           int'Image (Enc_Result));
                        Mtmd_Input_Chunks_Free_Safe (Chunks);
                        Mtmd_Bitmap_Free_Safe (Bitmap);
                        return False;
                     end if;
                     --  Get the embeddings
                     Last_Image.Embeddings :=
                       Mtmd_Get_Output_Embd_Safe (Mtmd_Ctx);
                     Last_Image.N_Tokens :=
                       Natural (Mtmd_Input_Chunk_Get_N_Tokens_Safe (Chunk));
                  end;
               end if;
            end;
         end loop;
      end;

      --  Store the bitmap and chunks for later use
      Last_Image.Bitmap := Bitmap;
      Last_Image.Chunks := Chunks;
      Last_Image.Is_Valid := True;

      Ada.Text_IO.Put_Line
        ("[Image_Encoder] Image encoded successfully, tokens=" &
         Natural'Image (Last_Image.N_Tokens));
      return True;
   end Encode_Image;

   --  Encode an image from raw image bytes (JPEG, PNG, etc.)
   --  The mtmd helper decodes the image internally using stb_image.
   --  Returns: True on success, False on failure
   function Encode_Image_From_Buffer
     (Image_Data : System.Address;
      Image_Len  : size_t) return Boolean
   is
      Mtmd_Ctx : Mtmd_Context;
      Bitmap   : Mtmd_Bitmap;
      Chunks   : Mtmd_Input_Chunks;
      Marker   : constant String := Get_Marker;
      Prompt   : constant String := "Describe this image in detail." & Marker;
      Text_Ptr : chars_ptr;
      Result   : int;
   begin
      --  Clean up any previous encoding
      Free_Last_Image;

      --  Get the mtmd context
      Mtmd_Ctx := Model_Manager.Get_Mtmd_Context (Model_Types.MMProj);
      if Mtmd_Ctx = Null_Mtmd_Context then
         Ada.Text_IO.Put_Line ("[Image_Encoder] MMProj not loaded");
         return False;
      end if;

      --  Create bitmap from image buffer (JPEG/PNG decoded by stb_image)
      Bitmap := Mtmd_Helper_Bitmap_Init_From_Buf_Safe (Mtmd_Ctx, Image_Data, Image_Len);
      if Bitmap = Null_Mtmd_Bitmap then
         Ada.Text_IO.Put_Line ("[Image_Encoder] Failed to decode image buffer");
         return False;
      end if;

      --  Create input chunks list
      Chunks := Mtmd_Input_Chunks_Init_Safe;
      if Chunks = Mtmd_Input_Chunks (System.Null_Address) then
         Ada.Text_IO.Put_Line ("[Image_Encoder] Failed to create chunks");
         Mtmd_Bitmap_Free_Safe (Bitmap);
         return False;
      end if;

      --  Tokenize the prompt with the image
      Text_Ptr := New_String (Prompt);
      begin
         Result := Mtmd_Tokenize_Safe
           (Ctx           => Mtmd_Ctx,
            Output        => Chunks,
            Text          => Text_Ptr,
            Add_Special   => True,
            Parse_Special => True,
            Bitmaps       => Bitmap'Address,
            N_Bitmaps     => 1);
      end;
      Free (Text_Ptr);

      if Result /= 0 then
         Ada.Text_IO.Put_Line
           ("[Image_Encoder] mtmd_tokenize failed: " & int'Image (Result));
         Mtmd_Input_Chunks_Free_Safe (Chunks);
         Mtmd_Bitmap_Free_Safe (Bitmap);
         return False;
      end if;

      --  Iterate chunks and encode image chunks
      declare
         N_Chunks : constant size_t :=
           Mtmd_Input_Chunks_Size_Safe (Chunks);
      begin
         for I in 0 .. N_Chunks - 1 loop
            declare
               Chunk : constant Mtmd_Input_Chunk :=
                 Mtmd_Input_Chunks_Get_Safe (Chunks, I);
               Chunk_Type : constant int :=
                 Mtmd_Input_Chunk_Get_Type_Safe (Chunk);
            begin
               --  MTMD_INPUT_CHUNK_TYPE_IMAGE = 1
               if Chunk_Type = 1 then
                  declare
                     Enc_Result : constant int :=
                       Mtmd_Encode_Chunk_Safe (Mtmd_Ctx, Chunk);
                  begin
                     if Enc_Result /= 0 then
                        Ada.Text_IO.Put_Line
                          ("[Image_Encoder] mtmd_encode_chunk failed: " &
                           int'Image (Enc_Result));
                        Mtmd_Input_Chunks_Free_Safe (Chunks);
                        Mtmd_Bitmap_Free_Safe (Bitmap);
                        return False;
                     end if;
                     Last_Image.Embeddings :=
                       Mtmd_Get_Output_Embd_Safe (Mtmd_Ctx);
                     Last_Image.N_Tokens :=
                       Natural (Mtmd_Input_Chunk_Get_N_Tokens_Safe (Chunk));
                  end;
               end if;
            end;
         end loop;
      end;

      --  Store the bitmap and chunks for later use
      Last_Image.Bitmap := Bitmap;
      Last_Image.Chunks := Chunks;
      Last_Image.Is_Valid := True;

      Ada.Text_IO.Put_Line
        ("[Image_Encoder] Image from buffer encoded successfully, tokens=" &
         Natural'Image (Last_Image.N_Tokens));
      return True;
   end Encode_Image_From_Buffer;

   --  Encode an image from a file (supports PNG, JPG, etc.)
   --  Reads the file into a buffer and calls Encode_Image_From_Buffer.
   function Encode_Image_From_File
     (Filename : String) return Boolean
   is
      use Ada.Streams.Stream_IO;
      File   : File_Type;
      File_Size : Natural;
      Data   : System.Address;
   begin
      --  Open the file and get its size
      begin
         Open (File, In_File, Filename);
      exception
         when others =>
            Ada.Text_IO.Put_Line
              ("[Image_Encoder] Cannot open file: " & Filename);
            return False;
      end;

      File_Size := Natural (Ada.Streams.Stream_IO.Size (File));

      if File_Size = 0 then
         Ada.Text_IO.Put_Line
           ("[Image_Encoder] Empty file: " & Filename);
         Close (File);
         return False;
      end if;

      --  Read file contents into a buffer
      declare
         Buffer : Ada.Streams.Stream_Element_Array (1 .. Ada.Streams.Stream_Element_Count (File_Size));
         Last   : Ada.Streams.Stream_Element_Offset;
      begin
         Read (File, Buffer, Last);
         Close (File);

         --  Call Encode_Image_From_Buffer with the raw bytes
         return Encode_Image_From_Buffer
           (Buffer'Address, size_t (Last));
      end;
   end Encode_Image_From_File;

   --  Get the number of embedding tokens from the last encoded image
   function Get_Last_Image_Tokens return Natural is
   begin
      return Last_Image.N_Tokens;
   end Get_Last_Image_Tokens;

   --  Get the embedding data from the last encoded image
   --  Returns a pointer to the float array containing the embeddings
   function Get_Last_Image_Embeddings return System.Address is
   begin
      return Last_Image.Embeddings;
   end Get_Last_Image_Embeddings;

   --  Free the last encoded image data
   procedure Free_Last_Image is
   begin
      if Last_Image.Is_Valid then
         if Last_Image.Bitmap /= Null_Mtmd_Bitmap then
            Mtmd_Bitmap_Free_Safe (Last_Image.Bitmap);
            Last_Image.Bitmap := Null_Mtmd_Bitmap;
         end if;
         if Last_Image.Chunks /= Mtmd_Input_Chunks (System.Null_Address) then
            Mtmd_Input_Chunks_Free_Safe (Last_Image.Chunks);
            Last_Image.Chunks := Mtmd_Input_Chunks (System.Null_Address);
         end if;
         Last_Image.N_Tokens := 0;
         Last_Image.Embeddings := System.Null_Address;
         Last_Image.Is_Valid := False;
      end if;
   end Free_Last_Image;

end Image_Encoder;
