pragma SPARK_Mode (Off);
with Interfaces.C; use Interfaces.C;
with Interfaces.C.Strings; use Interfaces.C.Strings;
with System;
with Ada.Text_IO;
with Model_Manager;
with Mtmd_Interface; use Mtmd_Interface;

--  Image encoding pipeline for multimodal support.
--  Why: This module provides the interface for encoding images into embeddings
--       that can be used by the LLM for visual understanding. It wraps the
--       mtmd API calls and provides a clean Ada interface for image processing.
package Image_Encoder is

   --  Encode an image from raw RGB pixels into embeddings
   --  Input: Raw RGB pixel data (nx * ny * 3 bytes in RGBRGBRGB... format)
   --  Output: Embedding data written to the mtmd context
   --  Returns: True on success, False on failure
   function Encode_Image
     (Nx         : unsigned;
      Ny         : unsigned;
      Pixel_Data : System.Address) return Boolean;

   --  Encode an image from raw image bytes (JPEG, PNG, etc.)
   --  The mtmd helper decodes the image internally using stb_image.
   --  Returns: True on success, False on failure
   function Encode_Image_From_Buffer
     (Image_Data : System.Address;
      Image_Len  : size_t) return Boolean;

   --  Encode an image from a file (supports PNG, JPG, etc.)
   --  Reads the file into a buffer and calls Encode_Image_From_Buffer.
   function Encode_Image_From_File
     (Filename : String) return Boolean;

   --  Get the number of embedding tokens from the last encoded image
   function Get_Last_Image_Tokens return Natural;

   --  Get the embedding data from the last encoded image
   --  Returns a pointer to the float array containing the embeddings
   function Get_Last_Image_Embeddings return System.Address;

   --  Free the last encoded image data
   procedure Free_Last_Image;

end Image_Encoder;
