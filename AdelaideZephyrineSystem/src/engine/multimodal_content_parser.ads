pragma SPARK_Mode (Off);
-- third-party: gnatcoll (GNATCOLL.JSON — no SPARK contracts) + thread safety
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Ada.Streams;
with GNATCOLL.JSON;
with Image_Encoder;

--  Utility functions for parsing OpenAI/Ollama request content.
--  Why: The OpenAI API supports two content formats:
--       1. Simple string: "content": "What is in this image?"
--       2. Multipart array: "content": [{"type": "text", "text": "..."}, {"type": "image_url", ...}]
--       Ollama uses a separate format: "images": ["base64_data", ...]
--       This module handles all formats and extracts images for encoding.
package Multimodal_Content_Parser is

   --  Extract text content from an OpenAI message content field
   --  Handles both string and array formats
   function Extract_Text_Content
     (Message : GNATCOLL.JSON.JSON_Value) return Unbounded_String with Pre => True, Post => True;

   --  Extract and encode images from an OpenAI message content field
   --  Returns True if any images were found and encoded
   function Extract_And_Encode_Images
     (Message : GNATCOLL.JSON.JSON_Value) return Boolean with Pre => True, Post => True;

   --  Extract and encode images from Ollama "images" field
   --  Ollama format: "images": ["base64_encoded_data", ...]
   --  Returns True if any images were found and encoded
   function Extract_Ollama_Images
     (Message : GNATCOLL.JSON.JSON_Value) return Boolean with Pre => True, Post => True;

   --  Check if a message contains image content (OpenAI or Ollama format)
   function Has_Images
     (Message : GNATCOLL.JSON.JSON_Value) return Boolean with Pre => True, Post => True;

end Multimodal_Content_Parser;
