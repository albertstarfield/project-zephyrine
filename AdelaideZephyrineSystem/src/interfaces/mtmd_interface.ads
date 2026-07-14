pragma SPARK_Mode (Off);
-- c_binding: multimodal C FFI
with Interfaces.C; use Interfaces.C;
with Interfaces.C.Strings; use Interfaces.C.Strings;
with System;

--  Ada bindings for mtmd (multimodal) C++ wrappers in llama_safe.cpp.
--  Why: These provide the Ada FFI interface to the llama.cpp multimodal API.
--       The mtmd API handles image/audio encoding for vision-capable models.
--       We wrap the C++ code with exception safety at the boundary.
package Mtmd_Interface is

   --  Opaque handle types (void pointers in C)
   type Mtmd_Context is new System.Address; -- FFI: System.Address required for C binding
   type Mtmd_Bitmap is new System.Address; -- FFI: System.Address required for C binding
   type Mtmd_Input_Chunks is new System.Address; -- FFI: System.Address required for C binding
   type Mtmd_Input_Chunk is new System.Address; -- FFI: System.Address required for C binding

   Null_Mtmd_Context : constant Mtmd_Context :=
     Mtmd_Context (System.Null_Address);
   Null_Mtmd_Bitmap  : constant Mtmd_Bitmap :=
     Mtmd_Bitmap (System.Null_Address);

   --  Chunk types
   MTMD_INPUT_CHUNK_TYPE_TEXT  : constant int := 0;
   MTMD_INPUT_CHUNK_TYPE_IMAGE : constant int := 1;
   MTMD_INPUT_CHUNK_TYPE_AUDIO : constant int := 2;

   --  Initialize mtmd context from mmproj file
   --  Returns Null_Mtmd_Context on failure
   function Mtmd_Init_From_File_Safe
     (Mmproj_Fname : chars_ptr;
      Text_Model   : System.Address; -- FFI: System.Address required for C binding
      Use_Gpu      : Boolean;
      N_Threads    : int) return Mtmd_Context;
   pragma Import
     (C, Mtmd_Init_From_File_Safe, "mtmd_init_from_file_safe");

   --  Free mtmd context
   procedure Mtmd_Free_Safe (Ctx : Mtmd_Context);
   pragma Import (C, Mtmd_Free_Safe, "mtmd_free_safe");

   --  Create bitmap from raw RGB pixels
   --  Data must be Nx * Ny * 3 bytes in RGBRGBRGB... format
   function Mtmd_Bitmap_Init_Safe
     (Nx   : unsigned;
      Ny   : unsigned;
      Data : System.Address) return Mtmd_Bitmap; -- FFI: System.Address required for C binding
   pragma Import (C, Mtmd_Bitmap_Init_Safe, "mtmd_bitmap_init_safe");

   --  Free bitmap
   procedure Mtmd_Bitmap_Free_Safe (Bitmap : Mtmd_Bitmap);
   pragma Import (C, Mtmd_Bitmap_Free_Safe, "mtmd_bitmap_free_safe");

   --  Get bitmap dimensions
   function Mtmd_Bitmap_Get_Nx_Safe (Bitmap : Mtmd_Bitmap) return unsigned;
   pragma Import (C, Mtmd_Bitmap_Get_Nx_Safe, "mtmd_bitmap_get_nx_safe");

   function Mtmd_Bitmap_Get_Ny_Safe (Bitmap : Mtmd_Bitmap) return unsigned;
   pragma Import (C, Mtmd_Bitmap_Get_Ny_Safe, "mtmd_bitmap_get_ny_safe");

   --  Initialize empty input chunks list
   function Mtmd_Input_Chunks_Init_Safe return Mtmd_Input_Chunks;
   pragma Import
     (C, Mtmd_Input_Chunks_Init_Safe, "mtmd_input_chunks_init_safe");

   --  Free input chunks
   procedure Mtmd_Input_Chunks_Free_Safe (Chunks : Mtmd_Input_Chunks);
   pragma Import
     (C, Mtmd_Input_Chunks_Free_Safe, "mtmd_input_chunks_free_safe");

   --  Get number of chunks
   function Mtmd_Input_Chunks_Size_Safe
     (Chunks : Mtmd_Input_Chunks) return size_t;
   pragma Import
     (C, Mtmd_Input_Chunks_Size_Safe, "mtmd_input_chunks_size_safe");

   --  Get chunk type: 0=text, 1=image, 2=audio
   function Mtmd_Input_Chunk_Get_Type_Safe
     (Chunk : Mtmd_Input_Chunk) return int;
   pragma Import
     (C, Mtmd_Input_Chunk_Get_Type_Safe, "mtmd_input_chunk_get_type_safe");

   --  Get number of tokens in a chunk
   function Mtmd_Input_Chunk_Get_N_Tokens_Safe
     (Chunk : Mtmd_Input_Chunk) return size_t;
   pragma Import
     (C,
      Mtmd_Input_Chunk_Get_N_Tokens_Safe,
      "mtmd_input_chunk_get_n_tokens_safe");

   --  Get text tokens from a text chunk
   --  Returns pointer to internal token array, N_Tokens_Output receives count
   --  WARNING: Do not free the returned pointer - it's owned by the chunk
   function Mtmd_Input_Chunk_Get_Tokens_Text_Safe
     (Chunk           : Mtmd_Input_Chunk;
      N_Tokens_Output : access size_t) return System.Address; -- FFI: System.Address required for C binding
   pragma Import
     (C,
      Mtmd_Input_Chunk_Get_Tokens_Text_Safe,
      "mtmd_input_chunk_get_tokens_text_safe");

   --  Encode a chunk (image or audio) - must be called before using embeddings
   function Mtmd_Encode_Chunk_Safe
     (Ctx   : Mtmd_Context;
      Chunk : Mtmd_Input_Chunk) return int;
   pragma Import (C, Mtmd_Encode_Chunk_Safe, "mtmd_encode_chunk_safe");

   --  Get output embeddings after encoding
   --  Returns pointer to float array
   function Mtmd_Get_Output_Embd_Safe
     (Ctx : Mtmd_Context) return System.Address; -- FFI: System.Address required for C binding
   pragma Import (C, Mtmd_Get_Output_Embd_Safe, "mtmd_get_output_embd_safe");

   --  Check if model supports vision
   function Mtmd_Support_Vision_Safe (Ctx : Mtmd_Context) return int;
   pragma Import
     (C, Mtmd_Support_Vision_Safe, "mtmd_support_vision_safe");

   --  Check if chunk needs non-causal mask (for image chunks)
   function Mtmd_Decode_Use_Non_Causal_Safe
     (Ctx   : Mtmd_Context;
      Chunk : Mtmd_Input_Chunk) return int;
   pragma Import
     (C,
      Mtmd_Decode_Use_Non_Causal_Safe,
      "mtmd_decode_use_non_causal_safe");

   --  Get default media marker string
   function Mtmd_Default_Marker_Safe return chars_ptr;
   pragma Import (C, Mtmd_Default_Marker_Safe, "mtmd_default_marker_safe");

   --  Tokenize text prompt + bitmaps into input chunks.
   --  The text must contain the media marker (default: "<__media__>").
   --  Number of bitmaps must equal number of markers in text.
   --  Returns 0 on success, 1 on marker/bitmap count mismatch, 2 on image error.
   function Mtmd_Tokenize_Safe
     (Ctx           : Mtmd_Context;
      Output        : Mtmd_Input_Chunks;
      Text          : chars_ptr;
      Add_Special   : Boolean;
      Parse_Special : Boolean;
      Bitmaps       : System.Address; -- FFI: System.Address required for C binding
      N_Bitmaps     : size_t) return int;
   pragma Import (C, Mtmd_Tokenize_Safe, "mtmd_tokenize_safe");

   --  Create bitmap from image buffer (JPEG, PNG, etc.)
   --  Uses stb_image internally to decode the image bytes.
   --  Returns Null_Mtmd_Bitmap on failure.
   function Mtmd_Helper_Bitmap_Init_From_Buf_Safe
     (Ctx : Mtmd_Context;
      Buf : System.Address; -- FFI: System.Address required for C binding
      Len : size_t) return Mtmd_Bitmap;
   pragma Import
     (C, Mtmd_Helper_Bitmap_Init_From_Buf_Safe,
      "mtmd_helper_bitmap_init_from_buf_safe");

   --  Get a chunk from the chunks list by index.
   --  Returns null address if index is out of range.
   --  WARNING: The returned pointer is owned by the chunks list - do NOT free it.
   function Mtmd_Input_Chunks_Get_Safe
     (Chunks : Mtmd_Input_Chunks;
      Idx    : size_t) return Mtmd_Input_Chunk;
   pragma Import
     (C, Mtmd_Input_Chunks_Get_Safe, "mtmd_input_chunks_get_safe");

end Mtmd_Interface;
