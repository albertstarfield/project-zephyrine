pragma SPARK_Mode (Off);
--  KV Cache Manager — SSD Cache Spillover for llama.cpp
--
--  Saves and loads the llama.cpp KV cache to/from SSD between inference
--  sessions. This allows long-context conversations to resume without
--  recomputing the entire KV cache from scratch.
--
--  Uses llama.cpp's Llama_State_Save_File / Llama_State_Load_File APIs.
--  Cache files are stored in cache/kv/ with SHA-256 hash naming.
--  LRU eviction keeps at most Max_Cache_Files entries.

with Interfaces.C;
with System;
with Llama_Interface;

package KV_Cache_Manager is

   Max_Cache_Files : constant := 10;
   Cache_Dir       : constant String := "cache/kv/";

   --  Save KV cache to SSD file
   --  Tokens: pointer to token array (from llama_tokenize)
   --  N_Tokens: number of tokens
   --  File_Path: full path to save file
   --  Success: True if save completed successfully
   procedure Save_To_SSD
     (Context    : Llama_Interface.Llama_Context;
      Tokens     : System.Address;
      N_Tokens   : Interfaces.C.size_t;
      File_Path  : String;
      Success    : out Boolean);

   --  Load KV cache from SSD file
   --  Returns True if load completed successfully
   --  Tokens: pointer to token array (caller must free)
   --  N_Tokens: number of tokens loaded
   function Load_From_SSD
     (Context    : Llama_Interface.Llama_Context;
      File_Path  : String;
      Tokens     : out System.Address;
      N_Tokens   : out Interfaces.C.size_t) return Boolean;

   --  Check if SSD cache exists for a given prompt prefix hash
   function Has_Cached_Prefix (Prompt_Hash : String) return Boolean;

   --  Generate SHA-256 hash of first N tokens
   function Hash_Tokens
     (Tokens   : System.Address;
      N_Tokens  : Interfaces.C.size_t;
      Max_Hash  : Positive := 128) return String;

   --  Evict oldest cache files if exceeding Max_Cache_Files
   procedure Evict_Old_Cache;

   --  Initialize cache directory
   procedure Initialize;

end KV_Cache_Manager;
