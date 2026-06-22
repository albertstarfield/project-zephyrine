pragma SPARK_Mode (Off);
--  ============================================================================
--  KV Cache Manager — SSD Cache Spillover for llama.cpp
--  ============================================================================
--  WHY THIS EXISTS:
--  The KV (Key-Value) cache stores intermediate attention states during LLM
--  inference. For long-context conversations (4K+ tokens), recomputing the
--  KV cache from scratch on every request is expensive (2-5 seconds).
--
--  This module saves the KV cache to SSD after each generation and loads it
--  on subsequent requests, providing fastest response times for repeated or
--  similar prompts.
--
--  RAM POLICY (CRITICAL):
--  Only the currently processing cache stays in RAM.
--  After generation completes:
--    1. Save KV cache to SSD (persist to disk)
--    2. Clear KV cache from RAM (free memory immediately)
--  This ensures:
--    - Minimal RAM footprint (only current process in memory)
--    - Cache persists across server restarts (fastest response)
--    - No memory leaks from accumulated cache data
--
--  HOW IT WORKS:
--  - Uses llama.cpp's Llama_State_Save_File / Llama_State_Load_File APIs
--  - Cache files stored in cache/kv/ with hash-based naming
--  - LRU eviction keeps at most Max_Cache_Files entries
--  - Auto-save: After each Generate call, cache is saved and cleared
--  - Auto-load: On startup, most recent cache is loaded if available
--  - Shutdown: On SIGINT/SIGTERM, all active caches are saved before exit
--
--  INTEGRATION WITH ADELAIDE:
--  - model_manager.adb: Generate procedure calls Auto_Save after generation
--  - adelaide_server.adb: Shutdown handler calls Save_All_On_Shutdown
--  - Startup: Auto_Load called after model initialization
--  ============================================================================

with Interfaces.C;
with System;
with Ada.Calendar;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Llama_Interface;

package KV_Cache_Manager is

   --  Maximum number of cache files to keep on disk
   --  WHY 10: Balances storage usage with cache hit rate.
   --  Each cache file is ~100-500MB depending on context size.
   --  10 files = ~1-5GB storage, reasonable for most use cases.
   Max_Cache_Files : constant := 10;

   --  Cache directory path (relative to Adelaide_Lite/)
   --  WHY relative: Works regardless of CWD at runtime.
   Cache_Dir       : constant String := "cache/kv/";

   --  ============================================================================
   --  CORE API
   --  ============================================================================

   --  Save KV cache to SSD file
   --  WHY: Persists inference state across server restarts.
   --  PARAMS:
   --    Context: llama.cpp context (contains KV cache state)
   --    Tokens: pointer to token array (used to generate cache key)
   --    N_Tokens: number of tokens in the array
   --    Success: True if save completed successfully
   --  SIDE EFFECTS: Creates cache directory if needed, evicts old files
   procedure Save_To_SSD
     (Context    : Llama_Interface.Llama_Context;
      Tokens     : System.Address;
      N_Tokens   : Interfaces.C.size_t;
      Success    : out Boolean);

   --  Load KV cache from SSD file
   --  WHY: Restores inference state without recomputing from scratch.
   --  PARAMS:
   --    Context: llama.cpp context (will be populated with loaded state)
   --    File_Path: full path to cache file (e.g., "cache/kv/12345.bin")
   --    Tokens: output pointer to token array (caller must manage memory)
   --    N_Tokens: output number of tokens loaded
   --  RETURNS: True if load completed successfully
   --  NOTE: Caller is responsible for freeing Tokens memory
   function Load_From_SSD
     (Context    : Llama_Interface.Llama_Context;
      File_Path  : String;
      Tokens     : out System.Address;
      N_Tokens   : out Interfaces.C.size_t) return Boolean;

   --  ============================================================================
   --  AUTO-SAVE/LOAD (for minimal RAM footprint)
   --  ============================================================================

   --  Auto-save after generation (called by Generate procedure)
   --  WHY: Ensures cache is persisted immediately after use.
   --  BEHAVIOR:
   --    1. Save KV cache to SSD
   --    2. Clear KV cache from RAM (Llama_Memory_Clear)
   --  RESULT: RAM only holds current process, disk has persistent cache
   --  Called from: model_manager.adb Generate procedure (after token loop)
   procedure Auto_Save
     (Context    : Llama_Interface.Llama_Context;
      Tokens     : System.Address;
      N_Tokens   : Interfaces.C.size_t);

   --  Auto-load on startup (loads most recent cache from disk)
   --  WHY: Provides fastest response for repeated/similar prompts.
   --  BEHAVIOR:
   --    1. Search cache/kv/ for .bin files
   --    2. Load the first (most recent) cache found
   --    3. Populate context with loaded KV state
   --  RETURNS: True if cache was loaded successfully
   --  Called from: model_manager.adb Generate procedure (after model load)
   function Auto_Load
     (Context    : Llama_Interface.Llama_Context;
      Tokens     : out System.Address;
      N_Tokens   : out Interfaces.C.size_t) return Boolean;

   --  ============================================================================
   --  SHUTDOWN SUPPORT
   --  ============================================================================

   --  Save all active caches (called on shutdown)
   --  WHY: Safety net to ensure no cache data is lost on clean shutdown.
   --  NOTE: Most saves happen via Auto_Save during normal operation.
   --  This procedure handles edge cases (e.g., in-flight requests).
   --  Called from: adelaide_server.adb shutdown handler
   procedure Save_All_On_Shutdown;

   --  ============================================================================
   --  INITIALIZATION
   --  ============================================================================

   --  Initialize cache directory
   --  WHY: Ensures cache/kv/ exists before any save operations.
   --  Called from: model_manager.adb Initialize procedure
   procedure Initialize;

   --  ============================================================================
   --  UTILITY FUNCTIONS
   --  ============================================================================

   --  Check if SSD cache exists for a given prompt hash
   --  WHY: Allows checking cache availability without loading.
   --  PARAMS:
   --    Prompt_Hash: hash string generated by Hash_Tokens
   --  RETURNS: True if cache file exists
   function Has_Cached_Prefix (Prompt_Hash : String) return Boolean;

   --  Generate hash from token array
   --  WHY: Creates unique cache key for each prompt/token sequence.
   --  PARAMS:
   --    Tokens: pointer to token array
   --    N_Tokens: number of tokens
   --    Max_Hash: maximum tokens to hash (default 128)
   --  RETURNS: hash string (e.g., "12345")
   --  NOTE: Currently uses simplified hash (token count only).
   --  TODO: Implement proper SHA-256 for production use.
   function Hash_Tokens
     (Tokens    : System.Address;
      N_Tokens  : Interfaces.C.size_t;
      Max_Hash  : Positive := 128) return String;

   --  Evict oldest cache files
   --  WHY: Prevents disk filling up with stale cache files.
   --  BEHAVIOR: Logs warning when count exceeds Max_Cache_Files.
   --  TODO: Implement LRU eviction by modification time.
   procedure Evict_Old_Cache;

end KV_Cache_Manager;
