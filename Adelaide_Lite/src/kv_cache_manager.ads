pragma SPARK_Mode (Off);
--  ============================================================================
--  KV Cache Manager — DATACENTER SPEED ON SLOW HARDWARE
--  ============================================================================
--  BLACKMAGIC TRICKS FOR MAXIMUM SPEED:
--
--  1. ASYNC SAVE: Fire-and-forget background task, never block on disk I/O
--  2. LAZY LOAD: Only load when Generate is called, not at startup
--  3. PRE-PATH CACHE: Cache the most recent file path to skip directory scan
--  4. WRITE-BUFFER: Collect multiple saves into single write (batching)
--  5. DIRECT I/O: Bypass OS cache for large files (if available)
--  6. MEMORY-MAP: Use mmap for reads (zero-copy)
--  7. BACKGROUND EVICTION: Evict old files in background, not on save
--  8. AGGRESSIVE PREFETCH: Prefetch next likely cache file
--
--  WHY: On slow machines (HDD, old Mac), every millisecond counts.
--  These tricks can give 10-100x speedup on cold cache hits.
--  ============================================================================

with Interfaces.C;
with System;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Llama_Interface;

package KV_Cache_Manager is

   Cache_Dir : constant String := "cache/kv/";

   --  ============================================================================
   --  ASYNC SAVE (fire-and-forget, non-blocking)
   --  ============================================================================
   --  WHY: Never block on disk I/O during request handling.
   --  Background task saves to disk while we return response instantly.

    procedure Save_To_SSD_Async
      (Context    : Llama_Interface.Llama_Context;
       Tokens     : System.Address;
       N_Tokens   : Interfaces.C.size_t;
       Model_ID   : String);

    --  ============================================================================
    --  LAZY LOAD (on-demand only)
    --  ============================================================================
    --  WHY: Don't load at startup - only load when Generate is called.
    --  Use pre-path cache to skip directory scan when possible.

    function Load_From_SSD_Lazy
      (Context    : Llama_Interface.Llama_Context;
       Tokens     : out System.Address;
       N_Tokens   : out Interfaces.C.size_t;
       Model_ID   : String) return Boolean;

    --  ============================================================================
    --  PRE-PATH CACHE (blackmagic trick #1)
    --  ============================================================================
    --  WHY: Directory scans are slow on HDD. Cache the last used path.
    --  On same prompt, skip the scan entirely.

    procedure Cache_Last_Path (Path : String);
    function Get_Cached_Path return String;
    function Has_Cached_Path (Model_ID : String) return Boolean;

   --  ============================================================================
   --  PREFETCH (blackmagic trick #2)
   --  ============================================================================
   --  WHY: Prefetch next likely cache file into OS page cache.
   --  By the time we need it, it's already in RAM.

   procedure Prefetch_Cache_File (Path : String);

   --  ============================================================================
   --  WAIT FOR SAVE (blocking, call before Unload_Model)
   --  ============================================================================
   --  WHY: The model must stay loaded until the async save completes.
   --  After Save_To_SSD_Async, call Wait_For_Save to block until the
   --  background task finishes writing to disk. Then it's safe to unload.

   procedure Wait_For_Save;

   --  ============================================================================
   --  UTILITY
   --  ============================================================================

   procedure Initialize;
   function Has_Cache_Files return Boolean;

   --  ============================================================================
   --  METRICS (logged every 10 seconds)
   --  ============================================================================
   --  WHY: Track cache performance for optimization.
   --  Logs: Total Prefill tokens, Cached Tokens, Cache Hit Percentage.

   procedure Record_Prefill (N_Tokens : Interfaces.C.size_t);
   procedure Record_Cache_Hit (N_Tokens : Interfaces.C.size_t);
   procedure Record_Cache_Miss;

end KV_Cache_Manager;
