pragma SPARK_Mode (Off);
--  Speculative Cache — Cached Response Speculation
--
--  Stores (predicted_follow_up_query → pre_computed_answer) pairs.
--  The Hybrid_Generate pipeline checks this cache BEFORE invoking
--  the 9B target model.  If the user's prompt matches a cached
--  predicted query (via Jaccard word-overlap > 0.4), the cached
--  answer is returned immediately without any LLM inference.
--
--  Max 5 entries with LRU eviction.
--  Thread-safe via Ada protected object.

with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Ada.Calendar;

package Speculative_Cache is

   Max_Entries : constant := 5;

   type Cache_Entry is record
      Predicted_Query : Unbounded_String := Null_Unbounded_String;
      Cached_Answer   : Unbounded_String := Null_Unbounded_String;
      Timestamp       : Ada.Calendar.Time;
      Valid           : Boolean := False;
   end record;

   type Entry_Array is array (1 .. Max_Entries) of Cache_Entry;

   protected type Cache is
      --  Store a (predicted_query → answer) pair.
      --  Evicts the oldest entry if full.
      procedure Store (Predicted_Query : String; Answer : String);

      --  Look up a user prompt in the cache.
      --  Returns the cached answer if a match is found, "" otherwise.
      function Lookup (Query : String) return String;

      --  Invalidate all entries (e.g. on context switch or reload).
      procedure Invalidate;

      --  Number of valid entries currently in cache.
      function Count return Natural;
   private
      Entries : Entry_Array :=
        [others => (Predicted_Query => Null_Unbounded_String,
                    Cached_Answer   => Null_Unbounded_String,
                    Timestamp       => Ada.Calendar.Clock,
                    Valid           => False)];
   end Cache;

   --  Shared cache instance used by:
   --    - Proactive_Cache_Task  (ELP0 populator in Knowledge_Manager)
   --    - Hybrid_Generate      (ELP1 lookup in Model_Manager)
   Proactive_Cache : Cache;

end Speculative_Cache;
