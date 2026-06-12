pragma SPARK_Mode (Off);
--  ELP Queue — Unified serial queue for all priority levels
--
--  Architecture: "Volatus Damarae"
--  A departure from the Python-centric orchestration of Project Zephyrine.
--  This Ada-native queue manages four Elevated Level Privilege priorities:
--    ELP0: Deep cognitive reasoning (background indexing) — preemptible
--    ELP1: High-priority real-time inference (user-facing generation)
--    ELP2: Stella-Icarus Deterministic API response
--    ELP3: Deterministic light task — 1ms fixed nanosecond WCET
--
--  Serial processing prevents heap corruption from concurrent llama.cpp
--  FFI calls on shared contexts.
--  Capacity: 2^63 (effectively unlimited).
--  Priority: ELP3 > ELP2 > ELP1 > ELP0
--
--  Every 5 seconds, a monitor task prints queue depth as 0%-100%.

with Model_Types; use Model_Types;
with Interfaces; use Interfaces;

package ELP_Queue is

   procedure Initialize;

   --  Enqueue a request at the given priority level.
   --  Blocks if queue is full (practically never with 2^63 capacity).
   procedure Enqueue
     (Level  : ELP_Level;
      Kind   : Model_Type;
      Source : String := "Unknown");

   --  Dequeue the highest-priority request.
   --  Blocks if queue is empty.
   procedure Dequeue
     (Level : out ELP_Level;
      Kind  : out Model_Type);

   --  Explicit level-aware dequeue for Model_Manager
   procedure Dequeue_Level (Level : ELP_Level);

   --  Query current queue depth.
   function Depth return Long_Long_Integer;

   --  Query capacity (2^63).
   function Capacity return Unsigned_64;

   --  Query utilization as percentage (0.0 .. 100.0).
   function Utilization return Long_Long_Float;

end ELP_Queue;
