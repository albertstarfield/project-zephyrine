pragma SPARK_Mode (Off);
--  ELP Queue — Unified serial queue for all priority levels
--
--  Capacity: 2^63 (effectively unlimited)
--  Parallelism: 1 (serial processing — prevents heap corruption
--    from concurrent llama.cpp FFI calls on shared contexts)
--  Priority: ELP3 > ELP2 > ELP1 > ELP0
--
--  Every 5 seconds, a monitor task prints queue depth as 0%-100%.

with Model_Types; use Model_Types;
with Ada.Real_Time; use Ada.Real_Time;

package ELP_Queue is

   procedure Initialize;

   --  Enqueue a request at the given priority level.
   --  Blocks if queue is full (practically never with 2^63 capacity).
   procedure Enqueue (Level : ELP_Level; Kind : Model_Type);

   --  Dequeue the highest-priority request.
   --  Blocks if queue is empty.
   procedure Dequeue
     (Level : out ELP_Level;
      Kind  : out Model_Type);

   --  Query current queue depth.
   function Depth return Long_Long_Integer;

   --  Query capacity (2^63).
   function Capacity return Long_Long_Integer;

   --  Query utilization as percentage (0.0 .. 100.0).
   function Utilization return Long_Long_Float;

end ELP_Queue;
