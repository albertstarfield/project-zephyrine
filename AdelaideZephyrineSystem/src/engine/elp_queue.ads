pragma SPARK_Mode (Off);
-- thread: Priority queue requires protected type
 --  ELP Queue — Priority-based task queue for model execution requests
 --
 --  Architecture: "Volatus Damarae"
 --  A departure from the Python-centric orchestration of Project Zephyrine.
 --  This Ada-native queue manages four Elevated Level Privilege (ELP) priorities:
 --    
 --  Priority Levels (Highest to Lowest):
 --    ELP1: User-facing tasks (chat, API responses) — HIGH priority
 --         - Always preempts ELP0 tasks
 --         - Uses priority gate acquisition to ensure responsiveness
 --    ELP0: Background tasks (indexing, pre-warming) — NORMAL priority
 --         - Can be preempted by ELP1 tasks
 --         - Runs only when no user tasks are pending
 --    ELP2: Stella-Icarus Deterministic API response — LOW priority
 --    ELP3: Deterministic light task — LOWEST priority (1ms fixed WCET)
 --
 --  Priority Rules:
 --    1. ELP1 tasks always take precedence over ELP0 tasks
 --    2. When an ELP1 request arrives, any pending ELP0 tasks are blocked
 --    3. Background tasks (ELP0) can only run when:
 --         a) No user tasks (ELP1) are pending
 --         b) No user tasks (ELP1) are active
 --         c) Model is not busy
 --
 --  Implementation Notes:
 --    - Serial processing prevents heap corruption from concurrent llama.cpp FFI calls
 --    - Capacity: 2^63 (effectively unlimited)
 --    - Priority is enforced through both the queue and the Priority_Model_Gate
 --    - Monitor task reports queue state every 5 seconds
 --
 --  Fixed Issues:
 --    - ELP0 tasks could incorrectly acquire priority over pending ELP1 tasks
 --    - Priority escalation now works properly when user requests arrive
 --    - Background tasks properly yield to user-facing work

with Model_Types; use Model_Types;
with Interfaces; use Interfaces;

package ELP_Queue is

   --  Initialize the ELP queue and start the monitor task.
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
