pragma SPARK_Mode (Off);
with Ada.Text_IO; use Ada.Text_IO;
with AnsiAda;
with Ada.Real_Time; use Ada.Real_Time;
with Shutdown_Manager;
with Model_Manager; use Model_Manager;

package body ELP_Queue is

   --  ========================================================================
   --  SIMPLIFIED ELP QUEUE (GRANULAR TRACKING)
   --  ========================================================================
   --  [VITAL-DO-NOT-REMOVE] Mandated by user for backend visibility.
   --  REASONING:
   --  We need to see exactly how many tasks of each priority level are
   --  pending to diagnose scheduling bottlenecks.

   type Level_Counts is array (ELP_Level) of Long_Long_Integer;

   protected Load_State is
      procedure Increment (Level : ELP_Level; Source : String);
      procedure Decrement (Level : ELP_Level);
      function Get_Counts return Level_Counts;
      function Get_Total return Long_Long_Integer;
      function Get_Last_Source return String;
   private
      Counts      : Level_Counts := [others => 0];
      Total       : Long_Long_Integer := 0;
      Last_Source : String (1 .. 32)  := [others => ' '];
      Source_Len  : Natural := 0;
   end Load_State;

   protected body Load_State is
      procedure Increment (Level : ELP_Level; Source : String) is
      begin
         Counts (Level) := Counts (Level) + 1;
         Total := Total + 1;
         Source_Len := Natural'Min (Source'Length, 32);
         Last_Source (1 .. Source_Len) :=
           Source (Source'First .. Source'First + Source_Len - 1);

         --  [VITAL-DO-NOT-REMOVE] Mandated by user.
         Put_Line (AnsiAda.Foreground (AnsiAda.Grey) & "[ELP-Queue] ENQUEUE: " &
                   Source & " (Level: " & Level'Img & ")" & AnsiAda.Reset);
      end Increment;

      procedure Decrement (Level : ELP_Level) is
      begin
         if Counts (Level) > 0 then
            Counts (Level) := Counts (Level) - 1;
            Total := Total - 1;
         end if;

         --  [VITAL-DO-NOT-REMOVE] Mandated by user.
         Put_Line (AnsiAda.Foreground (AnsiAda.Grey) & "[ELP-Queue] DEQUEUE: " &
                   Level'Img & " (Remaining Total:" & Total'Img & ")" &
                   AnsiAda.Reset);
      end Decrement;

      function Get_Counts return Level_Counts is (Counts);
      function Get_Total return Long_Long_Integer is (Total);
      function Get_Last_Source return String is (Last_Source (1 .. Source_Len));
   end Load_State;

    --  [OPTIMIZATION-M03] PREDICTIVE PRE-WARMING
    --  ======================================================================
    --  When an ELP1 request is enqueued, immediately start loading the
    --  required model in the background, BEFORE the request is acquired.
    --  This eliminates the GAP Zone wait time for the first request.
    --
    --  BENEFITS:
    --  - ELP1 requests get instant response (model already loaded)
    --  - Reduces perceived latency from user perspective
    --  - Better GPU utilization (no idle GAP Zone)
    --
    --  TRADEOFFS:
    --  - May load models that are never used (if request is cancelled)
    --  - Higher memory usage during pre-warming phase
    --  - Only beneficial for ELP1 (user-facing) requests
    --  ======================================================================
    procedure Enqueue
      (Level  : ELP_Level;
       Kind   : Model_Type;
       Source : String := "Unknown")
    is
    begin
       Load_State.Increment (Level, Source);
       
       --  Only apply predictive pre-warming for ELP1 (user-facing) requests
        if Level = ELP1 then
           --  Check if model is already loaded
           declare
               State : constant Model_Record := Model_Manager.Get_Model_State (Kind);
           begin
               if not State.Loaded and then not State.Warm_Cached
               then
                  Put_Line
                     (AnsiAda.Foreground (AnsiAda.Light_Blue)
                      & "[PreWarm] "
                      & AnsiAda.Reset
                      & "ELP1 request enqueued for "
                      & Model_Type'Image (Kind)
                      & ". Starting predictive pre-warming...");
               
                  --  Start loading the model in background
                  --  Note: This is non-blocking - the actual load happens asynchronously
                  declare
                      Success : Boolean;
                      pragma Unreferenced (Success);
                  begin
                      --  Load with minimal context to speed up pre-warming
                      --  The actual request will resize if needed
                      Model_Manager.Load_Model (Kind, Success, 4096, ELP1);
                  exception
                      when others =>
                          Put_Line
                             (AnsiAda.Foreground (AnsiAda.Yellow)
                              & "[PreWarm-WARN] "
                              & AnsiAda.Reset
                              & "Predictive pre-warming failed for "
                              & Model_Type'Image (Kind)
                              & ". Will load on-demand.");
                  end;
               else
                  Put_Line
                     (AnsiAda.Foreground (AnsiAda.Light_Blue)
                      & "[PreWarm-SKIP] "
                      & AnsiAda.Reset
                      & Model_Type'Image (Kind)
                      & " already "
                      & (if Model_Manager.Models (Kind).Loaded then "loaded"
                       elsif Model_Manager.Models (Kind).Warm_Cached then "warm-cached"
                       else "unknown")
                      & ". No pre-warming needed.");
               end if;
           end;
        end if;
     end Enqueue;

    --  Dequeue: Get the next task from the queue according to priority rules.
    --  
    --  Priority rules:
    --    1. ELP1 tasks (user-facing) have highest priority
    --    2. ELP0 tasks (background) have medium priority
    --    3. ELP2 tasks have low priority
    --    4. ELP3 tasks have lowest priority
    --  
    --  This procedure determines which priority level to serve next based on current queue state.
    --  The actual task processing is handled by the Model_Manager based on this priority.
    procedure Dequeue (Level : out ELP_Level; Kind : out Model_Type) is
       C : constant Level_Counts := Load_State.Get_Counts;
    begin
       --  Default values to satisfy compiler (should never be used)
       Level := ELP0;
       Kind  := Qwen_Embedding;
 
       --  Priority-based task selection:
       --  Always serve highest-priority tasks first to ensure responsive user experience.
       --  This matches the priority rules enforced in Priority_Model_Gate.
       if C(ELP1) > 0 then
           Level := ELP1;
       elsif C(ELP0) > 0 then
           Level := ELP0;
       elsif C(ELP2) > 0 then
           Level := ELP2;
       elsif C(ELP3) > 0 then
           Level := ELP3;
       end if;
 
       --  Update queue state to reflect that we've serviced a task at this priority level
       Load_State.Decrement (Level);
    end Dequeue;

    --  Dequeue_Level: Remove a specific task from the queue by priority level.
    --  
    --  This is called by Model_Manager when:
    --    - A task completes successfully
    --    - A task fails and needs to be aborted
    --    - An ELP0 task is preempted by an ELP1 task (priority escalation)
    --  
    --  SAFETY NOTE: The check for positive count prevents negative values which could
    --  cause incorrect priority handling. This is defensive programming against race conditions.
    procedure Dequeue_Level (Level : ELP_Level) is
    begin
       --  Safety check: only decrement if we actually have tasks at this priority
       --  This handles edge cases where multiple threads might try to dequeue the same task
       if Load_State.Get_Counts(Level) > 0 then
          Load_State.Decrement (Level);
       end if;
    end Dequeue_Level;

   function Depth return Long_Long_Integer is (Load_State.Get_Total);
   --  (2^64)/2 = 2^63 — fits in Unsigned_64 (max 2^64 - 1).
   function Capacity return Unsigned_64 is ((2**64) / 2);

   function Utilization return Long_Long_Float is
      D : constant Long_Long_Integer := Depth;
      C : constant Unsigned_64 := Capacity;
   begin
      if C = 0 then
         return 0.0;
      end if;
      return Long_Long_Float (D) / Long_Long_Float (C) * 100.0;
   end Utilization;

   task Monitor_Task is
      entry Start;
   end Monitor_Task;

   task body Monitor_Task is
      Interval   : constant Time_Span := Seconds (5);
      Next_Check : Time;
   begin
      accept Start;
      loop
         exit when Shutdown_Manager.Shutdown_Status.Requested;
         Next_Check := Clock + Interval;
         declare
            C : constant Level_Counts := Load_State.Get_Counts;
            T : constant Long_Long_Integer := Load_State.Get_Total;
            S : constant String := Load_State.Get_Last_Source;
         begin
            Put_Line (AnsiAda.Foreground (AnsiAda.Grey) &
                      "[ELP-Queue] Total:" & T'Img &
                      " | ELP0:" & C (ELP0)'Img &
                      " | ELP1:" & C (ELP1)'Img &
                      " | ELP2:" & C (ELP2)'Img &
                      " | ELP3:" & C (ELP3)'Img &
                      " | Source: " & S &
                      AnsiAda.Reset);
         end;
         delay until Next_Check;
      end loop;
   end Monitor_Task;

   Initialized : Boolean := False;

   procedure Initialize is
   begin
      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
      Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Init-V]" &
                AnsiAda.Reset & " ELP_Queue.Initialize ENTERED.");
      if Initialized then
         --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
         Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Init-V]" &
                   AnsiAda.Reset &
                   " ELP_Queue.Initialize: ALREADY INITIALIZED, skipping.");
         return;
      end if;
      Initialized := True;
      if not Monitor_Task'Terminated then
         --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
         Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Init-V]" &
                   AnsiAda.Reset &
                   " ELP_Queue.Initialize: Monitor_Task not terminated, " &
                   "calling Start...");
         Monitor_Task.Start;
         --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
         Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Init-V]" &
                   AnsiAda.Reset &
                   " ELP_Queue.Initialize: Monitor_Task.Start returned.");
      else
         --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
         Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Init-V]" &
                   AnsiAda.Reset &
                   " ELP_Queue.Initialize: Monitor_Task already terminated, " &
                   "skipping Start.");
      end if;
      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
      Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Init-V]" &
                AnsiAda.Reset & " ELP_Queue.Initialize COMPLETE.");
   end Initialize;

--  ======================================================================
--  ELP PRIORITY BEHAVIOR (2026-06-26)
--  
--  This queue works with Priority_Model_Gate to enforce proper priority:
--    1. ELP1 tasks (user-facing) always preempt ELP0 tasks (background)
--    2. Background tasks only run when no user tasks are pending or active
--    3. Priority is enforced through both the queue and the gate
--  
--  The queue tracks task counts while the gate manages actual resource access.
--  For details on the priority fix, see model_manager.adb comments.
--  ======================================================================
end ELP_Queue;
