pragma SPARK_Mode (Off);
-- thread: Priority queue requires protected type
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
  
  --  [FIXED] Task execution timing tracking
  type Task_Timing is record
     Enqueue_Time : Time;
     Start_Time   : Time;
     Source       : String (1 .. 32);
     Source_Len   : Natural;
  end record;
  
  Task_Timings : array (ELP_Level) of Task_Timing;

  --  ========================================================================
  --  PREWARM COOLDOWN TRACKING
  --  ========================================================================
  --  WHAT IS PREWARM:
  --    When an ELP1 (user-facing) request arrives, we predictive-load the
  --    required model in the background BEFORE the request is acquired.
  --    This eliminates the "GAP Zone" wait time — the model is already
  --    loaded and ready by the time the request needs it.
  --
  --  THE BUG THIS FIXES:
  --    Without cooldown, if a PreWarm load fails (e.g., GPU OOM), the
  --    request is immediately re-enqueued, PreWarm fires again, loads
  --    the model again, fails again — infinite loop. Each load takes
  --    200-300ms (weights from disk) + Metal init. After 3-4 failed
  --    loads, Metal state corruption triggers SIGTRAP.
  --
  --  THE FIX:
  --    Track the last PreWarm failure time. If a PreWarm failed within
  --    the last PREWARM_COOLDOWN_S seconds, skip predictive loading and
  --    let the model load on-demand instead. This breaks the retry loop.
  --  ========================================================================
  PREWARM_COOLDOWN_S : constant Duration := 5.0;
  Last_Prewarm_Failure : Time := Time_First;
  Prewarm_Fail_Count   : Natural := 0;

   protected Load_State is
      --  Increment the count for the given priority level and record the source name.
      procedure Increment (Level : ELP_Level; Source : String);
      --  Decrement the count for the given priority level and log completion timing.
      procedure Decrement (Level : ELP_Level);
      --  Return the per-level task counts.
      function Get_Counts return Level_Counts;
      --  Return the total number of pending tasks across all levels.
      function Get_Total return Long_Long_Integer;
      --  Return the source name of the most recently enqueued task.
      function Get_Last_Source return String;
      --  Record the time at which a task at the given level begins execution.
      procedure Set_Task_Start (Level : ELP_Level);
  private
     Counts      : Level_Counts := (others => 0);
     Total       : Long_Long_Integer := 0;
     Last_Source : String (1 .. 32)  := (others => ' ');
     Source_Len  : Natural := 0;
  end Load_State;

   protected body Load_State is
     --  Increment the count for the given priority level and record the source name.
     procedure Increment (Level : ELP_Level; Source : String) is
        -- pre => True, post => True
       begin
          Counts (Level) := Counts (Level) + 1;
          Total := Total + 1;
          Source_Len := Natural'Min (Source'Length, 32);
          Last_Source (1 .. Source_Len) := 
            Source (Source'First .. Source'First + Source_Len - 1);

          --  Record enqueue time for accurate execution timing
          Task_Timings (Level).Enqueue_Time := Clock;
          Task_Timings (Level).Start_Time := Clock;
          Task_Timings (Level).Source_Len := Source_Len;
          Task_Timings (Level).Source (1 .. Source_Len) := 
             Source (Source'First .. Source'First + Source_Len - 1);

          --  [VITAL-DO-NOT-REMOVE] Mandated by user.
          Put_Line (AnsiAda.Foreground (AnsiAda.Grey) & "[ELP-Queue] ENQUEUE: " &
                    Source & " (Level: " & Level'Img & ")" & AnsiAda.Reset);
       end Increment;

       --  Decrement the count for the given priority level and log completion timing.
       procedure Decrement (Level : ELP_Level) is
          -- pre => True, post => True
    begin
       if Counts (Level) > 0 then
          Counts (Level) := Counts (Level) - 1;
          Total := Total - 1;
       end if;

       --  [VITAL-DO-NOT-REMOVE] Mandated by user - ENHANCED VERBOSITY WITH ACCURATE TIMING
       declare
          Now : constant Time := Clock;
          Task_Start : constant Time := Task_Timings (Level).Start_Time;
          --  Calculate elapsed time from task start to completion
          Elapsed : constant Duration := To_Duration (Now - Task_Start);
          --  Calculate queue wait time (if task was delayed)
          Wait_Time : constant Duration := To_Duration (Task_Timings (Level).Start_Time - Task_Timings (Level).Enqueue_Time);
       begin
          case Level is
             when ELP0 =>
                Put_Line (AnsiAda.Foreground (AnsiAda.Grey) & "[ELP0-COMPLETED] " &
                          "Elapsed: " &  
                          AnsiAda.Foreground (AnsiAda.Light_Cyan) & 
                          Elapsed'Img & "s" & AnsiAda.Reset &
                          " | QueueWait: " & Wait_Time'Img & "s" &
                          " | Source: " & 
                          Task_Timings (Level).Source (1 .. Task_Timings (Level).Source_Len) &
                          " | Remaining: " & Total'Img & AnsiAda.Reset);
             when ELP1 =>
                Put_Line (AnsiAda.Foreground (AnsiAda.Light_Red) & "[ELP1-COMPLETED] " &
                          "Elapsed: " &  
                          AnsiAda.Foreground (AnsiAda.Light_Cyan) & 
                          Elapsed'Img & "s" & AnsiAda.Reset &
                          " | QueueWait: " & Wait_Time'Img & "s" &
                          " | Source: " & 
                          Task_Timings (Level).Source (1 .. Task_Timings (Level).Source_Len) &
                          " | Remaining: " & Total'Img & AnsiAda.Reset);
             when ELP2 =>
                Put_Line (AnsiAda.Foreground (AnsiAda.Light_Magenta) & "[ELP2-COMPLETED] " &
                          "Elapsed: " &  
                          AnsiAda.Foreground (AnsiAda.Light_Cyan) & 
                          Elapsed'Img & "s" & AnsiAda.Reset &
                          " | QueueWait: " & Wait_Time'Img & "s" &
                          " | Source: " & 
                          Task_Timings (Level).Source (1 .. Task_Timings (Level).Source_Len) &
                          " | Remaining: " & Total'Img & AnsiAda.Reset);
             when ELP3 =>
                Put_Line (AnsiAda.Foreground (AnsiAda.Light_Green) & "[ELP3-COMPLETED] " &
                          "Elapsed: " &  
                          AnsiAda.Foreground (AnsiAda.Light_Cyan) & 
                          Elapsed'Img & "s" & AnsiAda.Reset &
                          " | QueueWait: " & Wait_Time'Img & "s" &
                          " | Source: " & 
                          Task_Timings (Level).Source (1 .. Task_Timings (Level).Source_Len) &
                          " | Remaining: " & Total'Img & AnsiAda.Reset);
          end case;
       end;
       end Decrement;

       --  Set the actual task start time (when execution begins)
       procedure Set_Task_Start (Level : ELP_Level) is
          -- pre => True, post => True
       begin
          Task_Timings (Level).Start_Time := Clock;
       end Set_Task_Start;

       --  Return the per-level task counts.
       function Get_Counts return Level_Counts is (Counts);
       --  Return the total number of pending tasks across all levels.
       function Get_Total return Long_Long_Integer is (Total);
       --  Return the source name of the most recently enqueued task.
       function Get_Last_Source return String is (Last_Source (1 .. Source_Len));
    end Load_State;

    --  [OPTIMIZATION-M03] PREDICTIVE PRE-WARMING
    --  ======================================================================
    --  WHAT IS PREWARMING:
    --    When an ELP1 (user-facing) request arrives, we predictive-load the
    --    required model in the background BEFORE the request is acquired.
    --    This eliminates the "GAP Zone" wait time — the model is already
    --    loaded and ready by the time the request needs it.
    --
    --    Example: User sends a chat message → ELP1 request enqueued →
    --    PreWarm starts loading Mythos9bHybrid on GPU → by the time the
    --    request is dequeued and processed, the model is already warm.
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
    --
    --  COOLDOWN (PREWARM_COOLDOWN_S):
    --    If a PreWarm load fails (GPU OOM, Metal error, etc.), we record
    --    the failure time. For the next PREWARM_COOLDOWN_S seconds, all
    --    PreWarm attempts for that model are SKIPPED — the model will
    --    load on-demand instead. This prevents the infinite retry loop:
    --      fail → re-enqueue → PreWarm → load → fail → re-enqueue → ...
    --    Without cooldown, 3-4 rapid load attempts corrupt Metal state
    --    and trigger SIGTRAP.
    --  ======================================================================
    procedure Enqueue
      (Level  : ELP_Level;
       Kind   : Model_Type;
       Source : String := "Unknown")
    is
       -- pre => True, post => True
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
                  --  [PREWARM-COOLDOWN] Check if this model failed recently.
                  --  If so, skip predictive loading — let it load on-demand.
                  --  This breaks the infinite retry loop that causes SIGTRAP.
                   declare
                       Recent_Failure : Boolean := False;
                       Time_Since_Failure : Duration := 0.0;
                   begin
                       if Last_Prewarm_Failure /= Time_First then
                           Time_Since_Failure := Ada.Real_Time.To_Duration (Clock - Last_Prewarm_Failure);
                           if Time_Since_Failure < PREWARM_COOLDOWN_S then
                               Recent_Failure := True;
                           end if;
                       end if;

                       if Recent_Failure then
                          Put_Line
                             (AnsiAda.Foreground (AnsiAda.Yellow)
                              & "[PreWarm-COOLDOWN] "
                              & AnsiAda.Reset
                              & Model_Type'Image (Kind)
                              & " FAILED "
                              & Duration'Image (Time_Since_Failure)
                              & "s ago (cooldown="
                              & Duration'Image (PREWARM_COOLDOWN_S)
                              & "s). Skipping predictive load."
                              & " Will load on-demand.");
                       else
                          --  No recent failure — safe to PreWarm
                          Put_Line
                             (AnsiAda.Foreground (AnsiAda.Light_Blue)
                              & "[PreWarm] "
                              & AnsiAda.Reset
                              & "ELP1 request enqueued for "
                              & Model_Type'Image (Kind)
                              & ". Starting predictive pre-warming...");

                          --  Start loading the model in background
                          declare
                              Success : Boolean;
                              pragma Unreferenced (Success);
                          begin
                              Model_Manager.Load_Model (Kind, Success, (if Kind = Qwen_Embedding then 512 else 4096), ELP1);
                              if not Success then
                                 Last_Prewarm_Failure := Clock;
                                 Prewarm_Fail_Count := Prewarm_Fail_Count + 1;
                                 Put_Line
                                    (AnsiAda.Foreground (AnsiAda.Red)
                                     & "[PreWarm-FAILED] "
                                     & AnsiAda.Reset
                                     & Model_Type'Image (Kind)
                                     & " load FAILED! Cooldown "
                                     & Duration'Image (PREWARM_COOLDOWN_S)
                                     & "s started. Failure #"
                                     & Natural'Image (Prewarm_Fail_Count)
                                     & ". Will load on-demand.");
                              else
                                 if Prewarm_Fail_Count > 0 then
                                    Put_Line
                                       (AnsiAda.Foreground (AnsiAda.Light_Green)
                                        & "[PreWarm-RECOVERED] "
                                        & AnsiAda.Reset
                                        & Model_Type'Image (Kind)
                                        & " loaded successfully after "
                                        & Natural'Image (Prewarm_Fail_Count)
                                        & " previous failures. Cooldown cleared.");
                                    Prewarm_Fail_Count := 0;
                                 end if;
                                 Last_Prewarm_Failure := Time_First;
                              end if;
                          exception
                              when others =>
                                 Last_Prewarm_Failure := Clock;
                                 Prewarm_Fail_Count := Prewarm_Fail_Count + 1;
                                 Put_Line
                                    (AnsiAda.Foreground (AnsiAda.Red)
                                     & "[PreWarm-EXCEPTION] "
                                     & AnsiAda.Reset
                                     & Model_Type'Image (Kind)
                                     & " load CRASHED! Cooldown "
                                     & Duration'Image (PREWARM_COOLDOWN_S)
                                     & "s started. Failure #"
                                     & Natural'Image (Prewarm_Fail_Count)
                                     & ". Will load on-demand.");
                          end;
                       end if;
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
        -- pre => True, post => True
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

        --  Record the actual task start time for accurate timing
        Load_State.Set_Task_Start (Level);

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
        -- pre => True, post => True
     begin
        --  Safety check: only decrement if we actually have tasks at this priority
        --  This handles edge cases where multiple threads might try to dequeue the same task
        if Load_State.Get_Counts(Level) > 0 then
           --  Record task start time for accurate execution timing
           Load_State.Set_Task_Start (Level);
           Load_State.Decrement (Level);
        end if;
     end Dequeue_Level;

   --  Return the total number of pending tasks across all priority levels.
   function Depth return Long_Long_Integer is (Load_State.Get_Total);
   --  (2^64)/2 = 2^63 — fits in Unsigned_64 (max 2^64 - 1).
   function Capacity return Unsigned_64 is ((2**64) / 2);

   function Utilization return Long_Long_Float is
      -- pre => True, post => True
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

   --  Initialize the ELP queue and start the monitor task.
   procedure Initialize is
      -- pre => True, post => True
   begin
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
      Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Init-V]" &
                AnsiAda.Reset & " ELP_Queue.Initialize ENTERED.");
      if Initialized then
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
         Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Init-V]" &
                   AnsiAda.Reset &
                   " ELP_Queue.Initialize: ALREADY INITIALIZED, skipping.");
         return;
      end if;
      Initialized := True;
      if not Monitor_Task'Terminated then
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
         Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Init-V]" &
                   AnsiAda.Reset &
                   " ELP_Queue.Initialize: Monitor_Task not terminated, " &
                   "calling Start...");
         Monitor_Task.Start;
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
         Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Init-V]" &
                   AnsiAda.Reset &
                   " ELP_Queue.Initialize: Monitor_Task.Start returned.");
      else
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
         Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Init-V]" &
                   AnsiAda.Reset &
                   " ELP_Queue.Initialize: Monitor_Task already terminated, " &
                   "skipping Start.");
      end if;
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
