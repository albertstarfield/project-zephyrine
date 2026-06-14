pragma SPARK_Mode (Off);
with Ada.Text_IO; use Ada.Text_IO;
with AnsiAda;
with Ada.Real_Time; use Ada.Real_Time;
with Shutdown_Manager;

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

   procedure Enqueue
     (Level  : ELP_Level;
      Kind   : Model_Type;
      Source : String := "Unknown")
   is
      pragma Unreferenced (Kind);
   begin
      Load_State.Increment (Level, Source);
   end Enqueue;

   procedure Dequeue (Level : out ELP_Level; Kind : out Model_Type) is
   begin
      --  Defaults to keep compiler happy
      Level := ELP0;
      Kind  := Qwen_Embedding;
      Load_State.Decrement (Level);
   end Dequeue;

   --  Explicit level-aware dequeue for Model_Manager
   procedure Dequeue_Level (Level : ELP_Level) is
   begin
      Load_State.Decrement (Level);
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
      Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Init-V]" &
                AnsiAda.Reset & " ELP_Queue.Initialize ENTERED.");
      if Initialized then
         --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
         Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Init-V]" &
                   AnsiAda.Reset &
                   " ELP_Queue.Initialize: ALREADY INITIALIZED, skipping.");
         return;
      end if;
      Initialized := True;
      if not Monitor_Task'Terminated then
         --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
         Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Init-V]" &
                   AnsiAda.Reset &
                   " ELP_Queue.Initialize: Monitor_Task not terminated, " &
                   "calling Start...");
         Monitor_Task.Start;
         --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
         Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Init-V]" &
                   AnsiAda.Reset &
                   " ELP_Queue.Initialize: Monitor_Task.Start returned.");
      else
         --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
         Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Init-V]" &
                   AnsiAda.Reset &
                   " ELP_Queue.Initialize: Monitor_Task already terminated, " &
                   "skipping Start.");
      end if;
      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
      Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Init-V]" &
                AnsiAda.Reset & " ELP_Queue.Initialize COMPLETE.");
   end Initialize;

end ELP_Queue;
