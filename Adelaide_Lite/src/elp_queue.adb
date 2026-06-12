pragma SPARK_Mode (Off);
with Ada.Text_IO; use Ada.Text_IO;
with AnsiAda;
with Ada.Real_Time; use Ada.Real_Time;

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
      Counts      : Level_Counts := (others => 0);
      Total       : Long_Long_Integer := 0;
      Last_Source : String (1 .. 32)  := (others => ' ');
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
   function Capacity return Long_Long_Integer is (1_000);

   function Utilization return Long_Long_Float is
      D : constant Long_Long_Integer := Depth;
   begin
      return Long_Long_Float (D) / 1000.0 * 100.0;
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

   procedure Initialize is
   begin
      if not Monitor_Task'Terminated then
         Monitor_Task.Start;
      end if;
   end Initialize;

end ELP_Queue;
