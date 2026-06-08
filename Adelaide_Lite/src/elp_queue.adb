pragma SPARK_Mode (Off);
with Ada.Text_IO; use Ada.Text_IO;
with AnsiAda;

package body ELP_Queue is

   Max_Capacity : constant Long_Long_Integer := 2**62;  --  2^63 safe max

   type Queue_Entry is record
      Level : ELP_Level := ELP0;
      Kind  : Model_Type := Qwen_Embedding;
   end record;

   type Ring_Buffer is array (1 .. 1_000) of Queue_Entry;
   --  Internal ring buffer for actual storage

   Buffer  : Ring_Buffer;
   Head    : Long_Long_Integer := 1;   --  next dequeue position
   Tail    : Long_Long_Integer := 1;   --  next enqueue position
   Count   : Long_Long_Integer := 0;   --  current items in queue

   procedure Enqueue (Level : ELP_Level; Kind : Model_Type) is
   begin
      while Count >= Max_Capacity loop
         delay 0.001;
      end loop;

      declare
         Idx : constant Long_Long_Integer :=
           ((Tail - 1) mod 1_000) + 1;
      begin
         Buffer (Integer (Idx)) := (Level => Level, Kind => Kind);
         Tail := Tail + 1;
         Count := Count + 1;
      end;
   end Enqueue;

   procedure Dequeue
     (Level : out ELP_Level;
      Kind  : out Model_Type)
   is
      Best_Idx  : Long_Long_Integer := 0;
      Best_Prio : Integer := -1;
      Found     : Boolean := False;
   begin
      --  Scan for highest-priority item (ELP3=3 > ELP2=2 > ELP1=1 > ELP0=0)
      while not Found loop
         for I in 1 .. Long_Long_Integer'Min (Count, 1_000) loop
            declare
               Pos : constant Long_Long_Integer :=
                 ((Head + I - 2) mod 1_000) + 1;
               Entry_Level : constant Integer :=
                 ELP_Level'Pos (Buffer (Integer (Pos)).Level);
            begin
               if Entry_Level > Best_Prio then
                  Best_Prio := Entry_Level;
                  Best_Idx  := I;
                  Found     := True;
               end if;
            end;
         end loop;

         if not Found then
            delay 0.001;
         end if;
      end loop;

      --  Extract best item
      declare
         Pos : constant Long_Long_Integer :=
           ((Head + Best_Idx - 2) mod 1_000) + 1;
      begin
         Level := Buffer (Integer (Pos)).Level;
         Kind  := Buffer (Integer (Pos)).Kind;

         --  Compact: shift remaining items
         if Best_Idx > 1 then
            for I in Best_Idx .. Count loop
               declare
                  Src : constant Long_Long_Integer :=
                    ((Head + I - 2) mod 1_000) + 1;
                  Dst : constant Long_Long_Integer :=
                    ((Head + I - Best_Idx - 1) mod 1_000) + 1;
               begin
                  Buffer (Integer (Dst)) := Buffer (Integer (Src));
               end;
            end loop;
         end if;

         Head := Head + 1;
         Count := Count - 1;
      end;
   end Dequeue;

   function Depth return Long_Long_Integer is
   begin
      return Count;
   end Depth;

   function Capacity return Long_Long_Integer is
   begin
      return Max_Capacity;
   end Capacity;

   function Utilization return Long_Long_Float is
   begin
      if Max_Capacity = 0 then
         return 0.0;
      end if;
      return Long_Long_Float (Count) / Long_Long_Float (Max_Capacity) * 100.0;
   end Utilization;

   --  Monitor task: prints queue capacity every 5 seconds
   task Monitor_Task is
      entry Start;
   end Monitor_Task;

   task body Monitor_Task is
      Interval : constant Time_Span := Seconds (5);
      Next_Check : Time;
   begin
      accept Start;
      loop
         Next_Check := Clock + Interval;
         declare
            D : constant Long_Long_Integer := Depth;
            C : constant Long_Long_Integer := Capacity;
            U : constant Long_Long_Float := Utilization;
            Pct : constant Long_Long_Integer :=
              Long_Long_Integer (U);
         begin
            Put_Line (AnsiAda.Foreground (AnsiAda.Grey) &
              "[ELP-Queue] Depth:" & D'Img &
              " /" & C'Img &
              " (" & Pct'Img & "%)" &
              AnsiAda.Reset);
         end;
         delay until Next_Check;
      end loop;
   end Monitor_Task;

   procedure Initialize is
   begin
      Monitor_Task.Start;
   end Initialize;

end ELP_Queue;
