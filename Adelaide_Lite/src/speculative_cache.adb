pragma SPARK_Mode (Off);
with Ada.Characters.Handling; use Ada.Characters.Handling;
with Ada.Strings.Fixed; use Ada.Strings.Fixed;
with Ada.Calendar; use type Ada.Calendar.Time;

package body Speculative_Cache is

   --  ─── Helper functions (outside protected body) ───────────────

   --  Normalize a string for matching:
   --  lowercase, strip punctuation, collapse whitespace.
   function Normalize (S : String) return String is
      Buf : String (1 .. S'Length);
      Len : Natural := 0;
      Last_Was_Space : Boolean := False;
   begin
      for I in S'Range loop
         declare
            C  : constant Character := S (I);
            LC : constant Character := To_Lower (C);
         begin
            if LC in 'a' .. 'z' or else LC in '0' .. '9' then
               Len := Len + 1;
               Buf (Len) := LC;
               Last_Was_Space := False;
            elsif C = ' ' or else C = ASCII.HT
               or else C = ASCII.LF or else C = ASCII.CR
            then
               if not Last_Was_Space and then Len > 0 then
                  Len := Len + 1;
                  Buf (Len) := ' ';
                  Last_Was_Space := True;
               end if;
            end if;
         end;
      end loop;
      if Len > 0 and then Buf (Len) = ' ' then
         Len := Len - 1;
      end if;
      return Buf (1 .. Len);
   end Normalize;

   --  Count words in a normalized string.
   function Word_Count (S : String) return Natural is
      C    : Natural := 0;
      In_W : Boolean := False;
   begin
      for I in S'Range loop
         if S (I) /= ' ' then
            if not In_W then
               C := C + 1;
               In_W := True;
            end if;
         else
            In_W := False;
         end if;
      end loop;
      return C;
   end Word_Count;

   --  Extract the N-th word from a normalized string (0-indexed).
   function Nth_Word (S : String; N : Natural) return String is
      Idx  : Natural := 0;
      In_W : Boolean := False;
      Start_Pos : Positive := S'First;
   begin
      for I in S'Range loop
         if S (I) /= ' ' then
            if not In_W then
               if Idx = N then
                  Start_Pos := I;
               end if;
               Idx := Idx + 1;
               In_W := True;
            end if;
         else
            if In_W and then Idx - 1 = N then
               return S (Start_Pos .. I - 1);
            end if;
            In_W := False;
         end if;
      end loop;
      if In_W and then Idx - 1 = N then
         return S (Start_Pos .. S'Last);
      end if;
      return "";
   end Nth_Word;

   --  Jaccard word-overlap similarity between two normalized strings.
   function Jaccard (A, B : String) return Float is
      WA : constant Natural := Word_Count (A);
      WB : constant Natural := Word_Count (B);
      Intersection : Natural := 0;
      Union_Size   : Natural;
   begin
      if WA = 0 and WB = 0 then
         return 1.0;
      end if;
      if WA = 0 or WB = 0 then
         return 0.0;
      end if;

      for I in 0 .. WA - 1 loop
         declare
            Word_A : constant String := Nth_Word (A, I);
         begin
            for J in 0 .. WB - 1 loop
               if Word_A = Nth_Word (B, J) then
                  Intersection := Intersection + 1;
                  exit;
               end if;
            end loop;
         end;
      end loop;

      Union_Size := WA + WB - Intersection;
      if Union_Size = 0 then
         return 0.0;
      end if;
      return Float (Intersection) / Float (Union_Size);
   end Jaccard;

   --  Check if Query matches Cached_Query.
   --  Uses substring check (fast path) + Jaccard word-overlap (>0.4).
   function Matches (Query : String; Cached_Query : String) return Boolean is
      Norm_Q : constant String := Normalize (Query);
      Norm_C : constant String := Normalize (Cached_Query);
   begin
      --  Fast path: substring match
      if Norm_Q'Length > 0 and then Norm_C'Length > 0
        and then (Ada.Strings.Fixed.Index (Norm_C, Norm_Q) > 0
                  or else Ada.Strings.Fixed.Index (Norm_Q, Norm_C) > 0)
      then
         return True;
      end if;

      --  Jaccard word-overlap similarity
      if Jaccard (Norm_Q, Norm_C) > 0.4 then
         return True;
      end if;

      return False;
   end Matches;

   --  Find index of oldest valid entry (LRU eviction target).
   function Oldest_Entry (Entries : Entry_Array) return Positive is
      Oldest : Positive := 1;
   begin
      for I in 2 .. Max_Entries loop
         if Entries (I).Valid then
            if not Entries (Oldest).Valid
              or else Entries (I).Timestamp < Entries (Oldest).Timestamp
            then
               Oldest := I;
            end if;
         end if;
      end loop;
      return Oldest;
   end Oldest_Entry;

   --  ─── Cache protected body ────────────────────────────────────

   protected body Cache is

      procedure Store (Predicted_Query : String; Answer : String) is
         Slot    : Positive;
         Found   : Boolean := False;
      begin
         --  Reuse an invalid slot if available
         for I in 1 .. Max_Entries loop
            if not Entries (I).Valid then
               Slot := I;
               Found := True;
               exit;
            end if;
         end loop;

         --  Otherwise evict the oldest
         if not Found then
            Slot := Oldest_Entry (Entries);
         end if;

         Entries (Slot) :=
           (Predicted_Query => To_Unbounded_String (Predicted_Query),
            Cached_Answer   => To_Unbounded_String (Answer),
            Timestamp       => Ada.Calendar.Clock,
            Valid           => True);
      end Store;

      function Lookup (Query : String) return String is
      begin
         for I in 1 .. Max_Entries loop
            if Entries (I).Valid
              and then Matches (Query, To_String (Entries (I).Predicted_Query))
            then
               return To_String (Entries (I).Cached_Answer);
            end if;
         end loop;
         return "";
      end Lookup;

      procedure Invalidate is
      begin
         for I in 1 .. Max_Entries loop
            Entries (I).Valid := False;
         end loop;
      end Invalidate;

      function Count return Natural is
         C : Natural := 0;
      begin
         for I in 1 .. Max_Entries loop
            if Entries (I).Valid then
               C := C + 1;
            end if;
         end loop;
         return C;
      end Count;

   end Cache;

end Speculative_Cache;
