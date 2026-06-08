pragma SPARK_Mode (On);

package body Fuzzy_Match is

   function To_Lower (C : Character) return Character is
   begin
      if C in 'A' .. 'Z' then
         return Character'Val (Character'Pos (C) + 32);
      end if;
      return C;
   end To_Lower;

   function Match (Haystack, Needle : String) return Float is
      H_Len  : constant Integer := Haystack'Length;
      N_Len  : constant Integer := Needle'Length;
      Matches : Integer := 0;
      J      : Integer := Needle'First;
   begin
      for I in Haystack'Range loop
         if J <= Needle'Last then
            if To_Lower (Haystack (I)) = To_Lower (Needle (J)) then
               Matches := Matches + 1;
               J := J + 1;
            end if;
         end if;
         pragma Loop_Invariant (J >= Needle'First);
      end loop;
      return Float (Matches) / Float (Integer'Max (H_Len, N_Len));
   end Match;

end Fuzzy_Match;
