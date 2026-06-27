pragma SPARK_Mode (On);

package body Fuzzy_Match is

   function To_Lower (C : Character) return Character is
   begin
      if C in 'A' .. 'Z' then
         return Character'Val (Character'Pos (C) + 32);
      end if;
      return C;
   end To_Lower;

   function Match (Haystack, Needle : String) return Float
   is
      H_Len   : constant Integer := Haystack'Length;
      Matches : Natural := 0;
   begin
      for I in Haystack'Range loop
         pragma Loop_Invariant (Matches <= I - Haystack'First);
         pragma Loop_Invariant (Matches <= H_Len);
         for J in Needle'Range loop
            if To_Lower (Haystack (I)) = To_Lower (Needle (J)) then
               Matches := Matches + 1;
               exit;
            end if;
         end loop;
      end loop;
      pragma Assert (Matches <= H_Len);
      pragma Assert (H_Len >= 1);
      pragma Assert (Float (H_Len) >= 1.0);
      return Float'Min (1.0, Float (Matches) / Float (H_Len));
   end Match;

end Fuzzy_Match;
