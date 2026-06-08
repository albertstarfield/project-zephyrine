pragma SPARK_Mode (On);

pragma SPARK_Mode (Off);

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
         for J in Needle'Range loop
            if To_Lower (Haystack (I)) = To_Lower (Needle (J)) then
               Matches := Matches + 1;
               exit;
            end if;
         end loop;
      end loop;
      return Float (Matches) / Float (H_Len);
   end Match;

end Fuzzy_Match;
