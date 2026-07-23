pragma SPARK_Mode (On);

package body Fuzzy_Match is

   --  To_Lower: Converts an uppercase character to lowercase.
   function To_Lower (C : Character) return Character is
   begin
      if C in 'A' .. 'Z' then
         return Character'Val (Character'Pos (C) + 32);
      end if;
      return C;
   end To_Lower;

   --  Match: Returns a fuzzy match score between 0.0 (no match) and 1.0 (exact match).
   function Match (Haystack, Needle : String) return Float
   is
      H_Len   : constant Integer := Haystack'Length;
      N_Len   : constant Integer := Needle'Length;
   begin
      if N_Len = 0 then
         return 0.0;
      end if;

      -- Check for case-insensitive substring match
      if H_Len >= N_Len then
         for I in Haystack'First .. Haystack'Last - N_Len + 1 loop
            declare
               Found : Boolean := True;
            begin
               for J in 0 .. N_Len - 1 loop
                  if To_Lower (Haystack (I + J)) /= To_Lower (Needle (Needle'First + J)) then
                     Found := False;
                     exit;
                  end if;
               end loop;
               if Found then
                  return 1.0;
               end if;
            end;
         end loop;
      end if;

      -- Fallback to character overlap ratio if not direct substring
      declare
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
      end;
   end Match;

end Fuzzy_Match;
