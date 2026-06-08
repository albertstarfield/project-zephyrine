pragma SPARK_Mode (On);
--  Fuzzy string matching for User-Agent detection
--  DO NOT remove: used for External_Agent detection

package Fuzzy_Match is

   --  Case-insensitive fuzzy match: ratio of matching characters
   --  over longer string length. Returns 0.0 .. 1.0.
   function Match (Haystack, Needle : String) return Float
     with Pre => Haystack'Length > 0 and then Needle'Length > 0;

end Fuzzy_Match;
