package Math_Utils is
   pragma Spark_Mode (On);

   type Vector is array (Positive range <>) of Float;

   function Cosine_Similarity (V1 : Vector; V2 : Vector) return Float
     with
       Pre => V1'Length = V2'Length and then
              V1'Length > 0 and then
              V1'Length <= 1_000_000,
       Post => Cosine_Similarity'Result >= -1.0 and then
               Cosine_Similarity'Result <= 1.0;

end Math_Utils;
