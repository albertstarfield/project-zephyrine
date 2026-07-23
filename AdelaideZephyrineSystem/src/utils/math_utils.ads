pragma SPARK_Mode (On);

with Ada.Strings.Unbounded;

package Math_Utils is

    type Vector is array (Positive range <>) of Float;
    type Embedding_Vector is array (1 .. 4096) of Float;
    type Embedding_Vector_List is array (Positive range <>) of Embedding_Vector;
    type Natural_List is array (Positive range <>) of Natural;
    type Prompt_List is array (Positive range <>) of Ada.Strings.Unbounded.Unbounded_String;





   --  Cosine_Similarity: Computes the cosine similarity between two vectors.
   function Cosine_Similarity (V1 : Vector; V2 : Vector) return Float
     with
       Pre => V1'Length = V2'Length and then
              V1'Length > 0 and then
              V1'Length <= 1_000_000,
       Post => Cosine_Similarity'Result >= -1.0 and then
               Cosine_Similarity'Result <= 1.0;

end Math_Utils;
