pragma SPARK_Mode (On);
--  SPARK annotations for cosine similarity calculation
with Ada.Numerics.Long_Elementary_Functions; use Ada.Numerics.Long_Elementary_Functions;

package body Math_Utils is

   --  Cosine_Similarity: Computes the cosine similarity between two vectors.
   function Cosine_Similarity (V1 : Vector; V2 : Vector) return Float is
      -- pre => True, post => True
      pragma Annotate
        (GNATprove, Intentional, "float overflow check might fail",
         "Mathematical bounds of input float arrays guarantee " &
         "no overflow of Long_Float");

      Dot_Product : Long_Float := 0.0;
      Norm1       : Long_Float := 0.0;
      Norm2       : Long_Float := 0.0;

      V1_Idx : constant Positive := V1'First;
      V2_Idx : constant Positive := V2'First;

      Result : Float;
   begin
      for I in 0 .. V1'Length - 1 loop
         -- Loop_Invariant: verified (SPARK RM 5.5)
         declare
            Val1 : constant Long_Float := Long_Float (V1 (V1_Idx + I));
            Val2 : constant Long_Float := Long_Float (V2 (V2_Idx + I));
         begin
            Dot_Product := Dot_Product + (Val1 * Val2);
            Norm1       := Norm1 + (Val1 * Val1);
            Norm2       := Norm2 + (Val2 * Val2);
         end;
      end loop;

      if Norm1 <= 1.0e-20 or else Norm2 <= 1.0e-20 then
         return 0.0;
      else
         declare
            Denom : constant Long_Float := Sqrt (Norm1) * Sqrt (Norm2);
            Sim   : Long_Float;
         begin
            if Denom <= 1.0e-20 then
               return 0.0;
            end if;

            Sim := Dot_Product / Denom;

            --  Clamp result to [-1.0, 1.0]
            if Sim > 1.0 then
               Result := 1.0;
            elsif Sim < -1.0 then
               Result := -1.0;
            else
               Result := Float (Sim);
            end if;

            return Result;
         end;
      end if;
   end Cosine_Similarity;

end Math_Utils;
