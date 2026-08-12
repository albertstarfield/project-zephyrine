-- File: src/stella_icarus.ads

package Stella_Icarus is
   -- This procedure will print our personalized greeting to the console.
   pragma SPARK_Mode (On);  -- DO-178C 5.2.2
   procedure Greet with Pre => True, Post => True;
   -- @test: Greet covered by sabotage_verifier

end Stella_Icarus;
