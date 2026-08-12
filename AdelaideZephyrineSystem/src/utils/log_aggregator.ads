pragma Style_Checks (Off);
package Log_Aggregator is
   pragma SPARK_Mode (On);  -- DO-178C 5.2.2
   -- Placeholder package for log aggregation.
   procedure Start with Pre => True, Post => True;
   -- @test: Start covered by sabotage_verifier
   procedure Append (Message : String) with Pre => True, Post => True;
   procedure Stop with Pre => True, Post => True;
end Log_Aggregator;
