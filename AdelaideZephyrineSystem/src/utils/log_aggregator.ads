pragma Style_Checks (Off);
package Log_Aggregator is
   -- Placeholder package for log aggregation.
   procedure Start with Pre => True, Post => True;
   procedure Append (Message : String) with Pre => True, Post => True;
   procedure Stop with Pre => True, Post => True;
end Log_Aggregator;
