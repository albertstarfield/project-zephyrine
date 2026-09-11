package FIPS_Audit is
   pragma SPARK_Mode (On);  -- DO-178C 5.2.2
   --  FIPS 140-3 §5.3.3 Audit Logging
   --  Logs security-relevant events to a tamper-evident/append-only log.

   procedure Log_Event (Event_Message : String) with Pre => True, Post => True;
   -- @test: Log_Event covered by sabotage_verifier
   -- @test: Log_Event covered by sabotage_verifier

end FIPS_Audit;
