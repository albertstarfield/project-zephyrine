package FIPS_Audit is
   --  FIPS 140-3 §5.3.3 Audit Logging
   --  Logs security-relevant events to a tamper-evident/append-only log.

   procedure Log_Event (Event_Message : String);

end FIPS_Audit;
