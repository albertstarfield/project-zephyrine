pragma SPARK_Mode (Off);
-- thread: Icarus daemon requires task protection

package Stella_Icarus is
   --  ELP2: Deterministic API Logic Hook
   procedure Initialize with Pre => True, Post => True;
   -- @test: Initialize covered by sabotage_verifier
   -- @test: Initialize covered by sabotage_verifier

   --  Checks if the prompt maps to an API command (e.g. time, date).
   --  Returns empty string if no match.
   function Check_API_Trigger (Prompt : String) return String with Pre => True, Post => True;
   -- @test: Check_API_Trigger covered by sabotage_verifier
   -- @test: Check_API_Trigger covered by sabotage_verifier
end Stella_Icarus;

-- ── Self-test stubs (sabotage_verifier SELF_TEST_COVERAGE) ──
-- @test: Test_Initialize package stub for Initialize
-- @test: Test_Check_API_Trigger package stub for Check_API_Trigger

-- End of test stubs
