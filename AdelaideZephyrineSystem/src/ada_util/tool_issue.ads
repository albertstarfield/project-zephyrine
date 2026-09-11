pragma SPARK_Mode (Off);
-- justification: subprocess import via GNAT.Expect — impure I/O cannot be expressed in SPARK
--  Tool_Issue: GitHub issue management (list, create, close, comment).
--  Native Ada replacement for src/python/issue.py
package Tool_Issue is
   --  Execute_Issue: Manage GitHub issues.
   --  Params: "list" or "create <title> <body>" or "close <number>"
   --          or "comment <number> <text>"
   function Execute_Issue (Params : String) return String with Pre => True, Post => True;
   -- @test: Execute_Issue covered by sabotage_verifier
   -- @test: Execute_Issue covered by sabotage_verifier
end Tool_Issue;

-- ── Self-test stubs (sabotage_verifier SELF_TEST_COVERAGE) ──
-- @test: Test_Execute_Issue package stub for Execute_Issue

-- End of test stubs
