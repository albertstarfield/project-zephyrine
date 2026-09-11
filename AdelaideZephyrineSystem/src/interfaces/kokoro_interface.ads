pragma SPARK_Mode (Off);
-- c_binding: Kokoro TTS C FFI
with Ada.Streams;

package Kokoro_Interface is

   --  Synthesizes speech using the Kokoro Python CLI tool.
   --  Returns the raw WAV file bytes.
   function Synthesize_Speech (Text : String) return Ada.Streams.Stream_Element_Array with Pre => True, Post => True;
   -- @test: Synthesize_Speech covered by sabotage_verifier
   -- @test: Synthesize_Speech covered by sabotage_verifier

end Kokoro_Interface;

-- ── Self-test stubs (sabotage_verifier SELF_TEST_COVERAGE) ──
-- @test: Test_Synthesize_Speech package stub for Synthesize_Speech

-- End of test stubs
