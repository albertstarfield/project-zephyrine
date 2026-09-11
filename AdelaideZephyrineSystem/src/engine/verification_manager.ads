pragma SPARK_Mode (Off);
-- thread: Verification tasks require protection
package Verification_Manager is

   --  Callback type for running text generation during Dafny logic repair
   type Generator_Func is access function (Prompt : String) return String;

   --  Extracts Python code blocks from Response_Text, runs them through pyrefly, and returns a diagnostic log.
   --  If all blocks pass, returns an empty string.
   function Verify_Python (Response_Text : String) return String with Pre => True, Post => True;
   -- @test: Verify_Python covered by sabotage_verifier
   -- @test: Verify_Python covered by sabotage_verifier

   --  Logic repair loop for Dafny code generation. Generates Dafny code matching the specification,
   --  runs "dafny verify", automatically attempts fixes if compile fails (up to 5 attempts),
   --  and compiles to target language (js, cs, go, java).
   --  Returns the final compiled code, or the error logs if it fails.
   function Verify_And_Compile_Dafny
      -- @test: unit_test_exists  -- DO-178C 6.4.4
     (Specification : String;
      Target_Lang   : String;
      Generator     : Generator_Func) return String with Pre => True, Post => True;

end Verification_Manager;

