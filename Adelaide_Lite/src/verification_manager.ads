package Verification_Manager is
   pragma Spark_Mode (Off);

   --  Callback type for running text generation during Dafny logic repair
   type Generator_Func is access function (Prompt : String) return String;

   --  Extracts Python code blocks from Response_Text, runs them through pyrefly, and returns a diagnostic log.
   --  If all blocks pass, returns an empty string.
   function Verify_Python (Response_Text : String) return String;

   --  Logic repair loop for Dafny code generation. Generates Dafny code matching the specification,
   --  runs "dafny verify", automatically attempts fixes if compile fails (up to 5 attempts),
   --  and compiles to target language (js, cs, go, java).
   --  Returns the final compiled code, or the error logs if it fails.
   function Verify_And_Compile_Dafny
     (Specification : String;
      Target_Lang   : String;
      Generator     : Generator_Func) return String;

end Verification_Manager;

