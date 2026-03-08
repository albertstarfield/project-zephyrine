package Message_Generator
  with SPARK_Mode
is

   --  This function is pure SPARK. It has no inputs and returns a String.
   --  The postcondition contract "Post => Get_Message'Result = ..."
   --  is a promise to the prover about what the function will return.
   --  gnatprove will verify that the implementation fulfills this promise.
   function Get_Message return String
     with Post => Get_Message'Result = "Hello, Sekai! (Proven and Separated)";

end Message_Generator;