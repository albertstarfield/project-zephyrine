pragma SPARK_Mode (Off);
-- thread: Benchmark execution requires protection
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;

package Accuracy_Benchmark_Manager is

   --  Benchmark API Key
   BENCHMARK_API_KEY : constant String := "IknowtheConsequencesAndWouldLockupTheServerForHours";

   --  Benchmark types
   type Benchmark_Type is (
      BENCH_MMLU,
      BENCH_MMLU_PRO,
      BENCH_KMMLU,
      BENCH_CMMLU,
      BENCH_JMMLU,
      BENCH_GSM8K,
      BENCH_MATHQA,
      BENCH_HUMANEval,
      BENCH_MBPP,
      BENCH_LIVECODEBENCH,
      BENCHHELLASWAG,
      BENCH_TRUTHFULQA,
      BENCH_ARC_CHALLENGE,
      BENCH_WINOGRANDE,
      BENCH_BBM,
      BENCH_SAFETYBENCH
   );

   --  Question result
   type Question_Result is record
      Question_Id : Unbounded_String;
      Correct : Boolean := False;
      Expected : Unbounded_String;
      Predicted : Unbounded_String;
      Raw_Response : Unbounded_String;
      Time_Seconds : Float := 0.0;
   end record;

   --  Benchmark result
   type Benchmark_Result is record
      Benchmark_Name : Unbounded_String;
      Accuracy : Float := 0.0;
      Total_Questions : Natural := 0;
      Correct_Count : Natural := 0;
      Failed_Count : Natural := 0;
      Time_Seconds : Float := 0.0;
      Failed_Question : Question_Result;
      Failed_Message : Unbounded_String;
   end record;

   --  Benchmark failure exception
   --  Raised when any answer is unparseable or incorrect
   --  Complete stop - no tolerance for failures
   Benchmark_Failure : exception;

   --  Validate API key
   function Validate_API_Key (Key : String) return Boolean;

   --  Run accuracy benchmark
   --  RAISES Benchmark_Failure if any answer is unparseable
   procedure Run_Accuracy_Benchmark (
      Benchmark : Benchmark_Type;
      Sample_Size : Natural := 0;  -- 0 = full dataset
      On_Progress : access procedure (Event : String);
      Result : out Benchmark_Result
   );

   --  Load bundled dataset from local JSONL file (OMLX pattern)
   function Download_Dataset (
      Repo_Id : String;
      Subset  : String;
      Cache_Dir : String;
      Split   : String := "test"
   ) return String;

   --  Call model chat endpoint
   function Call_Model_Chat (
      Prompt : String;
      Max_Tokens : Natural := 128;
      Temperature : Float := 0.0
   ) return String;

   --  Extract answer from model response
   function Extract_Answer (
      Response : String;
      Benchmark : Benchmark_Type
   ) return String;

   --  Check if answer is correct
   function Check_Answer (
      Predicted : String;
      Expected : String;
      Benchmark : Benchmark_Type
   ) return Boolean;

end Accuracy_Benchmark_Manager;
