pragma SPARK_Mode (Off);
with Ada.Text_IO; use Ada.Text_IO;
with Ada.Strings; use Ada.Strings;
with Ada.Strings.Fixed; use Ada.Strings.Fixed;
with Ada.Real_Time; use Ada.Real_Time;
with AnsiAda;
with GNAT.OS_Lib;
with Ada.Directories;

package body Accuracy_Benchmark_Manager is

   --  [DO NOT REMOVE] Validate API key
   function Validate_API_Key (Key : String) return Boolean is
   begin
      return Key = BENCHMARK_API_KEY;
   end Validate_API_Key;

   --  [DO NOT REMOVE] Download dataset from HuggingFace
   function Download_Dataset (
      Repo_Id : String;
      Filename : String;
      Cache_Dir : String
   ) return String is
      Output_File : constant String := Cache_Dir & "/" & Filename;
      Success : Boolean;
   begin
      --  Create cache directory if it doesn't exist
      if not Ada.Directories.Exists(Cache_Dir) then
         Ada.Directories.Create_Path(Cache_Dir);
      end if;

      --  Download if not cached
      if not Ada.Directories.Exists(Output_File) then
         Put_Line(AnsiAda.Foreground(AnsiAda.Cyan) &
                  "[Benchmark]" & AnsiAda.Reset &
                  " Downloading " & Filename & " from " & Repo_Id);

         GNAT.OS_Lib.Spawn (
            Program_Name => "huggingface-cli",
            Args => [new String'("download"),
                     new String'(Repo_Id),
                     new String'(Filename),
                     new String'("--local-dir"),
                     new String'(Cache_Dir)],
            Success => Success
         );

         if not Success then
            Put_Line(AnsiAda.Foreground(AnsiAda.Red) &
                     "[Benchmark]" & AnsiAda.Reset &
                     " Failed to download " & Filename);
            return "";
         end if;
      end if;

      return Output_File;
   end Download_Dataset;

   --  [DO NOT REMOVE] Call model chat endpoint via HTTP
   function Call_Model_Chat (
      Prompt : String;
      Max_Tokens : Natural := 128;
      Temperature : Float := 0.0
   ) return String is
      use GNAT.OS_Lib;
      Request_Body : constant String :=
        "{""model"": ""Snowball-Enaga""," &
        """messages"": [{""role"": ""user"", ""content"": """ & Prompt & """}]," &
        """max_tokens"":" & Natural'Image(Max_Tokens) & "," &
        """temperature"":" & Float'Image(Temperature) & "}";
      Output_File : constant String := "/tmp/benchmark_response.json";
      Success : Boolean;
   begin
      --  Call curl to hit the API
      GNAT.OS_Lib.Spawn (
         Program_Name => "curl",
         Args => [new String'("-s"),
                  new String'("-X"),
                  new String'("POST"),
                  new String'("http://127.0.0.1:11420/v1/chat/completions"),
                  new String'("-H"),
                  new String'("Content-Type: application/json"),
                  new String'("-d"),
                  new String'(Request_Body),
                  new String'("-o"),
                  new String'(Output_File)],
         Success => Success
      );

      if not Success then
         Put_Line(AnsiAda.Foreground(AnsiAda.Red) &
                  "[Benchmark]" & AnsiAda.Reset &
                  " Failed to call model chat endpoint");
         return "";
      end if;

      --  Read response
      declare
         File : File_Type;
         Content : Unbounded_String;
      begin
         Open(File, In_File, Output_File);
         while not End_Of_File(File) loop
            Append(Content, Get_Line(File));
         end loop;
         Close(File);
         return To_String(Content);
      exception
         when others =>
            if Is_Open(File) then
               Close(File);
            end if;
            return "";
      end;
   end Call_Model_Chat;

   --  [DO NOT REMOVE] Extract answer from model response
   function Extract_Answer (
      Response : String;
      Benchmark : Benchmark_Type
   ) return String is
      Response_Upper : Unbounded_String := To_Unbounded_String(Response);
   begin
      --  Convert to uppercase manually
      for I in 1 .. Length(Response_Upper) loop
         declare
            C : Character := Element(Response_Upper, I);
         begin
            if C in 'a' .. 'z' then
               Replace_Element(Response_Upper, I, Character'Val(Character'Pos(C) - 32));
            end if;
         end;
      end loop;

      declare
         Response_Upper_Str : constant String := To_String(Response_Upper);
      begin
         case Benchmark is
            when BENCH_MMLU | BENCH_MMLU_PRO | BENCH_KMMLU | BENCH_CMMLU | BENCH_JMMLU =>
               --  Extract multiple choice answer (A, B, C, D)
               for I in Response_Upper_Str'Range loop
                  if Response_Upper_Str(I) = 'A' or else
                     Response_Upper_Str(I) = 'B' or else
                     Response_Upper_Str(I) = 'C' or else
                     Response_Upper_Str(I) = 'D' then
                     return "" & Response_Upper_Str(I);
                  end if;
               end loop;
            return "";

         when BENCH_GSM8K | BENCH_MATHQA =>
            --  Extract numeric answer after ####
            declare
               Pos : Natural := Index(Response, "####");
            begin
               if Pos > 0 then
                  return Trim(Response(Pos + 4 .. Response'Last), Both);
               end if;
               --  Fallback: last number
               for I in reverse Response'Range loop
                  if Response(I) in '0' .. '9' then
                     declare
                        Num_End : Natural := I;
                        Num_Start : Natural := I;
                     begin
                        while Num_Start > Response'First and then
                              Response(Num_Start - 1) in '0' .. '9' loop
                           Num_Start := Num_Start - 1;
                        end loop;
                        return Trim(Response(Num_Start .. Num_End), Both);
                     end;
                  end if;
               end loop;
               return "";
            end;

         when BENCH_HUMANEval | BENCH_MBPP | BENCH_LIVECODEBENCH =>
            --  Extract code block
            declare
               Start_Pos : Natural := Index(Response, "```python");
               End_Pos : Natural;
            begin
               if Start_Pos > 0 then
                  Start_Pos := Start_Pos + 9;
                  End_Pos := Index(Response(Start_Pos .. Response'Last), "```");
                  if End_Pos > 0 then
                     return Trim(Response(Start_Pos .. Start_Pos + End_Pos - 2), Both);
                  end if;
               end if;
               return Response;
            end;

         when BENCHHELLASWAG | BENCH_WINOGRANDE =>
            --  Extract answer (1, 2, 3, 4)
            for I in Response_Upper_Str'Range loop
               if Response_Upper_Str(I) = '1' then return "1";
               elsif Response_Upper_Str(I) = '2' then return "2";
               elsif Response_Upper_Str(I) = '3' then return "3";
               elsif Response_Upper_Str(I) = '4' then return "4";
               end if;
            end loop;
            return "";

         when BENCH_TRUTHFULQA =>
            --  Extract yes/no or true/false
            if Index(Response_Upper_Str, "YES") > 0 then return "YES";
            elsif Index(Response_Upper_Str, "NO") > 0 then return "NO";
            elsif Index(Response_Upper_Str, "TRUE") > 0 then return "TRUE";
            elsif Index(Response_Upper_Str, "FALSE") > 0 then return "FALSE";
            end if;
            return "";

         when BENCH_ARC_CHALLENGE =>
            --  Extract multiple choice (A, B, C, D, E)
            for I in Response_Upper_Str'Range loop
               if Response_Upper_Str(I) = 'A' then return "A";
               elsif Response_Upper_Str(I) = 'B' then return "B";
               elsif Response_Upper_Str(I) = 'C' then return "C";
               elsif Response_Upper_Str(I) = 'D' then return "D";
               elsif Response_Upper_Str(I) = 'E' then return "E";
               end if;
            end loop;
            return "";

         when BENCH_BBM =>
            --  Extract multiple choice (A, B, C, D)
            for I in Response_Upper_Str'Range loop
               if Response_Upper_Str(I) = 'A' then return "A";
               elsif Response_Upper_Str(I) = 'B' then return "B";
               elsif Response_Upper_Str(I) = 'C' then return "C";
               elsif Response_Upper_Str(I) = 'D' then return "D";
               end if;
            end loop;
            return "";

         when BENCH_SAFETYBENCH =>
            --  Extract answer (1, 2, 3, 4)
            for I in Response_Upper_Str'Range loop
               if Response_Upper_Str(I) = '1' then return "1";
               elsif Response_Upper_Str(I) = '2' then return "2";
               elsif Response_Upper_Str(I) = '3' then return "3";
               elsif Response_Upper_Str(I) = '4' then return "4";
               end if;
            end loop;
            return "";
      end case;
      end;
   end Extract_Answer;

   --  [DO NOT REMOVE] Check if answer is correct
   function Check_Answer (
      Predicted : String;
      Expected : String;
      Benchmark : Benchmark_Type
   ) return Boolean is
   begin
      case Benchmark is
         when BENCH_MMLU | BENCH_MMLU_PRO | BENCH_KMMLU | BENCH_CMMLU | BENCH_JMMLU |
              BENCH_ARC_CHALLENGE | BENCH_BBM =>
            return Predicted = Expected;

         when BENCH_GSM8K | BENCH_MATHQA =>
            --  Numeric comparison
            begin
               return Float'Value(Predicted) = Float'Value(Expected);
            exception
               when others => return False;
            end;

         when BENCH_HUMANEval | BENCH_MBPP | BENCH_LIVECODEBENCH =>
            --  Code similarity (simplified: exact match for now)
            return Predicted = Expected;

         when BENCHHELLASWAG | BENCH_WINOGRANDE | BENCH_SAFETYBENCH =>
            return Predicted = Expected;

         when BENCH_TRUTHFULQA =>
            return Predicted = Expected;
      end case;
   end Check_Answer;

   --  [DO NOT REMOVE] Run accuracy benchmark
   procedure Run_Accuracy_Benchmark (
      Benchmark : Benchmark_Type;
      Sample_Size : Natural := 0;
      On_Progress : access procedure (Event : String);
      Result : out Benchmark_Result
   ) is
      Start_Time : constant Time := Clock;
      Dataset_File : Unbounded_String;
      Correct : Natural := 0;
      Total : Natural := 0;
   begin
      --  [DO NOT REMOVE] Log benchmark start
      Put_Line(AnsiAda.Foreground(AnsiAda.Cyan) &
               "[Benchmark]" & AnsiAda.Reset &
               " Starting accuracy benchmark");

      --  Download dataset
      case Benchmark is
         when BENCH_MMLU =>
            Dataset_File := To_Unbounded_String(
               Download_Dataset("cais/mmlu", "all/test.jsonl", "benchmark_data"));
         when BENCH_GSM8K =>
            Dataset_File := To_Unbounded_String(
               Download_Dataset("openai/gsm8k", "test.jsonl", "benchmark_data"));
         when BENCH_HUMANEval =>
            Dataset_File := To_Unbounded_String(
               Download_Dataset("openai_humaneval", "human_eval.jsonl", "benchmark_data"));
         when BENCHHELLASWAG =>
            Dataset_File := To_Unbounded_String(
               Download_Dataset("Rowan/hellaswag", "hellaswag_val.jsonl", "benchmark_data"));
         when BENCH_TRUTHFULQA =>
            Dataset_File := To_Unbounded_String(
               Download_Dataset("truthfulqa/truthful_qa", "validation.jsonl", "benchmark_data"));
         when BENCH_ARC_CHALLENGE =>
            Dataset_File := To_Unbounded_String(
               Download_Dataset("allenai/ai2_arc", "ARC-Challenge.jsonl", "benchmark_data"));
         when BENCH_WINOGRANDE =>
            Dataset_File := To_Unbounded_String(
               Download_Dataset("allenai/winogrande", "winogrande_xl.jsonl", "benchmark_data"));
         when BENCH_MATHQA =>
            Dataset_File := To_Unbounded_String(
               Download_Dataset("math_qa", "test.jsonl", "benchmark_data"));
         when BENCH_MBPP =>
            Dataset_File := To_Unbounded_String(
               Download_Dataset("google-research-datasets/mbpp", "mbpp.jsonl", "benchmark_data"));
         when BENCH_LIVECODEBENCH =>
            Dataset_File := To_Unbounded_String(
               Download_Dataset("livecodebench", "test.jsonl", "benchmark_data"));
         when BENCH_BBM =>
            Dataset_File := To_Unbounded_String(
               Download_Dataset("s-Q-wers/bbq", "bbq.jsonl", "benchmark_data"));
         when BENCH_SAFETYBENCH =>
            Dataset_File := To_Unbounded_String(
               Download_Dataset("safetybench", "test.jsonl", "benchmark_data"));
         when BENCH_MMLU_PRO =>
            Dataset_File := To_Unbounded_String(
               Download_Dataset("TIGER-Lab/MMLU-Pro", "test.jsonl", "benchmark_data"));
         when BENCH_KMMLU =>
            Dataset_File := To_Unbounded_String(
               Download_Dataset("AI-Hyperdata/KMMLU", "test.jsonl", "benchmark_data"));
         when BENCH_CMMLU =>
            Dataset_File := To_Unbounded_String(
               Download_Dataset("haonan-li/cmmlu", "test.jsonl", "benchmark_data"));
         when BENCH_JMMLU =>
            Dataset_File := To_Unbounded_String(
               Download_Dataset("polyzer/jmmlu", "test.jsonl", "benchmark_data"));
      end case;

      --  [DO NOT REMOVE] Log dataset download
      Put_Line(AnsiAda.Foreground(AnsiAda.Green) &
               "[Benchmark]" & AnsiAda.Reset &
               " Dataset downloaded: " & To_String(Dataset_File));

      --  TODO: Parse dataset, run questions, check answers
      --  For now, return placeholder result
      Total := 100;  -- Placeholder
      Correct := 0;

      --  [DO NOT REMOVE] Log benchmark completion
      declare
         Total_Duration : constant Duration := To_Duration(Clock - Start_Time);
      begin
         Put_Line(AnsiAda.Foreground(AnsiAda.Cyan) &
                  "[Benchmark]" & AnsiAda.Reset &
                  " Benchmark completed in" & Duration'Image(Total_Duration) & "s");

         Result := (
            Benchmark_Name => To_Unbounded_String("accuracy_benchmark"),
            Accuracy => Float(Correct) / Float(Total),
            Total_Questions => Total,
            Correct_Count => Correct,
            Time_Seconds => Float(Total_Duration)
         );
      end;
   end Run_Accuracy_Benchmark;

end Accuracy_Benchmark_Manager;
