pragma SPARK_Mode (Off);
-- thread: Verification tasks require protection
with Ada.Text_IO; use Ada.Text_IO;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Ada.Strings.Fixed; use Ada.Strings.Fixed;
with GNAT.OS_Lib;
with Ada.Directories;
with Ada.Numerics.Discrete_Random;
with Ada.Characters.Handling;
with AnsiAda;

package body Verification_Manager is

   --  Helper to run an external command and capture its output
   function Run_Command_Capture
     (Cmd : String; Args : GNAT.OS_Lib.Argument_List; Log_File : String) return Integer
   is
      use GNAT.OS_Lib;
      Path     : GNAT.OS_Lib.String_Access := GNAT.OS_Lib.Locate_Exec_On_Path (Cmd);
      Success  : Boolean;
      Ret_Code : Integer;
   begin
      if Path = null then
         return -1;
      end if;

      Spawn (Path.all, Args, Log_File, Success, Ret_Code);
      Free (Path);

      if Success then
         return Ret_Code;
      else
         return -2;
      end if;
   end Run_Command_Capture;

   --  Helper to read a whole file into a String/Unbounded_String
   function Read_File_Content (File_Path : String) return String is
      File : File_Type;
      Content : Unbounded_String := Null_Unbounded_String;
   begin
      if not Ada.Directories.Exists (File_Path) then
         return "";
      end if;
      Open (File, In_File, File_Path);
      while not End_Of_File (File) loop
         Append (Content, Get_Line (File));
         if not End_Of_File (File) then
            Append (Content, ASCII.LF);
         end if;
      end loop;
      Close (File);
      return To_String (Content);
   exception
      when others =>
         if Is_Open (File) then
            Close (File);
         end if;
         return "";
   end Read_File_Content;

   --  Helper to generate a random 8-character hex string for temp filenames
   function Get_Random_Suffix return String is
      subtype Rand_Range is Integer range 0 .. 15;
      package Rand_Pack is new Ada.Numerics.Discrete_Random (Rand_Range);
      Seed : Rand_Pack.Generator;
      Chars : constant String := "0123456789abcdef";
      Result : String (1 .. 8);
   begin
      Rand_Pack.Reset (Seed);
      for I in Result'Range loop
         Result (I) := Chars (Rand_Pack.Random (Seed) + 1);
      end loop;
      return Result;
   end Get_Random_Suffix;

   -------------------
   -- Verify_Python --
   -------------------
   function Verify_Python (Response_Text : String) return String is
      use GNAT.OS_Lib;
      I           : Positive := Response_Text'First;
      Block_Idx   : Natural := 0;
      Passed      : Boolean := True;
      Logs        : Unbounded_String := Null_Unbounded_String;
      Tag         : constant String := "```python";
      Close_Tag   : constant String := "```";
   begin
      loop
         declare
            Tag_Pos : constant Natural := Index (Response_Text, Tag, I);
            Next_Pos : Natural;
            End_Pos  : Natural;
            LF_Pos   : Natural;
         begin
            exit when Tag_Pos = 0;
            
            --  Find end of line containing "```python"
            LF_Pos := Index (Response_Text, String'(1 => ASCII.LF), Tag_Pos);
            if LF_Pos = 0 then
               LF_Pos := Tag_Pos + Tag'Length;
            else
               LF_Pos := LF_Pos + 1;
            end if;

            --  Find closing ```
            Next_Pos := Index (Response_Text, Close_Tag, LF_Pos);
            if Next_Pos = 0 then
               exit;
            end if;

            End_Pos := Next_Pos - 1;
            Block_Idx := Block_Idx + 1;

            declare
               Code : constant String := Response_Text (LF_Pos .. End_Pos);
               Suffix : constant String := Get_Random_Suffix;
               Temp_File : constant String := "obj/py_verify_" & Suffix & ".py";
               Log_File  : constant String := "obj/pyrefly_" & Suffix & ".log";
               Temp_File_IO : File_Type;
               Args : Argument_List (1 .. 2);
               Ret : Integer;
            begin
               --  Create temp file
               Create (Temp_File_IO, Out_File, Temp_File);
               Put (Temp_File_IO, Code);
               Close (Temp_File_IO);

               --  Run pyrefly check
               Args (1) := new String'("check");
               Args (2) := new String'(Temp_File);
               Ret := Run_Command_Capture ("pyrefly", Args, Log_File);
               Free (Args (1));
               Free (Args (2));

               if Ret /= 0 then
                  Passed := False;
                  declare
                     Err_Log : constant String := Read_File_Content (Log_File);
                  begin
                     Append (Logs, "Block " & Block_Idx'Img & " failed validation: " & Err_Log & ASCII.LF);
                  end;
               else
                  Append (Logs, "Block " & Block_Idx'Img & " passed validation." & ASCII.LF);
               end if;

               --  Clean up temp files
               if Ada.Directories.Exists (Temp_File) then
                  Ada.Directories.Delete_File (Temp_File);
               end if;
               if Ada.Directories.Exists (Log_File) then
                  Ada.Directories.Delete_File (Log_File);
               end if;
            end;

            I := Next_Pos + Close_Tag'Length;
         end;
      end loop;

      if Passed then
         return "";
      else
         return To_String (Logs);
      end if;
   end Verify_Python;

   --------------------------------
   -- Verify_And_Compile_Dafny --
   --------------------------------
   function Verify_And_Compile_Dafny
     (Specification : String;
      Target_Lang   : String;
      Generator     : Generator_Func) return String
   is
      use GNAT.OS_Lib;
      MAX_ATTEMPTS  : constant Positive := 5;
      Dafny_Code    : Unbounded_String := Null_Unbounded_String;
      Last_Errors   : Unbounded_String := Null_Unbounded_String;
      Target        : Unbounded_String;
      Lang_Lower    : constant String := Ada.Characters.Handling.To_Lower (Target_Lang);
      Suffix        : constant String := Get_Random_Suffix;
      Dfy_File      : constant String := "obj/dafny_" & Suffix & ".dfy";
      Log_File      : constant String := "obj/dafny_verify_" & Suffix & ".log";
      Build_Log     : constant String := "obj/dafny_build_" & Suffix & ".log";
   begin
      --  Map Target Language
      if Lang_Lower = "js" or else Lang_Lower = "javascript" then
         Target := To_Unbounded_String ("js");
      elsif Lang_Lower = "cs" or else Lang_Lower = "csharp" then
         Target := To_Unbounded_String ("cs");
      elsif Lang_Lower = "go" then
         Target := To_Unbounded_String ("go");
      elsif Lang_Lower = "java" then
         Target := To_Unbounded_String ("java");
      elsif Lang_Lower = "python" or else Lang_Lower = "py" then
         Target := To_Unbounded_String ("py");
      else
         Target := To_Unbounded_String ("js");
      end if;

      for Attempt in 1 .. MAX_ATTEMPTS loop
         Put_Line ("[*] Dafny Phase: Generation/Fix Attempt" & Attempt'Img & " /" & MAX_ATTEMPTS'Img & "...");
         
         declare
            Prompt : Unbounded_String;
         begin
            Prompt := To_Unbounded_String
              ("You are a formal verification expert. Generate Dafny code for the following specification:" & ASCII.LF &
               Specification & ASCII.LF & ASCII.LF &
               "IMPORTANT: Output ONLY the Dafny code wrapped in ```dafny ... ``` tags." & ASCII.LF &
               "Ensure the code is self-contained and includes all necessary lemmas, predicates, or method pre/post conditions for verification." & ASCII.LF &
               "The Dafny code should be optimized for compilation to " & Target_Lang & ".");

            if Length (Dafny_Code) > 0 and then Length (Last_Errors) > 0 then
               Append (Prompt, ASCII.LF & ASCII.LF &
                 "Your previous Dafny attempt failed verification with these errors:" & ASCII.LF &
                 To_String (Last_Errors) & ASCII.LF & ASCII.LF &
                 "Please fix the Dafny code and provide a corrected version.");
            end if;

            declare
               Resp_Text : constant String := Generator (To_String (Prompt));
               Dfy_Start : constant Natural := Index (Resp_Text, "```dafny");
               Dfy_End   : Natural;
               LF_Pos    : Natural;
            begin
               if Dfy_Start = 0 then
                  Last_Errors := To_Unbounded_String ("No ```dafny``` block found in response.");
                  goto Continue;
               end if;

               LF_Pos := Index (Resp_Text, String'(1 => ASCII.LF), Dfy_Start);
               if LF_Pos = 0 then
                  LF_Pos := Dfy_Start + 8;
               else
                  LF_Pos := LF_Pos + 1;
               end if;

               Dfy_End := Index (Resp_Text, "```", LF_Pos);
               if Dfy_End = 0 then
                  Last_Errors := To_Unbounded_String ("Closing ``` tag missing.");
                  goto Continue;
               end if;

               Dafny_Code := To_Unbounded_String (Resp_Text (LF_Pos .. Dfy_End - 1));

               --  Write code to dfy file
               declare
                  File : File_Type;
               begin
                  Create (File, Out_File, Dfy_File);
                  Put (File, To_String (Dafny_Code));
                  Close (File);
               end;

               --  Run dafny verify
               Put_Line ("[*] Dafny Phase: Verifying logical correctness...");
               declare
                  Verify_Args : Argument_List (1 .. 2);
                  Ret : Integer;
               begin
                  Verify_Args (1) := new String'("verify");
                  Verify_Args (2) := new String'(Dfy_File);
                  Ret := Run_Command_Capture ("dafny", Verify_Args, Log_File);
                  Free (Verify_Args (1));
                  Free (Verify_Args (2));

                  declare
                     Verify_Out : constant String := Read_File_Content (Log_File);
                  begin
                     if Ret = 0 and then (Index (Verify_Out, "0 errors") > 0 or else Verify_Out = "") then
                        Put_Line ("[+] Dafny Phase: Verification SUCCESS. Compiling to " & Target_Lang & "...");
                        
                        --  Dafny verify succeeded, now build/compile
                        declare
                           Build_Args : Argument_List (1 .. 4);
                           Build_Ret : Integer;
                        begin
                           Build_Args (1) := new String'("build");
                           Build_Args (2) := new String'("--target");
                           Build_Args (3) := new String'(To_String (Target));
                           Build_Args (4) := new String'(Dfy_File);
                           Build_Ret := Run_Command_Capture ("dafny", Build_Args, Build_Log);
                           Free (Build_Args (1));
                           Free (Build_Args (2));
                           Free (Build_Args (3));
                           Free (Build_Args (4));

                           if Build_Ret = 0 then
                              --  Search for the compiled output file
                              declare
                                 Out_Ext : constant String :=
                                   (if To_String (Target) = "js" then ".js"
                                    elsif To_String (Target) = "cs" then ".cs"
                                    elsif To_String (Target) = "go" then ".go"
                                    elsif To_String (Target) = "java" then ".java"
                                    elsif To_String (Target) = "py" then ".py"
                                    else ".js");
                                 Out_File : constant String := "obj/dafny_" & Suffix & Out_Ext;
                                 JS_Dir_File : constant String := "obj/dafny_" & Suffix & "-js/index.js";
                              begin
                                 if Ada.Directories.Exists (Out_File) then
                                    declare
                                       Result_Code : constant String := Read_File_Content (Out_File);
                                    begin
                                       --  Clean up all files
                                       if Ada.Directories.Exists (Dfy_File) then Ada.Directories.Delete_File (Dfy_File); end if;
                                       if Ada.Directories.Exists (Log_File) then Ada.Directories.Delete_File (Log_File); end if;
                                       if Ada.Directories.Exists (Build_Log) then Ada.Directories.Delete_File (Build_Log); end if;
                                       if Ada.Directories.Exists (Out_File) then Ada.Directories.Delete_File (Out_File); end if;
                                       return Result_Code;
                                    end;
                                 elsif To_String (Target) = "js" and then Ada.Directories.Exists (JS_Dir_File) then
                                    declare
                                       Result_Code : constant String := Read_File_Content (JS_Dir_File);
                                    begin
                                       --  Clean up files/directory
                                       if Ada.Directories.Exists (Dfy_File) then Ada.Directories.Delete_File (Dfy_File); end if;
                                       if Ada.Directories.Exists (Log_File) then Ada.Directories.Delete_File (Log_File); end if;
                                       if Ada.Directories.Exists (Build_Log) then Ada.Directories.Delete_File (Build_Log); end if;
                                       if Ada.Directories.Exists (JS_Dir_File) then Ada.Directories.Delete_File (JS_Dir_File); end if;
                                       return Result_Code;
                                    end;
                                 else
                                    Last_Errors := To_Unbounded_String ("Compilation succeeded but output file not found in obj.");
                                 end if;
                              end;
                           else
                              Last_Errors := To_Unbounded_String ("Compilation failed:" & ASCII.LF & Read_File_Content (Build_Log));
                           end if;
                        end;
                     else
                        Last_Errors := To_Unbounded_String (Verify_Out);
                         Put_Line (AnsiAda.Background (AnsiAda.Red)
                            & "[BUGCHECK] [!] Dafny Phase: Verification FAILED."
                            & AnsiAda.Reset);
                     end if;
                  end;
               end;
            end;
         end;

         <<Continue>>
         null;
      end loop;

      --  Final cleanup of temp files if they still exist
      if Ada.Directories.Exists (Dfy_File) then Ada.Directories.Delete_File (Dfy_File); end if;
      if Ada.Directories.Exists (Log_File) then Ada.Directories.Delete_File (Log_File); end if;
      if Ada.Directories.Exists (Build_Log) then Ada.Directories.Delete_File (Build_Log); end if;

      return "Failed to verify Dafny code after 5 attempts. Errors:" & ASCII.LF & To_String (Last_Errors);
   end Verify_And_Compile_Dafny;

end Verification_Manager;
