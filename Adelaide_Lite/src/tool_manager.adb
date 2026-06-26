pragma SPARK_Mode (Off);
with AnsiAda;
with Ada.Text_IO; use Ada.Text_IO;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with GNAT.OS_Lib;
with GNAT.Expect;
with SD_Manager;

package body Tool_Manager is

   function Execute_Tool (Name : String; Params : String) return Tool_Result is
      use GNAT.OS_Lib;
      Path : GNAT.OS_Lib.String_Access;
      Full_Cmd : Unbounded_String;
      Result : Tool_Result := (Success => False,
                               Output  => Null_Unbounded_String);
   begin
      Path := GNAT.OS_Lib.Locate_Exec_On_Path ("python3");
      if Path = null then
         Result.Output := To_Unbounded_String ("Error: python3 not found");
         return Result;
      end if;

      Put_Line (AnsiAda.Foreground (AnsiAda.Yellow) & "[Tool]" & AnsiAda.Reset &
                " Executing: " & Name & " with params: " & Params);

      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
      --  Tool routing: Maps tool names to Python scripts.
      --  All tools are in python/ directory relative to the server binary.
      if Name = "web_search" or else Name = "searchglobalref" or else Name = "search" then
         Full_Cmd := To_Unbounded_String ("python/searchglobalref.py");
      elsif Name = "local_search" then
         Full_Cmd := To_Unbounded_String ("python/searchlocalref.py");
      elsif Name = "math" then
         Full_Cmd := To_Unbounded_String ("python/math_tool.py");
      elsif Name = "code" then
         Full_Cmd := To_Unbounded_String ("python/code_tool.py");
      elsif Name = "cat" then
         Full_Cmd := To_Unbounded_String ("python/cat_tool.py");
      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
      --  NEW TOOLS: Git, File Edit, Directory, Test, Build, Issue, Review, Security, Hook
      elsif Name = "git" then
         Full_Cmd := To_Unbounded_String ("python/git.py");
      elsif Name = "file_edit" or else Name = "edit" or else Name = "write" then
         Full_Cmd := To_Unbounded_String ("python/file_edit.py");
      elsif Name = "dir" or else Name = "ls" or else Name = "find" or else Name = "tree" then
         Full_Cmd := To_Unbounded_String ("python/directory.py");
      elsif Name = "test" or else Name = "pytest" or else Name = "lint" then
         Full_Cmd := To_Unbounded_String ("python/test.py");
      elsif Name = "build" or else Name = "make" or else Name = "compile" then
         Full_Cmd := To_Unbounded_String ("python/build.py");
      elsif Name = "issue" or else Name = "gh" then
         Full_Cmd := To_Unbounded_String ("python/issue.py");
      elsif Name = "review" or else Name = "code_review" then
         Full_Cmd := To_Unbounded_String ("python/review.py");
      elsif Name = "security" or else Name = "scan" then
         Full_Cmd := To_Unbounded_String ("python/security.py");
      elsif Name = "hook" then
         Full_Cmd := To_Unbounded_String ("python/hook.py");
      else
         Free (Path);
         Result.Output := To_Unbounded_String ("Error: Unknown tool " & Name);
         return Result;
      end if;

      declare
         Arg_List : Argument_List (1 .. 2);
         Status : aliased Integer;
      begin
         Arg_List (1) := new String'(To_String (Full_Cmd));
         Arg_List (2) := new String'(Params);
         
         Result.Output := To_Unbounded_String
           (GNAT.Expect.Get_Command_Output (Path.all, Arg_List, "",
            Status'Access));

         for I in Arg_List'Range loop Free (Arg_List (I)); end loop;
         Free (Path);
         
         Result.Success := True;
         return Result;
      end;
   exception
      when others =>
         if Path /= null then Free (Path); end if;
         Result.Output := To_Unbounded_String ("Error executing tool");
         return Result;
   end Execute_Tool;

   --  ============================================================================
   --  IMAGINE TOOL: Direct Ada call to SD_Manager (no Python sidecar)
   --  ============================================================================
   --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
   --  Called from Hybrid_Generate when the model outputs [ACTION: imagine(prompt)].
   --  Generates an image using the two-stage FLUX + SD refinement pipeline.
   --  Returns the Base64-encoded PNG as the tool output.

   function Execute_Imagine_Tool (Prompt : String) return Tool_Result is
      Image_B64 : Unbounded_String := Null_Unbounded_String;
      Error_Msg : Unbounded_String := Null_Unbounded_String;
      Result    : Tool_Result := (Success => False,
                                  Output  => Null_Unbounded_String);
   begin
      Put_Line (AnsiAda.Foreground (AnsiAda.Yellow) & "[Tool-Imagine]" &
                AnsiAda.Reset & " Generating image for: " &
                Prompt (Prompt'First .. Integer'Min (Prompt'First + 79, Prompt'Last)));

      SD_Manager.Generate_Two_Stage
        (Prompt         => Prompt,
         Width          => 1024,
         Height         => 1024,
         Seed           => -1,
         Flux_Steps     => 4,
         Flux_Cfg       => 1.0,
         Refine_Enabled => True,
         Refine_Steps   => 8,
         Refine_Strength => 0.4,
         Image_B64      => Image_B64,
         Error_Msg      => Error_Msg);

      if Length (Error_Msg) > 0 then
         Put_Line (AnsiAda.Foreground (AnsiAda.Red) & "[Tool-Imagine] ERROR: " &
                   AnsiAda.Reset & To_String (Error_Msg));
         Result.Output := To_Unbounded_String ("Error: " & To_String (Error_Msg));
         return Result;
      end if;

      if Length (Image_B64) > 0 then
         Put_Line (AnsiAda.Foreground (AnsiAda.Green) & "[Tool-Imagine]" &
                   AnsiAda.Reset & " Image generated. Base64 length=" &
                   Integer'Image (Length (Image_B64)));
         Result.Success := True;
         Result.Output := Image_B64;
      else
         Result.Output := To_Unbounded_String ("Error: Image generation returned empty");
      end if;

      return Result;
   end Execute_Imagine_Tool;

end Tool_Manager;
