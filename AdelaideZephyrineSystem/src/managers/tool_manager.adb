pragma SPARK_Mode (Off);
-- thread: Tool execution requires task protection
with AnsiAda;
with Ada.Text_IO; use Ada.Text_IO;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Ada.Strings; use Ada.Strings;
with Ada.Strings.Fixed; use Ada.Strings.Fixed;
with GNAT.OS_Lib;
with GNAT.Expect;
with SD_Manager;
with Cronia_Scheduler;
with Proactive_Engine;
with Ada.Calendar; use Ada.Calendar;
with Ada.Calendar.Formatting;
with Adelaide_Trace;
with Zenith_Orion;

package body Tool_Manager is

   --  ------------------------------------------------------------------------
   --  ASYNC TOOL EXECUTION TASK
   --  ------------------------------------------------------------------------
   --  Spawns a Python tool subprocess in a background Ada task so the caller
   --  can poll for completion with a configurable heartbeat (every 30 s).
   --
   --  Usage pattern in Execute_Tool:
   --
   --     declare
   --        task Runner is
   --           entry Get_Result (Output : out Unbounded_String;
   --                             Status : out Integer);
   --        end Runner;
   --        task body Runner is ... end Runner;
   --     begin
   --        loop
   --           select
   --              Runner.Get_Result (Result.Output, Ex_Status);
   --              Result.Success := (Ex_Status = 0);
   --              exit;
   --           or
   --              delay 30.0;
   --              Adelaide_Trace.Trace_Print (Name, "STILL_RUNNING",
   --                "elapsed: " & Integer'Image (Adelaide_Trace.Uptime) & "s");
   --           end select;
   --        end loop;
   --     end;
   --  ------------------------------------------------------------------------

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

      Adelaide_Trace.Trace_Print (Toolcall => "dispatch:" & Name,
        Message => "params: " & Params);

--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
      --  Tool routing: Maps tool names to Python scripts.
      --  All tools are in python/ directory relative to the server binary.
      if Name = "web_search" or else Name = "searchglobalref" or else Name = "search" then
         Full_Cmd := To_Unbounded_String ("src/python/searchglobalref.py");
      elsif Name = "local_search" then
         Full_Cmd := To_Unbounded_String ("src/python/searchlocalref.py");
      elsif Name = "math" then
         Full_Cmd := To_Unbounded_String ("src/python/math_tool.py");
      elsif Name = "code" then
         Full_Cmd := To_Unbounded_String ("src/python/code_tool.py");
      elsif Name = "cat" then
         Full_Cmd := To_Unbounded_String ("src/python/cat_tool.py");
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
      --  NEW TOOLS: Git, File Edit, Directory, Test, Build, Issue, Review, Security, Hook
      elsif Name = "git" then
         Full_Cmd := To_Unbounded_String ("src/python/git.py");
      elsif Name = "file_edit" or else Name = "edit" or else Name = "write" then
         Full_Cmd := To_Unbounded_String ("src/python/file_edit.py");
      elsif Name = "dir" or else Name = "ls" or else Name = "find" or else Name = "tree" then  --  MC/DC: each sub-expression independently toggles decision
         Full_Cmd := To_Unbounded_String ("src/python/directory.py");
      elsif Name = "test" or else Name = "pytest" or else Name = "lint" then
         Full_Cmd := To_Unbounded_String ("src/python/test.py");
      elsif Name = "build" or else Name = "make" or else Name = "compile" then
         Full_Cmd := To_Unbounded_String ("src/python/build.py");
      elsif Name = "issue" or else Name = "gh" then
         Full_Cmd := To_Unbounded_String ("src/python/issue.py");
      elsif Name = "review" or else Name = "code_review" then
         Full_Cmd := To_Unbounded_String ("src/python/review.py");
      elsif Name = "security" or else Name = "scan" then
         Full_Cmd := To_Unbounded_String ("src/python/security.py");
      elsif Name = "hook" then
         Full_Cmd := To_Unbounded_String ("src/python/hook.py");
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
      --  PACKAGE TOOL: Install system packages (apt, brew, pacman, winget, etc.)
      elsif Name = "package" or else Name = "install" or else Name = "pkg" then
         Full_Cmd := To_Unbounded_String ("src/python/package.py");
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
      --  GREP TOOL: Search file contents (agentic code tool)
      elsif Name = "grep" or else Name = "search_content" then
         Full_Cmd := To_Unbounded_String ("src/python/grep.py");
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
      --  TODO TOOL: Task management (agentic code tool)
      elsif Name = "todo" or else Name = "task" then
         Full_Cmd := To_Unbounded_String ("src/python/todo.py");
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
      --  KILLSHELL TOOL: Process management (agentic code tool)
      elsif Name = "kill" or else Name = "killshell" or else Name = "process" then
         Full_Cmd := To_Unbounded_String ("src/python/killshell.py");
      --  CRONIA TOOL: Schedule timed answers (native Ada)
      elsif Name = "cronia" or else Name = "timed_cronia_answer" or else Name = "schedule_answer" then
         Free (Path);
         return Execute_Cronia_Tool (Params);
      --  PROACTIVE TOOL: Handless mode and proactive questions (native Ada)
      elsif Name = "proactive" or else Name = "proactive_question" or else Name = "handless" then
         Free (Path);
         return Execute_Proactive_Tool (Params);
      --  ROS2 TOOL: Actuate and Telemetry trigger
      elsif Name = "ros2" or else Name = "actuator" then
         Free (Path);
         return Execute_ROS2_Tool (Params);
      else
         Free (Path);
         Result.Output := To_Unbounded_String ("Error: Unknown tool " & Name);
         return Result;
      end if;

      --  Async execution with 30 s heartbeat
      declare
         Cmd_Str  : constant String := To_String (Full_Cmd);
         Params_Str : constant String := Params;

         task Runner is
            entry Get_Result (Output : out Unbounded_String;
                              Status : out Integer);
         end Runner;

         task body Runner is
            use GNAT.OS_Lib;
            use GNAT.Expect;
            Local_Args : Argument_List (1 .. 2);
            Ex_Status  : aliased Integer;
         begin
            Local_Args (1) := new String'(Cmd_Str);
            Local_Args (2) := new String'(Params_Str);

            declare
               Out_Str : constant String :=
                 Get_Command_Output (Cmd_Str, Local_Args, "",
                                     Ex_Status'Access);
            begin
               accept Get_Result (Output : out Unbounded_String;
                                  Status : out Integer) do
                  Output := To_Unbounded_String (Out_Str);
                  Status := Ex_Status;
               end Get_Result;
            end;

            for I in Local_Args'Range loop
               Free (Local_Args (I));
            end loop;
         end Runner;

         Status : aliased Integer;
         Heartbeat_Count : Natural := 0;
      begin
         --  Wait loop with 30 s heartbeat
         loop
            select
               Runner.Get_Result (Result.Output, Status);
               Adelaide_Trace.Trace_Result (Name,
                 Success => (Status = 0),
                 Detail  => "duration: " &
                   Integer'Image (Adelaide_Trace.Uptime) & "s" &
                   " exit_code: " & Integer'Image (Status));
               Result.Success := (Status = 0);
               exit;
            or
               delay 30.0;
               Heartbeat_Count := Heartbeat_Count + 1;
               Adelaide_Trace.Trace_Print (Name, "STILL_RUNNING",
                 "heartbeat #" & Natural'Image (Heartbeat_Count) &
                 " elapsed: " & Integer'Image (Adelaide_Trace.Uptime) & "s");
            end select;
         end loop;

         Free (Path);
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
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
   --  Called from Hybrid_Generate when the model outputs [ACTION: imagine(prompt)].
   --  Generates an image using the two-stage FLUX + SD refinement pipeline.
   --  Returns the Base64-encoded PNG as the tool output.

   function Execute_Imagine_Tool (Prompt : String) return Tool_Result is
      Image_B64 : Unbounded_String := Null_Unbounded_String;
      Error_Msg : Unbounded_String := Null_Unbounded_String;
      Result    : Tool_Result := (Success => False,
                                   Output  => Null_Unbounded_String);
      Truncated_Prompt : constant String :=
        Prompt (Prompt'First .. Integer'Min (Prompt'First + 79, Prompt'Last));
   begin
      Adelaide_Trace.Trace_Print ("imagine", "generating",
        "prompt: """ & Truncated_Prompt & """");

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
         Adelaide_Trace.Trace_Print ("imagine", "error",
           To_String (Error_Msg));
         Result.Output := To_Unbounded_String ("Error: " & To_String (Error_Msg));
         return Result;
      end if;

      if Length (Image_B64) > 0 then
         Adelaide_Trace.Trace_Result ("imagine", Success => True,
           Detail => "Base64 length=" & Integer'Image (Length (Image_B64)));
         Result.Success := True;
         Result.Output := Image_B64;
      else
         Adelaide_Trace.Trace_Result ("imagine", Success => False,
           Detail => "image generation returned empty");
         Result.Output := To_Unbounded_String ("Error: Image generation returned empty");
      end if;

      return Result;
   end Execute_Imagine_Tool;

   --  ============================================================================
   --  CRONIA TOOL: Schedule a timed answer on ELP0
   --  ============================================================================
   --  Params format: "name|time_iso|prompt" for one-shot
   --                 "name|repeat_seconds|prompt" for repeating
   --                 "cancel|name" to cancel a job
   --  Example: "weather_check|2026-06-27T08:00:00|What's the weather today?"
   --           "hourly_reminder|3600|Check on the user"
   --  ============================================================================
   function Execute_Cronia_Tool (Params : String) return Tool_Result is
      Result : Tool_Result := (Success => False, Output => Null_Unbounded_String);
      Sep_Pos : Natural;
      Name    : Unbounded_String;
      Rest    : Unbounded_String;
   begin
      Adelaide_Trace.Trace_Print (Toolcall => "cronia",
        Message => "params: " & Params);

      --  Parse: "name|rest"
      Sep_Pos := Index (Params, "|");
      if Sep_Pos = 0 then
         Result.Output := To_Unbounded_String ("Error: Invalid format. Use: name|time_or_repeat|prompt");
         return Result;
      end if;

      Name := To_Unbounded_String (Params (Params'First .. Sep_Pos - 1));
      Rest := To_Unbounded_String (Params (Sep_Pos + 1 .. Params'Last));

      --  Check for cancel command
      if To_String (Name) = "cancel" then
         Cronia_Scheduler.Cancel (To_String (Rest));
         Result.Success := True;
         Result.Output := To_Unbounded_String ("Cancelled: " & To_String (Rest));
         return Result;
      end if;

      declare
         Rest_Str      : constant String := To_String (Rest);
         Sep_Pos2      : Natural;
         Time_Or_Repeat: Unbounded_String;
         Prompt_Str    : Unbounded_String;
      begin
         --  Parse second separator: "time_or_repeat|prompt"
         Sep_Pos2 := Index (Rest_Str, "|");
         if Sep_Pos2 = 0 then
            Result.Output := To_Unbounded_String ("Error: Missing prompt. Use: name|time_or_repeat|prompt");
            return Result;
         end if;

         Time_Or_Repeat := To_Unbounded_String (Rest_Str (Rest_Str'First .. Sep_Pos2 - 1));
         Prompt_Str     := To_Unbounded_String (Rest_Str (Sep_Pos2 + 1 .. Rest_Str'Last));

         --  Try to parse as repeat interval (numeric seconds)
         begin
            declare
               Repeat_Secs : constant Duration := Duration'Value (To_String (Time_Or_Repeat));
            begin
               Cronia_Scheduler.Schedule_Repeating
                 (Name     => To_String (Name),
                  Interval => Repeat_Secs,
                  Prompt   => To_String (Prompt_Str));
               Result.Success := True;
               Result.Output := To_Unbounded_String (
                 "Scheduled repeating job: " & To_String (Name) &
                 " every " & Duration'Image (Repeat_Secs) & "s");
            end;
         exception
            when others =>
               --  Try to parse as ISO time
               begin
                  declare
                     Target_Time : constant Time := Ada.Calendar.Formatting.Value (To_String (Time_Or_Repeat));
                  begin
                     Cronia_Scheduler.Schedule_If_Past
                       (Name    => To_String (Name),
                        At_Time => Target_Time,
                        Prompt  => To_String (Prompt_Str));
                     Result.Success := True;
                     Result.Output := To_Unbounded_String (
                       "Scheduled job: " & To_String (Name) &
                       " at " & To_String (Time_Or_Repeat) &
                       " (server-sleep compensation enabled)");
                  end;
               exception
                  when others =>
                     Result.Output := To_Unbounded_String (
                       "Error: Cannot parse time '" & To_String (Time_Or_Repeat) &
                       "'. Use ISO format (2026-06-27T08:00:00) or seconds (3600).");
               end;
         end;
      end;

      return Result;
   end Execute_Cronia_Tool;

   --  ============================================================================
   --  PROACTIVE TOOL: Handless mode and proactive questions
   --  ============================================================================
   --  Params format: "activate_handless" to enable handless mode
   --                 "deactivate_handless" to disable
   --                 "acoustic_trigger" to fire acoustic curiosity
   --                 "schedule_question|time_iso|topic" to schedule a question
   --  ============================================================================
   function Execute_Proactive_Tool (Params : String) return Tool_Result is
      Result : Tool_Result := (Success => False, Output => Null_Unbounded_String);
   begin
      Adelaide_Trace.Trace_Print (Toolcall => "proactive",
        Message => "params: " & Params);

      if Params = "activate_handless" then
         Proactive_Engine.Activate_Handless_Mode;
         Result.Success := True;
         Result.Output := To_Unbounded_String ("Handless mode activated. Adelaide will greet you!");

      elsif Params = "deactivate_handless" then
         Proactive_Engine.Deactivate_Handless_Mode;
         Result.Success := True;
         Result.Output := To_Unbounded_String ("Handless mode deactivated.");

      elsif Params = "acoustic_trigger" then
         Proactive_Engine.Trigger_Acoustic_Question;
         Result.Success := True;
         Result.Output := To_Unbounded_String ("Acoustic curiosity triggered.");

      elsif Index (Params, "|") > 0 then
         --  Parse: "schedule_question|time_iso|topic"
         declare
            Sep1  : constant Natural := Index (Params, "|");
            Sep2  : constant Natural := Index (Params (Sep1 + 1 .. Params'Last), "|");
         begin
            if Sep2 = 0 then
               Result.Output := To_Unbounded_String ("Error: Use: schedule_question|time_iso|topic");
               return Result;
            end if;

            declare
               Command   : constant String := Params (Params'First .. Sep1 - 1);
               Time_Str  : constant String := Params (Sep1 + 1 .. Sep2 - 1);
               Topic     : constant String := Params (Sep2 + 1 .. Params'Last);
            begin
               if Command = "schedule_question" then
                  declare
                     Target_Time : constant Time := Ada.Calendar.Formatting.Value (Time_Str);
                  begin
                     Proactive_Engine.Schedule_Question (At_Time => Target_Time, Topic => Topic);
                     Result.Success := True;
                     Result.Output := To_Unbounded_String (
                       "Question scheduled at " & Time_Str & " Topic: " & Topic);
                  end;
               elsif Command = "repeat_question" then
                  declare
                     Interval : constant Duration := Duration'Value (Time_Str);
                  begin
                     Proactive_Engine.Schedule_Repeating_Question (Interval => Interval, Topic => Topic);
                     Result.Success := True;
                     Result.Output := To_Unbounded_String (
                       "Repeating question every " & Duration'Image (Interval) & "s Topic: " & Topic);
                  end;
               else
                  Result.Output := To_Unbounded_String ("Error: Unknown command " & Command);
               end if;
            end;
         end;

      else
         Result.Output := To_Unbounded_String (
           "Error: Unknown proactive command. Use: activate_handless, deactivate_handless, " &
           "acoustic_trigger, schedule_question|time|topic, repeat_question|seconds|topic");
      end if;

      return Result;
   end Execute_Proactive_Tool;

   --  ROS2 TOOL: Trigger native Ada ROS2 actuator via ELP3
   --  Params format: "servo_id|angle"
   function Execute_ROS2_Tool (Params : String) return Tool_Result is
      Result : Tool_Result := (Success => False, Output => Null_Unbounded_String);
      Pipe_Idx : Natural := Index (Params, "|");
   begin
      if Pipe_Idx = 0 or else Pipe_Idx = Params'First or else Pipe_Idx = Params'Last then
         Result.Output := To_Unbounded_String ("Error: Invalid ROS2 tool parameters. Expected 'servo_id|angle'.");
         return Result;
      end if;

      declare
         Servo_ID : constant String := Params (Params'First .. Pipe_Idx - 1);
         Angle_Str : constant String := Params (Pipe_Idx + 1 .. Params'Last);
         Angle : Float;
      begin
         Angle := Float'Value (Angle_Str);
         -- Push to ZenithOrion Buffer for ELP3 execution
         Zenith_Orion.ROS2_Command_Buffer.Push_Command (Servo_ID, Angle);

         Result.Success := True;
         Result.Output := To_Unbounded_String ("ROS2 Command pushed to fast-path buffer successfully.");
      exception
         when others =>
            Result.Output := To_Unbounded_String ("Error: Could not parse Angle as Float.");
            return Result;
      end;
   end Execute_ROS2_Tool;

end Tool_Manager;
