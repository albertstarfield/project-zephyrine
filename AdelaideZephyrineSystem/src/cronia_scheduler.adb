pragma SPARK_Mode (Off);
with Ada.Text_IO;           use Ada.Text_IO;
with Ada.Strings;           use Ada.Strings;
with Ada.Strings.Fixed;     use Ada.Strings.Fixed;
with Ada.Real_Time;
with Ada.Calendar;          use Ada.Calendar;
with Ada.Exceptions;
with Model_Manager;
with AnsiAda;
with Kokoro_Interface;
with Proactive_Engine;
with Ada.Streams;

package body Cronia_Scheduler is

   use type Ada.Real_Time.Time;

   --  Job storage
   Jobs       : array (1 .. Max_Cron_Jobs) of Cron_Job;
   Job_Count  : Natural := 0;
   Init_Time  : Ada.Real_Time.Time;

   --  Elapsed time since init (for logging)
   function "+" (Left : Ada.Calendar.Time; Right : Duration) return Ada.Calendar.Time renames Ada.Calendar."+";
   function "-" (Left : Ada.Calendar.Time; Right : Duration) return Ada.Calendar.Time renames Ada.Calendar."-";
   function "-" (Left, Right : Ada.Calendar.Time) return Duration renames Ada.Calendar."-";

   function Uptime return Duration is
   begin
      return Ada.Real_Time.To_Duration (Ada.Real_Time.Clock - Init_Time);
   end Uptime;

   procedure Initialize is
   begin
      Init_Time := Ada.Real_Time.Clock;
      Job_Count := 0;
      Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Cronia]" &
                AnsiAda.Reset & " Scheduler initialized.");
   end Initialize;

   --  Find a job by name, return index or 0 if not found
   function Find_Job (Name : String) return Natural is
   begin
      for I in 1 .. Job_Count loop
         if To_String (Jobs (I).Name) = Name then
            return I;
         end if;
      end loop;
      return 0;
   end Find_Job;

   --  Add a new job to the array
   procedure Add_Job (Job : Cron_Job) is
   begin
      if Job_Count < Max_Cron_Jobs then
         Job_Count := Job_Count + 1;
         Jobs (Job_Count) := Job;
         Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Cronia]" &
                   AnsiAda.Reset & " Scheduled: " & To_String (Job.Name));
      else
         Put_Line (AnsiAda.Foreground (AnsiAda.Yellow) & "[Cronia]" &
                   AnsiAda.Reset & " WARNING: Max cron jobs reached, cannot add: " & To_String (Job.Name));
      end if;
   end Add_Job;

   procedure Schedule_At (Name : String; At_Time : Ada.Calendar.Time; Prompt : String) is
      New_Job : Cron_Job;
   begin
      New_Job.Name            := To_Unbounded_String (Name);
      New_Job.State           := Scheduled;
      New_Job.Scheduled_Time  := At_Time;
      New_Job.Repeat_Interval := 0.0;
      New_Job.Prompt          := To_Unbounded_String (Prompt);
      Add_Job (New_Job);
   end Schedule_At;

   procedure Schedule_Repeating (Name : String; Interval : Duration; Prompt : String) is
      New_Job : Cron_Job;
   begin
      New_Job.Name            := To_Unbounded_String (Name);
      New_Job.State           := Scheduled;
      New_Job.Scheduled_Time  := Ada.Calendar.Clock + Interval;
      New_Job.Repeat_Interval := Interval;
      New_Job.Prompt          := To_Unbounded_String (Prompt);
      Add_Job (New_Job);
   end Schedule_Repeating;

   procedure Schedule_If_Past (Name : String; At_Time : Time; Prompt : String) is
      New_Job : Cron_Job;
   begin
      New_Job.Name            := To_Unbounded_String (Name);
      New_Job.Repeat_Interval := 0.0;
      New_Job.Prompt          := To_Unbounded_String (Prompt);

      --  Server-sleep compensation: if scheduled time already passed,
      --  mark as Scheduled so Tick() fires it immediately.
      if Ada.Calendar.Clock >= At_Time then
         Put_Line (AnsiAda.Foreground (AnsiAda.Yellow) & "[Cronia]" &
                   AnsiAda.Reset & " Schedule_If_Past: time already passed for " &
                   Name & ", will fire on next Tick.");
         New_Job.State          := Scheduled;
         New_Job.Scheduled_Time := At_Time;  --  Keep original for record
      else
         New_Job.State          := Scheduled;
         New_Job.Scheduled_Time := At_Time;
      end if;

      Add_Job (New_Job);
   end Schedule_If_Past;

   procedure Cancel (Name : String) is
      Idx : constant Natural := Find_Job (Name);
   begin
      if Idx > 0 then
         --  Shift remaining jobs down
         for I in Idx .. Job_Count - 1 loop
            Jobs (I) := Jobs (I + 1);
         end loop;
         Jobs (Job_Count) := (others => <>);
         Job_Count := Job_Count - 1;
         Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Cronia]" &
                   AnsiAda.Reset & " Cancelled: " & Name);
      end if;
   end Cancel;

   procedure Tick is
      Now : constant Time := Ada.Calendar.Clock;
   begin
      for I in 1 .. Job_Count loop
         if Jobs (I).State = Scheduled and then Now >= Jobs (I).Scheduled_Time then
            Jobs (I).State := Running;
            Put_Line (AnsiAda.Foreground (AnsiAda.Green) & "[Cronia]" &
                      AnsiAda.Reset & " FIRING: " & To_String (Jobs (I).Name) &
                      " | Uptime=" & Duration'Image (Uptime) & "s");

            --  Execute the job via Model_Manager.Hybrid_Generate at ELP0
            declare
               Result : Unbounded_String;
            begin
               Model_Manager.Hybrid_Generate
                 (Prompt => To_String (Jobs (I).Prompt),
                  Result => Result,
                  Level  => ELP0,
                  Agentic => True,
                  Raw_Prompt => True);

               if Length (Result) > 0 then
                  declare
                     PCM_Data : constant Ada.Streams.Stream_Element_Array :=
                       Kokoro_Interface.Synthesize_Speech (To_String (Result));
                  begin
                     if PCM_Data'Length > 0 then
                        declare
                           Result_Str : String (1 .. Natural (PCM_Data'Length));
                        begin
                           for I in PCM_Data'Range loop
                              Result_Str (Natural (I) - Natural (PCM_Data'First) + 1) := Character'Val (PCM_Data (I));
                           end loop;
                           Proactive_Engine.Queue_Audio (Result_Str);
                        end;
                     end if;
                  end;
               end if;
            exception
               when E : others =>
                  Put_Line (AnsiAda.Foreground (AnsiAda.Red) & "[Cronia]" &
                            AnsiAda.Reset & " ERROR executing " &
                            To_String (Jobs (I).Name) & ": " &
                            Ada.Exceptions.Exception_Message (E));
            end;

            Jobs (I).Last_Executed := Now;

            --  Handle repeat or mark completed
            if Jobs (I).Repeat_Interval > 0.0 then
               Jobs (I).Scheduled_Time := Now + Jobs (I).Repeat_Interval;
               Jobs (I).State := Scheduled;
               Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Cronia]" &
                         AnsiAda.Reset & " Re-scheduled: " & To_String (Jobs (I).Name) &
                         " in " & Duration'Image (Jobs (I).Repeat_Interval) & "s");
            else
               Jobs (I).State := Completed;
               Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Cronia]" &
                         AnsiAda.Reset & " Completed: " & To_String (Jobs (I).Name));
            end if;
         end if;
      end loop;
   end Tick;

   function Active_Job_Count return Natural is
      Count : Natural := 0;
   begin
      for I in 1 .. Job_Count loop
         if Jobs (I).State = Scheduled then
            Count := Count + 1;
         end if;
      end loop;
      return Count;
   end Active_Job_Count;

   function Get_Job (Index : Positive) return Cron_Job is
   begin
      if Index <= Job_Count then
         return Jobs (Index);
      else
         return (others => <>);
      end if;
   end Get_Job;

end Cronia_Scheduler;
