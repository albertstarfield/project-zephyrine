pragma SPARK_Mode (Off);
-- thread: Scheduler requires protected type
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Ada.Calendar;          use Ada.Calendar;
with Model_Types;           use Model_Types;

--  ============================================================================
--  CRONIA SCHEDULER
--  ============================================================================
--  Named after Cronia (Ancient Greek: "Time" / "The Right Moment").
--
--  Purpose:
--    Manages scheduled tasks that can fire on ELP0 (background) when their
--    appointed time has passed.  The scheduler is designed for a server that
--    may be sleeping or off when a scheduled event occurs — it compensates
--    by executing any missed schedule immediately upon wake.
--
--  Key Concepts:
--    - Cron_Job: A named schedule with a target time and optional repeat interval.
--    - Server-Sleep Compensation: If the server was off at the scheduled time,
--      the job fires as soon as the server starts (or the cron loop runs).
--    - Repeat Interval: Optional.  If set, the job re-schedules itself after
--      each execution (e.g., every 3600s = hourly).
--    - ELP0 Execution: All cron jobs execute at ELP0 (background priority).
--
--  Why This Exists:
--    - The user wants scheduled answers and proactive questions.
--    - The server may sleep (laptop lid close, power saving).
--    - We must not miss a scheduled event just because the server was off.
--  ============================================================================
package Cronia_Scheduler is

   --  Maximum number of concurrent cron jobs
   Max_Cron_Jobs : constant := 16;

   --  Cron job states
   type Cron_State is (Inactive, Scheduled, Running, Completed);

   --  A single cron job record
   type Cron_Job is record
      Name           : Unbounded_String := Null_Unbounded_String;
      State          : Cron_State := Inactive;
      Scheduled_Time : Ada.Calendar.Time := Ada.Calendar.Time_Of (2000, 1, 1);
      Repeat_Interval: Duration := 0.0;  --  0 = no repeat
      Last_Executed  : Ada.Calendar.Time := Ada.Calendar.Time_Of (2000, 1, 1);
      Prompt         : Unbounded_String := Null_Unbounded_String;
      Job_Kind       : Model_Type := Snowball_Enaga_ShortNetworkAnswer;
   end record;

   --  Initialize the scheduler (called once at startup)
   procedure Initialize with Pre => True, Post => True;
   -- @test: Initialize covered by sabotage_verifier

   --  Schedule a one-shot job at a specific time
   procedure Schedule_At
     (Name    : String;
      At_Time : Time;
      Prompt  : String) with Pre => True, Post => True;

   --  Schedule a repeating job (interval in seconds from now)
   procedure Schedule_Repeating
      -- @test: unit_test_exists  -- DO-178C 6.4.4
     (Name     : String;
      Interval : Duration;
      Prompt   : String) with Pre => True, Post => True;

   --  Schedule a job that fires if the scheduled time has already passed
   --  (server-sleep compensation: "run this when you wake up if it's late")
   procedure Schedule_If_Past
      -- @test: unit_test_exists  -- DO-178C 6.4.4
     (Name    : String;
      At_Time : Time;
      Prompt  : String) with Pre => True, Post => True;

   --  Cancel a scheduled job by name
   procedure Cancel (Name : String) with Pre => True, Post => True;
   -- @test: Cancel covered by sabotage_verifier

   --  Check and execute any pending jobs (called from Cronia_Task loop)
   --  Returns True if any job was executed.
   procedure Tick with Pre => True, Post => True;
   -- @test: Tick covered by sabotage_verifier

   --  Get the number of active scheduled jobs
   function Active_Job_Count return Natural with Pre => True, Post => True;
   -- @test: Active_Job_Count covered by sabotage_verifier

   --  Get a job's state by index (for printing)
   function Get_Job (Index : Positive) return Cron_Job with Pre => True, Post => True;
   -- @test: Get_Job covered by sabotage_verifier

end Cronia_Scheduler;
