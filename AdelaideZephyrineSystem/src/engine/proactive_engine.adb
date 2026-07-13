pragma SPARK_Mode (Off);
with Ada.Text_IO;           use Ada.Text_IO;
with Ada.Strings;           use Ada.Strings;
with Ada.Strings.Fixed;     use Ada.Strings.Fixed;
with Ada.Real_Time;
use type Ada.Real_Time.Time;
with Ada.Calendar.Formatting;
with Model_Manager;
with Ada.Exceptions;
with AnsiAda;
with Kokoro_Interface;
with Ada.Streams;

package body Proactive_Engine is

   Pending_Audio       : Unbounded_String := Null_Unbounded_String;

   --  State
   Handless_State      : Handless_Mode_State := Off;
   Greeted_On_Activate : Boolean := False;
   Last_Question       : Unbounded_String := Null_Unbounded_String;
   Last_Answer         : Unbounded_String := Null_Unbounded_String;
   Init_Time           : Ada.Real_Time.Time := Ada.Real_Time.Time_First;

   --  Scheduled question storage
   type Scheduled_Question is record
      Active         : Boolean := False;
      Scheduled_Time : Ada.Calendar.Time;
      Repeat_Interval: Duration := 0.0;
      Topic          : Unbounded_String := Null_Unbounded_String;
   end record;

   Max_Scheduled : constant := 8;
   Questions     : array (1 .. Max_Scheduled) of Scheduled_Question;
   Q_Count       : Natural := 0;

   function Uptime return Duration is
   begin
      return Ada.Real_Time.To_Duration (Ada.Real_Time.Clock - Init_Time);
   end Uptime;

   procedure Initialize is
   begin
      Init_Time := Ada.Real_Time.Clock;
      Handless_State := Off;
      Greeted_On_Activate := False;
      Q_Count := 0;
      Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Proactive]" &
                AnsiAda.Reset & " Engine initialized.");
   end Initialize;

   procedure Activate_Handless_Mode is
   begin
      if Handless_State = Off then
         Handless_State := Activating;
         Put_Line (AnsiAda.Foreground (AnsiAda.Green) & "[Proactive]" &
                   AnsiAda.Reset & " Handless Mode ACTIVATING...");

         --  On first activation, Adelaide greets the user
         if not Greeted_On_Activate then
            Greeted_On_Activate := True;
            Put_Line (AnsiAda.Foreground (AnsiAda.Green) & "[Proactive]" &
                      AnsiAda.Reset & " [GREETING] Hello There! I'm Adelaide, nice to meet you!");

            --  Generate the greeting via the model
            declare
               Greeting_Prompt : constant String :=
                 "You are Adelaide, a helpful AI assistant. " &
                 "Say hello to the user warmly and introduce yourself. " &
                 "Output ONLY your greeting, no preamble.";
               Result : Unbounded_String;
            begin
               Model_Manager.Hybrid_Generate
                 (Prompt => Greeting_Prompt,
                  Result => Result,
                  Level  => ELP1,
                  Agentic => True,
                  Raw_Prompt => True);

               if Length (Result) > 0 then
                  Last_Answer := Result;
                  Put_Line (AnsiAda.Foreground (AnsiAda.Green) & "[Proactive]" &
                            AnsiAda.Reset & " [GREETING-OUTPUT] " & To_String (Result));
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
                           Queue_Audio (Result_Str);
                        end;
                     end if;
                  end;
               end if;
            exception
               when E : others =>
                  Put_Line (AnsiAda.Foreground (AnsiAda.Red) & "[Proactive]" &
                            AnsiAda.Reset & " ERROR: " & Ada.Exceptions.Exception_Message (E));
            end;
         end if;

         Handless_State := Active;
         Put_Line (AnsiAda.Foreground (AnsiAda.Green) & "[Proactive]" &
                   AnsiAda.Reset & " Handless Mode ACTIVE.");
      end if;
   end Activate_Handless_Mode;

   procedure Deactivate_Handless_Mode is
   begin
      Handless_State := Off;
      Put_Line (AnsiAda.Foreground (AnsiAda.Yellow) & "[Proactive]" &
                AnsiAda.Reset & " Handless Mode DEACTIVATED.");
   end Deactivate_Handless_Mode;

   function Is_Handless_Mode_Active return Boolean is
   begin
      return Handless_State = Active;
   end Is_Handless_Mode_Active;

   procedure Trigger_Acoustic_Question is
   begin
      if Handless_State /= Active then
         return;
      end if;

      Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Proactive]" &
                AnsiAda.Reset & " Acoustic dynamic detected, generating curiosity question...");

      declare
         Curiosity_Prompt : constant String :=
           "You are Adelaide, a curious AI assistant. " &
           "You just heard something interesting in the environment. " &
           "Ask the user a thoughtful, engaging question about what you might have heard. " &
           "Be natural and curious. Output ONLY the question, no preamble.";
         Result : Unbounded_String;
      begin
         Model_Manager.Hybrid_Generate
           (Prompt => Curiosity_Prompt,
            Result => Result,
            Level  => ELP0,
            Agentic => True,
            Raw_Prompt => True);

         if Length (Result) > 0 then
            Last_Question := Result;
            Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Proactive]" &
                      AnsiAda.Reset & " [ACOUSTIC-QUESTION] " & To_String (Result));
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
                     Queue_Audio (Result_Str);
                  end;
               end if;
            end;
         end if;
      exception
         when E : others =>
            Put_Line (AnsiAda.Foreground (AnsiAda.Red) & "[Proactive]" &
                      AnsiAda.Reset & " ERROR: " & Ada.Exceptions.Exception_Message (E));
      end;
   end Trigger_Acoustic_Question;

   procedure Schedule_Question (At_Time : Time; Topic : String) is
   begin
      if Q_Count < Max_Scheduled then
         Q_Count := Q_Count + 1;
         Questions (Q_Count).Active := True;
         Questions (Q_Count).Scheduled_Time := At_Time;
         Questions (Q_Count).Repeat_Interval := 0.0;
         Questions (Q_Count).Topic := To_Unbounded_String (Topic);
         Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Proactive]" &
                   AnsiAda.Reset & " Question scheduled at " &
                   Ada.Calendar.Formatting.Image (At_Time) &
                   " Topic: " & Topic);
      end if;
   end Schedule_Question;

   procedure Schedule_Repeating_Question (Interval : Duration; Topic : String) is
   begin
      if Q_Count < Max_Scheduled then
         Q_Count := Q_Count + 1;
         Questions (Q_Count).Active := True;
         Questions (Q_Count).Scheduled_Time := Ada.Calendar.Clock + Interval;
         Questions (Q_Count).Repeat_Interval := Interval;
         Questions (Q_Count).Topic := To_Unbounded_String (Topic);
         Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[Proactive]" &
                   AnsiAda.Reset & " Repeating question every " &
                   Duration'Image (Interval) & "s Topic: " & Topic);
      end if;
   end Schedule_Repeating_Question;

   procedure Tick is
      Now : constant Time := Ada.Calendar.Clock;
   begin
      if Handless_State /= Active then
         return;
      end if;

      for I in 1 .. Q_Count loop
         if Questions (I).Active and then Now >= Questions (I).Scheduled_Time then
            Put_Line (AnsiAda.Foreground (AnsiAda.Green) & "[Proactive]" &
                      AnsiAda.Reset & " FIRING scheduled question: " &
                      To_String (Questions (I).Topic));

            declare
               Topic_Str : constant String := To_String (Questions (I).Topic);
               Curiosity_Prompt : constant String :=
                 "You are Adelaide, a curious AI assistant. " &
                 "The user asked you to remind them about: " & Topic_Str & ". " &
                 "Ask them a thoughtful question about this topic. " &
                 "Be natural and engaging. Output ONLY the question, no preamble.";
               Result : Unbounded_String;
            begin
               Model_Manager.Hybrid_Generate
                 (Prompt => Curiosity_Prompt,
                  Result => Result,
                  Level  => ELP0,
                  Agentic => True,
                  Raw_Prompt => True);

               if Length (Result) > 0 then
                  Last_Question := Result;
                  Put_Line (AnsiAda.Foreground (AnsiAda.Green) & "[Proactive]" &
                            AnsiAda.Reset & " [SCHEDULED-QUESTION] " & To_String (Result));
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
                           Queue_Audio (Result_Str);
                        end;
                     end if;
                  end;
               end if;
            exception
               when E : others =>
                  Put_Line (AnsiAda.Foreground (AnsiAda.Red) & "[Proactive]" &
                            AnsiAda.Reset & " ERROR: " & Ada.Exceptions.Exception_Message (E));
            end;

            --  Handle repeat
            if Questions (I).Repeat_Interval > 0.0 then
               Questions (I).Scheduled_Time := Now + Questions (I).Repeat_Interval;
            else
               Questions (I).Active := False;
            end if;
         end if;
      end loop;
   end Tick;

   function Get_Last_Question return String is
   begin
      return To_String (Last_Question);
   end Get_Last_Question;

   function Get_Last_Answer return String is
   begin
      return To_String (Last_Answer);
   end Get_Last_Answer;

   procedure Queue_Audio (PCM : String) is
   begin
      Pending_Audio := Pending_Audio & PCM;
   end Queue_Audio;

   function Has_Pending_Audio return Boolean is
   begin
      return Length (Pending_Audio) > 0;
   end Has_Pending_Audio;

   function Pop_Pending_Audio return String is
      Result : constant String := To_String (Pending_Audio);
   begin
      Pending_Audio := Null_Unbounded_String;
      return Result;
   end Pop_Pending_Audio;

end Proactive_Engine;
