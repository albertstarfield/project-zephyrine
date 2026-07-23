pragma SPARK_Mode (Off);
-- thread: Icarus daemon requires task protection
with Ada.Calendar;
with Ada.Calendar.Formatting;
with Ada.Characters.Handling;
with Ada.Strings.Fixed; use Ada.Strings.Fixed;

package body Stella_Icarus is

   --  Initialize: Initializes the Stella Icarus subsystem.
   procedure Initialize is
   begin
      null;
   end Initialize;

   --  Check_API_Trigger: Checks if the prompt matches a deterministic API trigger.
   function Check_API_Trigger (Prompt : String) return String is
      Lower_Prompt : constant String := Ada.Characters.Handling.To_Lower (Prompt);
   begin
      if Index (Lower_Prompt, "what time is it") > 0 or else Index (Lower_Prompt, "current time") > 0 then
         declare
            Now : constant Ada.Calendar.Time := Ada.Calendar.Clock;
         begin
            return "[StellaIcarus-ELP2] The current time is " & Ada.Calendar.Formatting.Image (Now) & ".";
         end;
      elsif Index (Lower_Prompt, "system status") > 0 then
         return "[StellaIcarus-ELP2] All deterministic API hooks are online and nominal.";
      end if;
      return "";
   end Check_API_Trigger;

end Stella_Icarus;
