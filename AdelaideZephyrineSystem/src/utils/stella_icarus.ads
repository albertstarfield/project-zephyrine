pragma SPARK_Mode (Off);
-- thread: Icarus daemon requires task protection

package Stella_Icarus is
   --  ELP2: Deterministic API Logic Hook
   procedure Initialize with Pre => True, Post => True;

   --  Checks if the prompt maps to an API command (e.g. time, date).
   --  Returns empty string if no match.
   function Check_API_Trigger (Prompt : String) return String with Pre => True, Post => True;
end Stella_Icarus;
