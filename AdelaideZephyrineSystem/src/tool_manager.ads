pragma SPARK_Mode (Off);
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;

package Tool_Manager is

   type Tool_Result is record
      Success : Boolean;
      Output  : Unbounded_String;
   end record;

   function Execute_Tool (Name : String; Params : String) return Tool_Result;

--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
   --  Execute the "imagine" tool directly via SD_Manager (no Python sidecar).
   --  Returns the Base64-encoded PNG image data.
   --  Called from Hybrid_Generate when the model outputs [ACTION: imagine(prompt)].
   function Execute_Imagine_Tool (Prompt : String) return Tool_Result;

   --  CRONIA TOOL: Schedule a timed answer on ELP0
   --  Params format: "name|time_iso|prompt" or "name|repeat_seconds|prompt"
   function Execute_Cronia_Tool (Params : String) return Tool_Result;

   --  PROACTIVE TOOL: Trigger proactive question or handless mode
   --  Params format: "activate_handless" or "acoustic_trigger" or "schedule_question|time_iso|topic"
   function Execute_Proactive_Tool (Params : String) return Tool_Result;

   --  ROS2 TOOL: Trigger native Ada ROS2 actuator via ELP3
   --  Params format: "servo_id|angle"
   function Execute_ROS2_Tool (Params : String) return Tool_Result;

end Tool_Manager;
