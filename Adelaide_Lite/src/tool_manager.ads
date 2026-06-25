pragma SPARK_Mode (Off);
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;

package Tool_Manager is

   type Tool_Result is record
      Success : Boolean;
      Output  : Unbounded_String;
   end record;

   function Execute_Tool (Name : String; Params : String) return Tool_Result;

   --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
   --  Execute the "imagine" tool directly via SD_Manager (no Python sidecar).
   --  Returns the Base64-encoded PNG image data.
   --  Called from Hybrid_Generate when the model outputs [ACTION: imagine(prompt)].
   function Execute_Imagine_Tool (Prompt : String) return Tool_Result;

end Tool_Manager;
