with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;

package Tool_Manager is
   pragma Spark_Mode (Off);

   type Tool_Result is record
      Success : Boolean;
      Output  : Unbounded_String;
   end record;

   function Execute_Tool (Name : String; Params : String) return Tool_Result;

end Tool_Manager;
