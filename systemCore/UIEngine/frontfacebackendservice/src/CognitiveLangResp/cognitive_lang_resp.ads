pragma Ada_2022;

package Cognitive_Lang_Resp is

   -- Sets the target URL (e.g. "http://localhost:11434")
   procedure Configure_Backend (Base_URL : String);
   
   -- NEW: Allows Handlers to see where to forward requests
   function Get_Backend_URL return String;

   function Process_Input 
     (Input_Sequence : String;
      Model_ID       : String := "Snowball-Enaga") return String;

end Cognitive_Lang_Resp;