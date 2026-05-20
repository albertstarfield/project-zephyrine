with Llama_Interface;
with Ada.Real_Time; use Ada.Real_Time;
with Math_Utils;

package Model_Manager is
   pragma Spark_Mode (Off);

   type Model_Type is (Qwen_0_8B, Qwen_4B, Qwen_Embedding, MMProj);

   procedure Initialize;

   procedure Load_Model (Kind : Model_Type; Success : out Boolean; Requested_Ctx : Positive := 4096);

   procedure Unload_Model (Kind : Model_Type);

   function Get_Context (Kind : Model_Type) return Llama_Interface.Llama_Context;

   function Get_Model (Kind : Model_Type) return Llama_Interface.Llama_Model;

   --  Perform inference (simplified for now)
   function Generate (Kind : Model_Type; Prompt : String; Requested_Ctx : Positive := 4096) return String;

   --  Perform multi-hop reasoning (0.8b thinking -> 4b final)
   function Hybrid_Generate (Prompt : String) return String;

   function Get_Embedding (Prompt : String) return Math_Utils.Vector;

   function Get_Kind_For_Model_Name (Name : String) return Model_Type;

   function Is_Loaded (Kind : Model_Type) return Boolean;

end Model_Manager;
