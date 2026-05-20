with Llama_Interface; use Llama_Interface;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Interfaces.C.Strings; use Interfaces.C.Strings;
with Ada.Text_IO;
with Math_Utils;

package Model_Manager is
   pragma Spark_Mode (Off);

   type Model_Type is (Qwen_0_8B, Qwen_4B, Qwen_Embedding, MMProj);

   procedure Initialize;
   
   procedure Load_Model (Kind : Model_Type; Success : out Boolean);
   
   procedure Unload_Model (Kind : Model_Type);
   
   function Get_Context (Kind : Model_Type) return Llama_Context;
   
   function Get_Model (Kind : Model_Type) return Llama_Model;

   --  Perform inference (simplified for now)
   function Generate (Kind : Model_Type; Prompt : String) return String;

   --  Perform multi-hop reasoning (0.8b thinking -> 4b final)
   function Hybrid_Generate (Prompt : String) return String;

   function Get_Embedding (Prompt : String) return Math_Utils.Vector;

   function Get_Kind_For_Model_Name (Name : String) return Model_Type;

end Model_Manager;
