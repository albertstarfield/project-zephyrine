pragma SPARK_Mode (Off);
with Interfaces.C;
with Interfaces.C.Strings;
with Moonshine_Bindings;

package Moonshine_Interface is

   procedure Init_Moonshine (Model_Path : String);
   procedure Free_Moonshine;
   
   --  Transcribe expects raw 16KHz floats
   function Transcribe_Raw_PCM (Audio_Data : access Float; Audio_Length : Interfaces.Unsigned_64) return String;
   
end Moonshine_Interface;
