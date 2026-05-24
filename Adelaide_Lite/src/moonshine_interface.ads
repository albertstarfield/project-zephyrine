with Interfaces.C;
with Interfaces.C.Strings;
with Moonshine_Bindings;

package Moonshine_Interface is

   procedure Init_Moonshine (Model_Path : String);
   
   --  Transcribe expects raw 16KHz floats
   function Transcribe_Raw_PCM (Audio_Data : access float; Audio_Length : Interfaces.Unsigned_64) return String;
   
end Moonshine_Interface;
