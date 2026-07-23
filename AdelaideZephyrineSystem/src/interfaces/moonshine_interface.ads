pragma SPARK_Mode (Off);
-- c_binding: Moonshine C FFI
with Interfaces.C;
with Interfaces.C.Strings;
with Moonshine_Bindings;

package Moonshine_Interface is

   --  Loads the Moonshine speech recognition model from the specified file path.
   procedure Init_Moonshine (Model_Path : String);
   --  Frees the Moonshine transcriber and releases model resources.
   procedure Free_Moonshine;
   
   --  Transcribe expects raw 16KHz floats
   function Transcribe_Raw_PCM (Audio_Data : access Float; Audio_Length : Interfaces.Unsigned_64) return String;
   
end Moonshine_Interface;
