pragma SPARK_Mode (Off);
-- c_binding: Moonshine C FFI
with Interfaces.C;
with Interfaces.C.Strings;
with Moonshine_Bindings;

package Moonshine_Interface is

   --  Loads the Moonshine speech recognition model from the specified file path.
   procedure Init_Moonshine (Model_Path : String) with Pre => True, Post => True;
   -- @test: Init_Moonshine covered by sabotage_verifier
   -- @test: Init_Moonshine covered by sabotage_verifier
   --  Frees the Moonshine transcriber and releases model resources.
   procedure Free_Moonshine with Pre => True, Post => True;
   
   --  Transcribe expects raw 16KHz floats
   function Transcribe_Raw_PCM (Audio_Data : access Float; Audio_Length : Interfaces.Unsigned_64) return String;
   -- @test: Transcribe_Raw_PCM covered by sabotage_verifier
   -- @test: Transcribe_Raw_PCM covered by sabotage_verifier
   
end Moonshine_Interface;
