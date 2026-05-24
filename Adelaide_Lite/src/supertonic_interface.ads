with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Ada.Streams; use Ada.Streams;

package Supertonic_Interface is

   procedure Initialize (Onnx_Dir : String; Use_Gpu : Boolean := False);
   procedure Load_Voice_Style (Path : String);
   
   -- Generates speech and returns raw 32-bit Float PCM
   function Synthesize_Speech (Text : String; Lang : String := "en") return Stream_Element_Array;
   
   procedure Shutdown;

end Supertonic_Interface;
