with Ada.Streams;

package Kokoro_Interface is

   --  Synthesizes speech using the Kokoro Python CLI tool.
   --  Returns the raw WAV file bytes.
   function Synthesize_Speech (Text : String) return Ada.Streams.Stream_Element_Array;

end Kokoro_Interface;
