pragma SPARK_Mode (Off);
-- c_binding: Kokoro TTS C FFI
with GNAT.OS_Lib;
with Ada.Streams.Stream_IO;
with Ada.Directories;
with Ada.Text_IO; use Ada.Text_IO;
with Ada.Exceptions;
with AnsiAda;

package body Kokoro_Interface is

   function Synthesize_Speech (Text : String) return Ada.Streams.Stream_Element_Array is
      File_Name : constant String := "kokoro_temp.wav";
      
      Success : Boolean;
      Empty_Array : Ada.Streams.Stream_Element_Array (1 .. 0);
   begin
      GNAT.OS_Lib.Spawn (
         Program_Name => "vendor/tts_kokoro_component/venv/bin/python",
         Args         => [new String'("vendor/tts_kokoro_component/stereo_cloner.py"),
                          new String'("--text"), new String'(Text),
                          new String'("--ref"), new String'("src/sampleAdeltts_refAudioSpeech.dat"),
                          new String'("--out"), new String'(File_Name)],
         Success      => Success
      );
      
      if not Success then
         Put_Line (AnsiAda.Foreground (AnsiAda.Red) & "[Kokoro] Failed to execute python sidecar CLI.");
         return Empty_Array;
      end if;
      
      if not Ada.Directories.Exists (File_Name) then
         Put_Line (AnsiAda.Foreground (AnsiAda.Red) & "[Kokoro] TTS output file was not generated.");
         return Empty_Array;
      end if;
      
      -- Read File_Name into Stream_Element_Array
      declare
         File : Ada.Streams.Stream_IO.File_Type;
         Size : Ada.Streams.Stream_Element_Offset;
      begin
         Ada.Streams.Stream_IO.Open (File, Ada.Streams.Stream_IO.In_File, File_Name);
         Size := Ada.Streams.Stream_Element_Offset (Ada.Streams.Stream_IO.Size (File));
         
         declare
            Stream_Arr : Ada.Streams.Stream_Element_Array (1 .. Size);
            Last       : Ada.Streams.Stream_Element_Offset;
         begin
            Ada.Streams.Stream_IO.Read (File, Stream_Arr, Last);
            Ada.Streams.Stream_IO.Close (File);
            
            -- Clean up the temporary file
            Ada.Directories.Delete_File (File_Name);
            
            return Stream_Arr (1 .. Last);
         end;
      exception
         when E : others =>
            if Ada.Streams.Stream_IO.Is_Open (File) then
               Ada.Streams.Stream_IO.Close (File);
            end if;
            Put_Line (AnsiAda.Foreground (AnsiAda.Red) & "[Kokoro] Failed to read TTS file: " & 
                      Ada.Exceptions.Exception_Message (E));
            return Empty_Array;
      end;
   end Synthesize_Speech;

end Kokoro_Interface;
