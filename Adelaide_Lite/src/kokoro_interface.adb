with GNATCOLL.JSON; use GNATCOLL.JSON;
with AWS.Client;
with AWS.Response;
with AWS.Messages; use AWS.Messages;
with Ada.Text_IO; use Ada.Text_IO;
with Ada.Exceptions;
with AnsiAda;

package body Kokoro_Interface is

   function Synthesize_Speech (Text : String) return Ada.Streams.Stream_Element_Array is
      Payload : constant JSON_Value := Create_Object;
      Response : AWS.Response.Data;
      Empty_Array : Ada.Streams.Stream_Element_Array (1 .. 0);
   begin
      Set_Field (Payload, "text", Text);
      Set_Field (Payload, "voice", "af_sarah");

      begin
         Response := AWS.Client.Post (
            URL          => "http://127.0.0.1:11421/tts",
            Data         => Write (Payload),
            Content_Type => "application/json"
         );
         
         if AWS.Response.Status_Code (Response) = AWS.Messages.S200 then
            declare
               Body_Str : constant String := AWS.Response.Message_Body (Response);
               Stream_Arr : Ada.Streams.Stream_Element_Array (1 .. Body_Str'Length);
            begin
               for I in Body_Str'Range loop
                  Stream_Arr (Ada.Streams.Stream_Element_Offset (I - Body_Str'First + 1)) := 
                     Character'Pos (Body_Str (I));
               end loop;
               return Stream_Arr;
            end;
         else
            Put_Line (AnsiAda.Foreground (AnsiAda.Red) & "[Kokoro] TTS Sidecar returned status code " & 
                      AWS.Messages.Status_Code'Image (AWS.Response.Status_Code (Response)));
            return Empty_Array;
         end if;
      exception
         when E : others =>
            Put_Line (AnsiAda.Foreground (AnsiAda.Red) & "[Kokoro] Failed to reach TTS Sidecar: " & 
                      Ada.Exceptions.Exception_Message (E));
            return Empty_Array;
      end;
   end Synthesize_Speech;

end Kokoro_Interface;
