pragma SPARK_Mode (Off);
-- c_binding: Moonshine C FFI
with Ada.Text_IO;
with Ada.Strings.Unbounded;
with System;

package body Moonshine_Interface is

   use Interfaces.C;
   use Interfaces.C.Strings;

   Handle : int := -1;

   procedure Init_Moonshine (Model_Path : String) is
      C_Path : chars_ptr := New_String (Model_Path);
   begin
      -- Using Tiny Streaming
      Handle := Moonshine_Bindings.Load_Transcriber_From_Files
        (Path              => C_Path,
         Model_Arch        => Interfaces.Unsigned_32 (Moonshine_Bindings.MOONSHINE_MODEL_ARCH_TINY_STREAMING),
         Options           => System.Null_Address,
         Options_Count     => 0,
         Moonshine_Version => Moonshine_Bindings.MOONSHINE_HEADER_VERSION);

      Free (C_Path);

      if Handle < 0 then
         Ada.Text_IO.Put_Line ("Failed to load Moonshine model!");
      else
         Ada.Text_IO.Put_Line ("Loaded Moonshine model successfully!");
      end if;
   end Init_Moonshine;

   procedure Free_Moonshine is
   begin
      if Handle >= 0 then
         Moonshine_Bindings.Free_Transcriber (Handle);
         Handle := -1;
         Ada.Text_IO.Put_Line ("Freed Moonshine transcriber resources.");
      end if;
   end Free_Moonshine;

   function Transcribe_Raw_PCM (Audio_Data : access Float; Audio_Length : Interfaces.Unsigned_64) return String is
      Transcript_Ptr : aliased Moonshine_Bindings.Transcript_Ptr := null;
      Result : int;
      Transcription_Result : Ada.Strings.Unbounded.Unbounded_String;
      use type Interfaces.Unsigned_64;
      use type Moonshine_Bindings.Transcript_Ptr;
   begin
      if Handle < 0 then
         return "Moonshine model not initialized";
      end if;

      Result := Moonshine_Bindings.Transcribe_Without_Streaming
        (Transcriber_Handle => Handle,
         Audio_Data         => Audio_Data,
         Audio_Length       => Audio_Length,
         Sample_Rate        => 16000,
         Flags              => 0,
         Out_Transcript     => Transcript_Ptr'Access);

      if Result /= 0 then
         return "Transcription failed";
      end if;

      if Transcript_Ptr /= null and then Transcript_Ptr.Lines /= null then
         declare
            L_Count : constant Interfaces.Unsigned_64 := Transcript_Ptr.Line_Count;
            type Line_Array is array (0 .. Natural (L_Count) - 1) of aliased Moonshine_Bindings.Transcript_Line;
            Lines_Access : Line_Array with Import, Address => Transcript_Ptr.Lines.all'Address;
            Full_Text : Ada.Strings.Unbounded.Unbounded_String;
         begin
            if L_Count > 0 then
               for I in 0 .. Natural (L_Count) - 1 loop
                  declare
                     Text_Str : constant String := Interfaces.C.Strings.Value (Lines_Access (I).Text);
                  begin
                     Ada.Strings.Unbounded.Append (Full_Text, Text_Str);
                     if I < Natural (L_Count) - 1 then
                        Ada.Strings.Unbounded.Append (Full_Text, " ");
                     end if;
                  end;
               end loop;
               return Ada.Strings.Unbounded.To_String (Full_Text);
            end if;
         end;
      end if;

      return "";
   end Transcribe_Raw_PCM;

end Moonshine_Interface;
