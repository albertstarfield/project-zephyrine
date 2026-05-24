with Interfaces.C; use Interfaces.C;
with Interfaces.C.Strings; use Interfaces.C.Strings;
with Supertonic_Bindings; use Supertonic_Bindings;
with System; use type System.Address;
with Ada.Text_IO; use Ada.Text_IO;

package body Supertonic_Interface is

   Current_TTS   : SupertonicTTS := SupertonicTTS (System.Null_Address);
   Current_Style : SupertonicStyle := SupertonicStyle (System.Null_Address);

   procedure Initialize (Onnx_Dir : String; Use_Gpu : Boolean := False) is
      C_Dir : chars_ptr := New_String (Onnx_Dir);
      C_Gpu : int := (if Use_Gpu then 1 else 0);
   begin
      if Current_TTS /= SupertonicTTS (System.Null_Address) then
         Free (Current_TTS);
      end if;
      Current_TTS := Init (C_Dir, C_Gpu);
      Free (C_Dir);
      if Current_TTS = SupertonicTTS (System.Null_Address) then
         Put_Line ("Failed to initialize Supertonic.");
      else
         Put_Line ("Supertonic initialized successfully.");
      end if;
   end Initialize;

   procedure Load_Voice_Style (Path : String) is
      C_Path : chars_ptr := New_String (Path);
      type Chars_Ptr_Array is array (0 .. 0) of aliased chars_ptr;
      Paths_Array : aliased Chars_Ptr_Array := [0 => C_Path];
   begin
      if Current_Style /= SupertonicStyle (System.Null_Address) then
         Free_Style (Current_Style);
      end if;
      
      Current_Style := Load_Style (Paths_Array'Address, 1);
      Free (C_Path);
      
      if Current_Style = SupertonicStyle (System.Null_Address) then
         Put_Line ("Failed to load Supertonic style from " & Path);
      else
         Put_Line ("Supertonic style loaded successfully.");
      end if;
   end Load_Voice_Style;

   function Synthesize_Speech (Text : String; Lang : String := "en") return Stream_Element_Array is
      C_Text : chars_ptr := New_String (Text);
      C_Lang : chars_ptr := New_String (Lang);
      Out_Samples : aliased size_t := 0;
      Audio_Ptr : System.Address;
      Empty_Array : Stream_Element_Array (1 .. 0);
   begin
      if Current_TTS = SupertonicTTS (System.Null_Address) or else Current_Style = SupertonicStyle (System.Null_Address) then
         Put_Line ("TTS or Style not initialized.");
         Free (C_Text);
         Free (C_Lang);
         return Empty_Array;
      end if;

      Audio_Ptr := Synthesize
        (TTS              => Current_TTS,
         Text             => C_Text,
         Lang             => C_Lang,
         Style            => Current_Style,
         Total_Step       => 10,
         Speed            => 1.0,
         Silence_Duration => 0.3,
         Out_Samples      => Out_Samples'Access);

      Free (C_Text);
      Free (C_Lang);

      if Audio_Ptr = System.Null_Address or else Out_Samples = 0 then
         return Empty_Array;
      end if;

      declare
         Total_Bytes : constant Stream_Element_Offset := Stream_Element_Offset (Out_Samples * 4);
         Result      : Stream_Element_Array (1 .. Total_Bytes);
         
         type Byte_Array is array (1 .. Total_Bytes) of aliased Stream_Element;
         Audio_Bytes : Byte_Array with Import, Address => Audio_Ptr;
      begin
         for I in 1 .. Total_Bytes loop
            Result (I) := Audio_Bytes (I);
         end loop;
         
         Free_Audio (Audio_Ptr);
         return Result;
      end;
   end Synthesize_Speech;

   procedure Shutdown is
   begin
      if Current_Style /= SupertonicStyle (System.Null_Address) then
         Free_Style (Current_Style);
         Current_Style := SupertonicStyle (System.Null_Address);
      end if;
      if Current_TTS /= SupertonicTTS (System.Null_Address) then
         Free (Current_TTS);
         Current_TTS := SupertonicTTS (System.Null_Address);
      end if;
   end Shutdown;

end Supertonic_Interface;
