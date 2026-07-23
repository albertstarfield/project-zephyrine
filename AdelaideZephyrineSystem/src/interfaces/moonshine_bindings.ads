pragma SPARK_Mode (Off);
-- c_binding: Moonshine C FFI
with Interfaces.C;
with Interfaces.C.Strings;
with System;

package Moonshine_Bindings is

   use Interfaces.C;
   use Interfaces.C.Strings;

   -- Constants
   MOONSHINE_HEADER_VERSION : constant int := 20000;
   
   MOONSHINE_MODEL_ARCH_TINY             : constant int := 0;
   MOONSHINE_MODEL_ARCH_BASE             : constant int := 1;
   MOONSHINE_MODEL_ARCH_TINY_STREAMING   : constant int := 2;
   MOONSHINE_MODEL_ARCH_BASE_STREAMING   : constant int := 3;

   MOONSHINE_FLAG_FORCE_UPDATE : constant unsigned := 1;

   -- Structs
   type Moonshine_Option is record
      Name  : chars_ptr;
      Value : chars_ptr;
   end record;
   pragma Convention (C, Moonshine_Option);

   type Transcript_Line is record
      Text                          : chars_ptr;
      Audio_Data                    : access float;
      Audio_Data_Count              : size_t;
      Start_Time                    : float;
      Duration                      : float;
      Id                            : Interfaces.Unsigned_64;
      Is_Complete                   : Interfaces.Integer_8;
      Is_Updated                    : Interfaces.Integer_8;
      Is_New                        : Interfaces.Integer_8;
      Has_Text_Changed              : Interfaces.Integer_8;
      Has_Speaker_Id                : Interfaces.Integer_8;
      Speaker_Id                    : Interfaces.Unsigned_64;
      Speaker_Index                 : Interfaces.Unsigned_32;
      Last_Transcription_Latency_Ms : Interfaces.Unsigned_32;
      Words                         : System.Address; -- FFI: System.Address required for C binding
      Word_Count                    : Interfaces.Unsigned_64;
   end record;
   pragma Convention (C, Transcript_Line);

   type Transcript is record
      Lines      : access Transcript_Line;
      Line_Count : Interfaces.Unsigned_64;
   end record;
   pragma Convention (C, Transcript);

   type Transcript_Ptr is access all Transcript;

   -- Functions
   function Load_Transcriber_From_Files
     (Path              : chars_ptr;
      Model_Arch        : Interfaces.Unsigned_32;
      Options           : System.Address; -- FFI: System.Address required for C binding
      Options_Count     : Interfaces.Unsigned_64;
      Moonshine_Version : int) return int;
   pragma Import (C, Load_Transcriber_From_Files, "moonshine_load_transcriber_from_files");

   --  Frees the transcriber handle and releases all associated resources.
   procedure Free_Transcriber (Transcriber_Handle : int);
   pragma Import (C, Free_Transcriber, "moonshine_free_transcriber");

   --  Creates a new streaming transcription session for the given transcriber handle.
   --  Returns a stream handle on success, negative on failure.
   function Create_Stream
     (Transcriber_Handle : int;
      Flags              : Interfaces.Unsigned_32) return int;
   pragma Import (C, Create_Stream, "moonshine_create_stream");

   --  Frees the streaming session and releases its resources.
   function Free_Stream
     (Transcriber_Handle : int;
      Stream_Handle      : int) return int;
   pragma Import (C, Free_Stream, "moonshine_free_stream");

   --  Begins audio capture on the given streaming session for real-time transcription.
   function Start_Stream
     (Transcriber_Handle : int;
      Stream_Handle      : int) return int;
   pragma Import (C, Start_Stream, "moonshine_start_stream");

   --  Stops audio capture on the given streaming session.
   function Stop_Stream
     (Transcriber_Handle : int;
      Stream_Handle      : int) return int;
   pragma Import (C, Stop_Stream, "moonshine_stop_stream");

   --  Appends new audio samples to the streaming transcription buffer at the given sample rate.
   function Transcribe_Add_Audio_To_Stream
     (Transcriber_Handle : int;
      Stream_Handle      : int;
      New_Audio_Data     : access float;
      Audio_Length       : Interfaces.Unsigned_64;
      Sample_Rate        : int;
      Flags              : Interfaces.Unsigned_32) return int;
   pragma Import (C, Transcribe_Add_Audio_To_Stream, "moonshine_transcribe_add_audio_to_stream");

   --  Transcribes accumulated audio from the stream, returning a Transcript pointer
   --  with the recognized text lines.
   function Transcribe_Stream
     (Transcriber_Handle : int;
      Stream_Handle      : int;
      Flags              : Interfaces.Unsigned_32;
      Out_Transcript     : access Transcript_Ptr) return int;
   pragma Import (C, Transcribe_Stream, "moonshine_transcribe_stream");

   --  Transcribes an entire audio buffer in one-shot mode without streaming.
   --  Audio_Data must be 16KHz mono float samples.
   function Transcribe_Without_Streaming
     (Transcriber_Handle : int;
      Audio_Data         : access float;
      Audio_Length       : Interfaces.Unsigned_64;
      Sample_Rate        : int;
      Flags              : Interfaces.Unsigned_32;
      Out_Transcript     : access Transcript_Ptr) return int;
   pragma Import (C, Transcribe_Without_Streaming, "moonshine_transcribe_without_streaming");

end Moonshine_Bindings;
