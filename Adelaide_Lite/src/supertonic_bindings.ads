with Interfaces.C; use Interfaces.C;
with Interfaces.C.Strings; use Interfaces.C.Strings;
with System;

package Supertonic_Bindings is
   pragma Preelaborate;

   type SupertonicTTS is new System.Address;
   type SupertonicStyle is new System.Address;

   -- SupertonicTTS* supertonic_init(const char* onnx_dir, int use_gpu);
   function Init (Onnx_Dir : chars_ptr; Use_Gpu : int) return SupertonicTTS
     with Import => True, Convention => C, External_Name => "supertonic_init";

   -- void supertonic_free(SupertonicTTS* tts);
   procedure Free (TTS : SupertonicTTS)
     with Import => True, Convention => C, External_Name => "supertonic_free";

   -- SupertonicStyle* supertonic_load_style(const char** voice_style_paths, int num_paths);
   function Load_Style (Voice_Style_Paths : System.Address; Num_Paths : int) return SupertonicStyle
     with Import => True, Convention => C, External_Name => "supertonic_load_style";

   -- void supertonic_free_style(SupertonicStyle* style);
   procedure Free_Style (Style : SupertonicStyle)
     with Import => True, Convention => C, External_Name => "supertonic_free_style";

   -- float* supertonic_synthesize(...)
   function Synthesize
     (TTS              : SupertonicTTS;
      Text             : chars_ptr;
      Lang             : chars_ptr;
      Style            : SupertonicStyle;
      Total_Step       : int;
      Speed            : C_float;
      Silence_Duration : C_float;
      Out_Samples      : access size_t) return System.Address
     with Import => True, Convention => C, External_Name => "supertonic_synthesize";

   -- void supertonic_free_audio(float* audio);
   procedure Free_Audio (Audio : System.Address)
     with Import => True, Convention => C, External_Name => "supertonic_free_audio";

   -- int supertonic_get_sample_rate(SupertonicTTS* tts);
   function Get_Sample_Rate (TTS : SupertonicTTS) return int
     with Import => True, Convention => C, External_Name => "supertonic_get_sample_rate";

end Supertonic_Bindings;
