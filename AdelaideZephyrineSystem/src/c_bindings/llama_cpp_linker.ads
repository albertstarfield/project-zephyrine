pragma SPARK_Mode (Off);
-- c_binding: Llama.cpp linker FFI
package Llama_Cpp_Linker is
   pragma Linker_Options ("-L/Users/albertstarfield/LibraryTube/OpenIntellegentiaPlatform/llama.cpp/build/src");
   pragma Linker_Options ("-lllama");
   pragma Linker_Options ("-L/Users/albertstarfield/LibraryTube/OpenIntellegentiaPlatform/llama.cpp/build/ggml/src");
   pragma Linker_Options ("-lggml");
   pragma Linker_Options ("-lggml-base");
   pragma Linker_Options ("-lggml-cpu");
   pragma Linker_Options ("-L/Users/albertstarfield/LibraryTube/OpenIntellegentiaPlatform/llama.cpp/build/ggml/src/ggml-metal");
   pragma Linker_Options ("-lggml-metal");
   pragma Linker_Options ("-L/Users/albertstarfield/LibraryTube/OpenIntellegentiaPlatform/llama.cpp/build/ggml/src/ggml-blas");
   pragma Linker_Options ("-lggml-blas");
   pragma Linker_Options ("-L/opt/homebrew/opt/sqlite/lib");
   pragma Linker_Options ("-lsqlite3");
   pragma Linker_Options ("-Wl,-syslibroot,/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk");
   pragma Linker_Options ("-mmacosx-version-min=15.0");
   pragma Linker_Options ("-framework");
   pragma Linker_Options ("Accelerate");
   pragma Linker_Options ("-framework");
   pragma Linker_Options ("Metal");
   pragma Linker_Options ("-framework");
   pragma Linker_Options ("Foundation");
   pragma Linker_Options ("-lc++");
end Llama_Cpp_Linker;
