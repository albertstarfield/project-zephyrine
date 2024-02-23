# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/Users/user/Documents/misc/AI/LLM/alpaca-electron-zephyrine/usr/vendor/gemma.cpp/build/_deps/sentencepiece-src"
  "/Users/user/Documents/misc/AI/LLM/alpaca-electron-zephyrine/usr/vendor/gemma.cpp/build/_deps/sentencepiece-build"
  "/Users/user/Documents/misc/AI/LLM/alpaca-electron-zephyrine/usr/vendor/gemma.cpp/build/_deps/sentencepiece-subbuild/sentencepiece-populate-prefix"
  "/Users/user/Documents/misc/AI/LLM/alpaca-electron-zephyrine/usr/vendor/gemma.cpp/build/_deps/sentencepiece-subbuild/sentencepiece-populate-prefix/tmp"
  "/Users/user/Documents/misc/AI/LLM/alpaca-electron-zephyrine/usr/vendor/gemma.cpp/build/_deps/sentencepiece-subbuild/sentencepiece-populate-prefix/src/sentencepiece-populate-stamp"
  "/Users/user/Documents/misc/AI/LLM/alpaca-electron-zephyrine/usr/vendor/gemma.cpp/build/_deps/sentencepiece-subbuild/sentencepiece-populate-prefix/src"
  "/Users/user/Documents/misc/AI/LLM/alpaca-electron-zephyrine/usr/vendor/gemma.cpp/build/_deps/sentencepiece-subbuild/sentencepiece-populate-prefix/src/sentencepiece-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/Users/user/Documents/misc/AI/LLM/alpaca-electron-zephyrine/usr/vendor/gemma.cpp/build/_deps/sentencepiece-subbuild/sentencepiece-populate-prefix/src/sentencepiece-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/Users/user/Documents/misc/AI/LLM/alpaca-electron-zephyrine/usr/vendor/gemma.cpp/build/_deps/sentencepiece-subbuild/sentencepiece-populate-prefix/src/sentencepiece-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
