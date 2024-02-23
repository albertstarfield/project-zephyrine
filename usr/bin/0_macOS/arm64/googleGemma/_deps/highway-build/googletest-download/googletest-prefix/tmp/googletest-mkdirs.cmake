# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/Users/user/Documents/misc/AI/LLM/alpaca-electron-zephyrine/usr/vendor/gemma.cpp/build/_deps/highway-build/googletest-src"
  "/Users/user/Documents/misc/AI/LLM/alpaca-electron-zephyrine/usr/vendor/gemma.cpp/build/_deps/highway-build/googletest-build"
  "/Users/user/Documents/misc/AI/LLM/alpaca-electron-zephyrine/usr/vendor/gemma.cpp/build/_deps/highway-build/googletest-download/googletest-prefix"
  "/Users/user/Documents/misc/AI/LLM/alpaca-electron-zephyrine/usr/vendor/gemma.cpp/build/_deps/highway-build/googletest-download/googletest-prefix/tmp"
  "/Users/user/Documents/misc/AI/LLM/alpaca-electron-zephyrine/usr/vendor/gemma.cpp/build/_deps/highway-build/googletest-download/googletest-prefix/src/googletest-stamp"
  "/Users/user/Documents/misc/AI/LLM/alpaca-electron-zephyrine/usr/vendor/gemma.cpp/build/_deps/highway-build/googletest-download/googletest-prefix/src"
  "/Users/user/Documents/misc/AI/LLM/alpaca-electron-zephyrine/usr/vendor/gemma.cpp/build/_deps/highway-build/googletest-download/googletest-prefix/src/googletest-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/Users/user/Documents/misc/AI/LLM/alpaca-electron-zephyrine/usr/vendor/gemma.cpp/build/_deps/highway-build/googletest-download/googletest-prefix/src/googletest-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/Users/user/Documents/misc/AI/LLM/alpaca-electron-zephyrine/usr/vendor/gemma.cpp/build/_deps/highway-build/googletest-download/googletest-prefix/src/googletest-stamp${cfgdir}") # cfgdir has leading slash
endif()
