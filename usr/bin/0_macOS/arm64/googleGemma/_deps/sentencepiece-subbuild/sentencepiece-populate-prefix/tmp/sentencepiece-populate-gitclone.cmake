# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

if(EXISTS "/Users/user/adelaide-zephyrine-charlotte-assistant/usr/vendor/gemma.cpp/build/_deps/sentencepiece-subbuild/sentencepiece-populate-prefix/src/sentencepiece-populate-stamp/sentencepiece-populate-gitclone-lastrun.txt" AND EXISTS "/Users/user/adelaide-zephyrine-charlotte-assistant/usr/vendor/gemma.cpp/build/_deps/sentencepiece-subbuild/sentencepiece-populate-prefix/src/sentencepiece-populate-stamp/sentencepiece-populate-gitinfo.txt" AND
  "/Users/user/adelaide-zephyrine-charlotte-assistant/usr/vendor/gemma.cpp/build/_deps/sentencepiece-subbuild/sentencepiece-populate-prefix/src/sentencepiece-populate-stamp/sentencepiece-populate-gitclone-lastrun.txt" IS_NEWER_THAN "/Users/user/adelaide-zephyrine-charlotte-assistant/usr/vendor/gemma.cpp/build/_deps/sentencepiece-subbuild/sentencepiece-populate-prefix/src/sentencepiece-populate-stamp/sentencepiece-populate-gitinfo.txt")
  message(STATUS
    "Avoiding repeated git clone, stamp file is up to date: "
    "'/Users/user/adelaide-zephyrine-charlotte-assistant/usr/vendor/gemma.cpp/build/_deps/sentencepiece-subbuild/sentencepiece-populate-prefix/src/sentencepiece-populate-stamp/sentencepiece-populate-gitclone-lastrun.txt'"
  )
  return()
endif()

execute_process(
  COMMAND ${CMAKE_COMMAND} -E rm -rf "/Users/user/adelaide-zephyrine-charlotte-assistant/usr/vendor/gemma.cpp/build/_deps/sentencepiece-src"
  RESULT_VARIABLE error_code
)
if(error_code)
  message(FATAL_ERROR "Failed to remove directory: '/Users/user/adelaide-zephyrine-charlotte-assistant/usr/vendor/gemma.cpp/build/_deps/sentencepiece-src'")
endif()

# try the clone 3 times in case there is an odd git clone issue
set(error_code 1)
set(number_of_tries 0)
while(error_code AND number_of_tries LESS 3)
  execute_process(
    COMMAND "/usr/bin/git"
            clone --no-checkout --config "advice.detachedHead=false" "https://github.com/google/sentencepiece" "sentencepiece-src"
    WORKING_DIRECTORY "/Users/user/adelaide-zephyrine-charlotte-assistant/usr/vendor/gemma.cpp/build/_deps"
    RESULT_VARIABLE error_code
  )
  math(EXPR number_of_tries "${number_of_tries} + 1")
endwhile()
if(number_of_tries GREATER 1)
  message(STATUS "Had to git clone more than once: ${number_of_tries} times.")
endif()
if(error_code)
  message(FATAL_ERROR "Failed to clone repository: 'https://github.com/google/sentencepiece'")
endif()

execute_process(
  COMMAND "/usr/bin/git"
          checkout "53de76561cfc149d3c01037f0595669ad32a5e7c" --
  WORKING_DIRECTORY "/Users/user/adelaide-zephyrine-charlotte-assistant/usr/vendor/gemma.cpp/build/_deps/sentencepiece-src"
  RESULT_VARIABLE error_code
)
if(error_code)
  message(FATAL_ERROR "Failed to checkout tag: '53de76561cfc149d3c01037f0595669ad32a5e7c'")
endif()

set(init_submodules TRUE)
if(init_submodules)
  execute_process(
    COMMAND "/usr/bin/git" 
            submodule update --recursive --init 
    WORKING_DIRECTORY "/Users/user/adelaide-zephyrine-charlotte-assistant/usr/vendor/gemma.cpp/build/_deps/sentencepiece-src"
    RESULT_VARIABLE error_code
  )
endif()
if(error_code)
  message(FATAL_ERROR "Failed to update submodules in: '/Users/user/adelaide-zephyrine-charlotte-assistant/usr/vendor/gemma.cpp/build/_deps/sentencepiece-src'")
endif()

# Complete success, update the script-last-run stamp file:
#
execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy "/Users/user/adelaide-zephyrine-charlotte-assistant/usr/vendor/gemma.cpp/build/_deps/sentencepiece-subbuild/sentencepiece-populate-prefix/src/sentencepiece-populate-stamp/sentencepiece-populate-gitinfo.txt" "/Users/user/adelaide-zephyrine-charlotte-assistant/usr/vendor/gemma.cpp/build/_deps/sentencepiece-subbuild/sentencepiece-populate-prefix/src/sentencepiece-populate-stamp/sentencepiece-populate-gitclone-lastrun.txt"
  RESULT_VARIABLE error_code
)
if(error_code)
  message(FATAL_ERROR "Failed to copy script-last-run stamp file: '/Users/user/adelaide-zephyrine-charlotte-assistant/usr/vendor/gemma.cpp/build/_deps/sentencepiece-subbuild/sentencepiece-populate-prefix/src/sentencepiece-populate-stamp/sentencepiece-populate-gitclone-lastrun.txt'")
endif()
