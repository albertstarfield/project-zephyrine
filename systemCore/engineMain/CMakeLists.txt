# Minimum version of CMake required
cmake_minimum_required(VERSION 3.10)

# Project name and version
project(AdelaideandAlbertEngine VERSION 1.0)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 20) #Newest Bleeding edge C++ standard!
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Setting main Zephyrine Compilation extra features for your specific Processor!")
set(CMAKE_CXX_FLAGS "-Wall -Wextra") # -Wall (Warning all for unoptimized or wasteful coding)
message(STATUS "I forgot what is -g flag means?")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
message(STATUS "Allowing the code to be as close as the CPU architecture allow to run the most optimized, minimizing compatibility cross machine binary execution!")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(mainEngineCode "${CMAKE_SOURCE_DIR}/main.cpp")


# Add your source files
set(SOURCES
    main.cpp
    # Add more source files here if needed
)

#--------- Definition of the Modules/Library (Mixed with Python + C++) 

#Raw C++ Library +_+_+_+_+_+_+_
#Main directory for Modules Library directory
message(STATUS "Setting main Zephyrine Library Module for C++ Management Engine LIBRARY ${LIBRARY}")
set(LIBRARY "${CMAKE_SOURCE_DIR}/Library")

message(STATUS "Setting main curl Zephyrine Library Module for C++ Management Engine")
include_directories(${Library}/curl/include/curl/)
# Add the path to the compiled libcurl library
# Prevent issue Undefined symbols for architecture arm64:
link_directories(${LIBRARY}/curl/lib)  # Ensure this path points to where libcurl.a or libcurl.so is located

message(STATUS "Setting jpeg2png Libary Zephyrine Library Module for C++ Management Engine ProgressBar etc")
include_directories(${Library}/jpeg2png/)

message(STATUS "Setting SQLite Library Zephyrine Library Module for SQLITE Database amalgamation 3470100")
set(SQLITE_DIR ${LIBRARY}/sqlite-amalgamation-3470100)
add_library(sqlite3 STATIC
    ${SQLITE_DIR}/sqlite3.c
)
target_include_directories(sqlite3 PUBLIC ${SQLITE_DIR})


# Specify the path to your Boost Library 
message(STATUS "Setting (1/3) Boost Universal Library Zephyrine Library Module for C++ Management Engine")
set(BOOST_ROOT "${CMAKE_SOURCE_DIR}/Library/boost_1_86_0")

# Add the Boost root include directory
include_directories(${BOOST_ROOT})

# Add the Boost include directory
#/Users/albertstarfield/Documents/misc/AI/ProjectZephy/systemCore/engineMain/./Library/crow.h:227:10: fatal error: 'boost/algorithm/string/predicate.hpp' file not found
#include <boost/algorithm/string/predicate.hpp>
message(STATUS "Setting (2/3) main Zephyrine Library Module for C++ Management Engine")
include_directories(${BOOST_ROOT}/boost)


message(STATUS "Setting (3/3) main Zephyrine Library nlohmann json Module for C++ Management Engine")
include_directories(${LIBRARY}/jsonLibNlohMann/include)

#Python mixed c++ Library +_+_+_+_+_+_+_

# Detect Python with a version constraint between 3.10 and 3.12
message(STATUS "Checking Python3 main Zephyrine Library Module for Python x C++ Mixed Session Management Engine")
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
if (Python3_VERSION VERSION_GREATER_EQUAL "3.10" AND Python3_VERSION VERSION_LESS "3.13")
    message(STATUS "Found suitable Python3 version: ${Python3_VERSION}")
else()
    message(FATAL_ERROR "Python version must be between 3.10 and 3.12. Found: ${Python3_VERSION}")
endif()

# Determine the correct executable paths and site-packages based on the platform
message(STATUS "Setting main Zephyrine Library Module for Python x C++ Mixed Session Management Engine")
set(PYTHON_VENV_PATH "${LIBRARY}/pythonBridgeRuntime")


# Define a list of preferred Python versions
set(preferred_python_versions "python3.12" "python3.11" "python3.10" "python3")

# Initialize an empty variable for the chosen Python interpreter
set(chosen_python_interpreter "")


# Iterate over preferred versions and find the available one
foreach(version ${preferred_python_versions})
    find_program(PYTHON_EXECUTABLE NAMES ${version})
    if(PYTHON_EXECUTABLE)
        set(chosen_python_interpreter ${PYTHON_EXECUTABLE})
        break()
    endif()
endforeach()

# Ensure a suitable Python interpreter was found
if(NOT chosen_python_interpreter)
    message(FATAL_ERROR "No suitable Python interpreter found for versions 3.10, 3.11, or 3.12.")
endif()

message(STATUS "Using Global Python interpreter: ${chosen_python_interpreter}")
# Create a virtual environment using the chosen Python interpreter
message(STATUS "Creating containerization for Python for Status-Quo ML python-only Libraries Module")
# Create a virtual environment named pythonBridgeRuntime
execute_process(
    COMMAND ${chosen_python_interpreter} -m venv ${PYTHON_VENV_PATH}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)



# Verify and print the site-packages path
set(PYTHON_VENV_PYTHON3_BIN_PATH "${PYTHON_VENV_BIN_PATH}/python3") # Ensure this uses the detected version range

# Verify and print the site-pacset(PYTHON_VENV_PYTHON3_BIN_PATH "${PYTHON_VENV_BIN_PATH}/python3") # Ensure this uses the detected version range
execute_process(
    COMMAND ${PYTHON_VENV_PYTHON3_BIN_PATH} -c "import sysconfig; print(sysconfig.get_paths())"
    OUTPUT_VARIABLE PYTHON_PATHS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

#Checking if the binary version is within 3.10 - 3.12 and if it's not then reject


message(STATUS "Python paths: ${PYTHON_PATHS}")

# Library for Python Venv
message(STATUS "Setting SITE_PACKAGES...")

# Check the platform and set the Python virtual environment path accordingly
if(WIN32)
    # For Windows, set the path to 'Scripts'
    set(PYTHON_VENV_BIN_PATH "${LIBRARY}/pythonBridgeRuntime/Scripts")
    message(STATUS "Detected Windows platform. Using PYTHON_VENV_BIN_PATH: ${PYTHON_VENV_BIN_PATH}")
else()
    # For Unix-like systems, set the path to 'bin'
    set(PYTHON_VENV_BIN_PATH "${LIBRARY}/pythonBridgeRuntime/bin")
    message(STATUS "Detected Unix-like platform. Using PYTHON_VENV_BIN_PATH: ${PYTHON_VENV_BIN_PATH}")
endif()

# Debugging message to ensure the correct path is being set
message(STATUS "PYTHON_VENV_BIN_PATH is set to: ${PYTHON_VENV_BIN_PATH}")

set(PYTHON_VENV_PYTHON3_BIN_PATH "${LIBRARY}/pythonBridgeRuntime/bin/python3") #Change this to enforce python 3.10-3.12

# Upgrade pip inside the virtual environment
execute_process(
    COMMAND ${PYTHON_VENV_BIN_PATH}/pip install --upgrade pip
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
#set PYTHON_SITE_PACKAGES variable through venv python3 bin path 
execute_process(
    COMMAND ${PYTHON_VENV_PYTHON3_BIN_PATH} -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "set SITE_PACKAGES to ${PYTHON_SITE_PACKAGES}")
message(STATUS "If somehow you are having some issue please delete or reset the Library/pythonBridgeRuntime")

## Define python and c++ bridge Library
#pybind11
# Correctly adding the include directory for pybind11
message(STATUS "Setting Bridge pybind11 Zephyrine Library Module for Python x C++ Mixed Session Management Engine")
include_directories(${PYTHON_SITE_PACKAGES}/pybind11/include)
#Corectly include 
message(STATUS "Setting Bridge pybind11 Zephyrine Library Module for Python x C++ Mixed Session Management Engine")
# Add pybind11 and Python include directories
include_directories(${PYTHON_SITE_PACKAGES}/pybind11/include)
include_directories(${Python3_INCLUDE_DIRS})

#--=-=-=- Downloading Python PyPIP standard procedure (all cloud stuff)

#-=-=-=-= Detect Hardware capabilities
# Detect Platform and Hardware Capabilities

message(STATUS "Detecting platform and hardware capabilities for llama-cpp-python installation")

set(CMAKE_ARGS_LIST "")

# Base CMake arguments
list(APPEND CMAKE_ARGS_LIST "-DGGML_BLAS=ON" "-DGGML_BLAS_VENDOR=OpenBLAS")

# Handle platform specific options
if(APPLE)
    message(STATUS "Detected macOS platform")
    list(APPEND CMAKE_ARGS_LIST "-DGGML_METAL=on")
elseif(UNIX)
    message(STATUS "Detected Linux platform")
    if(EXISTS "/opt/intel/oneapi/setvars.sh")
        message(STATUS "Intel oneAPI detected, adding SYCL and RPC support")
        set(SYCL_ENV_SCRIPT "/opt/intel/oneapi/setvars.sh")
        list(APPEND CMAKE_ARGS_LIST "-DGGML_SYCL=on" "-DCMAKE_C_COMPILER=icx" "-DCMAKE_CXX_COMPILER=icpx" "-DGGML_RPC=on")
    endif()
elseif(WIN32)
    message(STATUS "Detected Windows platform")
    list(APPEND CMAKE_ARGS_LIST "-DGGML_CUDA=on") # Assuming CUDA is available
endif()

# Add checks for additional hardware-specific capabilities
# (You might need to customize this section based on the hardware checking tools you have)

# Example: Check for CUDA availability on Linux
if(UNIX AND EXISTS "/usr/local/cuda")
    message(STATUS "CUDA found, enabling CUDA support")
    list(APPEND CMAKE_ARGS_LIST "-DGGML_CUDA=on")
endif()

# Create a string from CMAKE_ARGS_LIST for pip installation
string(JOIN " " CMAKE_ARGS ${CMAKE_ARGS_LIST})

message(STATUS "Downloading Zephyrine pre-requisite Library Status-Quo python-only Libraries Module")
execute_process(
    COMMAND ${PYTHON_VENV_BIN_PATH}/pip install --upgrade pip
    COMMAND ${PYTHON_VENV_BIN_PATH}/pip install pep517 pybind11 llama-index pandas numpy scikit-learn diffusers llama-cpp-python torch keras huggingface
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    RESULT_VARIABLE result
)
# Check if the previous commands were successful
if(NOT ${result} EQUAL 0)
    message(FATAL_ERROR "Failed to install required Python packages in the virtual environment.")
endif()
# Status
message(STATUS "Enforcing python root dir to use locally not Globally...")
set(Python_ROOT_DIR "${PYTHON_VENV_PATH}")

# refer to pybind11 documentation
#   https://pybind11.readthedocs.io/en/stable/compiling.html

# https://github.com/pybind/pybind11/blob/master/tools/pybind11Config.cmake.in

message(STATUS "Configuring pybind11...")
message(STATUS "Adding discovery pybind11 using local python venv...")
set(pybind11_DIR "${PYTHON_SITE_PACKAGES}/pybind11/share/cmake/pybind11/")
message(STATUS "Registering pybind11 using Documented https://pybind11.readthedocs.io/en/stable/compiling.html classic find_package ${pybind11_DIR}")
find_package(pybind11 CONFIG REQUIRED)
message(STATUS "Adding library module via pybind11 to C++")


# Commands are from this documentation https://github.com/pybind/pybind11/blob/master/tools/pybind11Config.cmake.in

#pybind11_add_module(llama-index ${mainEngineCode})
#target_link_libraries(llama-index PUBLIC pybind11::module)
#pybind11_add_module(pandas ${mainEngineCode})
#target_link_libraries(pandas PUBLIC pybind11::module)
#pybind11_add_module(numpy ${mainEngineCode})
#target_link_libraries(numpy PUBLIC pybind11::module)
#pybind11_add_module(scikit-learn ${mainEngineCode})
#target_link_libraries(scikit-learn PUBLIC pybind11::module)
#pybind11_add_module(diffusers ${mainEngineCode})
#target_link_libraries(diffusers PUBLIC pybind11::module)
#pybind11_add_module(llama-cpp-python ${mainEngineCode})
#target_link_libraries(llama-cpp-python PUBLIC pybind11::module)
#pybind11_add_module(torch ${mainEngineCode})
#target_link_libraries(torch PUBLIC pybind11::module)
#pybind11_add_module(keras ${mainEngineCode})
#target_link_libraries(keras PUBLIC pybind11::module)
#pybind11_add_module(huggingface ${mainEngineCode})
#target_link_libraries(huggingface PUBLIC pybind11::module)


#Debug Variable
message(STATUS "Make sure that you execute compile.sh or CMAKELISTS on the ./systemCore/engineMain directory!")
#---------

# Upgrade pip and install pybind11, llamaindex, and llama-cpp-python inside the virtual environment
#Required Python install module for c++ usage (Ironic isn't it?)

message(STATUS "Adding definitions for Python site packages")

# Add definitions for virtual environment site-packages path
add_definitions(-DPYTHON_SITE_PACKAGES="${PYTHON_SITE_PACKAGES}")

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#Compilation

# Add the executable target
add_executable(main ${mainEngineCode})




# Link the curl library
target_link_libraries(main PRIVATE curl)  

# Find the SQLite3 library
#find_library(SQLITE3_LIBRARIES sqlite3 REQUIRED) ( We don't need sqlite3 global system dependent library anymore because we have local and portable library already)

# Link the SQLite3 library
target_link_libraries(main PRIVATE sqlite3) #imported from local

#embed python3 interpreter into the ./main engine using pybind11
target_link_libraries(main PUBLIC pybind11::embed)

# Link Python libraries
target_link_libraries(main PRIVATE Python3::Python)

# Include directories for your libraries
target_include_directories(main PRIVATE ${Python3_INCLUDE_DIRS} ${PYBIND11_INCLUDE_DIRS} ${CURL_INCLUDE_DIRS})

