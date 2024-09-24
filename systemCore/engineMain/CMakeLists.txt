# Minimum version of CMake required
cmake_minimum_required(VERSION 3.10)

# Project name and version
project(MyProject VERSION 1.0)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

#--------- Definition of the Modules/Library (Mixed with Python + C++) 

#Raw C++ Library +_+_+_+_+_+_+_
#Main directory for Modules Library directory
message(STATUS "Setting main Zephyrine Library Module for C++ Management Engine LIBRARY ${LIBRARY}")
set(LIBRARY "${CMAKE_SOURCE_DIR}/Library")

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

# Optionally specify Boost libraries if you need to link specific Boost components
message(STATUS "Setting (3/3) main Zephyrine Library Module for C++ Management Engine")
link_directories(${BOOST_ROOT}/stage/lib)  # Adjust this if Boost libraries are built in a different location


#Python mixed c++ Library +_+_+_+_+_+_+_

# Detect Python
message(STATUS "Checking Python3 main Zephyrine Library Module for Python x C++ Mixed Session Management Engine")
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Determine the correct executable paths and site-packages based on the platform
message(STATUS "Setting main Zephyrine Library Module for Python x C++ Mixed Session Management Engine")
set(PYTHON_VENV_PATH "${LIBRARY}/pythonBridgeRuntime")

message(STATUS "Creating containerization for Python for Status-Quo ML python-only Libraries Module")
# Create a virtual environment named pythonBridgeRuntime
execute_process(
    COMMAND python3 -m venv ${PYTHON_VENV_PATH}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
message(STATUS "Setting SITE_PACKAGES...")
set(PYTHON_VENV_BIN_PATH "${LIBRARY}/pythonBridgeRuntime/bin")

set(PYTHON_VENV_PYTHON3_BIN_PATH "${LIBRARY}/pythonBridgeRuntime/bin/python3")

execute_process(
    COMMAND ${PYTHON_VENV_PYTHON3_BIN_PATH} -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "set SITE_PACKAGES to ${PYTHON_SITE_PACKAGES}")
#set(PYTHON_SITE_PACKAGES "${LIBRARY}/pythonBridgeRuntime/lib/python3.12/site-packages") #How to make sure its adaptive and not locked to python 3.12

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


message(STATUS "Downloading Zephyrine pre-requisite Library Status-Quo python-only Libraries Module")
execute_process(
    COMMAND ${PYTHON_VENV_BIN_PATH}/pip install --upgrade pip
    COMMAND ${PYTHON_VENV_BIN_PATH}/pip install pybind11 llama-index pandas numpy scikit-learn diffusers llama-cpp-python torch keras huggingface llama-cpp-python
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    RESULT_VARIABLE result
)

# Check if the previous commands were successful
if(NOT ${result} EQUAL 0)
    message(FATAL_ERROR "Failed to install required Python packages in the virtual environment.")
endif()
# Status

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
add_executable(main main.cpp)

# Link Python libraries
target_link_libraries(main PRIVATE Python3::Python)
