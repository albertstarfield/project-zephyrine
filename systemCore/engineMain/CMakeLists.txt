# Minimum version of CMake required
cmake_minimum_required(VERSION 3.10)

# Project name and version
project(MyProject VERSION 1.0)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Specify the path to your Boost installation
set(BOOST_ROOT "${CMAKE_SOURCE_DIR}/systemCore/engineMain/Library/boost_1_86_0")

# Add the Boost include directory
include_directories(${BOOST_ROOT})

# Optionally specify Boost libraries if you need to link specific Boost components
# link_directories(${BOOST_ROOT}/stage/lib)  # Adjust this if Boost libraries are built in a different location

# Detect Python
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Determine the correct executable paths and site-packages based on the platform
if(WIN32)
    # For Windows
    set(PYTHON_VENV_PATH "${CMAKE_BINARY_DIR}/pythonBridgeRuntime/Scripts")
    set(PYTHON_SITE_PACKAGES "${CMAKE_BINARY_DIR}/pythonBridgeRuntime/Lib/site-packages")
else()
    # For Unix-like systems
    set(PYTHON_VENV_PATH "${CMAKE_BINARY_DIR}/pythonBridgeRuntime/bin")
    execute_process(
        COMMAND ${PYTHON_VENV_PATH}/python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"
        OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
endif()

# Create a virtual environment named pythonBridgeRuntime
execute_process(
    COMMAND ${Python3_EXECUTABLE} -m venv ${CMAKE_BINARY_DIR}/pythonBridgeRuntime
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Upgrade pip and install pybind11, llamaindex, and llama-cpp-python inside the virtual environment
execute_process(
    COMMAND ${PYTHON_VENV_PATH}/pip install --upgrade pip
    COMMAND ${PYTHON_VENV_PATH}/pip install pybind11 llamaindex llama-cpp-python
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    RESULT_VARIABLE result
)

# Check if the previous commands were successful
if(NOT ${result} EQUAL 0)
    message(FATAL_ERROR "Failed to install required Python packages in the virtual environment.")
endif()

# Add pybind11 and Python include directories
include_directories(${PYTHON_SITE_PACKAGES}/pybind11/include)
include_directories(${Python3_INCLUDE_DIRS})

# Add definitions for virtual environment site-packages path
add_definitions(-DPYTHON_SITE_PACKAGES="${PYTHON_SITE_PACKAGES}")

# Add the executable target
add_executable(MyExecutable main.cpp)

# Link Python libraries
target_link_libraries(MyExecutable PRIVATE Python3::Python)

# Link Boost libraries if needed (example shown for Boost system library)
# target_link_libraries(MyExecutable PRIVATE Boost::system)
