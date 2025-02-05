# Minimum version of CMake required
cmake_minimum_required(VERSION 3.10)

# Project name and version
project(AdelaideandAlbertEngine VERSION 1.0)

# Since we're only building a Python application, we don't need to set C++ standards
# set(CMAKE_CXX_STANDARD 20)
# set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set the build type if not already set
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

#--------- Python Environment Setup ---------

# Detect Python with a version constraint between 3.10 and 3.12
message(STATUS "Checking for suitable Python interpreter...")

# Define a list of preferred Python versions
set(preferred_python_versions "3.12" "3.11" "3.10")

# Initialize variables
set(chosen_python_interpreter "")
set(chosen_python_version "")

# Iterate over preferred versions and find an available one
foreach(version ${preferred_python_versions})
    find_program(PYTHON_EXECUTABLE NAMES python${version})
    if(PYTHON_EXECUTABLE)
        execute_process(
            COMMAND ${PYTHON_EXECUTABLE} -c "import sys; print(sys.version[:4])"
            OUTPUT_VARIABLE temp_version
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )

        if(temp_version VERSION_GREATER_EQUAL "3.10" AND temp_version VERSION_LESS "3.13")
            set(chosen_python_interpreter ${PYTHON_EXECUTABLE})
            set(chosen_python_version ${temp_version})
            break()
        endif()
    endif()
endforeach()

# Ensure a suitable Python interpreter was found
if(NOT chosen_python_interpreter)
    message(FATAL_ERROR "No suitable Python interpreter found for versions 3.10, 3.11, or 3.12.")
else()
    message(STATUS "Found suitable Python version: ${chosen_python_version} with interpreter: ${chosen_python_interpreter}")
endif()

# Set Python virtual environment path
set(PYTHON_VENV_PATH "${CMAKE_SOURCE_DIR}/Library/pythonBridgeRuntime")

# Function to delete the pythonBridgeRuntime directory if it exists
function(delete_python_bridge_runtime)
  if(EXISTS "${PYTHON_VENV_PATH}")
    message(STATUS "Deleting existing pythonBridgeRuntime directory...")
    file(REMOVE_RECURSE "${PYTHON_VENV_PATH}")
  endif()
endfunction()

# Create a virtual environment using the chosen Python interpreter
execute_process(
    COMMAND ${chosen_python_interpreter} -m venv ${PYTHON_VENV_PATH}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    RESULT_VARIABLE CREATE_VENV_RESULT
)

# Check if virtual environment creation was successful
if(NOT ${CREATE_VENV_RESULT} EQUAL 0)
  # If creation failed, delete the directory and try again
  delete_python_bridge_runtime()
  execute_process(
      COMMAND ${chosen_python_interpreter} -m venv ${PYTHON_VENV_PATH}
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )
endif()

# Determine the correct executable paths and site-packages based on the platform
if(WIN32)
    set(PYTHON_VENV_BIN_PATH "${PYTHON_VENV_PATH}/Scripts")
else()
    set(PYTHON_VENV_BIN_PATH "${PYTHON_VENV_PATH}/bin")
endif()

set(PYTHON_VENV_PYTHON3_BIN_PATH "${PYTHON_VENV_BIN_PATH}/python3")

# Upgrade pip inside the virtual environment
execute_process(
    COMMAND ${PYTHON_VENV_BIN_PATH}/pip install --upgrade pip
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    RESULT_VARIABLE UPGRADE_PIP_RESULT
)

# Check if pip upgrade was successful
if(NOT ${UPGRADE_PIP_RESULT} EQUAL 0)
  delete_python_bridge_runtime()
  message(FATAL_ERROR "Failed to upgrade pip in the virtual environment. Deleted pythonBridgeRuntime.")
endif()

# Get the site-packages path from the virtual environment
execute_process(
    COMMAND ${PYTHON_VENV_PYTHON3_BIN_PATH} -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

#--------- Hardware Acceleration Detection ---------

# Detect CUDA
find_program(CUDA_EXECUTABLE NAMES nvcc)
if(CUDA_EXECUTABLE)
    message(STATUS "CUDA detected. Enabling CUDA for llama-cpp-python and stable-diffusion-cpp-python.")
    set(LLAMA_CPP_CMAKE_ARGS "-DGGML_CUDA=on" CACHE STRING "CMake arguments for llama-cpp-python" FORCE)
    set(STABLE_DIFFUSION_CMAKE_ARGS "-DSD_CUDA=ON" CACHE STRING "CMake arguments for stable-diffusion-cpp-python" FORCE)
else()
    message(STATUS "CUDA not found.")
    # Detect ROCm (AMD)
    find_program(ROCM_EXECUTABLE NAMES hipcc)
    if(ROCM_EXECUTABLE)
        message(STATUS "ROCm detected. Enabling ROCm/hipBLAS for llama-cpp-python and stable-diffusion-cpp-python.")
        # Get AMD GPU target - Replace gfxXXXX with your GPU architecture if known, or detect it in a more sophisticated way if needed.
        # Detecting AMD GPU target in CMake can be complex and might require external scripts. A simple placeholder is used here.
        set(AMD_GPU_TARGET "gfx90a" CACHE STRING "AMD GPU Target" FORCE)
        set(LLAMA_CPP_CMAKE_ARGS "-DGGML_HIPBLAS=on" CACHE STRING "CMake arguments for llama-cpp-python" FORCE)
        set(STABLE_DIFFUSION_CMAKE_ARGS "-G Ninja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DSD_HIPBLAS=ON -DCMAKE_BUILD_TYPE=Release -DAMDGPU_TARGETS=${AMD_GPU_TARGET}" CACHE STRING "CMake arguments for stable-diffusion-cpp-python" FORCE)
    else()
        message(STATUS "ROCm not found.")
        # Detect Metal (Apple)
        if(APPLE)
            message(STATUS "Apple platform detected. Enabling Metal for llama-cpp-python and stable-diffusion-cpp-python.")
            set(LLAMA_CPP_CMAKE_ARGS "-DGGML_METAL=on" CACHE STRING "CMake arguments for llama-cpp-python" FORCE)
            set(STABLE_DIFFUSION_CMAKE_ARGS "-DSD_METAL=ON" CACHE STRING "CMake arguments for stable-diffusion-cpp-python" FORCE)
        else()
            message(STATUS "Apple Metal not available.")
            # Detect SYCL (Intel)
            find_program(INTEL_ONEAPI_EXECUTABLE NAMES setvars.sh PATHS "/opt/intel/oneapi/" NO_DEFAULT_PATH) # Adjust path if necessary
            if (INTEL_ONEAPI_EXECUTABLE)
                message(STATUS "Intel oneAPI detected. Enabling SYCL for llama-cpp-python and stable-diffusion-cpp-python.")
                execute_process(
                    COMMAND bash -c "source ${INTEL_ONEAPI_EXECUTABLE} && env"
                    OUTPUT_VARIABLE env_output
                )
                string(REGEX MATCHALL "([^=]+)=([^\n]+)" SYCL_ENV_LIST "${env_output}")
                foreach(entry ${SYCL_ENV_LIST})
                    string(REGEX REPLACE "([^=]+)=([^\n]+)" "\\1" key "${entry}")
                    string(REGEX REPLACE "([^=]+)=([^\n]+)" "\\2" value "${entry}")
                    set(ENV{${key}} "${value}")
                endforeach()
                set(LLAMA_CPP_CMAKE_ARGS "-DGGML_SYCL=on -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx" CACHE STRING "CMake arguments for llama-cpp-python" FORCE)
                set(STABLE_DIFFUSION_CMAKE_ARGS "-DSD_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx" CACHE STRING "CMake arguments for stable-diffusion-cpp-python" FORCE)
            else()
               message(STATUS "Intel oneAPI not found.")
                # Fallback to OpenBLAS
                message(STATUS "No specific GPU acceleration detected. Enabling OpenBLAS for llama-cpp-python.")
                set(LLAMA_CPP_CMAKE_ARGS "-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS" CACHE STRING "CMake arguments for llama-cpp-python" FORCE)
                set(STABLE_DIFFUSION_CMAKE_ARGS "-DGGML_OPENBLAS=ON" CACHE STRING "CMake arguments for stable-diffusion-cpp-python" FORCE)

            endif()
        endif()
    endif()
endif()
#--------- Install Required Python Packages ---------

message(STATUS "Installing required Python packages into virtual environment...")

# Set environment variables for llama-cpp-python and stable-diffusion-cpp-python installation
set(ENV{CMAKE_ARGS} "${LLAMA_CPP_CMAKE_ARGS}")
set(ENV{CMAKE_ARGS} "${STABLE_DIFFUSION_CMAKE_ARGS}")

# Install PyInstaller and other necessary packages
execute_process(
    COMMAND ${PYTHON_VENV_BIN_PATH}/pip install pyinstaller fuzzywuzzy stable-diffusion-cpp-python python-Levenshtein recoll pep517 pybind11 tiktoken transformers llama-index pandas numpy scikit-learn diffusers llama-cpp-python torch keras huggingface langchain langchain_community langchain_experimental colored faiss-cpu
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    RESULT_VARIABLE INSTALL_PACKAGES_RESULT
)

if(NOT ${INSTALL_PACKAGES_RESULT} EQUAL 0)
  delete_python_bridge_runtime()
  message(FATAL_ERROR "Failed to install required Python packages. Deleted pythonBridgeRuntime.")
endif()

#--------- Build Executable with PyInstaller ---------

# Find the path to your Python script
set(PYTHON_SCRIPT "${CMAKE_SOURCE_DIR}/experimentalpythoninference.py") # Adjust if your script is elsewhere

# Set the output directory for the PyInstaller-built executable
set(PYINSTALLER_OUTPUT_DIR "${CMAKE_BINARY_DIR}/dist")

# Create a custom target to build the executable using PyInstaller
add_custom_target(build_python_app
    COMMAND ${PYTHON_VENV_BIN_PATH}/pyinstaller --onefile --name AdelaideandAlbertEngine "${PYTHON_SCRIPT}"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Building Python application with PyInstaller..."
)

# Optional: Add a message to indicate completion
add_custom_command(TARGET build_python_app POST_BUILD
    COMMAND ;
    COMMENT "Python application built successfully in ${PYINSTALLER_OUTPUT_DIR}"
)

#--------- Clean Target ---------

# Add a custom target to clean up the PyInstaller build artifacts and virtual environment
add_custom_target(clean_python_build
    COMMAND ${CMAKE_COMMAND} -E remove_directory "${PYINSTALLER_OUTPUT_DIR}"
    COMMAND ${CMAKE_COMMAND} -E remove_directory "${CMAKE_BINARY_DIR}/build"
    COMMAND ${CMAKE_COMMAND} -E remove_directory "${PYTHON_VENV_PATH}"  
    COMMENT "Cleaning Python build artifacts and virtual environment..."
)