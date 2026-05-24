cmake_minimum_required(VERSION 3.15)
project(Supertonic_C_Wrapper)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG -ffast-math")

# Location of Supertonic clone
set(SUPERTONIC_DIR "../../../../supertonic")

# Add Supertonic C++ source files
set(SUPERTONIC_SRC
    supertonic_c_api.cpp
    ${SUPERTONIC_DIR}/cpp/helper.cpp
)

# Include directories
include_directories(${SUPERTONIC_DIR}/cpp)

# Find ONNX Runtime (using multiple methods similar to Supertonic's CMake)
find_package(PkgConfig REQUIRED)
find_package(onnxruntime QUIET CONFIG)

if(NOT onnxruntime_FOUND)
    pkg_check_modules(ONNXRUNTIME QUIET libonnxruntime)
    if(ONNXRUNTIME_FOUND)
        set(ONNXRUNTIME_INCLUDE_DIR ${ONNXRUNTIME_INCLUDE_DIRS})
        set(ONNXRUNTIME_LIB ${ONNXRUNTIME_LIBRARIES})
    else()
        find_path(ONNXRUNTIME_INCLUDE_DIR 
            NAMES onnxruntime_cxx_api.h
            PATHS /usr/local/include /opt/homebrew/include /usr/include ${CMAKE_PREFIX_PATH}/include
            PATH_SUFFIXES onnxruntime
        )
        find_library(ONNXRUNTIME_LIB
            NAMES onnxruntime libonnxruntime
            PATHS /usr/local/lib /opt/homebrew/lib /usr/lib ${CMAKE_PREFIX_PATH}/lib
        )
    endif()
endif()

find_package(nlohmann_json REQUIRED)
find_package(OpenMP)

if(NOT onnxruntime_FOUND)
    include_directories(${ONNXRUNTIME_INCLUDE_DIR})
endif()

# Build the shared wrapper library
add_library(supertonic_c SHARED ${SUPERTONIC_SRC})

if(onnxruntime_FOUND)
    target_link_libraries(supertonic_c
        onnxruntime::onnxruntime
        nlohmann_json::nlohmann_json
    )
else()
    target_link_libraries(supertonic_c
        ${ONNXRUNTIME_LIB}
        nlohmann_json::nlohmann_json
    )
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(supertonic_c OpenMP::OpenMP_CXX)
endif()
