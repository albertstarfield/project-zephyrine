cmake_minimum_required(VERSION 3.10)
project(DummyZephy)

set(CMAKE_CXX_STANDARD 17)

# Include the current directory (or the directory where crow.h is located)
include_directories(${CMAKE_SOURCE_DIR})

add_executable(my_crow_app dummy.cpp)

# Link with Boost
find_package(Boost REQUIRED)
target_link_libraries(my_crow_app Boost::boost Boost::system)
