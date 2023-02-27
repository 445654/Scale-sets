cmake_minimum_required(VERSION 3.22)
project(scale-sets)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include(FetchContent)
FetchContent_Declare(
        argparse
        GIT_REPOSITORY https://github.com/p-ranav/argparse.git
)
FetchContent_MakeAvailable(argparse)

add_executable(scale-sets main.cpp)
target_link_libraries(scale-sets ${OpenCV_LIBS})
target_link_libraries(scale-sets argparse)