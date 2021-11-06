cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(task1_1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(task1_1 task1_1.cpp)

add_library(helper INTERFACE ../helper/helpers.hpp)

target_include_directories(task1_1 PRIVATE ../helper)
target_link_libraries(task1_1 helper)
