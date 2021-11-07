cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(task1_2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(task1_2 task1_2.cpp)

add_library(helper INTERFACE ../helper/helpers.hpp)

target_include_directories(task1_2 PRIVATE ../helper)
target_link_libraries(task1_2 helper)

