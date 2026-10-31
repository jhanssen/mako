cmake_minimum_required(VERSION 4.3)
project(mako LANGUAGES CXX)

if(NOT APPLE OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    message(FATAL_ERROR "mako targets Apple Silicon only")
endif()

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_OSX_DEPLOYMENT_TARGET 26.0)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
    message(STATUS "CMAKE_BUILD_TYPE not set: using RelWithDebInfo")
endif()

add_compile_options(-Wall -Wextra -Wpedantic)
# Warnings fail the build (C++ and Metal); cmake --compile-no-warning-error turns that off for one configure.
set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

# The Jinja engine's files come with mako-common's llama.cpp sources (set before it is added).
include(cmake/jinja_files.cmake)
if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/external/mako-common/CMakeLists.txt)
    message(FATAL_ERROR "external/mako-common is missing: git submodule update --init")
endif()
add_subdirectory(external/mako-common)
include(cmake/jinja.cmake)

add_subdirectory(src)
add_subdirectory(bench)
add_subdirectory(tools)
# Model-backed tests (ctest) need the first shard of the GGUF; without it they report as skipped.
set(MAKO_MODEL "$ENV{MAKO_MODEL}" CACHE FILEPATH "First GGUF shard for the tests")
set(MAKO_DS41_MODEL "$ENV{MAKO_DS41_MODEL}" CACHE FILEPATH "First DeepSeek-V4.1 GGUF shard for its tests")
set(MAKO_DS41_ENGRAM "$ENV{MAKO_DS41_ENGRAM}" CACHE FILEPATH "A GGUF with other Engram tables for DeepSeek-V4.1's tests (default: the model's own)")
set(MAKO_DS41_ABLATION "$ENV{MAKO_DS41_ABLATION}" CACHE FILEPATH "An ablation file for DeepSeek-V4.1 (optional test input)")
set(MAKO_DS41_DSPARK "$ENV{MAKO_DS41_DSPARK}" CACHE PATH "DeepSeek-V4.1's DSpark drafter directory (optional test input)")
enable_testing()
add_subdirectory(tests)
