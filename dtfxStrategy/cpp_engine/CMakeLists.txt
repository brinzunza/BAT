cmake_minimum_required(VERSION 3.15)
project(dtfx_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
  OUTPUT_VARIABLE pybind11_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 CONFIG REQUIRED)
find_package(Threads REQUIRED)

pybind11_add_module(dtfx_engine
  src/pivots.cpp
  src/structure_fsm.cpp
  src/zone_detect.cpp
  src/backtest.cpp
  src/monte_carlo.cpp
  src/features.cpp
  src/bindings.cpp
)

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mcpu=native" HAS_MCPU_NATIVE)
if(HAS_MCPU_NATIVE)
  target_compile_options(dtfx_engine PRIVATE -mcpu=native)
else()
  check_cxx_compiler_flag("-march=native" HAS_MARCH_NATIVE)
  if(HAS_MARCH_NATIVE)
    target_compile_options(dtfx_engine PRIVATE -march=native)
  endif()
endif()

target_compile_options(dtfx_engine PRIVATE -O3 -Wall -Wextra)
target_link_libraries(dtfx_engine PRIVATE Threads::Threads)
