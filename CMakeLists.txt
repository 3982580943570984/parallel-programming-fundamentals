cmake_minimum_required(VERSION 3.31)

project(parallel_programming CXX)

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_LINKER_TYPE MOLD)

add_subdirectory(gas_station)
add_subdirectory(racing_competition)
add_subdirectory(mpi_basics)