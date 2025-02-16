find_package(MPI REQUIRED)

add_executable(race_simple race_simple.cpp)
add_executable(race race.cpp)
add_executable(multiplication_simple multiplication_simple.cpp)
add_executable(multiplication multiplication.cpp)

target_link_libraries(race_simple openmpi::openmpi)
target_link_libraries(race openmpi::openmpi)
target_link_libraries(multiplication_simple openmpi::openmpi)
target_link_libraries(multiplication openmpi::openmpi)