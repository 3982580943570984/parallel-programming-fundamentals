find_package(MPI)

add_executable(test test.cpp)
add_executable(linear linear.cpp)
add_executable(linear_other linear_other.cpp)
add_executable(ring ring.cpp)
add_executable(ring_other ring_other.cpp)

target_link_libraries(test openmpi::openmpi)
target_link_libraries(linear openmpi::openmpi)
target_link_libraries(linear_other openmpi::openmpi)
target_link_libraries(ring openmpi::openmpi)
target_link_libraries(ring_other openmpi::openmpi)