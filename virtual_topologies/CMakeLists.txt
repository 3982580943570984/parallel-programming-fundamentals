find_package(MPI)

add_executable(linear linear.cpp)
add_executable(ring ring.cpp)

target_link_libraries(linear openmpi::openmpi)
target_link_libraries(ring openmpi::openmpi)