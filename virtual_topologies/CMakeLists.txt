find_package(MPI)

add_executable(test test.cpp)
add_executable(flow_graph flow_graph.cpp)
add_executable(dependency_graph dependency_graph.cpp)
add_executable(odd_even_sort odd_even_sort.cpp)
add_executable(linear linear.cpp)
add_executable(linear_other linear_other.cpp)
add_executable(ring ring.cpp)
add_executable(ring_other ring_other.cpp)

target_link_libraries(test openmpi::openmpi)
target_link_libraries(flow_graph openmpi::openmpi)
target_link_libraries(dependency_graph openmpi::openmpi)
target_link_libraries(odd_even_sort openmpi::openmpi)
target_link_libraries(linear openmpi::openmpi)
target_link_libraries(linear_other openmpi::openmpi)
target_link_libraries(ring openmpi::openmpi)
target_link_libraries(ring_other openmpi::openmpi)