find_package(nlohmann_json REQUIRED)

add_executable(gas_station main.cpp)

target_link_libraries(gas_station nlohmann_json::nlohmann_json)