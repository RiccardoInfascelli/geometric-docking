cmake_minimum_required(VERSION 3.20)

project(geometric_docking LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 REQUIRED)

add_subdirectory(
    ../chem-interactions
    ${CMAKE_BINARY_DIR}/chem-interactions
)

add_executable(
    geometric_docking
    src/main.cpp
    src/molecule_io.cpp
    src/interaction_areas.cpp
)

target_include_directories(
    geometric_docking
    PRIVATE
    include
)

target_link_libraries(
    geometric_docking
    PRIVATE
    libinteraction
    Eigen3::Eigen
)