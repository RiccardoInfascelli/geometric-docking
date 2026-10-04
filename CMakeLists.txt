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

# --------------------------------------------------
# Core library
# --------------------------------------------------

add_library(
    docking_core
    src/molecule_io.cpp
    src/interaction_areas.cpp
    src/geometry.cpp
    src/triplet_generation.cpp
    src/matching_areas.cpp
    src/matching_grid.cpp
    src/protein_grid.cpp
    src/pose_generation.cpp
    src/pose_targets.cpp
)

target_include_directories(
    docking_core
    PUBLIC
    include
)

target_link_libraries(
    docking_core
    PUBLIC
    libinteraction
    Eigen3::Eigen
)

# --------------------------------------------------
# Main application
# --------------------------------------------------

add_executable(
    geometric_docking
    apps/docking_main.cpp
)

target_link_libraries(
    geometric_docking
    PRIVATE
    docking_core
)