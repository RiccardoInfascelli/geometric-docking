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
    src/interaction_scoring.cpp
    src/clash.cpp
    src/diagnostics.cpp
    src/pose_ranking.cpp
    src/run_config.cpp
    src/pipeline.cpp
    src/report.cpp
    src/pose_output.cpp
    src/benchmark_csv.cpp
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

# PROV
add_executable(
    triplet_failure_diagnostic
    apps/triplet_failure_diagnostic_main.cpp
)

target_link_libraries(
    triplet_failure_diagnostic
    PRIVATE docking_core
)

#PROV
add_executable(
    clash_threshold_experiment
    apps/clash_threshold_experiment_main.cpp
)

target_link_libraries(
    clash_threshold_experiment
    PRIVATE
        docking_core
)

bsl_set_cxx_flags(
    clash_threshold_experiment
)
add_executable(
    pose_target_shift_experiment
    apps/pose_target_shift_experiment_main.cpp
)

target_link_libraries(
    pose_target_shift_experiment
    PRIVATE
        docking_core
)

bsl_set_cxx_flags(
    pose_target_shift_experiment
)

add_executable(
    offline_profile_experiment
    apps/offline_profile_experiment_main.cpp
)

target_link_libraries(
    offline_profile_experiment
    PRIVATE
        docking_core
)

bsl_set_cxx_flags(
    offline_profile_experiment
)

add_executable(
    pocket_area_experiment
    apps/pocket_area_experiment_main.cpp
)

target_link_libraries(
    pocket_area_experiment
    PRIVATE
        docking_core
)

bsl_set_cxx_flags(
    pocket_area_experiment
)

add_executable(
    area_accessibility_experiment
    apps/area_accessibility_experiment_main.cpp
)

target_link_libraries(
    area_accessibility_experiment
    PRIVATE
        docking_core
)

bsl_set_cxx_flags(
    area_accessibility_experiment
)

add_executable(
    clearance_pruning_experiment
    apps/clearance_pruning_experiment_main.cpp
)

target_link_libraries(
    clearance_pruning_experiment
    PRIVATE
        docking_core
)

bsl_set_cxx_flags(
    clearance_pruning_experiment
)

add_executable(
    pocket_openness_experiment
    apps/pocket_openness_experiment_main.cpp
)

target_link_libraries(
    pocket_openness_experiment
    PRIVATE
        docking_core
)

bsl_set_cxx_flags(
    pocket_openness_experiment
)

# ============================================================
# Serial reference docking
# ============================================================

add_executable(
    serial_docking
    apps/serial_main.cpp
    src/serial_docking.cpp
)

target_link_libraries(
    serial_docking
    PRIVATE
        docking_core
)

bsl_set_cxx_flags(
    serial_docking
)
