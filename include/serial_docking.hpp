#pragma once

#include "geometry.hpp"
#include "interaction_scoring.hpp"
#include "run_config.hpp"

#include <cstddef>
#include <vector>


struct DockedPose
{
    RigidTransform transform{};

    double fast_score{};

    ExactInteractionScore exact_score{};

    std::size_t occupied_atoms{};
};


struct SerialDockingTimings
{
    double offline_target_preparation_seconds{};

    double ligand_preparation_seconds{};

    double protein_triplet_index_seconds{};
    double matching_kabsch_seconds{};
    double grid_clash_seconds{};

    double fast_scoring_seconds{};
    double exact_scoring_seconds{};

    double online_total_seconds{};
    double total_seconds{};
};


struct SerialDockingResult
{
    std::vector<DockedPose> top_poses;

    SerialDockingTimings timings;
};


SerialDockingResult run_serial_docking(
    const RunConfig& config,
    std::size_t top_n = 20
);
