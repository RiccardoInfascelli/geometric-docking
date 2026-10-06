#ifndef PIPELINE_HPP
#define PIPELINE_HPP

#include "diagnostics.hpp"
#include "pose_generation.hpp"
#include "pose_ranking.hpp"
#include "run_config.hpp"

#include <cstddef>
#include <string>


struct RunTimings
{
    // Preparazione riutilizzabile del target:
    // protein/pocket -> areas -> grid -> pocket region -> pose targets
    double offline_target_preparation_seconds{};

    // Preparazione specifica del ligando:
    // load ligand -> interaction/matching areas -> diameter -> triplets
    double online_ligand_preparation_seconds{};

    // Ricerca delle pose:
    // protein-triplet index + triplet lookup + transforms + grid/clash
    //
    // NOTA: il protein-triplet index è qui perché attualmente
    // dipende dal ligand_diameter.
    double online_pose_search_seconds{};

    // Fast scoring + prefilter + exact scoring + ranking
    double online_scoring_seconds{};

    // ligand preparation + pose search + scoring
    double online_total_seconds{};

    // offline + online
    double total_seconds{};
};


struct RunResult
{
    std::string complex_id;

    std::size_t protein_atoms{};
    std::size_t pocket_atoms{};
    std::size_t ligand_atoms{};

    std::size_t protein_areas{};
    std::size_t pocket_protein_areas{};
    std::size_t ligand_areas{};

    std::size_t protein_matching_areas{};
    std::size_t ligand_matching_areas{};

    double ligand_diameter{};
    std::size_t ligand_triplets{};

    double pocket_size_x{};
    double pocket_size_y{};
    double pocket_size_z{};

    int grid_nx{};
    int grid_ny{};
    int grid_nz{};

    std::size_t total_voxels{};
    std::size_t occupied_voxels{};
    std::size_t interaction_voxels{};
    std::size_t multi_interaction_voxels{};

    std::size_t crystal_grid_total{};
    std::size_t crystal_grid_in_grid{};
    std::size_t crystal_grid_occupied{};
    std::size_t crystal_grid_compatible{};

    std::size_t crystal_exact_interactions{};
    std::size_t crystal_exact_ligand_areas{};

    std::size_t pocket_exact_interactions{};
    std::size_t pocket_exact_ligand_areas{};

    std::size_t pocket_matching_areas{};
    std::size_t pose_targets{};
    std::size_t pose_targets_missing{};

    MatchingAreaDistanceDiagnostic
        matching_area_distance_diagnostic;

    PoseTargetCrystalDiagnostic
        pose_target_crystal_diagnostic;

    PoseGenerationResult
        pose_generation;

    PoseRankingResult
        pose_ranking;

    RunTimings timings;
};


RunResult run_pipeline(
    const RunConfig& config
);

#endif