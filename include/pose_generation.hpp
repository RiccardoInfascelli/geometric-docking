#ifndef POSE_GENERATION_HPP
#define POSE_GENERATION_HPP

#include "geometry.hpp"
#include "interaction_scoring.hpp"
#include "matching_areas.hpp"
#include "pose_targets.hpp"
#include "protein_grid.hpp"
#include "triplet_generation.hpp"

#include <GraphMol/ROMol.h>

#include <array>
#include <cstddef>
#include <limits>
#include <vector>


struct RankedPoseDiagnostic
{
    std::size_t compatible_areas{};
    std::size_t occupied_atoms{};

    double clash_fraction{};
    double rmsd{};

    RigidTransform transform{};

    double fast_weighted_score{};

    ExactInteractionScore exact_score{};
};

struct PoseSearchTimings
{
    double protein_triplet_index_seconds{};

    double matching_kabsch_seconds{};

    double grid_clash_seconds{};

    // Diagnostica/oracle: NON fa parte
    // dell'algoritmo produttivo.
    double oracle_rmsd_seconds{};
};


struct PoseGenerationResult
{
    std::vector<RankedPoseDiagnostic>
        surviving_poses;


    // Protein triplet hash
    std::size_t protein_triplet_records{};
    std::size_t protein_hash_keys{};


    // Triplet lookup
    std::size_t ligand_triplets_with_match{};
    std::size_t total_candidate_matches{};
    std::size_t largest_bucket{};


    // Transform generation
    std::size_t transforms_generated{};
    std::size_t transforms_valid{};
    std::size_t poses_with_grid_support{};
    std::size_t best_grid_score{};


    // Clash threshold diagnostic
    std::array<std::size_t, 4>
        poses_below_clash_threshold{};

    std::array<double, 4>
        best_rmsd_below_clash_threshold{
            std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::infinity()
        };


    // Absolute clash diagnostic
    std::size_t poses_zero_clash{};
    std::size_t poses_one_clash{};
    std::size_t poses_two_clashes{};
    std::size_t poses_three_to_five_clashes{};
    std::size_t poses_over_five_clashes{};

    std::size_t best_grid_score_zero_clash{};

    double best_zero_clash_rmsd =
        std::numeric_limits<double>::infinity();

    std::vector<std::size_t>
        zero_clash_grid_histogram;


    double generation_seconds{};

    PoseSearchTimings timings;
};


bool rigid_transform_from_points(
    const std::array<interaction::point3D, 3>& source,
    const std::array<interaction::point3D, 3>& target,
    RigidTransform& transform
);

PoseGenerationResult generate_candidate_poses_from_index(
    const RDKit::ROMol& ligand,
    const std::vector<MatchingArea>& ligand_matching_areas,
    const std::vector<Triplet>& ligand_triplets,
    const std::vector<PoseTarget>& pose_targets,
    const ProteinGrid& protein_grid,
    const FlatTripletIndex& protein_triplet_index,
    double triplet_bin_size,
    double operational_clash_threshold,
    const std::array<double, 4>& clash_thresholds
);


PoseGenerationResult generate_candidate_poses(
    const RDKit::ROMol& ligand,
    const std::vector<MatchingArea>& ligand_matching_areas,
    const std::vector<Triplet>& ligand_triplets,
    const std::vector<PoseTarget>& pose_targets,
    const ProteinGrid& protein_grid,
    double ligand_diameter,
    double triplet_bin_size,
    double operational_clash_threshold,
    const std::array<double, 4>& clash_thresholds
);

#endif