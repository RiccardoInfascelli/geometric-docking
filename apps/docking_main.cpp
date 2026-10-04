#include "molecule_io.hpp"
#include "interaction_areas.hpp"
#include "geometry.hpp"
#include "matching_areas.hpp"
#include "triplet_generation.hpp"
#include "protein_grid.hpp"
#include "pose_targets.hpp"
#include "pose_generation.hpp"

#include <interaction/find_interaction.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>


namespace
{

// ============================================================
// Constants
// ============================================================

constexpr double GRID_VOXEL_SIZE = 0.5;
constexpr double GRID_MARGIN = 6.5;
constexpr double TRIPLET_BIN_SIZE = 0.5;


// ============================================================
// Geometry utility
// ============================================================

double point_distance(
    const interaction::point3D& a,
    const interaction::point3D& b)
{
    return std::sqrt(
        static_cast<double>(
            interaction::distance2(a, b)
        )
    );
}


// ============================================================
// Protein-grid diagnostics
// ============================================================

std::size_t count_occupied_voxels(
    const ProteinGrid& grid)
{
    std::size_t count = 0;

    for (const auto& voxel : grid.voxels) {

        if (voxel.occupied)
            ++count;
    }

    return count;
}


std::size_t count_interaction_voxels(
    const ProteinGrid& grid)
{
    std::size_t count = 0;

    for (const auto& voxel : grid.voxels) {

        if (voxel.interactions != 0)
            ++count;
    }

    return count;
}


std::size_t count_multi_interaction_voxels(
    const ProteinGrid& grid)
{
    std::size_t count = 0;

    for (const auto& voxel : grid.voxels) {

        InteractionMask mask =
            voxel.interactions;

        std::size_t bits = 0;

        while (mask != 0) {

            bits +=
                static_cast<std::size_t>(
                    mask & InteractionMask{1}
                );

            mask >>= 1;
        }

        if (bits > 1)
            ++count;
    }

    return count;
}


// ============================================================
// Crystal diagnostic for PoseTargets
//
// Ground truth is used ONLY here for evaluation.
// ============================================================

void print_pose_target_crystal_diagnostic(
    const std::vector<PoseTarget>& pose_targets,
    const std::vector<MatchingArea>& pocket_matching_areas,
    const std::vector<MatchingArea>& ligand_matching_areas,
    const std::vector<interaction::formed_interaction>& exact_interactions)
{
    std::size_t exact_with_target = 0;
    std::size_t exact_without_target = 0;

    std::unordered_set<std::size_t>
        exact_ligand_areas;

    std::unordered_set<std::size_t>
        retained_ligand_areas;


    double target_error_sum = 0.0;
    double target_error_max = 0.0;

    double original_error_sum = 0.0;
    double quality_target_error_sum = 0.0;

    std::size_t quality_count = 0;

    std::size_t improved = 0;
    std::size_t worsened = 0;
    std::size_t unchanged = 0;


    for (const auto& formed :
         exact_interactions) {

        const std::size_t protein_index =
            formed.index_protein_interaction_area;

        const std::size_t ligand_index =
            formed.index_ligand_interaction_area;


        if (
            ligand_index >=
            ligand_matching_areas.size()
        ) {
            continue;
        }


        exact_ligand_areas.insert(
            ligand_index
        );


        const InteractionKey required_key =
            ligand_matching_areas[
                ligand_index
            ].key;


        // ----------------------------------------------------
        // Corresponding PoseTarget
        // ----------------------------------------------------

        const PoseTarget* found_target =
            nullptr;


        for (const auto& target :
             pose_targets) {

            if (
                target.source_area_index ==
                    protein_index
                &&
                target.key ==
                    required_key
            ) {

                found_target =
                    &target;

                break;
            }
        }


        if (found_target == nullptr) {

            ++exact_without_target;

            continue;
        }


        ++exact_with_target;

        retained_ligand_areas.insert(
            ligand_index
        );


        const auto& crystal_position =
            ligand_matching_areas[
                ligand_index
            ].centroid;


        const double target_error =
            point_distance(
                found_target->position,
                crystal_position
            );


        target_error_sum +=
            target_error;

        target_error_max =
            std::max(
                target_error_max,
                target_error
            );


        // ----------------------------------------------------
        // Original MatchingArea
        // ----------------------------------------------------

        const MatchingArea* original_area =
            nullptr;


        for (const auto& area :
             pocket_matching_areas) {

            if (
                area.source_area_index ==
                    protein_index
                &&
                area.key ==
                    required_key
            ) {

                original_area =
                    &area;

                break;
            }
        }


        if (original_area == nullptr)
            continue;


        const double original_error =
            point_distance(
                original_area->centroid,
                crystal_position
            );


        original_error_sum +=
            original_error;

        quality_target_error_sum +=
            target_error;

        ++quality_count;


        constexpr double EPS = 1e-6;


        if (
            target_error <
            original_error - EPS
        ) {

            ++improved;
        }
        else if (
            target_error >
            original_error + EPS
        ) {

            ++worsened;
        }
        else {

            ++unchanged;
        }
    }


    // ========================================================
    // Coverage
    // ========================================================

    std::cout
        << "\n=== Pose target / crystal diagnostic ===\n"

        << "Exact interactions: "
        << exact_interactions.size()
        << '\n'

        << "Exact interactions with target: "
        << exact_with_target
        << '\n'

        << "Exact interactions without target: "
        << exact_without_target
        << '\n'

        << "Exact ligand areas with target: "
        << retained_ligand_areas.size()
        << " / "
        << exact_ligand_areas.size()
        << '\n';


    if (exact_with_target > 0) {

        std::cout
            << "Mean target error: "
            << (
                target_error_sum /
                static_cast<double>(
                    exact_with_target
                )
            )
            << " A\n"

            << "Max target error: "
            << target_error_max
            << " A\n";
    }


    // ========================================================
    // Repositioning quality
    // ========================================================

    if (quality_count > 0) {

        std::cout
            << "\n=== Repositioning quality ===\n"

            << "Mean original error: "
            << (
                original_error_sum /
                static_cast<double>(
                    quality_count
                )
            )
            << " A\n"

            << "Mean target error: "
            << (
                quality_target_error_sum /
                static_cast<double>(
                    quality_count
                )
            )
            << " A\n"

            << "Improved interactions: "
            << improved
            << " / "
            << quality_count
            << '\n'

            << "Worsened interactions: "
            << worsened
            << " / "
            << quality_count
            << '\n'

            << "Unchanged interactions: "
            << unchanged
            << " / "
            << quality_count
            << '\n';
    }
}

} // namespace


// ============================================================
// MAIN
// ============================================================

int main(int argc, char* argv[])
{
    try {

        std::cout
            << std::setprecision(6);


        // ====================================================
        // Input
        // ====================================================

        std::filesystem::path complex_dir =
            "/workspace/datasets/refined-set/1a1e";


        if (argc > 1) {

            complex_dir =
                std::filesystem::absolute(
                    argv[1]
                ).lexically_normal();


            if (complex_dir.filename().empty()) {

                complex_dir =
                    complex_dir.parent_path();
            }
        }


        const std::string complex_id =
            complex_dir.filename().string();


        const std::string protein_path =
            (
                complex_dir /
                (complex_id + "_protein.pdb")
            ).string();


        const std::string ligand_path =
            (
                complex_dir /
                (complex_id + "_ligand.mol2")
            ).string();


        const std::string pocket_path =
            (
                complex_dir /
                (complex_id + "_pocket.pdb")
            ).string();


        // ====================================================
        // Load molecules
        // ====================================================

        auto protein =
            load_protein(
                protein_path
            );

        auto ligand =
            load_ligand(
                ligand_path
            );

        auto pocket =
            load_pocket(
                pocket_path
            );


        std::cout
            << "=== Molecules ===\n"

            << "Protein atoms: "
            << protein->getNumAtoms()
            << '\n'

            << "Pocket atoms: "
            << pocket->getNumAtoms()
            << '\n'

            << "Ligand atoms: "
            << ligand->getNumAtoms()
            << '\n';


        // ====================================================
        // Interaction areas
        // ====================================================

        const auto protein_areas =
            get_protein_areas(
                *protein
            );


        const auto ligand_areas =
            get_ligand_areas(
                *ligand
            );


        const auto protein_matching_areas =
            build_protein_matching_areas(
                protein_areas
            );


        const auto ligand_matching_areas =
            build_ligand_matching_areas(
                ligand_areas
            );


        std::cout
            << "\n=== Interaction areas ===\n"

            << "Protein interaction areas: "
            << protein_areas.size()
            << '\n'

            << "Ligand interaction areas: "
            << ligand_areas.size()
            << '\n'

            << "Protein matching areas: "
            << protein_matching_areas.size()
            << '\n'

            << "Ligand matching areas: "
            << ligand_matching_areas.size()
            << '\n';


        // ====================================================
        // Ligand geometry
        // ====================================================

        const double ligand_diameter =
            compute_ligand_diameter(
                ligand_areas
            );


        const auto ligand_triplets =
            generate_triplets(
                ligand_matching_areas,
                ligand_diameter
            );


        std::cout
            << "\n=== Ligand geometry ===\n"

            << "Ligand diameter: "
            << ligand_diameter
            << " A\n"

            << "Ligand triplets: "
            << ligand_triplets.size()
            << '\n';


        // ====================================================
        // Protein Interaction Grid
        // ====================================================

        const GridBounds pocket_bounds =
            compute_pocket_bounds(
                *pocket
            );


        auto protein_grid =
            build_empty_protein_grid(
                *pocket,
                GRID_VOXEL_SIZE,
                GRID_MARGIN
            );


        mark_protein_occupancy(
            *protein,
            protein_grid
        );


        mark_interaction_regions(
            protein_matching_areas,
            protein_grid
        );


        const std::size_t occupied_voxels =
            count_occupied_voxels(
                protein_grid
            );


        const std::size_t interaction_voxels =
            count_interaction_voxels(
                protein_grid
            );


        const std::size_t multi_interaction_voxels =
            count_multi_interaction_voxels(
                protein_grid
            );


        std::cout
            << "\n=== Protein grid ===\n"

            << "Pocket size: "
            << (
                pocket_bounds.xmax -
                pocket_bounds.xmin
            )
            << " x "
            << (
                pocket_bounds.ymax -
                pocket_bounds.ymin
            )
            << " x "
            << (
                pocket_bounds.zmax -
                pocket_bounds.zmin
            )
            << " A\n"

            << "Grid dimensions: "
            << protein_grid.nx
            << " x "
            << protein_grid.ny
            << " x "
            << protein_grid.nz
            << '\n'

            << "Total voxels: "
            << protein_grid.voxels.size()
            << '\n'

            << "Occupied voxels: "
            << occupied_voxels
            << '\n'

            << "Interaction voxels: "
            << interaction_voxels
            << '\n'

            << "Multi-interaction voxels: "
            << multi_interaction_voxels
            << '\n';


        // ====================================================
        // Crystal ligand on fast grid
        //
        // Oracle diagnostic only.
        // ====================================================

        const auto crystal_grid_score =
            score_ligand_on_grid(
                ligand_matching_areas,
                protein_grid
            );


        std::cout
            << "\n=== Crystal ligand / fast grid ===\n"

            << "Areas in grid: "
            << crystal_grid_score.areas_in_grid
            << " / "
            << crystal_grid_score.total_areas
            << '\n'

            << "Areas on occupied voxel: "
            << crystal_grid_score
                   .areas_on_occupied_voxel
            << '\n'

            << "Grid-compatible ligand areas: "
            << crystal_grid_score.compatible_areas
            << " / "
            << crystal_grid_score.total_areas
            << '\n';


        // ====================================================
        // Exact crystal interactions
        //
        // Oracle diagnostic only.
        // ====================================================

        const auto exact_interactions =
            interaction::find_interactions(
                protein_areas,
                ligand_areas
            );


        std::unordered_set<std::size_t>
            exact_ligand_area_indices;


        for (const auto& formed :
             exact_interactions) {

            exact_ligand_area_indices.insert(
                formed
                    .index_ligand_interaction_area
            );
        }


        std::cout
            << "\n=== Crystal exact interactions ===\n"

            << "Exact interactions: "
            << exact_interactions.size()
            << '\n'

            << "Exact interacting ligand areas: "
            << exact_ligand_area_indices.size()
            << '\n';


        // ====================================================
        // Strict pocket MatchingAreas
        // ====================================================

        const auto pocket_matching_areas =
            filter_matching_areas_in_pocket(
                protein_matching_areas,
                protein_areas,
                *pocket
            );


        // ====================================================
        // Pose targets
        // ====================================================

        const auto pose_targets =
            generate_pose_targets(
                pocket_matching_areas,
                protein_grid,
                *protein,
                *pocket
            );


        std::cout
            << "\n=== Pose targets ===\n"

            << "Protein matching areas: "
            << protein_matching_areas.size()
            << '\n'

            << "Matching areas in pocket: "
            << pocket_matching_areas.size()
            << '\n'

            << "Pose targets generated: "
            << pose_targets.size()
            << '\n'

            << "Pocket areas without target: "
            << (
                pocket_matching_areas.size() -
                pose_targets.size()
            )
            << '\n';


        print_pose_target_crystal_diagnostic(
            pose_targets,
            pocket_matching_areas,
            ligand_matching_areas,
            exact_interactions
        );


        // ====================================================
        // Protein PoseTarget triplet hash
        // ====================================================

        const auto protein_triplet_hash =
            build_protein_triplet_hash_table(
                pose_targets,
                ligand_diameter,
                TRIPLET_BIN_SIZE
            );


        std::size_t protein_triplet_records =
            0;


        for (const auto& [key, bucket] :
             protein_triplet_hash) {

            (void)key;

            protein_triplet_records +=
                bucket.size();
        }


        std::cout
            << "\n=== Protein target triplets ===\n"

            << "Pose targets: "
            << pose_targets.size()
            << '\n'

            << "Protein triplets indexed: "
            << protein_triplet_records
            << '\n'

            << "Hash keys: "
            << protein_triplet_hash.size()
            << '\n';


        // ====================================================
        // 27-neighbor lookup diagnostic
        // ====================================================

        std::size_t ligand_triplets_with_match =
            0;

        std::size_t total_candidate_matches =
            0;

        std::size_t largest_bucket =
            0;


        for (const auto& ligand_triplet :
             ligand_triplets) {

            const auto canonical =
                canonicalize_triplet(
                    ligand_triplet,
                    TRIPLET_BIN_SIZE
                );


            const auto buckets =
                find_neighbor_triplet_buckets(
                    protein_triplet_hash,
                    canonical
                );


            if (buckets.empty())
                continue;


            ++ligand_triplets_with_match;


            for (const auto* bucket :
                 buckets) {

                total_candidate_matches +=
                    bucket->size();


                largest_bucket =
                    std::max(
                        largest_bucket,
                        bucket->size()
                    );
            }
        }


        std::cout
            << "\n=== 27-neighbor triplet lookup ===\n"

            << "Ligand triplets: "
            << ligand_triplets.size()
            << '\n'

            << "Ligand triplets with match: "
            << ligand_triplets_with_match
            << '\n'

            << "Candidate matches: "
            << total_candidate_matches
            << '\n'

            << "Largest bucket: "
            << largest_bucket
            << '\n';


        // ====================================================
        // Candidate pose generation
        //
        // 27-neighbor lookup
        // -> equivalent ligand orders
        // -> Kabsch best-fit
        // -> interaction-grid score
        // -> fast occupancy clash
        // ====================================================

        std::size_t transforms_generated =
            0;

        std::size_t transforms_valid =
            0;

        std::size_t poses_with_grid_support =
            0;

        std::size_t best_grid_score =
            0;


        // ----------------------------------------------------
        // Fast clash histogram
        // ----------------------------------------------------

        std::size_t poses_zero_clash =
            0;

        std::size_t poses_one_clash =
            0;

        std::size_t poses_two_clashes =
            0;

        std::size_t poses_three_to_five_clashes =
            0;

        std::size_t poses_over_five_clashes =
            0;


        std::size_t best_grid_score_zero_clash =
            0;


        // ----------------------------------------------------
        // Oracle RMSD diagnostic
        // ----------------------------------------------------

        double best_zero_clash_rmsd =
            std::numeric_limits<double>::infinity();


        // ----------------------------------------------------
        // Grid-score histogram for zero-clash poses
        //
        // index = compatible_areas
        // ----------------------------------------------------

        std::vector<std::size_t>
            grid_score_histogram(
                ligand_matching_areas.size() + 1,
                0
            );


        const auto& ligand_conformer =
            ligand->getConformer();


        // ====================================================
        // Candidate traversal
        // ====================================================

        for (const auto& ligand_triplet :
             ligand_triplets) {

            const auto canonical =
                canonicalize_triplet(
                    ligand_triplet,
                    TRIPLET_BIN_SIZE
                );


            const auto buckets =
                find_neighbor_triplet_buckets(
                    protein_triplet_hash,
                    canonical
                );


            for (const auto* bucket :
                 buckets) {

                for (const auto& protein_record :
                     *bucket) {


                    // ----------------------------------------
                    // Protein target triangle
                    //
                    // ProteinTripletRecord stores indices
                    // into pose_targets in canonical order.
                    // ----------------------------------------

                    const std::array<
                        interaction::point3D,
                        3
                    > target_points = {

                        pose_targets[
                            protein_record.indices[0]
                        ].position,

                        pose_targets[
                            protein_record.indices[1]
                        ].position,

                        pose_targets[
                            protein_record.indices[2]
                        ].position
                    };


                    // ----------------------------------------
                    // Equivalent ligand correspondences
                    // ----------------------------------------

                    for (
                        std::size_t order_index = 0;
                        order_index <
                            canonical
                                .equivalent_order_count;
                        ++order_index
                    ) {

                        const auto& ligand_order =
                            canonical
                                .equivalent_orders[
                                    order_index
                                ];


                        const std::array<
                            interaction::point3D,
                            3
                        > source_points = {

                            ligand_matching_areas[
                                ligand_order[0]
                            ].centroid,

                            ligand_matching_areas[
                                ligand_order[1]
                            ].centroid,

                            ligand_matching_areas[
                                ligand_order[2]
                            ].centroid
                        };


                        ++transforms_generated;


                        // ------------------------------------
                        // Kabsch best-fit
                        // ------------------------------------

                        RigidTransform transform;


                        if (
                            !rigid_transform_from_points(
                                source_points,
                                target_points,
                                transform
                            )
                        ) {

                            continue;
                        }


                        ++transforms_valid;


                        // ------------------------------------
                        // Transform interaction centroids
                        // ------------------------------------

                        const auto transformed_areas =
                            transform_matching_areas(
                                ligand_matching_areas,
                                transform
                            );


                        const auto grid_score =
                            score_ligand_on_grid(
                                transformed_areas,
                                protein_grid
                            );


                        if (
                            grid_score.compatible_areas > 0
                        ) {

                            ++poses_with_grid_support;
                        }


                        best_grid_score =
                            std::max(
                                best_grid_score,
                                grid_score.compatible_areas
                            );


                        // ------------------------------------
                        // Transform ligand atoms.
                        //
                        // At the same time:
                        // 1. fast occupancy clash
                        // 2. oracle RMSD diagnostic
                        // ------------------------------------

                        std::size_t occupied_ligand_atoms =
                            0;


                        double rmsd_sum_squared =
                            0.0;


                        for (
                            unsigned int atom_index = 0;
                            atom_index <
                                ligand->getNumAtoms();
                            ++atom_index
                        ) {

                            const auto position =
                                ligand_conformer.getAtomPos(
                                    atom_index
                                );


                            const interaction::point3D
                                original_point{

                                static_cast<
                                    interaction::coordinate_type
                                >(position.x),

                                static_cast<
                                    interaction::coordinate_type
                                >(position.y),

                                static_cast<
                                    interaction::coordinate_type
                                >(position.z)
                            };


                            const auto transformed_point =
                                transform_point(
                                    original_point,
                                    transform
                                );


                            // --------------------------------
                            // Oracle RMSD
                            // --------------------------------

                            const double dx =
                                static_cast<double>(
                                    transformed_point.x
                                ) -
                                position.x;

                            const double dy =
                                static_cast<double>(
                                    transformed_point.y
                                ) -
                                position.y;

                            const double dz =
                                static_cast<double>(
                                    transformed_point.z
                                ) -
                                position.z;


                            rmsd_sum_squared +=
                                dx * dx +
                                dy * dy +
                                dz * dz;


                            // --------------------------------
                            // Fast occupancy clash
                            // --------------------------------

                            const int gx =
                                protein_grid.to_grid(
                                    transformed_point.x,
                                    protein_grid.xmin
                                );

                            const int gy =
                                protein_grid.to_grid(
                                    transformed_point.y,
                                    protein_grid.ymin
                                );

                            const int gz =
                                protein_grid.to_grid(
                                    transformed_point.z,
                                    protein_grid.zmin
                                );


                            if (
                                !protein_grid.in_bounds(
                                    gx,
                                    gy,
                                    gz
                                )
                            ) {

                                continue;
                            }


                            const std::size_t voxel_index =
                                protein_grid.index(
                                    gx,
                                    gy,
                                    gz
                                );


                            if (
                                protein_grid
                                    .voxels[
                                        voxel_index
                                    ]
                                    .occupied
                            ) {

                                ++occupied_ligand_atoms;
                            }
                        }


                        // ====================================
                        // Clash distribution
                        // ====================================

                        if (
                            occupied_ligand_atoms == 0
                        ) {

                            ++poses_zero_clash;


                            best_grid_score_zero_clash =
                                std::max(
                                    best_grid_score_zero_clash,
                                    grid_score.compatible_areas
                                );


                            // --------------------------------
                            // Oracle RMSD
                            // --------------------------------

                            const double rmsd =
                                std::sqrt(
                                    rmsd_sum_squared /
                                    static_cast<double>(
                                        ligand->getNumAtoms()
                                    )
                                );


                            best_zero_clash_rmsd =
                                std::min(
                                    best_zero_clash_rmsd,
                                    rmsd
                                );


                            ++grid_score_histogram[
                                grid_score.compatible_areas
                            ];
                        }
                        else if (
                            occupied_ligand_atoms == 1
                        ) {

                            ++poses_one_clash;
                        }
                        else if (
                            occupied_ligand_atoms == 2
                        ) {

                            ++poses_two_clashes;
                        }
                        else if (
                            occupied_ligand_atoms <= 5
                        ) {

                            ++poses_three_to_five_clashes;
                        }
                        else {

                            ++poses_over_five_clashes;
                        }
                    }
                }
            }
        }


        // ====================================================
        // Candidate-pose summary
        // ====================================================

        std::cout
            << "\n=== Candidate pose generation ===\n"

            << "Transforms generated: "
            << transforms_generated
            << '\n'

            << "Valid transforms: "
            << transforms_valid
            << '\n'

            << "Poses with grid support: "
            << poses_with_grid_support
            << '\n'

            << "Best grid-compatible areas: "
            << best_grid_score
            << " / "
            << ligand_matching_areas.size()
            << '\n';


        // ====================================================
        // Fast clash summary
        // ====================================================

        std::cout
            << "\n=== Fast atom clash diagnostic ===\n"

            << "0 occupied atoms: "
            << poses_zero_clash
            << '\n'

            << "1 occupied atom: "
            << poses_one_clash
            << '\n'

            << "2 occupied atoms: "
            << poses_two_clashes
            << '\n'

            << "3-5 occupied atoms: "
            << poses_three_to_five_clashes
            << '\n'

            << ">5 occupied atoms: "
            << poses_over_five_clashes
            << '\n'

            << "Best grid score with 0 occupied atoms: "
            << best_grid_score_zero_clash
            << " / "
            << ligand_matching_areas.size()
            << '\n';


        // ====================================================
        // Zero-clash pose quality
        //
        // RMSD is oracle diagnostic only.
        // ====================================================

        std::cout
            << "\n=== Zero-clash pose quality ===\n"

            << "Zero-clash poses: "
            << poses_zero_clash
            << '\n';


        if (
            std::isfinite(
                best_zero_clash_rmsd
            )
        ) {

            std::cout
                << "Best RMSD to crystal: "
                << best_zero_clash_rmsd
                << " A\n";
        }
        else {

            std::cout
                << "Best RMSD to crystal: N/A\n";
        }


        std::cout
            << "\nGrid score distribution:\n";


        for (
            std::size_t grid_score = 0;
            grid_score <
                grid_score_histogram.size();
            ++grid_score
        ) {

            if (
                grid_score_histogram[
                    grid_score
                ] == 0
            ) {

                continue;
            }


            std::cout
                << "  "
                << grid_score
                << " / "
                << ligand_matching_areas.size()
                << ": "
                << grid_score_histogram[
                    grid_score
                ]
                << '\n';
        }


        return 0;
    }
    catch (const std::exception& e) {

        std::cerr
            << "Error: "
            << e.what()
            << '\n';

        return 1;
    }
}