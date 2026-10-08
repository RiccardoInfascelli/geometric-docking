#include "geometry.hpp"
#include "interaction_areas.hpp"
#include "matching_areas.hpp"
#include "molecule_io.hpp"
#include "pose_generation.hpp"
#include "pose_targets.hpp"
#include "protein_grid.hpp"
#include "run_config.hpp"
#include "triplet_generation.hpp"

#include <interaction/elements.hpp>

#include <GraphMol/Atom.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>


namespace
{

double atom_vdw_radius(
    const RDKit::Atom& atom)
{
    const auto element =
        interaction::parse_chemical_element(
            atom.getSymbol()
        );

    if (element.has_value()) {
        return interaction::get(
            element.value()
        ).vdw_radius;
    }

    return 1.7;
}


double distance_to_protein_surface(
    const interaction::point3D& point,
    const RDKit::ROMol& protein)
{
    const auto& conformer =
        protein.getConformer();

    double best_distance =
        std::numeric_limits<double>::infinity();


    for (
        unsigned int atom_index = 0;
        atom_index < protein.getNumAtoms();
        ++atom_index
    ) {
        const auto position =
            conformer.getAtomPos(
                atom_index
            );

        const double dx =
            static_cast<double>(point.x) -
            position.x;

        const double dy =
            static_cast<double>(point.y) -
            position.y;

        const double dz =
            static_cast<double>(point.z) -
            position.z;

        const double center_distance =
            std::sqrt(
                dx * dx +
                dy * dy +
                dz * dz
            );

        const double radius =
            atom_vdw_radius(
                *protein.getAtomWithIdx(
                    atom_index
                )
            );

        best_distance =
            std::min(
                best_distance,
                center_distance - radius
            );
    }

    return best_distance;
}


std::size_t threshold_index(
    const std::array<double, 4>& thresholds,
    double wanted)
{
    for (
        std::size_t i = 0;
        i < thresholds.size();
        ++i
    ) {
        if (
            std::abs(
                thresholds[i] -
                wanted
            ) < 1e-9
        ) {
            return i;
        }
    }

    throw std::runtime_error(
        "Requested clash threshold not found"
    );
}

} // namespace


int main(int argc, char* argv[])
{
    try {

        const auto config =
            parse_run_config(
                argc,
                argv
            );


        // ====================================================
        // Same target preparation as production
        // ====================================================

        auto protein =
            load_protein(
                config.protein_path.string()
            );

        auto pocket =
            load_pocket(
                config.pocket_path.string()
            );


        const auto protein_areas =
            get_protein_areas(
                *protein
            );


        const auto protein_matching_areas =
            build_protein_matching_areas(
                protein_areas
            );


        auto protein_grid =
            build_empty_protein_grid(
                *pocket,
                config.grid_voxel_size,
                config.grid_margin
            );


        mark_protein_occupancy(
            *protein,
            protein_grid
        );


        mark_interaction_regions(
            protein_matching_areas,
            protein_grid
        );


        const auto pocket_matching_areas =
            filter_matching_areas_near_pocket(
                protein_matching_areas,
                *pocket,
                config.pocket_centroid_distance
            );


        const auto pose_targets =
            generate_pose_targets(
                pocket_matching_areas,
                protein_grid,
                *protein,
                *pocket
            );


        // ====================================================
        // Clearance of each generated PoseTarget
        //
        // generate_pose_targets() already chose the
        // maximum-clearance voxel for each area.
        // ====================================================

        std::vector<double>
            target_clearances;

        target_clearances.reserve(
            pose_targets.size()
        );


        for (const auto& target :
             pose_targets) {

            target_clearances.push_back(
                distance_to_protein_surface(
                    target.position,
                    *protein
                )
            );
        }


        // ====================================================
        // Ligand preparation - identical for every experiment
        // ====================================================

        auto ligand =
            load_ligand(
                config.ligand_path.string()
            );


        const auto ligand_areas =
            get_ligand_areas(
                *ligand
            );


        const auto ligand_matching_areas =
            build_ligand_matching_areas(
                ligand_areas
            );


        const double ligand_diameter =
            compute_ligand_diameter(
                ligand_areas
            );


        const auto ligand_triplets =
            generate_triplets(
                ligand_matching_areas,
                ligand_diameter
            );


        // ====================================================
        // Clash indices
        // ====================================================

        const std::size_t clash_5_index =
            threshold_index(
                config.clash_thresholds,
                0.05
            );


        const std::size_t clash_10_index =
            threshold_index(
                config.clash_thresholds,
                0.10
            );


        // ====================================================
        // Clearance pruning sweep
        // ====================================================

        constexpr std::array<double, 5>
            CLEARANCE_THRESHOLDS{
                0.00,
                0.75,
                1.00,
                1.25,
                1.50
            };


        std::cout
            << std::fixed
            << std::setprecision(6);


        std::cout
            << "=== CLEARANCE PRUNING EXPERIMENT ===\n\n"

            << "Protein atoms: "
            << protein->getNumAtoms()
            << '\n'

            << "Pocket matching areas: "
            << pocket_matching_areas.size()
            << '\n'

            << "Baseline pose targets: "
            << pose_targets.size()
            << '\n'

            << "Ligand areas: "
            << ligand_matching_areas.size()
            << '\n'

            << "Ligand triplets: "
            << ligand_triplets.size()
            << "\n\n";


        std::cout
            << "clearance"
            << "\ttargets"
            << "\tprotein_triplets"
            << "\tcandidate_matches"
            << "\ttransforms"
            << "\tsurvivors_5pct"
            << "\tbest_rmsd_5pct"
            << "\tbest_rmsd_10pct"
            << "\tindex_s"
            << "\tsearch_s"
            << '\n';


        for (
            const double clearance_threshold :
            CLEARANCE_THRESHOLDS
        ) {

            std::vector<PoseTarget>
                filtered_targets;

            filtered_targets.reserve(
                pose_targets.size()
            );


            for (
                std::size_t i = 0;
                i < pose_targets.size();
                ++i
            ) {
                if (
                    target_clearances[i] >=
                    clearance_threshold
                ) {
                    filtered_targets.push_back(
                        pose_targets[i]
                    );
                }
            }


            const auto result =
                generate_candidate_poses(
                    *ligand,
                    ligand_matching_areas,
                    ligand_triplets,
                    filtered_targets,
                    protein_grid,
                    ligand_diameter,
                    config.triplet_bin_size,
                    config.operational_clash_threshold,
                    config.clash_thresholds
                );


            std::cout
                << clearance_threshold
                << '\t'

                << filtered_targets.size()
                << '\t'

                << result.protein_triplet_records
                << '\t'

                << result.total_candidate_matches
                << '\t'

                << result.transforms_generated
                << '\t'

                << result
                    .poses_below_clash_threshold[
                        clash_5_index
                    ]
                << '\t'

                << result
                    .best_rmsd_below_clash_threshold[
                        clash_5_index
                    ]
                << '\t'

                << result
                    .best_rmsd_below_clash_threshold[
                        clash_10_index
                    ]
                << '\t'

                << result.timings
                    .protein_triplet_index_seconds
                << '\t'

                << result.generation_seconds
                << '\n';
        }


        return 0;
    }
    catch (const std::exception& error) {

        std::cerr
            << "Error: "
            << error.what()
            << '\n';

        return 1;
    }
}
