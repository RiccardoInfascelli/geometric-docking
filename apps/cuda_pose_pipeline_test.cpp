#include "cuda_pose_pipeline.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>


namespace
{

bool close(
    double a,
    double b,
    double tolerance = 1e-10
)
{
    return
        std::abs(a - b) <=
        tolerance;
}


std::size_t grid_index(
    int x,
    int y,
    int z
)
{
    constexpr int nx = 4;
    constexpr int ny = 4;

    return
        static_cast<std::size_t>(
            x +
            nx * (
                y +
                ny * z
            )
        );
}

} // namespace


int main()
{
    // ========================================================
    // Ligand matching triangle
    // ========================================================

    const std::vector<GpuPoint3>
        ligand_matching_points = {

            {0.0, 0.0, 0.0},
            {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0}
        };


    GpuPointArrayDevice
        device_ligand_matching_points;


    if (
        !cuda_upload_points(
            ligand_matching_points,
            device_ligand_matching_points
        )
    ) {
        std::cerr
            << "Ligand matching point upload failed\n";

        return 1;
    }


    // ========================================================
    // Protein target triangles
    //
    // 0,1,2 -> ligand triangle translated +1 X
    // 3,4,5 -> ligand triangle translated +2 X
    // ========================================================

    const std::vector<GpuPoint3>
        protein_target_points = {

            {1.0, 0.0, 0.0},
            {2.0, 0.0, 0.0},
            {1.0, 1.0, 0.0},

            {2.0, 0.0, 0.0},
            {3.0, 0.0, 0.0},
            {2.0, 1.0, 0.0}
        };


    GpuPointArrayDevice
        device_protein_target_points;


    if (
        !cuda_upload_points(
            protein_target_points,
            device_protein_target_points
        )
    ) {
        std::cerr
            << "Protein target point upload failed\n";

        cuda_free_points(
            device_ligand_matching_points
        );

        return 1;
    }


    // ========================================================
    // Alignment candidates
    // ========================================================

    GpuAlignmentCandidate
        alignment0;

    alignment0.ligand_indices[0] = 0;
    alignment0.ligand_indices[1] = 1;
    alignment0.ligand_indices[2] = 2;

    alignment0.protein_indices[0] = 0;
    alignment0.protein_indices[1] = 1;
    alignment0.protein_indices[2] = 2;


    GpuAlignmentCandidate
        alignment1;

    alignment1.ligand_indices[0] = 0;
    alignment1.ligand_indices[1] = 1;
    alignment1.ligand_indices[2] = 2;

    alignment1.protein_indices[0] = 3;
    alignment1.protein_indices[1] = 4;
    alignment1.protein_indices[2] = 5;


    const std::vector<GpuAlignmentCandidate>
        alignments = {

            alignment0,
            alignment1
        };


    // ========================================================
    // Ligand atoms
    //
    // For +1 X:
    //
    // atom 0 -> voxel (1,0,0), free
    // atom 1 -> voxel (2,0,0), occupied
    // atom 2 -> outside grid
    //
    // clash = 1/3
    //
    // For +2 X:
    //
    // atom 0 -> voxel (2,0,0), occupied
    // atom 1 -> voxel (3,0,0), occupied
    // atom 2 -> outside grid
    //
    // clash = 2/3
    // ========================================================

    const std::vector<GpuPoint3>
        ligand_atoms = {

            {0.2, 0.2, 0.2},
            {1.2, 0.2, 0.2},
            {10.0, 10.0, 10.0}
        };


    GpuPointArrayDevice
        device_ligand_atoms;


    if (
        !cuda_upload_points(
            ligand_atoms,
            device_ligand_atoms
        )
    ) {
        std::cerr
            << "Ligand atom upload failed\n";

        cuda_free_points(
            device_protein_target_points
        );

        cuda_free_points(
            device_ligand_matching_points
        );

        return 1;
    }


    // ========================================================
    // Protein grid
    // ========================================================

    GpuProteinGridHost
        host_grid;


    host_grid.xmin = 0.0;
    host_grid.ymin = 0.0;
    host_grid.zmin = 0.0;

    host_grid.voxel_size = 1.0;

    host_grid.nx = 4;
    host_grid.ny = 4;
    host_grid.nz = 4;

    host_grid.voxels.resize(
        4 * 4 * 4
    );


    constexpr std::uint16_t
        interaction_bit =
            std::uint16_t{1} << 0;


    /*
     * Free interaction voxel used by
     * surviving +1 X pose.
     */
    host_grid
        .voxels[
            grid_index(
                1,
                0,
                0
            )
        ]
        .interactions =
            interaction_bit;


    /*
     * Clash voxels.
     */
    host_grid
        .voxels[
            grid_index(
                2,
                0,
                0
            )
        ]
        .occupied = 1;


    host_grid
        .voxels[
            grid_index(
                3,
                0,
                0
            )
        ]
        .occupied = 1;


    GpuProteinGridDevice
        device_grid;


    if (
        !cuda_upload_protein_grid(
            host_grid,
            device_grid
        )
    ) {
        std::cerr
            << "Protein grid upload failed\n";

        cuda_free_points(
            device_ligand_atoms
        );

        cuda_free_points(
            device_protein_target_points
        );

        cuda_free_points(
            device_ligand_matching_points
        );

        return 1;
    }


    // ========================================================
    // Fast interaction areas
    //
    // For +1 X:
    //
    // area 0 -> (1,0,0): free + compatible
    // area 1 -> (2,0,0): occupied
    // area 2 -> outside
    // ========================================================

    const std::vector<GpuFastMatchingArea>
        host_fast_areas = {

            {
                {0.2, 0.2, 0.2},
                interaction_bit,
                {},
                1.5
            },

            {
                {1.2, 0.2, 0.2},
                interaction_bit,
                {},
                2.5
            },

            {
                {10.0, 10.0, 10.0},
                interaction_bit,
                {},
                3.5
            }
        };


    GpuFastMatchingAreaArrayDevice
        device_fast_areas;


    if (
        !cuda_upload_fast_matching_areas(
            host_fast_areas,
            device_fast_areas
        )
    ) {
        std::cerr
            << "Fast-area upload failed\n";

        cuda_free_protein_grid(
            device_grid
        );

        cuda_free_points(
            device_ligand_atoms
        );

        cuda_free_points(
            device_protein_target_points
        );

        cuda_free_points(
            device_ligand_matching_points
        );

        return 1;
    }


    // ========================================================
    // Pipeline
    //
    // threshold 0.5:
    //
    // +1 X -> 1/3 -> survives
    // +2 X -> 2/3 -> rejected
    // ========================================================

    GpuPosePipelineResult
        result;


    if (
        !cuda_run_pose_pipeline(
            device_ligand_matching_points,
            device_protein_target_points,
            device_ligand_atoms,
            device_fast_areas,
            device_grid,
            alignments,
            0.5,
            result
        )
    ) {
        std::cerr
            << "CUDA pose pipeline failed\n";

        cuda_free_fast_matching_areas(
            device_fast_areas
        );

        cuda_free_protein_grid(
            device_grid
        );

        cuda_free_points(
            device_ligand_atoms
        );

        cuda_free_points(
            device_protein_target_points
        );

        cuda_free_points(
            device_ligand_matching_points
        );

        return 1;
    }


    // ========================================================
    // Pipeline-level checks
    // ========================================================

    if (
        result.input_alignment_count != 2 ||
        result.valid_transform_count != 2 ||
        result.clash_survivor_count != 1 ||
        result.poses.size() != 1
    ) {
        std::cerr
            << "Unexpected pipeline counts\n";

        return 1;
    }


    const auto& pose =
        result.poses[0];


    if (
        pose.alignment_index != 0
    ) {
        std::cerr
            << "Unexpected surviving alignment\n";

        return 1;
    }


    // ========================================================
    // Kabsch check
    // ========================================================

    if (
        !close(
            pose.transform.translation[0],
            1.0
        ) ||
        !close(
            pose.transform.translation[1],
            0.0
        ) ||
        !close(
            pose.transform.translation[2],
            0.0
        )
    ) {
        std::cerr
            << "Unexpected pipeline Kabsch transform\n";

        return 1;
    }


    std::cout
        << "CUDA pipeline Kabsch stage passed\n";


    // ========================================================
    // Clash check
    // ========================================================

    if (
        pose.clash.occupied_atoms != 1 ||
        pose.clash.total_atoms != 3 ||
        !close(
            pose.clash.fraction,
            1.0 / 3.0
        )
    ) {
        std::cerr
            << "Unexpected pipeline clash result\n";

        return 1;
    }


    std::cout
        << "CUDA pipeline clash filtering passed\n";


    // ========================================================
    // Fast scoring check
    // ========================================================

    if (
        pose.fast_score.total_areas != 3 ||
        pose.fast_score.areas_in_grid != 2 ||
        pose.fast_score
            .areas_on_occupied_voxel != 1 ||
        pose.fast_score.compatible_areas != 1 ||
        !close(
            pose.fast_score.weighted_score,
            1.5
        )
    ) {
        std::cerr
            << "Unexpected pipeline fast score\n";

        return 1;
    }


    std::cout
        << "CUDA pipeline fast scoring passed\n";


    // ========================================================
    // Cleanup
    // ========================================================

    cuda_free_fast_matching_areas(
        device_fast_areas
    );

    cuda_free_protein_grid(
        device_grid
    );

    cuda_free_points(
        device_ligand_atoms
    );

    cuda_free_points(
        device_protein_target_points
    );

    cuda_free_points(
        device_ligand_matching_points
    );


    std::cout
        << "CUDA pose pipeline tests passed\n";


    return 0;
}