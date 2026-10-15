#include "cuda_pose_pipeline.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>


bool cuda_run_pose_pipeline(
    const GpuPointArrayDevice& ligand_matching_points,
    const GpuPointArrayDevice& protein_target_points,
    const GpuPointArrayDevice& ligand_atoms,
    const GpuFastMatchingAreaArrayDevice& ligand_fast_areas,
    const GpuProteinGridDevice& protein_grid,
    const std::vector<GpuAlignmentCandidate>& alignments,
    double clash_threshold,
    GpuPosePipelineResult& result
)
{
    result =
        GpuPosePipelineResult{};


    if (
        !std::isfinite(
            clash_threshold
        ) ||
        clash_threshold < 0.0 ||
        clash_threshold > 1.0
    ) {
        return false;
    }


    if (
        alignments.size() >
        std::numeric_limits<
            std::uint32_t
        >::max()
    ) {
        return false;
    }


    result.input_alignment_count =
        static_cast<std::uint32_t>(
            alignments.size()
        );


    if (alignments.empty()) {
        return true;
    }


        // ========================================================
    // 1. Kabsch - device resident
    // ========================================================

    GpuRigidTransformArrayDevice
        device_transforms;


    if (
        !cuda_compute_kabsch_transforms_device(
            ligand_matching_points,
            protein_target_points,
            alignments,
            device_transforms
        )
    ) {
        return false;
    }


    if (
        device_transforms.count !=
        alignments.size()
    ) {
        cuda_free_rigid_transforms(
            device_transforms
        );

        return false;
    }


    // ========================================================
    // 2. Grid clash - consume Kabsch output directly on GPU
    // ========================================================

    GpuClashScoreArrayDevice
        device_clash_scores;


    if (
        !cuda_score_grid_clash_device(
            ligand_atoms,
            protein_grid,
            device_transforms,
            device_clash_scores
        )
    ) {
        cuda_free_rigid_transforms(
            device_transforms
        );

        return false;
    }


    if (
        device_clash_scores.count !=
        device_transforms.count
    ) {
        cuda_free_clash_scores(
            device_clash_scores
        );

        cuda_free_rigid_transforms(
            device_transforms
        );

        return false;
    }


        // ========================================================
    // 3. GPU clash compaction
    //
    // Keep only valid transforms below the clash threshold.
    // The full transform/clash arrays remain on the GPU.
    // ========================================================

    GpuClashSurvivorArrayDevice
        device_survivors{};


    std::uint32_t
        valid_transform_count = 0;


    if (
        !cuda_compact_clash_survivors(
            device_transforms,
            device_clash_scores,
            clash_threshold,
            device_survivors,
            valid_transform_count
        )
    ) {
        cuda_free_clash_scores(
            device_clash_scores
        );

        cuda_free_rigid_transforms(
            device_transforms
        );

        return false;
    }


    result.valid_transform_count =
        valid_transform_count;


    result.clash_survivor_count =
        device_survivors.count;


    // Full Kabsch/clash arrays are no longer needed.
    cuda_free_clash_scores(
        device_clash_scores
    );


    cuda_free_rigid_transforms(
        device_transforms
    );


    if (
        result.clash_survivor_count == 0
    ) {
        cuda_free_clash_survivors(
            device_survivors
        );

        return true;
    }


    // ========================================================
    // Temporary survivor download
    //
    // Fast scoring still uses the legacy host-vector API.
    // Therefore only the compacted survivor set is downloaded.
    // ========================================================

    std::vector<GpuRigidTransform>
        survivor_transforms;


    std::vector<std::uint32_t>
        survivor_alignment_indices;


    std::vector<GpuClashScore>
        survivor_clash_scores;


    if (
        !cuda_download_clash_survivors(
            device_survivors,
            survivor_transforms,
            survivor_clash_scores,
            survivor_alignment_indices
        )
    ) {
        cuda_free_clash_survivors(
            device_survivors
        );

        return false;
    }


    cuda_free_clash_survivors(
        device_survivors
    );


    if (
        survivor_transforms.size() !=
            result.clash_survivor_count ||

        survivor_clash_scores.size() !=
            result.clash_survivor_count ||

        survivor_alignment_indices.size() !=
            result.clash_survivor_count
    ) {
        return false;
    }

    // ========================================================
    // 4. Fast interaction scoring
    // ========================================================

    std::vector<GpuFastScore>
        fast_scores;


    if (
        !cuda_score_fast_interactions(
            ligand_fast_areas,
            protein_grid,
            survivor_transforms,
            fast_scores
        )
    ) {
        return false;
    }


    if (
        fast_scores.size() !=
        survivor_transforms.size()
    ) {
        return false;
    }


    // ========================================================
    // 5. Assemble surviving poses
    // ========================================================

    result.poses.reserve(
        survivor_transforms.size()
    );


    for (
        std::size_t i = 0;
        i < survivor_transforms.size();
        ++i
    ) {
        if (
            fast_scores[i].valid == 0
        ) {
            return false;
        }


        GpuPipelinePose pose;


        pose.alignment_index =
            survivor_alignment_indices[i];

        pose.transform =
            survivor_transforms[i];

        pose.clash =
            survivor_clash_scores[i];

        pose.fast_score =
            fast_scores[i];


        result.poses.push_back(
            pose
        );
    }


    return true;
}