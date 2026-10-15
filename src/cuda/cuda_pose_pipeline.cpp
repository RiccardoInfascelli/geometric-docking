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
    // Temporary host download
    //
    // For now filtering remains on CPU.
    // The important change is that there is no host round-trip
    // between Kabsch and clash anymore.
    // ========================================================

    std::vector<GpuRigidTransform>
        transforms;


    std::vector<GpuClashScore>
        clash_scores;


    if (
        !cuda_download_rigid_transforms(
            device_transforms,
            transforms
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


    if (
        !cuda_download_clash_scores(
            device_clash_scores,
            clash_scores
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


    cuda_free_clash_scores(
        device_clash_scores
    );


    cuda_free_rigid_transforms(
        device_transforms
    );


    if (
        transforms.size() !=
        alignments.size() ||
        clash_scores.size() !=
        alignments.size()
    ) {
        return false;
    }


    for (const auto& transform :
         transforms) {

        if (transform.valid != 0) {

            ++result
                .valid_transform_count;
        }
    }


    // ========================================================
    // 3. Clash filtering
    //
    // Keep the mapping to the original alignment index.
    // ========================================================

    std::vector<GpuRigidTransform>
        survivor_transforms;

    std::vector<std::uint32_t>
        survivor_alignment_indices;

    std::vector<GpuClashScore>
        survivor_clash_scores;


    survivor_transforms.reserve(
        transforms.size()
    );

    survivor_alignment_indices.reserve(
        transforms.size()
    );

    survivor_clash_scores.reserve(
        transforms.size()
    );


    for (
        std::size_t i = 0;
        i < transforms.size();
        ++i
    ) {
        const auto& transform =
            transforms[i];

        const auto& clash =
            clash_scores[i];


        if (
            transform.valid == 0 ||
            clash.valid == 0
        ) {
            continue;
        }


        if (
            clash.fraction >
            clash_threshold
        ) {
            continue;
        }


        survivor_transforms.push_back(
            transform
        );

        survivor_alignment_indices.push_back(
            static_cast<std::uint32_t>(
                i
            )
        );

        survivor_clash_scores.push_back(
            clash
        );
    }


    result.clash_survivor_count =
        static_cast<std::uint32_t>(
            survivor_transforms.size()
        );


    if (
        survivor_transforms.empty()
    ) {
        return true;
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