#include "cuda_pose_pipeline.hpp"

#include <cuda_runtime.h>

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
    const GpuAlignmentCandidateArrayDevice& alignments,
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


    result.input_alignment_count =
        alignments.count;


    if (alignments.count == 0) {
        return true;
    }


    if (alignments.alignments == nullptr) {
        return false;
    }


    // ========================================================
    // 1. Kabsch - consume alignment candidates directly on GPU
    // ========================================================

    GpuRigidTransformArrayDevice
        device_transforms{};


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
        alignments.count
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
        device_clash_scores{};


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


    // Full pre-compaction arrays are no longer needed.
    cuda_free_clash_scores(
        device_clash_scores
    );


    cuda_free_rigid_transforms(
        device_transforms
    );


    if (
        device_survivors.count == 0
    ) {
        cuda_free_clash_survivors(
            device_survivors
        );

        return true;
    }


    // ========================================================
    // 4. Device-resident fast interaction scoring
    // ========================================================

    const GpuRigidTransformArrayDevice
        survivor_transform_view{
            device_survivors.transforms,
            device_survivors.count
        };


    GpuFastScoreArrayDevice
        device_fast_scores{};


    if (
        !cuda_score_fast_interactions_device(
            ligand_fast_areas,
            protein_grid,
            survivor_transform_view,
            device_fast_scores
        )
    ) {
        cuda_free_clash_survivors(
            device_survivors
        );

        return false;
    }


    if (
        device_fast_scores.count !=
        device_survivors.count
    ) {
        cuda_free_fast_scores(
            device_fast_scores
        );

        cuda_free_clash_survivors(
            device_survivors
        );

        return false;
    }


    // ========================================================
    // Final survivor download
    // ========================================================

    std::vector<GpuRigidTransform>
        survivor_transforms;


    std::vector<std::uint32_t>
        survivor_alignment_indices;


    std::vector<GpuClashScore>
        survivor_clash_scores;


    std::vector<GpuFastScore>
        fast_scores;


    if (
        !cuda_download_clash_survivors(
            device_survivors,
            survivor_transforms,
            survivor_clash_scores,
            survivor_alignment_indices
        )
    ) {
        cuda_free_fast_scores(
            device_fast_scores
        );

        cuda_free_clash_survivors(
            device_survivors
        );

        return false;
    }


    if (
        !cuda_download_fast_scores(
            device_fast_scores,
            fast_scores
        )
    ) {
        cuda_free_fast_scores(
            device_fast_scores
        );

        cuda_free_clash_survivors(
            device_survivors
        );

        return false;
    }


    cuda_free_fast_scores(
        device_fast_scores
    );


    cuda_free_clash_survivors(
        device_survivors
    );


    if (
        survivor_transforms.size() !=
            result.clash_survivor_count ||

        survivor_clash_scores.size() !=
            result.clash_survivor_count ||

        survivor_alignment_indices.size() !=
            result.clash_survivor_count ||

        fast_scores.size() !=
            result.clash_survivor_count
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


// ============================================================
// Legacy host-vector API
//
// Used by the existing synthetic pipeline test.
// The production/device path above does not perform this
// host-to-device alignment copy.
// ============================================================

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
    if (
        alignments.size() >
        std::numeric_limits<
            std::uint32_t
        >::max()
    ) {
        return false;
    }


    /*
     * Preserve the index validation performed by the
     * previous host-vector Kabsch path.
     */
    for (const auto& alignment :
         alignments) {

        for (int i = 0; i < 3; ++i) {

            if (
                alignment.ligand_indices[i] >=
                    ligand_matching_points.count ||

                alignment.protein_indices[i] >=
                    protein_target_points.count
            ) {
                return false;
            }
        }
    }


    GpuAlignmentCandidateArrayDevice
        device_alignments{};


    if (!alignments.empty()) {

        const std::size_t bytes =
            alignments.size() *
            sizeof(GpuAlignmentCandidate);


        if (
            cudaMalloc(
                reinterpret_cast<void**>(
                    &device_alignments.alignments
                ),
                bytes
            ) != cudaSuccess
        ) {
            return false;
        }


        device_alignments.count =
            static_cast<std::uint32_t>(
                alignments.size()
            );


        if (
            cudaMemcpy(
                device_alignments.alignments,
                alignments.data(),
                bytes,
                cudaMemcpyHostToDevice
            ) != cudaSuccess
        ) {
            cuda_free_alignment_candidates(
                device_alignments
            );

            return false;
        }
    }


    const bool ok =
        cuda_run_pose_pipeline(
            ligand_matching_points,
            protein_target_points,
            ligand_atoms,
            ligand_fast_areas,
            protein_grid,
            device_alignments,
            clash_threshold,
            result
        );


    cuda_free_alignment_candidates(
        device_alignments
    );


    return ok;
}