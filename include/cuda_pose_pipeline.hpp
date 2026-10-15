#ifndef CUDA_POSE_PIPELINE_HPP
#define CUDA_POSE_PIPELINE_HPP

#include "cuda_backend.hpp"

#include <cstdint>
#include <vector>


struct GpuPipelinePose
{
    std::uint32_t alignment_index{};

    GpuRigidTransform transform{};

    GpuClashScore clash{};

    GpuFastScore fast_score{};
};


struct GpuPosePipelineResult
{
    std::uint32_t input_alignment_count{};
    std::uint32_t valid_transform_count{};
    std::uint32_t clash_survivor_count{};

    std::vector<GpuPipelinePose> poses;
};

bool cuda_run_pose_pipeline(
    const GpuPointArrayDevice& ligand_matching_points,
    const GpuPointArrayDevice& protein_target_points,
    const GpuPointArrayDevice& ligand_atoms,
    const GpuFastMatchingAreaArrayDevice& ligand_fast_areas,
    const GpuProteinGridDevice& protein_grid,
    const GpuAlignmentCandidateArrayDevice& alignments,
    double clash_threshold,
    GpuPosePipelineResult& result
);


bool cuda_run_pose_pipeline(
    const GpuPointArrayDevice& ligand_matching_points,
    const GpuPointArrayDevice& protein_target_points,
    const GpuPointArrayDevice& ligand_atoms,
    const GpuFastMatchingAreaArrayDevice& ligand_fast_areas,
    const GpuProteinGridDevice& protein_grid,
    const std::vector<GpuAlignmentCandidate>& alignments,
    double clash_threshold,
    GpuPosePipelineResult& result
);


#endif